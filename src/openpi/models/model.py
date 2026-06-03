import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
import typing
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# array 类型的类型变量（JAX array、PyTorch tensor 或 numpy array）。
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """支持的模型类型。"""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# 模型始终期望这些图像。
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# 如果以后发布 small model，这里可能需要调整。
IMAGE_RESOLUTION = (224, 224)


def normalize_pytorch_state_dict_for_loading(
    state_dict: dict[str, torch.Tensor],
    *,
    source_label: str,
) -> dict[str, torch.Tensor]:
    """为 PyTorch checkpoint 归一化 torch.compile 包装片段和 tied weights。"""
    embed_tokens_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"

    normalized_state_dict = {}
    normalized_keys = 0
    for key, value in state_dict.items():
        normalized_key = ".".join(part for part in key.split(".") if part != "_orig_mod")
        if normalized_key != key:
            normalized_keys += 1
        if normalized_key in normalized_state_dict:
            raise ValueError(
                f"{source_label} contains duplicate logical key '{normalized_key}' "
                f"after stripping compile wrapper segments."
            )
        normalized_state_dict[normalized_key] = value

    if normalized_keys > 0:
        logger.info(
            "Normalized %d compiled key(s) from %s for compatibility.",
            normalized_keys,
            source_label,
        )

    if embed_tokens_key not in normalized_state_dict and lm_head_key in normalized_state_dict:
        logger.info(f"Copying tied weight: {lm_head_key} -> {embed_tokens_key}")
        normalized_state_dict[embed_tokens_key] = normalized_state_dict[lm_head_key]

    return normalized_state_dict


# 数据格式
#
# 数据 transform 会生成嵌套字典形式的模型输入，后续会转换为 `Observation` 和 `Actions` 对象。
# 见下方说明。
#
# 字典形式的数据应如下所示：
# {
#     # observation 数据。
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB 图像，范围为 [-1, 1] 或 [0, 255]
#         ...  # 额外相机视角
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # 图像有效时为 True
#         ...  # 额外视角的 mask
#     },
#     "state": float32[*b, s],  # 低维机器人状态
#     "tokenized_prompt": int32[*b, l],  # 可选，tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # 可选，tokenized prompt 的 mask
#     "token_ar_mask": int32[*b, l],  # 可选，FAST model 的 autoregressive mask
#     "token_loss_mask": bool[*b, l],  # 可选，FAST model 的 loss mask
#
#      # actions 数据。
#      "actions": float32[*b ah ad]
# }
# 其中：
#   *b = batch 维度
#   h,w = 图像高/宽
#   s = state 维度
#   l = sequence length
#
ImageArray: typing.TypeAlias = at.Float[ArrayT, "*b h w c"] | at.UInt8[ArrayT, "*b h w c"]


def _contains_cuda_tensor(tree: typing.Any) -> bool:
    if isinstance(tree, dict):
        return any(_contains_cuda_tensor(value) for value in tree.values())
    if isinstance(tree, list | tuple):
        return any(_contains_cuda_tensor(value) for value in tree)
    return isinstance(tree, torch.Tensor) and tree.is_cuda


@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """保存 observation，也就是模型输入。

    期望的字典形式见 `Observation.from_dict`。这是 data transforms 应该生成的格式。
    """

    # 图像：可能是归一化到 [-1, 1] 的 float32，也可能是 PyTorch 设备转移前的原始 uint8。
    images: dict[str, ImageArray]
    # 图像 mask，key 与 images 相同。
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # 低维机器人状态。
    state: at.Float[ArrayT, "*b s"]

    # 可选：每个 batch 元素的离散 skill id（chunk-level 或按帧聚合）。
    # 仅在某些微调设置中使用（例如 skill-level auxiliary heads）；
    # 不消费该字段的模型会忽略它。
    skill_id: at.Int[ArrayT, "*b"] | None = None

    # 可选：chunk-level soft skill label distribution，用于 soft-label supervision。
    # shape 通常为 [*b, K]，其中 K 是 skill 类别数。
    skill_soft: at.Float[ArrayT, "*b k"] | None = None

    # tokenized prompt。
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # tokenized prompt mask。
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model 专用字段。

    # token auto-regressive mask（用于 FAST autoregressive model）。
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # token loss mask（用于 FAST autoregressive model）。
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(
        cls,
        data: at.PyTree[ArrayT],
        *,
        normalize_torch_images: bool = True,
    ) -> "Observation[ArrayT]":
        """定义从非结构化数据（即嵌套 dict）到结构化 Observation 格式的映射。"""
        # 确保 tokenized_prompt 和 tokenized_prompt_mask 同时提供。
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # 如果图像是 uint8，则转换为 [-1, 1] float32。
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif (
                normalize_torch_images
                and hasattr(data["image"][key], "dtype")
                and data["image"][key].dtype == torch.uint8
            ):
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            elif (
                normalize_torch_images and isinstance(data["image"][key], torch.Tensor) and data["image"][key].ndim == 4
            ):
                image = data["image"][key].to(torch.float32)
                if image.shape[-1] == 3:
                    image = image.permute(0, 3, 1, 2)
                data["image"][key] = image
        kwargs = {
            "images": data["image"],
            "image_masks": data["image_mask"],
            "state": data["state"],
            "skill_id": data.get("skill_id"),
            "skill_soft": data.get("skill_soft"),
            "tokenized_prompt": data.get("tokenized_prompt"),
            "tokenized_prompt_mask": data.get("tokenized_prompt_mask"),
            "token_ar_mask": data.get("token_ar_mask"),
            "token_loss_mask": data.get("token_loss_mask"),
        }
        # jaxtyping/beartype 可能会在 generic dataclass 构造过程中拒绝有效的 CUDA torch.bool mask，
        # 即使同样的 tensor 在下游可以正常工作。其他路径仍保留运行时类型检查，
        # 但对这个仅 GPU 触发的路径绕过检查。
        if _contains_cuda_tensor(kwargs):
            with at.disable_typechecking():
                return cls(**kwargs)
        return cls(**kwargs)

    def to_dict(self) -> at.PyTree[ArrayT]:
        """将 Observation 转换为嵌套 dict。"""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# 定义 actions 的格式。该字段会以 "actions" 形式包含在 data transforms 生成的字典中。
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """预处理 observation：执行图像增强（如果 train=True）、必要时缩放，并在必要时填充默认 image mask。
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # 为 augmax 从 [-1, 1] 转换到 [0, 1]。
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # 转回 [-1, 1]。
            image = image * 2.0 - 1.0

        out_images[key] = image

    # 获取 mask。
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 默认不屏蔽。
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        skill_id=observation.skill_id,
        skill_soft=observation.skill_soft,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """所有模型共享的配置。

    具体模型应继承此类，并实现 `create` 方法以创建对应模型。
    """

    # action 空间维度。
    action_dim: int
    # action 序列长度。
    action_horizon: int
    # tokenized prompt 最大长度。
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """模型类型。"""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """创建新模型并初始化参数。"""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """使用给定参数创建模型。"""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")

        # 重要：使用 float32 创建 model config，避免权重加载期间发生精度损失。
        # 权重加载完成后，policy_config.py 会将模型转换为 bfloat16。
        model_config = dataclasses.replace(train_config.model, dtype="float32")

        model = pi0_pytorch.PI0Pytorch(config=model_config)

        # 在加载权重前注入 ControlAttention（inject-then-load）。
        # 推理所用 checkpoint 已经用 CA 训练，因此保存的 key 已具有
        # .origin./* 和 object_branch./* 结构，与注入后的模型匹配。
        if getattr(model_config, "control_attention_enabled", False):
            model.enable_control_attention()

        # 归一化 state dict：去掉 compile wrapper 片段，并修复 tied embed_tokens weight。
        state_dict = safetensors.torch.load_file(weight_path)
        new_state_dict = normalize_pytorch_state_dict_for_loading(
            state_dict,
            source_label="PyTorch policy checkpoint",
        )

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

        # 将 missing keys 分为预期缺失（depth/skill 模块可能不在 checkpoint 中）和非预期缺失。
        expected_missing, unexpected_missing = [], []
        for key in missing_keys:
            if key.startswith(("depth", "skill_head.")):
                expected_missing.append(key)
            else:
                unexpected_missing.append(key)

        if expected_missing:
            logger.debug(f"Missing keys (expected, new modules not yet in checkpoint): {expected_missing}")

        # 如果 config 与 checkpoint 的 control-attention 结构不匹配，则报错。
        ca_missing = [k for k in unexpected_missing if ".origin." in k or "object_branch" in k]
        ca_unexpected = [k for k in unexpected_keys if "object_branch" in k]
        if ca_missing or ca_unexpected:
            lines = []
            if ca_missing:
                lines.append(
                    f"  Config has control_attention_enabled=True but checkpoint is missing "
                    f"{len(ca_missing)} control-attention key(s), e.g. '{ca_missing[0]}'."
                )
            if ca_unexpected:
                lines.append(
                    f"  Config has control_attention_enabled=False but checkpoint contains "
                    f"{len(ca_unexpected)} control-attention key(s), e.g. '{ca_unexpected[0]}'."
                )
            raise ValueError(
                "Checkpoint structure does not match config:\n"
                + "\n".join(lines)
                + "\nUpdate the config's control_attention_enabled to match the checkpoint."
            )

        other_missing = [k for k in unexpected_missing if k not in ca_missing]
        other_unexpected = [k for k in unexpected_keys if k not in ca_unexpected]
        if other_missing:
            logger.warning(f"Missing keys (unexpected): {other_missing}")
        if other_unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {other_unexpected}")

        logger.info("Model loaded successfully.")
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """返回模型输入规格。值为 jax.ShapeDtypeStruct。"""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """所有模型实现的基类。

    具体模型应继承此类，并调用 super().__init__() 初始化共享属性
    （action_dim、action_horizon 和 max_token_len）。
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """从 checkpoint 恢复非结构化 params PyTree。

    这既适用于 openpi 训练过程中通过 `save_state` 保存的 checkpoint
    （见 `training/checkpoints.py`），也适用于 openpi 发布的预训练 checkpoint。

    Args:
        params_path: checkpoint 目录的本地路径。
        restore_type: 恢复 params 时使用的类型。可设为 `np.ndarray`，以 numpy array 形式加载参数。
        dtype: 恢复所有 params 时使用的 dtype。若未提供，则使用 checkpoint 中的原始 dtype。
        sharding: params 使用的 sharding。若未提供，则参数会复制到所有设备。

    Returns:
        恢复后的 params。
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
