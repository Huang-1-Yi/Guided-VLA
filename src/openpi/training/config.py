"""可用 config 列表见 _CONFIGS。"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.calvin_policy as calvin_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.attention_map as _attention_map
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# 规避 tyro 直接使用 nnx.filterlib.Filter 时的问题。
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """确定用于搭建 data pipeline 的 assets 位置，例如 norm stats。

    这些 assets 会复制到 checkpoint 内的 `assets/asset_id` 目录下。

    这可用于从其他 checkpoint（例如 base model checkpoint）或其他集中位置加载 assets。
    例如，微调时如果要从 base model checkpoint 加载 Trossen 机器人的 norm stats，可使用：

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets 目录。未提供时使用 config 的 assets_dirs。适合从其他 checkpoint
    #（例如 base model checkpoint）或集中位置加载 assets。
    assets_dir: str | None = None

    # Asset id。未提供时使用 repo id。它允许用户引用描述不同机器人平台的 assets。
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id。若为 None，则创建 fake data。
    repo_id: str | None = None
    # assets 目录中包含 data assets 的子目录。
    asset_id: str | None = None
    # 包含预计算的 normalization stats。若为 None，则不执行归一化。
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 将数据集特定输入适配为 data transforms 期望的通用格式。
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms，通常包含机器人特定转换，会在数据归一化前应用。
    # 归一化后的数据格式见 `model.Observation` 和 `model.Actions`。
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 模型特定 transforms，会在数据归一化后应用。
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 为 true 时使用分位数归一化；否则使用普通 z-score 归一化。
    use_quantile_norm: bool = False

    # 通过 `delta_timestamps` 按 horizon 长度序列加载的 key 名称。
    # 包括主 action tensor，以及必须与 action horizon 对齐的辅助标签，
    # 例如在线构造 `skill_soft` 时使用的 `observation.skill_id`。
    horizon_sequence_keys: Sequence[str] = ("actions",)

    # 为 true 时使用 LeRobot 数据集 task 定义 prompt。
    prompt_from_task: bool = False

    # 为 true 时，训练代码会使用数据集中的 attention/object maps 计算辅助 object-mask supervision loss。
    use_object_loss: bool = False

    # 为 true 时，训练代码会使用数据集中预计算的 soft skill labels 计算辅助 skill-level loss。
    use_skill_loss: bool = False

    # LeRobotDataset 的本地根目录。
    local_root_dir: str | None = None

    # 多数据集加载时使用的可选数据集条目。
    multi_datasets: Sequence[dict[str, str | None]] | None = None

    # 训练/验证切分比例。默认 0.9（90% train，10% val）。
    # 设为 1.0 时，全部数据用于训练（不切分验证集）。
    train_val_split: float = 0.93

    # 用于可复现 train/val 切分的随机种子。
    split_seed: int = 42

    # RLDS 专用 data loader 字段。
    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """创建 group。"""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """为标准 pi0 模型创建 model transforms。"""

    # 数据集样本未提供 prompt 时注入的默认 prompt。
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # LeRobot repo id。
    repo_id: str = tyro.MISSING
    # 决定 assets 的加载方式。
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # 将由 factory 更新的 base config。
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建 data config。"""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


def with_optional_skill_sequence_key(
    base_keys: Sequence[str],
    model_config: _model.BaseModelConfig,
) -> tuple[str, ...]:
    keys = tuple(base_keys)
    if getattr(model_config, "use_skill_loss", False) and "observation.skill_id" not in keys:
        return (*keys, "observation.skill_id")
    return keys


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # data transforms 的 factory。
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # model transforms 的 factory。
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # 为 true 时，将 joint 维度转换为相对当前 state 的 delta 后再传给模型。
    # gripper 维度保持绝对值。
    use_delta_joint_actions: bool = True
    # 若提供，且输入数据中不存在 "prompt" key，则注入该 prompt。
    default_prompt: str | None = None
    # 为 true 时，将 joint 和 gripper 值从标准 Aloha 空间转换到
    # pretrained model 使用的 base PI action normalization 空间。
    # 使用标准 Aloha 数据时应设为 true。
    adapt_to_pi: bool = False
    # ALOHA 数据的本地 LeRobot 数据集根目录。CLI/base_config 覆盖项仍优先。
    local_root_dir: str | None = None

    # Repack transforms。
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # 应从数据集中按 horizon 长度序列加载的 keys。
    horizon_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        base = self.create_base_config(assets_dirs, model_config)
        effective_root = base.local_root_dir if base.local_root_dir is not None else self.local_root_dir

        return dataclasses.replace(
            base,
            local_root_dir=effective_root,
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            horizon_sequence_keys=self.horizon_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobotwinDataConfig(DataConfigFactory):
    """RoboTwin 数据沿用 ALOHA observation 布局，但保持在 RoboTwin joint 空间中。

    policy 使用 14-D 绝对 joint targets，grippers 归一化到 [0, 1]。
    训练保持 `use_delta_joint_actions=True`，推理时再将模型输出转回绝对 RoboTwin joint targets。
    """

    repo_id: str = "robotwin"
    base_config: tyro.conf.Suppress[DataConfig | None] = dataclasses.field(
        default_factory=lambda: DataConfig(prompt_from_task=True)
    )

    # 为 true 时，将 joint 维度转换为相对当前 state 的 delta 后再传给模型。
    # gripper 维度保持绝对值。
    use_delta_joint_actions: bool = True
    # 若提供，且输入数据中不存在 "prompt" key，则注入该 prompt。
    default_prompt: str | None = None
    # RoboTwin 数据的本地 LeRobot 数据集根目录。CLI/base_config 覆盖项仍优先。
    local_root_dir: str | None = None
    # 可选的多数据集加载配置。
    multi_datasets: Sequence[dict[str, str | None]] | None = None
    # 应从数据集中按 horizon 长度序列加载的 keys。
    horizon_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        input_transforms: list = []
        if model_config.use_skill_loss:
            num_classes = model_config.skill_num_classes
            input_transforms.append(_transforms.ComputeSkillSoftLabel(num_classes=num_classes))
        input_transforms.append(aloha_policy.AlohaInputs(adapt_to_pi=False))
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=False)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        base = self.create_base_config(assets_dirs, model_config)
        effective_root = base.local_root_dir if base.local_root_dir is not None else self.local_root_dir
        effective_multi_datasets = self.multi_datasets if self.multi_datasets is not None else base.multi_datasets

        return dataclasses.replace(
            base,
            local_root_dir=effective_root,
            multi_datasets=effective_multi_datasets,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                            "skill_id": "observation.skill_id",
                            "attention_map": {key: key for key in _attention_map.ROBOTWIN_OBJECT_MAP_KEY_TO_VIEW},
                        },
                        optional_keys=frozenset(["attention_map", "skill_id"]),
                    )
                ]
            ),
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            horizon_sequence_keys=with_optional_skill_sequence_key(self.horizon_sequence_keys, model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """LIBERO 风格 LeRobot 数据集使用的 data transforms。"""

    extra_delta_transform: bool = False

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                        "skill_id": "observation.skill_id",
                        "attention_map": {key: key for key in _attention_map.LIBERO_OBJECT_MAP_KEY_TO_VIEW},
                    },
                    optional_keys=frozenset(["attention_map", "skill_id"]),
                )
            ]
        )
    )

    # 应从数据集中按 horizon 长度序列加载的 keys。
    horizon_sequence_keys: Sequence[str] = ("actions",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        input_transforms: list = []
        # 启用 skill loss 时，根据通过 delta_timestamps 在 action horizon 上加载的 skill_id 序列
        # 计算 soft skill labels（GuidedVLA Eq. 4）。
        if getattr(model_config, "use_skill_loss", False):
            num_classes = getattr(model_config, "skill_num_classes", 8)
            input_transforms.append(_transforms.ComputeSkillSoftLabel(num_classes=num_classes))
        input_transforms.append(libero_policy.LiberoInputs(model_type=model_config.model_type))
        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[libero_policy.LiberoOutputs()],
        )

        # LIBERO actions 已经是 deltas；该选项用于支持训练时额外做过 delta conversion 的 checkpoint。
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            horizon_sequence_keys=with_optional_skill_sequence_key(self.horizon_sequence_keys, model_config),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # 可选字典，用于将 RLDS episode ids 映射到需要保留的 timestep 范围。
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader 返回绝对 joint position actions，这里转换为 delta actions 用于训练。
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # 这里假设使用 joint *velocity* actions，因此不应再额外应用 delta transform。
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotCalvinDataConfig(DataConfigFactory):
    """CALVIN LeRobot 数据集的数据 transform。

    默认数据集是 InternRobotics/InternData-Calvin_ABC，它将 CALVIN observations 和 actions
    存储在拆分字段中。policy adapter 会把这些字段打包到 OpenPI 的标准 state/action key 中。
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "image": "video.image_base",
                        "wrist_image": "video.image_wrist",
                        "state_ee_pos": "state.ee_pos",
                        "state_ee_rot": "state.ee_rot",
                        "state_gripper": "state.gripper",
                        "action_delta_ee_pos": "action.delta_ee_pos",
                        "action_delta_ee_rot": "action.delta_ee_rot",
                        "action_gripper": "action.gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[calvin_policy.CalvinInputs(model_type=model_config.model_type)],
            outputs=[calvin_policy.CalvinOutputs()],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            horizon_sequence_keys=("action.delta_ee_pos", "action.delta_ee_rot", "action.gripper"),
            use_quantile_norm=False,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # config 名称，必须唯一，用于引用该 config。
    name: tyro.conf.Suppress[str]
    # 项目名称。
    project_name: str = "guidedvla"
    # 实验名称，用于命名 metadata 和 checkpoint 目录。
    exp_name: str = tyro.MISSING

    # 定义模型 config。部分属性（action_dim、action_horizon、max_token_len）由所有模型共享，
    # 见 BaseModelConfig。具体模型实现（例如 Pi0Config）继承 BaseModelConfig，
    # 并可定义额外属性。
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # weight loader 可在模型初始化后，从磁盘可选加载权重（可以是部分权重）。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # 可选的 PyTorch checkpoint 路径，用于加载权重。
    pytorch_weight_path: str | None = None

    # PyTorch 训练精度。
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"
    # 启用 PyTorch 训练的 gradient checkpointing，以降低内存/显存占用。
    use_gradient_checkpointing: bool = False
    # DDP find_unused_parameters：False 可减少约 20% allreduce 开销，
    # 但要求所有参数都参与每次 forward/backward。当 control_attention 始终启用，
    # 且 skill/object 分支始终使用时，可以安全设为 False。
    ddp_find_unused_parameters: bool = False
    # 每个 backbone 的 LR 缩放：设为 <1.0（例如 0.1）可让新 head 比 backbone 学得更快。
    backbone_lr_scale: float = 1.0

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # 指定哪些权重应被冻结。
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 决定训练所用数据。
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # config assets 的基础目录，例如 norm stats。
    assets_base_dir: str = "./assets"
    # checkpoints 的基础目录。
    checkpoint_base_dir: str = "./checkpoints"

    # 训练中随机生成器使用的随机种子。
    seed: int = 42
    # 每个 optimizer step 的全局 batch size。
    batch_size: int = 32
    # data loader 使用的 worker 数量。增大该值可加快数据加载，但会增加内存和 CPU 使用。
    num_workers: int = 15
    # 运行的训练 step（batch）数量。
    num_train_steps: int = 30_000

    # 记录训练指标的频率（按 step）。
    log_interval: int = 100
    # 保存 checkpoint 的频率（按 step）。
    save_interval: int = 5000
    # 运行验证的频率（按 step）。若为 None，则默认使用 save_interval。
    val_interval: int | None = 500
    # 验证时最多使用的 batch 数。设为 None 时使用全部验证数据。
    val_max_batches: int = 8
    # 应用于 object-mask supervision loss 的权重。
    object_loss_weight: float = 0.1
    # 启用 skill auxiliary loss 时应用的权重。
    skill_loss_weight: float = 0.1
    # 若设置，任何满足 step % keep_period == 0 的已有 checkpoints 都不会被删除。
    keep_period: int | None = 5000

    # 为 true 时，如果 checkpoint 目录已存在则覆盖。
    overwrite: bool = False
    # 为 true 时，从最后一个 checkpoint 恢复训练。
    resume: bool = False

    # 为 true 时启用 wandb logging。
    wandb_enabled: bool = True

    # 用于向 policy server 传递 metadata。
    policy_metadata: dict[str, Any] | None = None

    # 仅 JAX trainer 使用：每个 FSDP group 跨多少设备进行 sharding。
    # PyTorch trainer 使用 DDP，会忽略该值。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """获取该 config 的 assets 目录。"""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """获取该 config 的 checkpoint 目录。"""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """获取 trainable parameters 的 filter。"""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# 如需在代码中按名称获取 config，请使用 `get_config`。
_CONFIGS = [
    #
    # Aloha 推理 configs。
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # DROID 推理 configs。
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Libero 微调 configs。
    #
    TrainConfig(
        name="pi0_libero",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",  # 替换为你的 LeRobot 数据集 repo
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        },
                    )
                ]
            ),
        ),
        pytorch_training_precision="float32",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_object_depth_skill",
        model=pi0_config.Pi0Config(
            guided_layer_indices=[9, 10, 11, 12],
            control_attention_enabled=True,
            control_attention_num_heads=8,
            use_object_loss=True,
            object_head_indices=[0, 1],
            object_use_control=False,
            use_depth=True,
            depth_model_name="path/to/da3-small",
            depth_head_indices=[4, 5],
            depth_use_control=True,
            use_skill_loss=True,
            skill_num_classes=4,
            skill_head_indices=[6, 7],
            skill_use_control=False,
        ),
        # 已发布的 GuidedVLA LIBERO 数据集，包含辅助 head 使用的 object labels 和 skill labels。
        data=LeRobotLiberoDataConfig(
            repo_id="ybwowen/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                use_object_loss=True,
                use_skill_loss=True,
            ),
            extra_delta_transform=True,
        ),
        pytorch_training_precision="float32",
        object_loss_weight=0.001,
        skill_loss_weight=0.001,
        batch_size=64,
        num_workers=8,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_object",
        model=pi0_config.Pi0Config(
            control_attention_enabled=True,
            control_attention_num_heads=8,
            guided_layer_indices=[9, 10, 11, 12],
            use_object_loss=True,
            object_use_control=True,
            object_head_indices=[0, 1],
        ),
        # 已发布的 GuidedVLA LIBERO 数据集，包含辅助 object head 使用的 object labels。
        data=LeRobotLiberoDataConfig(
            repo_id="ybwowen/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                use_object_loss=True,
            ),
            extra_delta_transform=True,
        ),
        pytorch_training_precision="float32",
        object_loss_weight=1e-3,
        num_workers=8,
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_depth",
        model=pi0_config.Pi0Config(
            control_attention_enabled=True,
            control_attention_num_heads=8,
            guided_layer_indices=[9, 10, 11, 12],
            use_depth=True,
            depth_use_control=True,
            depth_model_name="path/to/da3-small",  # 设为你的本地 DA3-SMALL checkpoint
            depth_head_indices=[4, 5],
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="ybwowen/libero",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        pytorch_training_precision="float32",
        batch_size=64,
        num_workers=8,
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_skill",
        model=pi0_config.Pi0Config(
            control_attention_enabled=True,
            control_attention_num_heads=8,
            guided_layer_indices=[9, 10, 11, 12],
            use_skill_loss=True,
            skill_num_classes=4,
            skill_head_indices=[6, 7],
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="ybwowen/libero",
            base_config=DataConfig(
                prompt_from_task=True,
                use_skill_loss=True,
            ),
            extra_delta_transform=True,
        ),
        skill_loss_weight=0.001,
        pytorch_training_precision="float32",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="your-hf-username/your-dataset",  # 替换为你的 LeRobot 数据集 repo
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    #
    # CALVIN 微调 configs。
    #
    TrainConfig(
        name="pi05_calvin",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotCalvinDataConfig(
            repo_id="InternRobotics/InternData-Calvin_ABC",
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        num_workers=8,
    ),
    #
    # Aloha 微调 configs。
    #
    # 在自定义 ALOHA LeRobot 数据集上微调的示例 config。
    # 数据转换和训练说明见 examples/aloha_real/README.md。
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # RoboTwin 微调 configs。这些配置与 RoboTwin 的 `policy/pi05` 分支对齐：
    # - 14-D 绝对 qpos actions
    # - ALOHA 风格 camera/state 布局
    # - RoboTwin-normalized grippers，因此 `adapt_to_pi=False`
    # - 通过 LeRobotRobotwinDataConfig 默认值保持 delta-action training 启用
    #
    TrainConfig(
        name="pi05_aloha_robotwin_full",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotRobotwinDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    TrainConfig(
        name="pi05_aloha_robotwin_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRobotwinDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
    ),
    # 与 RoboTwin `policy/pi05` 分支名称匹配的向后兼容别名。
    TrainConfig(
        name="pi05_aloha_full_base",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotRobotwinDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    TrainConfig(
        name="pi05_base_aloha_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRobotwinDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
    ),
    TrainConfig(
        name="pi0_base_aloha_robotwin_full",
        model=pi0_config.Pi0Config(),
        data=LeRobotRobotwinDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        pytorch_training_precision="float32",
        num_train_steps=30_000,
        batch_size=16,
        fsdp_devices=4,
    ),
    TrainConfig(
        name="pi0_base_aloha_robotwin_lora",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRobotwinDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
    ),
    TrainConfig(
        name="pi0_fast_aloha_robotwin_full",
        model=pi0_fast.Pi0FASTConfig(),
        data=LeRobotRobotwinDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_fast_aloha_robotwin_lora",
        model=pi0_fast.Pi0FASTConfig(
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotRobotwinDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        fsdp_devices=2,
    ),
    # PyTorch trainer 使用的 GuidedVLA 风格 RoboTwin configs。
    # Object supervision 使用可选的 `*_attention_object` maps。
    # skill config 还需要 `observation.skill_id`。
    TrainConfig(
        name="pi0_base_aloha_robotwin_object_depth_skill",
        model=pi0_config.Pi0Config(
            max_token_len=56,
            guided_layer_indices=[9, 10, 11, 12],
            control_attention_enabled=True,
            control_attention_num_heads=8,
            use_object_loss=True,
            object_head_indices=[0, 1],
            object_use_control=False,
            use_depth=True,
            depth_model_name="path/to/da3-small",  # 设为你的本地 DA3-SMALL checkpoint
            depth_head_indices=[4, 5],
            depth_use_control=True,
            use_skill_loss=True,
            skill_num_classes=8,  # 如果你的数据集 skill vocabulary 更小，请相应调整
            skill_head_indices=[6, 7],
            skill_use_control=False,
        ),
        data=LeRobotRobotwinDataConfig(
            base_config=DataConfig(
                prompt_from_task=True,
                use_object_loss=True,
                use_skill_loss=True,
            ),
        ),
        pytorch_training_precision="float32",
        object_loss_weight=0.01,
        skill_loss_weight=0.01,
        batch_size=32,
        num_workers=8,
        num_train_steps=30_000,
    ),
    # 用于多数据集加载路径的轻量 smoke-test config。
    TrainConfig(
        name="pi0_base_aloha_robotwin_full_multi",
        model=pi0_config.Pi0Config(),
        data=SimpleDataConfig(
            repo_id="fake",
            data_transforms=lambda _model_config: _transforms.Group(),
            model_transforms=lambda _model_config: _transforms.Group(),
            base_config=DataConfig(
                multi_datasets=(
                    {"repo_id": "fake"},
                    {"repo_id": "fake"},
                ),
            ),
        ),
        batch_size=4,
        num_train_steps=10,
        save_interval=100,
        overwrite=True,
        exp_name="robotwin_multi_debug",
        wandb_enabled=False,
    ),
    #
    # ALOHA Sim configs。该 config 用于演示如何在简单仿真环境中训练。
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # 调试 configs。
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # RoboArena configs。
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """按名称获取 config。"""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
