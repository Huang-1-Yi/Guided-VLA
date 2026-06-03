from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """对数据应用 transform。

        Args:
            data: 待应用 transform 的数据。它可能是嵌套字典，包含未 batch 的数据元素。
                每个叶子节点都应是 numpy array。虽然可以使用 JAX array，但不建议这样做，
                因为它可能导致 data loader worker 进程中产生额外的 GPU 显存占用。

        Returns:
            transform 后的数据。可以是被原地修改的输入 `data`，也可以是新的数据结构。
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """一组 transforms。"""

    # 应用于模型输入数据的 transforms。
    inputs: Sequence[DataTransformFn] = ()

    # 应用于模型输出数据的 transforms。
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """向组中追加 transforms，并返回新的组。

        Args:
            inputs: 追加到当前 input transforms 的末尾。
            outputs: 追加到当前 output transforms 的开头。

        Returns:
            包含追加 transforms 的新组。
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """按顺序应用一组 transforms 的组合 transform。"""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """将一组 transforms 组合为单个 transform。"""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class ComputeSkillSoftLabel(DataTransformFn):
    """根据 action horizon 上的一段 skill ID 序列计算 soft skill 分布。

    实现 GuidedVLA 论文（Eq. 3-4）中的 ground-truth soft label y，用于表示未来 horizon H
    上的 skill 分布：
        y_k = count(skill_k in horizon) / H

    该 transform 期望 ``skill_id`` 是形状为 (H,) 的数组，包含通过 delta_timestamps 加载的
    整数 skill ID。它会产生：
      - ``skill_soft``: 形状为 (num_classes,) 的 float32 数组，表示归一化概率分布
      - ``skill_id``:   序列中间时间步对应的 int 标量
    """

    num_classes: int

    def __call__(self, data: DataDict) -> DataDict:
        skill_id_seq = data.get("skill_id")
        if skill_id_seq is None:
            return data

        skill_id_arr = np.asarray(skill_id_seq)

        if skill_id_arr.ndim == 0:
            current_skill = int(skill_id_arr)
            soft = np.zeros(self.num_classes, dtype=np.float32)
            soft[current_skill] = 1.0
        else:
            center_idx = skill_id_arr.shape[0] // 2
            current_skill = int(skill_id_arr[center_idx])
            counts = np.bincount(skill_id_arr.astype(np.int64), minlength=self.num_classes).astype(np.float32)
            soft = counts / counts.sum()

        data = dict(data)
        data["skill_soft"] = soft
        data["skill_id"] = current_skill
        return data


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """将输入字典重新打包为新的字典。

    repack 规则由一个字典定义：key 是新的 key，value 是旧 key 的扁平化路径。
    扁平化时使用 '/' 作为分隔符。

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }

    Args:
        structure: 将新 key 映射到旧扁平化路径的 PyTree。
        optional_keys: 可选的顶层 key 集合。如果源 key 不存在于数据中，会跳过该 key，
            而不是抛出 KeyError。
    """

    structure: at.PyTree[str]
    optional_keys: frozenset[str] = dataclasses.field(default_factory=frozenset)

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        result = {}
        for key, value in self.structure.items():
            if key in self.optional_keys:
                # 对可选 key，检查所有必需的源 key 是否存在。
                source_keys = jax.tree.leaves(value)
                if not all(k in flat_item for k in source_keys):
                    continue
            result[key] = jax.tree.map(lambda k: flat_item[k], value)
        return result


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # 为 true 时使用分位数归一化；否则使用普通 z-score 归一化。
    use_quantiles: bool = False
    # 为 true 时，如果 norm stats 中任意 key 不存在于数据中，就抛出错误。
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # 为 true 时使用分位数归一化；否则使用普通 z-score 归一化。
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # 确保 norm stats 中的所有 key 都存在于数据中。
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """将绝对动作重新打包到 delta action 空间。"""

    # 布尔 mask，用于指定哪些动作维度需要重新打包到 delta action 空间。
    # 长度可以小于实际维度数。若为 None，则该 transform 不执行任何操作。
    # 更多细节见 `make_bool_mask`。
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """将 delta actions 重新打包到绝对动作空间。"""

    # 布尔 mask，用于指定哪些动作维度需要重新打包到绝对动作空间。
    # 长度可以小于实际维度数。若为 None，则该 transform 不执行任何操作。
    # 更多细节见 `make_bool_mask`。
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # 模型输出保存在 "actions" 中，但对 FAST 模型来说它们表示 tokens。
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """从当前 LeRobot 数据集任务中提取 prompt。"""

    # 包含 LeRobot 数据集任务（dataset.meta.tasks）。
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        # 较新的 LeRobot 版本可能会直接暴露 task 字符串；较旧版本会暴露 task index，
        # 需要通过 dataset metadata 解析。
        if "task" in data:
            return {**data, "prompt": data["task"]}

        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task" or "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """将 states 和 actions 用 0 padding 到模型动作维度。"""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """扁平化嵌套字典，使用 '/' 作为分隔符。"""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """将扁平化字典还原为嵌套字典，假设使用 '/' 作为分隔符。"""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """使用一组 patterns 转换嵌套字典的结构。

    转换规则由 `patterns` 字典定义。key 是需要匹配的输入 key，value 是输出字典中的新名称。
    如果 value 为 None，则会移除对应输入 key。

    key 和 value 都应表示使用 '/' 分隔的扁平化路径。key 可以是正则表达式，
    value 可以包含对匹配组的反向引用（更多细节见 `re.sub`）。注意正则表达式必须匹配整个 key。

    `patterns` 字典中的顺序很重要。只会使用第一个匹配输入 key 的 pattern。

    更多示例见单元测试。

    Args:
        patterns: 从旧 key 到新 key 的映射。
        tree: 待转换的嵌套字典。

    Returns:
        转换后的嵌套字典。
    """
    data = flatten_dict(tree)

    # 编译 patterns。
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # 如果没有匹配项，则使用原始 key。
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # 校验输出结构，确保它可以被还原为嵌套字典。
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """沿指定 axis 将数组 padding 到目标维度。"""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """为给定维度生成布尔 mask。

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: 用于生成 mask 的维度。

    Returns:
        布尔值元组。
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
