from collections.abc import Sequence
from contextlib import nullcontext as _nullcontext
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """初始化 Policy。

        Args:
            model: 用于 action sampling 的模型。
            rng: JAX 模型使用的随机数生成器 key。PyTorch 模型会忽略该参数。
            transforms: 推理前应用的输入数据 transformations。
            output_transforms: 推理后应用的输出数据 transformations。
            sample_kwargs: 传给 model.sample_actions 的额外关键字参数。
            metadata: 随 policy 一起保存的额外 metadata。
            pytorch_device: PyTorch 模型使用的设备，例如 "cpu"、"cuda:0"。
                          仅当 is_pytorch=True 时相关。
            is_pytorch: 模型是否为 PyTorch 模型。若为 False，则视为 JAX 模型。
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX 模型设置。
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # 复制一份，因为 transformations 可能会原地修改输入。
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # 构造 batch 并转换为 jax.Array。
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # 将输入转换为 PyTorch tensors，并移动到正确设备。
            # 当输入已经是 numpy 时，使用 np.ascontiguousarray 避免不必要的拷贝。
            inputs = jax.tree.map(
                lambda x: torch.as_tensor(np.ascontiguousarray(x)).to(self._pytorch_device)[None, ...], inputs
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # 准备传给 sample_actions 的 kwargs。
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # 若 noise 是 (action_horizon, action_dim)，则添加 batch 维度。
                noise = noise[None, ...]  # 转为 (1, action_horizon, action_dim)。
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        _inference_ctx = torch.inference_mode() if self._is_pytorch_model else _nullcontext()
        # 当 torch.compile 使用 CUDAGraphs（mode="reduce-overhead"）时，上一次运行的 tensor 输出 buffer
        # 可能被下一次运行覆盖。每次调用前执行 cudagraph_mark_step_begin() 可告知 CUDAGraphs
        # 一个新的独立 step 即将开始，从而避免 stale-buffer 错误。
        if (
            self._is_pytorch_model
            and hasattr(torch, "compiler")
            and hasattr(torch.compiler, "cudagraph_mark_step_begin")
        ):
            torch.compiler.cudagraph_mark_step_begin()
        with _inference_ctx:
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
            }

        model_time = time.monotonic() - start_time

        # 将输出转换为 numpy。
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """将 policy 行为记录到磁盘。"""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
