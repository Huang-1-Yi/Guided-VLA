from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from padp.data.libero_batch_adapter import LiberoPadpBatchAdapter
from padp.data.libero_batch_adapter import ensure_torch
from padp.data.libero_batch_adapter import repeat_obs_steps
from padp.data.libero_batch_adapter import unpack_openpi_batch


@dataclass(frozen=True)
class TaskConditionSpec:
    """Preset task-conditioning contract for PADP-VA LIBERO batches."""

    num_tasks: int = 10
    obs_key: str = "task_onehot"
    id_source: str = "skill_id"
    strict_task_id: bool = True
    fallback_task_id: int | None = None


@dataclass(frozen=True)
class LiberoTaskPadpBatchAdapter:
    """Add a preset-size task one-hot low-dim observation to PADP batches."""

    horizon: int = 40
    n_obs_steps: int = 1
    action_dim: int = 7
    state_dim: int = 8
    task: TaskConditionSpec = TaskConditionSpec()

    def __post_init__(self) -> None:
        if self.task.num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {self.task.num_tasks}")
        object.__setattr__(
            self,
            "_base_adapter",
            LiberoPadpBatchAdapter(
                horizon=self.horizon,
                n_obs_steps=self.n_obs_steps,
                action_dim=self.action_dim,
                state_dim=self.state_dim,
            ),
        )

    def __call__(self, openpi_batch: Any) -> dict[str, Any]:
        obs, actions, extra = unpack_openpi_batch(openpi_batch)
        return self.adapt(obs, actions, extra=extra)

    def adapt(self, obs: Any, actions: torch.Tensor, extra: Any | None = None) -> dict[str, Any]:
        batch = self._base_adapter.adapt(obs, actions, extra=extra)
        batch_size = int(batch["action"].shape[0])
        task_ids = extract_task_ids(obs, extra, batch_size=batch_size, spec=self.task)
        task_onehot = torch.nn.functional.one_hot(task_ids, num_classes=self.task.num_tasks).to(torch.float32)
        batch["obs"][self.task.obs_key] = repeat_obs_steps(task_onehot, self.n_obs_steps)
        return batch


def extract_task_ids(
    obs: Any,
    extra: Any | None,
    *,
    batch_size: int,
    spec: TaskConditionSpec,
) -> torch.Tensor:
    value = _find_task_value(obs, extra, spec.id_source)
    if value is None:
        if spec.fallback_task_id is None:
            raise KeyError(
                f"Could not find task id source {spec.id_source!r} in openpi batch. "
                "Run padp.training.diagnose_libero_task_condition first, or set fallback_task_id."
            )
        value = spec.fallback_task_id

    task_ids = _coerce_task_ids(value, batch_size=batch_size)
    bad = (task_ids < 0) | (task_ids >= int(spec.num_tasks))
    if torch.any(bad):
        unique = sorted(int(x) for x in task_ids.unique().tolist())
        message = f"Task ids {unique} exceed preset num_tasks={spec.num_tasks}."
        if spec.strict_task_id:
            raise ValueError(message)
        task_ids = torch.clamp(task_ids, min=0, max=int(spec.num_tasks) - 1)
    return task_ids


def _find_task_value(obs: Any, extra: Any | None, source: str) -> Any | None:
    candidates = [source, "skill_id", "task_id", "task_index"]

    for key in candidates:
        if hasattr(obs, key):
            value = getattr(obs, key)
            if value is not None:
                return value

    if isinstance(obs, Mapping):
        for key in candidates:
            if key in obs and obs[key] is not None:
                return obs[key]

    if isinstance(extra, Mapping):
        for key in candidates:
            if key in extra and extra[key] is not None:
                return extra[key]

    return None


def _coerce_task_ids(value: Any, *, batch_size: int) -> torch.Tensor:
    tensor = ensure_torch(value).detach().cpu()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1).repeat(batch_size)
    elif tensor.ndim == 1:
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.repeat(batch_size)
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} task ids, got shape {tuple(tensor.shape)}")
    else:
        if tensor.shape[0] != batch_size:
            raise ValueError(f"Expected first task-id dim {batch_size}, got shape {tuple(tensor.shape)}")
        tensor = tensor.reshape(batch_size, -1)[:, 0]
    return tensor.round().to(torch.long)
