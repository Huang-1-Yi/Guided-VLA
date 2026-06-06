from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from torch.utils.data import IterableDataset

from padp.common.normalize_util import get_identity_normalizer_from_stat
from padp.common.normalize_util import get_image_range_normalizer
from padp.common.normalize_util import get_range_normalizer_from_stat
from padp.data.libero_task_batch_adapter import LiberoTaskPadpBatchAdapter
from padp.data.libero_task_batch_adapter import TaskConditionSpec
from padp.data.openpi_libero_loader import OpenPiLiberoLoaderConfig
from padp.data.openpi_libero_loader import StreamingFeatureStats
from padp.data.openpi_libero_loader import create_openpi_libero_loader
from padp.data.openpi_libero_loader import load_padp_normalizer
from padp.data.openpi_libero_loader import save_padp_normalizer
from padp.data.openpi_libero_loader import update_streaming_stats
from padp.model.common.normalizer import LinearNormalizer


@dataclass(frozen=True)
class OpenPiLiberoTaskLoaderConfig(OpenPiLiberoLoaderConfig):
    task_num_tasks: int = 10
    task_obs_key: str = "task_onehot"
    task_id_source: str = "skill_id"
    strict_task_id: bool = True
    fallback_task_id: int | None = None


class OpenPiLiberoTaskPadpDataset(IterableDataset):
    """OpenPI LIBERO iterator that emits task-conditioned PADP batches."""

    def __init__(
        self,
        openpi_config_name: str = "pi0_libero_object",
        repo_id: str = "ybwowen/libero",
        local_root_dir: str | None = None,
        batch_size: int = 8,
        num_workers: int = 0,
        shuffle: bool = False,
        split: str = "train",
        num_batches: int | None = None,
        skip_norm_stats: bool = True,
        seed: int = 42,
        horizon: int = 40,
        n_obs_steps: int = 1,
        action_dim: int = 7,
        state_dim: int = 8,
        task_num_tasks: int = 10,
        task_obs_key: str = "task_onehot",
        task_id_source: str = "skill_id",
        strict_task_id: bool = True,
        fallback_task_id: int | None = None,
    ) -> None:
        self.config = OpenPiLiberoTaskLoaderConfig(
            openpi_config_name=openpi_config_name,
            repo_id=repo_id,
            local_root_dir=local_root_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            split=split,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            seed=seed,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            action_dim=action_dim,
            state_dim=state_dim,
            task_num_tasks=task_num_tasks,
            task_obs_key=task_obs_key,
            task_id_source=task_id_source,
            strict_task_id=strict_task_id,
            fallback_task_id=fallback_task_id,
        )
        self.adapter = LiberoTaskPadpBatchAdapter(
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            action_dim=action_dim,
            state_dim=state_dim,
            task=TaskConditionSpec(
                num_tasks=task_num_tasks,
                obs_key=task_obs_key,
                id_source=task_id_source,
                strict_task_id=strict_task_id,
                fallback_task_id=fallback_task_id,
            ),
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        loader = create_openpi_libero_loader(self.config)
        for openpi_batch in loader:
            yield self.adapter(openpi_batch)

    def get_normalizer(self, num_batches: int = 8, log_every: int | None = 100) -> LinearNormalizer:
        return fit_task_padp_normalizer(
            iter(self),
            num_batches=num_batches,
            log_every=log_every,
            task_obs_key=self.config.task_obs_key,
        )


def fit_task_padp_normalizer(
    batches: Iterator[dict[str, Any]],
    *,
    num_batches: int = 8,
    log_every: int | None = 100,
    task_obs_key: str = "task_onehot",
) -> LinearNormalizer:
    obs_accumulators: dict[str, StreamingFeatureStats] = {}
    image_keys: set[str] = set()
    action_accumulator: StreamingFeatureStats | None = None
    consumed_batches = 0

    for idx, batch in enumerate(batches):
        consumed_batches = idx + 1
        action = batch["action"].detach().cpu()
        action_accumulator = update_streaming_stats(action_accumulator, action, last_dim=1)

        for key, value in batch["obs"].items():
            if key.endswith("image"):
                image_keys.add(key)
                continue
            obs_accumulators[key] = update_streaming_stats(
                obs_accumulators.get(key),
                value.detach().cpu(),
                last_dim=1,
            )

        if log_every is not None and log_every > 0 and (idx + 1 == 1 or (idx + 1) % log_every == 0):
            print(f"  task normalizer batches: {idx + 1}/{num_batches}", flush=True)

        if idx + 1 >= num_batches:
            break

    if action_accumulator is None:
        raise RuntimeError("Cannot fit PADP task normalizer from an empty iterator.")

    if consumed_batches < num_batches:
        print(
            f"  task normalizer iterator ended after {consumed_batches} batches; requested {num_batches}.",
            flush=True,
        )

    normalizer = LinearNormalizer()
    normalizer["action"] = get_range_normalizer_from_stat(action_accumulator.to_stat())

    for key in sorted(image_keys):
        normalizer[key] = get_image_range_normalizer()

    for key, accumulator in sorted(obs_accumulators.items()):
        stat = accumulator.to_stat()
        if key.endswith("quat") or key == task_obs_key:
            normalizer[key] = get_identity_normalizer_from_stat(stat)
        else:
            normalizer[key] = get_range_normalizer_from_stat(stat)

    return normalizer


__all__ = [
    "OpenPiLiberoTaskPadpDataset",
    "OpenPiLiberoTaskLoaderConfig",
    "fit_task_padp_normalizer",
    "load_padp_normalizer",
    "save_padp_normalizer",
]
