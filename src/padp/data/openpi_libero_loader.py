from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

from openpi.training import config as openpi_config
from openpi.training import data_loader as openpi_data_loader
from padp.common.normalize_util import array_to_stats
from padp.common.normalize_util import get_identity_normalizer_from_stat
from padp.common.normalize_util import get_image_range_normalizer
from padp.common.normalize_util import get_range_normalizer_from_stat
from padp.data.libero_batch_adapter import LiberoPadpBatchAdapter
from padp.model.common.normalizer import LinearNormalizer


@dataclass(frozen=True)
class OpenPiLiberoLoaderConfig:
    openpi_config_name: str = "pi0_libero_object"
    repo_id: str = "ybwowen/libero"
    local_root_dir: str | None = None
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = False
    split: str = "train"
    num_batches: int | None = None
    skip_norm_stats: bool = True
    seed: int = 42
    horizon: int = 40
    n_obs_steps: int = 1
    action_dim: int = 7
    state_dim: int = 8


class OpenPiLiberoPadpDataset(IterableDataset):
    """Batch iterator that reuses GuidedVLA/openpi LIBERO loading and emits PADP batches."""

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
    ) -> None:
        self.config = OpenPiLiberoLoaderConfig(
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
        )
        self.adapter = LiberoPadpBatchAdapter(
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            action_dim=action_dim,
            state_dim=state_dim,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        loader = create_openpi_libero_loader(self.config)
        for openpi_batch in loader:
            yield self.adapter(openpi_batch)

    def get_normalizer(self, num_batches: int = 8, log_every: int | None = 100) -> LinearNormalizer:
        return fit_padp_normalizer(iter(self), num_batches=num_batches, log_every=log_every)


def create_openpi_libero_loader(config: OpenPiLiberoLoaderConfig):
    configure_torch_dataloader_runtime(config.num_workers)
    train_config = make_openpi_train_config(config)
    return openpi_data_loader.create_data_loader(
        train_config,
        framework="pytorch",
        split=config.split,
        shuffle=config.shuffle,
        num_batches=config.num_batches,
        skip_norm_stats=config.skip_norm_stats,
    )


def configure_torch_dataloader_runtime(num_workers: int) -> None:
    if num_workers <= 0:
        return

    # Many large torch tensors are passed from workers to the parent process.
    # The default "file_descriptor" strategy can exhaust low ulimit -n values
    # when num_workers is high, especially with LIBERO image batches.
    torch.multiprocessing.set_sharing_strategy("file_system")
    os.environ.setdefault("DATALOADER_PREFETCH_FACTOR", "1")


def make_openpi_train_config(config: OpenPiLiberoLoaderConfig):
    train_config = openpi_config.get_config(config.openpi_config_name)
    data_factory = train_config.data

    if not dataclasses.is_dataclass(data_factory):
        raise TypeError(f"Expected a dataclass data factory, got {type(data_factory)!r}")

    base_config = getattr(data_factory, "base_config", None)
    if base_config is not None and dataclasses.is_dataclass(base_config):
        base_config = dataclasses.replace(base_config, local_root_dir=config.local_root_dir)

    replace_kwargs: dict[str, Any] = {"repo_id": config.repo_id}
    if base_config is not None:
        replace_kwargs["base_config"] = base_config
    if hasattr(data_factory, "local_root_dir"):
        replace_kwargs["local_root_dir"] = config.local_root_dir

    data_factory = dataclasses.replace(data_factory, **replace_kwargs)
    return dataclasses.replace(
        train_config,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        data=data_factory,
    )


def fit_padp_normalizer(
    batches: Iterator[dict[str, Any]],
    *,
    num_batches: int = 8,
    log_every: int | None = 100,
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
            print(f"  normalizer batches: {idx + 1}/{num_batches}", flush=True)

        if idx + 1 >= num_batches:
            break

    if action_accumulator is None:
        raise RuntimeError("Cannot fit PADP normalizer from an empty iterator.")

    if consumed_batches < num_batches:
        print(
            f"  normalizer iterator ended after {consumed_batches} batches; requested {num_batches}.",
            flush=True,
        )

    normalizer = LinearNormalizer()
    normalizer["action"] = get_range_normalizer_from_stat(action_accumulator.to_stat())

    for key in sorted(image_keys):
        normalizer[key] = get_image_range_normalizer()

    for key, accumulator in sorted(obs_accumulators.items()):
        if key.endswith("quat"):
            normalizer[key] = get_identity_normalizer_from_stat(accumulator.to_stat())
        else:
            normalizer[key] = get_range_normalizer_from_stat(accumulator.to_stat())

    return normalizer


@dataclass
class StreamingFeatureStats:
    count: int
    min: torch.Tensor
    max: torch.Tensor
    sum: torch.Tensor
    sumsq: torch.Tensor

    def update(self, flat: torch.Tensor) -> "StreamingFeatureStats":
        flat = flat.to(dtype=torch.float64)
        self.count += int(flat.shape[0])
        self.min = torch.minimum(self.min, flat.min(dim=0).values)
        self.max = torch.maximum(self.max, flat.max(dim=0).values)
        self.sum += flat.sum(dim=0)
        self.sumsq += (flat * flat).sum(dim=0)
        return self

    def to_stat(self) -> dict[str, np.ndarray]:
        mean = self.sum / self.count
        var = torch.clamp(self.sumsq / self.count - mean * mean, min=0.0)
        std = torch.sqrt(var)
        return {
            "min": self.min.numpy().astype(np.float32),
            "max": self.max.numpy().astype(np.float32),
            "mean": mean.numpy().astype(np.float32),
            "std": std.numpy().astype(np.float32),
        }


def update_streaming_stats(
    accumulator: StreamingFeatureStats | None,
    tensor: torch.Tensor,
    *,
    last_dim: int,
) -> StreamingFeatureStats:
    flat = flatten_feature(tensor, last_dim=last_dim).to(dtype=torch.float64)
    if accumulator is None:
        return StreamingFeatureStats(
            count=int(flat.shape[0]),
            min=flat.min(dim=0).values,
            max=flat.max(dim=0).values,
            sum=flat.sum(dim=0),
            sumsq=(flat * flat).sum(dim=0),
        )
    return accumulator.update(flat)


def flatten_feature(tensor: torch.Tensor, *, last_dim: int) -> torch.Tensor:
    if last_dim <= 0:
        return tensor.reshape(-1, 1)

    feature_dim = 1
    for size in tensor.shape[-last_dim:]:
        feature_dim *= int(size)
    return tensor.reshape(-1, feature_dim)


def tensor_to_stat(tensor: torch.Tensor, *, last_dim: int) -> dict[str, Any]:
    flat = flatten_feature(tensor, last_dim=last_dim)
    return array_to_stats(flat.numpy())


def save_padp_normalizer(
    normalizer: LinearNormalizer,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "normalizer_state_dict": normalizer.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_padp_normalizer(path: str | Path, *, map_location: str | torch.device = "cpu") -> LinearNormalizer:
    payload = torch.load(Path(path), map_location=map_location)
    normalizer = LinearNormalizer()
    state_dict = payload["normalizer_state_dict"] if "normalizer_state_dict" in payload else payload
    normalizer.load_state_dict(state_dict)
    return normalizer
