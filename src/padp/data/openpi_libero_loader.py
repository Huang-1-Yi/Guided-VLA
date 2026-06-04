from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

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

    def get_normalizer(self, num_batches: int = 8) -> LinearNormalizer:
        return fit_padp_normalizer(iter(self), num_batches=num_batches)


def create_openpi_libero_loader(config: OpenPiLiberoLoaderConfig):
    train_config = make_openpi_train_config(config)
    return openpi_data_loader.create_data_loader(
        train_config,
        framework="pytorch",
        split=config.split,
        shuffle=config.shuffle,
        num_batches=config.num_batches,
        skip_norm_stats=config.skip_norm_stats,
    )


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
) -> LinearNormalizer:
    chunks: list[dict[str, Any]] = []
    for idx, batch in enumerate(batches):
        chunks.append(batch)
        if idx + 1 >= num_batches:
            break
    if not chunks:
        raise RuntimeError("Cannot fit PADP normalizer from an empty iterator.")

    obs_chunks = {key: [] for key in chunks[0]["obs"]}
    action_chunks = []
    for batch in chunks:
        for key, value in batch["obs"].items():
            obs_chunks[key].append(value.detach().cpu())
        action_chunks.append(batch["action"].detach().cpu())

    normalizer = LinearNormalizer()
    normalizer["action"] = get_range_normalizer_from_stat(
        tensor_to_stat(torch.cat(action_chunks, dim=0), last_dim=1),
    )

    for key, values in obs_chunks.items():
        tensor = torch.cat(values, dim=0)
        if key.endswith("image"):
            normalizer[key] = get_image_range_normalizer()
        elif key.endswith("quat"):
            normalizer[key] = get_identity_normalizer_from_stat(tensor_to_stat(tensor, last_dim=1))
        else:
            normalizer[key] = get_range_normalizer_from_stat(tensor_to_stat(tensor, last_dim=1))

    return normalizer


def tensor_to_stat(tensor: torch.Tensor, *, last_dim: int) -> dict[str, Any]:
    if last_dim <= 0:
        flat = tensor.reshape(-1, 1)
    else:
        feature_dim = 1
        for size in tensor.shape[-last_dim:]:
            feature_dim *= int(size)
        flat = tensor.reshape(-1, feature_dim)
    return array_to_stats(flat.numpy())
