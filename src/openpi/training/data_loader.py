from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Any, Literal, Protocol, SupportsIndex, TypeVar

import etils.epath as epath
import jax
import jax.numpy as jnp
import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
from torch.utils.data import Subset

import openpi.models.model as _model
import openpi.shared.attention_map as _attention_map
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """支持随机访问的数据集接口。"""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """可迭代数据集接口。"""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """data loader 接口。"""

    def data_config(self) -> _config.DataConfig:
        """获取该 data loader 对应的数据配置。"""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class ConcatDataset(Dataset[T_co]):
    """将多个数据集合并为单个数据集。"""

    def __init__(self, datasets: Sequence[Dataset]):
        self._datasets = list(datasets)
        if not self._datasets:
            raise ValueError("At least one dataset is required for ConcatDataset.")
        self._cumulative_sizes = self._compute_cumulative_sizes()

    def _compute_cumulative_sizes(self) -> list[int]:
        """计算累积大小，便于快速索引查找。"""
        cumulative_sizes = []
        total = 0
        for dataset in self._datasets:
            total += len(dataset)
            cumulative_sizes.append(total)
        return cumulative_sizes

    def __getitem__(self, index: SupportsIndex) -> T_co:
        idx = index.__index__()
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {index} is out of range for dataset of size {len(self)}")

        # 找到该 index 属于哪个数据集。
        dataset_idx = 0
        for cumulative_size in self._cumulative_sizes:
            if idx < cumulative_size:
                break
            dataset_idx += 1

        # 计算它在选中数据集内部的 index。
        sample_idx = idx if dataset_idx == 0 else idx - self._cumulative_sizes[dataset_idx - 1]
        return self._datasets[dataset_idx][sample_idx]

    def __len__(self) -> int:
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # transforms 设计为作用在单个样本上，因此这里需要先把 batch 拆成单样本，
                # 再逐个应用 transform。
                batch_size = next(v.shape[0] for v in sample.values())

                # 使用 tree_map 将 batch 拆分成单个样本。
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # 转换每个样本。
                transformed = [self._transform(s) for s in individual_samples]

                # 使用 tree_map 将样本重新组合成 batch。
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # 移除 batch 维度。
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                value = jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            elif spec.dtype == jnp.int32:
                value = jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            else:
                value = jnp.zeros(shape=shape, dtype=spec.dtype)
            # PyTorch 的 default_collate 无法 batch JAX arrays，但 numpy arrays
            # 同时适用于 JAX 和 PyTorch 训练路径。
            return np.asarray(value).copy()

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    split: Literal["train", "val", "all"] = "all",
) -> Dataset:
    """创建用于训练或验证的数据集。

    Args:
        data_config: 数据配置。
        action_horizon: action horizon。
        model_config: 模型配置。
        split: 返回哪个 split。"train" 表示训练集，"val" 表示验证集，
               "all" 表示完整数据集（不切分）。
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=data_config.local_root_dir)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=data_config.local_root_dir,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.horizon_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    train_val_split = getattr(data_config, "train_val_split", 0.9)
    split_seed = getattr(data_config, "split_seed", 42)

    if split != "all" and train_val_split < 1.0:
        dataset = _split_dataset(dataset, train_val_split, split_seed, split)
    elif split != "all" and train_val_split >= 1.0:
        logging.warning(
            f"Requested split='{split}' but train_val_split={train_val_split}. "
            "Returning full dataset. Set train_val_split < 1.0 to enable splitting."
        )

    return dataset


def _split_dataset(
    dataset: Dataset,
    train_ratio: float,
    seed: int,
    split: Literal["train", "val"],
) -> Dataset:
    """将数据集切分为训练集和验证集。

    Args:
        dataset: 待切分的数据集。
        train_ratio: 用于训练的数据比例，例如 0.9 表示 90% train。
        seed: 用于可复现切分的随机种子。
        split: 返回哪个 split（"train" 或 "val"）。

    Returns:
        原始数据集的一个 Subset。
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)

    # 生成可复现的随机 indices。
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    if split == "train":
        logging.info(f"Using train split: {len(train_indices)} samples ({train_ratio * 100:.1f}%)")
        return Subset(dataset, train_indices)
    logging.info(f"Using val split: {len(val_indices)} samples ({(1 - train_ratio) * 100:.1f}%)")
    return Subset(dataset, val_indices)


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # 目前 RLDS 数据集只支持 DROID。
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """通过应用 data transforms 来转换数据集。"""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """通过应用 data transforms 来转换数据集。"""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    split: Literal["train", "val", "all"] = "all",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建用于训练的 data loader。

    Args:
        config: 训练配置。
        sharding: data loader 使用的 sharding（仅 JAX）。
        shuffle: 是否打乱数据。
        num_batches: 指定返回的 batch 数量。
        skip_norm_stats: 是否跳过数据归一化。
        framework: 使用的框架（"jax" 或 "pytorch"）。
        split: 使用哪个 split（"train"、"val" 或 "all"）。
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")
    object_view_filter = getattr(config.model, "object_supervision_views", None)

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            model_config=config.model,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
            allowed_object_views=object_view_filter,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        split=split,
        allowed_object_views=object_view_filter,
        assets_dirs=config.assets_dirs,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    split: Literal["train", "val", "all"] = "all",
    allowed_object_views: tuple[str, ...] | None = None,
    assets_dirs: pathlib.Path | str | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建用于训练的 data loader。

    Args:
        data_config: 数据配置。
        action_horizon: action horizon。
        batch_size: batch size。
        sharding: data loader 使用的 sharding。若为 None，data loader 将使用单设备 sharding。
        skip_norm_stats: 是否跳过数据归一化。
        shuffle: 是否打乱数据。
        num_batches: 指定返回的 batch 数量。如果该数值超过数据集中的 batch 数，
            data loader 会循环遍历数据集。未提供时会无限迭代数据集。
        num_workers: 使用的 worker 进程数。若为 0，则 data loader 在主进程中执行。
        seed: 打乱数据时使用的随机种子。
        split: 使用哪个 split（"train"、"val" 或 "all"）。
    """
    # 如果配置定义了多个数据集，则逐个构建并转换。
    multi = getattr(data_config, "multi_datasets", None)
    if multi:
        datasets: list[Dataset] = []
        for sub in multi:
            sub_repo = sub.get("repo_id", data_config.repo_id)
            sub_root = sub.get("local_root_dir", data_config.local_root_dir)

            sub_dc = dataclasses.replace(
                data_config,
                repo_id=sub_repo,
                local_root_dir=sub_root,
            )

            # 使用与 DataConfigFactory._load_norm_stats 相同的逻辑加载每个数据集的 norm_stats。
            asset_id = sub.get("asset_id") or sub_repo
            norm_stats = None
            if asset_id is not None and not skip_norm_stats and assets_dirs is not None:
                try:
                    data_assets_dir = str(epath.Path(assets_dirs) / asset_id)
                    norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
                    logging.info(f"Loaded norm stats from {data_assets_dir} for {sub_repo}")
                except FileNotFoundError:
                    logging.info(f"Norm stats not found in {data_assets_dir} for {sub_repo}, skipping.")

            try:
                sub_dc = dataclasses.replace(sub_dc, norm_stats=norm_stats)
            except Exception:
                sub_dc.norm_stats = norm_stats

            ds = create_torch_dataset(sub_dc, action_horizon, model_config, split=split)
            ds = transform_dataset(ds, sub_dc, skip_norm_stats=skip_norm_stats)
            datasets.append(ds)

        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    else:
        dataset = create_torch_dataset(data_config, action_horizon, model_config, split=split)
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # 两种 framework 都使用 TorchDataLoader。
    # 对 PyTorch DDP，创建 DistributedSampler，并按 world size 切分 batch size。
    # 对 JAX，按 process count 切分。
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            # 验证时希望每个 rank 评估同一份验证集，因此验证时设置 shuffle=False 且 drop_last=False。
            is_validation = split == "val"
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle and not is_validation,  # 不打乱验证数据。
                drop_last=not is_validation,  # 不丢弃最后一批验证样本。
            )
            # 训练和验证都在 DDP 各 rank 间切分 batch size，保证显存使用一致并便于聚合。
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")

    if framework == "pytorch":
        collate_fn = None
        worker_init_fn = None
    else:
        collate_fn = _collate_fn
        worker_init_fn = _worker_init_fn

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # 使用 sampler 时不再额外 shuffle。
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    return DataLoaderImpl(
        data_config,
        data_loader,
        framework=framework,
        allowed_object_views=allowed_object_views,
    )


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    model_config: _model.BaseModelConfig,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
    allowed_object_views: tuple[str, ...] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建用于训练的 RLDS data loader。

    注意：该 data loader 需要一些额外依赖，见 examples/droid/README_train.md。

    Args:
        data_config: 数据配置。
        action_horizon: action horizon。
        batch_size: batch size。
        sharding: data loader 使用的 sharding。若为 None，则 data loader 会使用单设备 sharding。
        skip_norm_stats: 是否跳过数据归一化。
        shuffle: 是否打乱数据。
        num_batches: 指定返回的 batch 数量。如果该数量超过数据集中的 batch 数，
            data loader 会循环遍历数据集。若未提供，则会无限迭代数据集。
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(
        data_config,
        data_loader,
        framework=framework,
        allowed_object_views=allowed_object_views,
    )


class TorchDataLoader:
    """Torch data loader 实现。"""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        collate_fn=None,
        worker_init_fn=None,
    ):
        """创建 PyTorch data loader。

        Args:
            dataset: 待加载的数据集。
            local_batch_size: 每个进程的本地 batch size。
            sharding: data loader 使用的 sharding。
            shuffle: 是否打乱数据。
            num_batches: 若提供，则指定返回的 batch 数量。如果该数值大于数据集中的 batch 数，
                data loader 会循环遍历数据集。未提供时会无限迭代数据集。
            num_workers: 使用的 worker 进程数。若为 0，则 data loader 在主进程中执行。
            seed: 打乱数据时使用的随机种子。
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # 保存 sharding：PyTorch 为 None，JAX 为 JAX sharding。
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # 仅 JAX 默认使用 data parallel sharding。
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            # forkserver 比 spawn 更快：worker 从预初始化 server fork，
            # 不需要重新 import 所有模块，可减少约 50-70% 启动时间。
            mp_context = multiprocessing.get_context("forkserver")

        generator = torch.Generator()
        generator.manual_seed(seed)

        final_collate_fn = collate_fn if collate_fn is not None else _collate_fn
        final_worker_init_fn = worker_init_fn if worker_init_fn is not None else _worker_init_fn

        if framework == "pytorch":
            final_collate_fn = _pytorch_collate_fn
            # 对 PyTorch，添加 worker init function，避免 worker 中初始化 CUDA。
            final_worker_init_fn = _pytorch_worker_init_fn if num_workers > 0 else None

        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # 使用 sampler 时不再额外 shuffle。
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            collate_fn=final_collate_fn,
            worker_init_fn=final_worker_init_fn,
            drop_last=True,
            generator=generator,
            prefetch_factor=int(os.environ.get("DATALOADER_PREFETCH_FACTOR", "4")) if num_workers > 0 else None,
        )

    def set_epoch(self, epoch: int):
        loader = self._data_loader
        if hasattr(loader, "sampler"):
            sampler = loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def __len__(self) -> int:
        if self._num_batches is not None:
            return self._num_batches
        return len(self._data_loader)

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        is_pytorch_framework = self._sharding is None

        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # 数据集已遍历完，创建新 iterator 并重新开始。
                num_items += 1
                # 对 JAX，转换为 sharded arrays；对 PyTorch，直接返回 torch tensors。
                if is_pytorch_framework:
                    yield batch
                else:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """将 batch 元素整理为 batched numpy arrays。"""
    # stack 前先转换为 numpy arrays，因为部分输入元素可能是 JAX arrays。
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


OBJECT_MAP_LEAF_KEYS = frozenset(
    (
        *_attention_map.SUPPORTED_OBJECT_VIEWS,
        *_attention_map.DATASET_OBJECT_MAP_KEY_TO_VIEW,
        *_attention_map.MODEL_ATTENTION_KEY_TO_VIEW,
    )
)


def _format_collate_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"


def _is_object_map_path(path: tuple[str, ...]) -> bool:
    return bool(path) and path[-1] in OBJECT_MAP_LEAF_KEYS


def _is_object_map_container_path(path: tuple[str, ...], value: Any) -> bool:
    if path and path[-1] == "attention_map":
        return True
    return isinstance(value, dict) and bool(value) and set(value).issubset(OBJECT_MAP_LEAF_KEYS)


def _pytorch_collate_fn(items, *, path: tuple[str, ...] = ()):
    """为 PyTorch 整理嵌套样本，同时保留可选的 ``None`` 叶子节点。"""
    if not items:
        raise ValueError("Cannot collate an empty batch.")

    if all(item is None for item in items):
        return None
    if _is_object_map_path(path):
        return list(items)
    if any(item is None for item in items):
        raise ValueError(f"Cannot collate mixed None and non-None values at {_format_collate_path(path)}")

    first = items[0]
    if isinstance(first, dict):
        if _is_object_map_container_path(path, first):
            keys = dict.fromkeys(key for item in items for key in item)
            return {key: _pytorch_collate_fn([item.get(key) for item in items], path=(*path, key)) for key in keys}
        return {key: _pytorch_collate_fn([item[key] for item in items], path=(*path, key)) for key in first}
    if isinstance(first, tuple):
        return tuple(_pytorch_collate_fn(list(values), path=path) for values in zip(*items, strict=True))
    if isinstance(first, list):
        return [_pytorch_collate_fn(list(values), path=path) for values in zip(*items, strict=True)]
    if isinstance(first, torch.Tensor):
        try:
            return torch.stack(items, dim=0)
        except RuntimeError as exc:
            shapes = [tuple(item.shape) for item in items]
            raise ValueError(f"Cannot collate tensors at {_format_collate_path(path)} with shapes {shapes}") from exc
    if isinstance(first, np.ndarray):
        # 避免走 PyTorch worker 侧的 numpy collation 路径；某些 array 来源会生成
        # 由不可调整大小 storage 支撑的 tensor。
        arrays = [np.asarray(item) for item in items]
        try:
            return torch.from_numpy(np.stack(arrays, axis=0))
        except ValueError as exc:
            shapes = [array.shape for array in arrays]
            raise ValueError(
                f"Cannot collate numpy arrays at {_format_collate_path(path)} with shapes {shapes}"
            ) from exc
    if isinstance(first, np.generic):
        return torch.as_tensor(np.asarray(items))
    return torch.utils.data.default_collate(items)


def _worker_init_fn(worker_id: int) -> None:
    """告知 worker 进程中的 JAX 不要预分配 GPU 显存。"""
    # NOTE: 这会在 worker 进程 import jax 后调用，因此该方式不适合用于选择 backend。
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def _pytorch_worker_init_fn(worker_id: int) -> None:
    """避免 torch workers 初始化 CUDA，并减少线程竞争。"""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(1)


class RLDSDataLoader:
    """围绕 DROID data loader 的轻量包装，使其兼容 openpi。

    所有 batching 已经在 DROID 数据集中完成，因此这里不需要再额外处理。
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # 默认使用 data parallel sharding。
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # 数据集已遍历完，创建新 iterator 并重新开始。
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


SUPPORTED_VIEWS = _attention_map.SUPPORTED_OBJECT_VIEWS
VIEW_NAME_TO_ID = {name: i for i, name in enumerate(SUPPORTED_VIEWS)}
MAX_VIEWS = len(SUPPORTED_VIEWS)
PATCH_SIZE = 16
NUM_PATCHES = PATCH_SIZE * PATCH_SIZE  # 256


class DataLoaderImpl(DataLoader):
    def __init__(
        self,
        data_config,
        data_loader,
        *,
        framework: Literal["jax", "pytorch"] = "jax",
        allowed_object_views: tuple[str, ...] | None = None,
    ):
        self._data_config = data_config
        self._data_loader = data_loader
        self._framework = framework
        self._logged_object_map_keys = False
        self._use_object_loss = bool(getattr(data_config, "use_object_loss", False))
        self._allowed_object_views = tuple(allowed_object_views) if allowed_object_views else None
        self._allowed_object_views_set = set(allowed_object_views) if allowed_object_views else None

    def data_config(self):
        return self._data_config

    def set_epoch(self, epoch: int):
        if hasattr(self._data_loader, "set_epoch"):
            self._data_loader.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._data_loader)

    def _extract_object_maps(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        object_map_dict = batch.get("attention_map")
        if isinstance(object_map_dict, dict):
            normalized = _attention_map.normalize_object_map_keys(object_map_dict)
            if normalized:
                return normalized

        return _attention_map.normalize_object_map_keys(batch)

    @staticmethod
    def _object_map_batch_metadata(map_value: Any) -> tuple[int, torch.device, bool]:
        if isinstance(map_value, list | tuple):
            for item in map_value:
                if isinstance(item, torch.Tensor):
                    return len(map_value), item.device, item.device.type == "cpu" and item.is_pinned()
            return len(map_value), torch.device("cpu"), False
        if isinstance(map_value, torch.Tensor):
            return map_value.shape[0], map_value.device, map_value.device.type == "cpu" and map_value.is_pinned()
        if isinstance(map_value, np.ndarray):
            return map_value.shape[0], torch.device("cpu"), False
        raise TypeError(f"Unsupported object map batch value type: {type(map_value)!r}")

    @staticmethod
    def _iter_object_map_samples(map_value: Any, batch_size: int) -> list[Any]:
        if isinstance(map_value, list | tuple):
            if len(map_value) != batch_size:
                raise ValueError(f"Expected {batch_size} object-map samples, got {len(map_value)}")
            return list(map_value)
        if isinstance(map_value, torch.Tensor | np.ndarray):
            if map_value.shape[0] != batch_size:
                raise ValueError(f"Expected leading batch dimension {batch_size}, got {map_value.shape[0]}")
            return [map_value[i] for i in range(batch_size)]
        raise TypeError(f"Unsupported object map value type: {type(map_value)!r}")

    @staticmethod
    def _pack_single_object_map(map_value: Any, view_name: str) -> torch.Tensor | None:
        if map_value is None:
            return None

        if isinstance(map_value, torch.Tensor):
            packed_view = map_value.to(dtype=torch.float32)
        elif isinstance(map_value, np.ndarray):
            packed_view = torch.from_numpy(np.array(map_value, copy=True)).to(dtype=torch.float32)
        else:
            packed_view = torch.as_tensor(map_value, dtype=torch.float32)

        if packed_view.dim() == 3 and packed_view.shape[0] == 1:
            packed_view = packed_view.squeeze(0)

        if packed_view.dim() == 2 and packed_view.shape[-2:] == (PATCH_SIZE, PATCH_SIZE):
            packed_view = packed_view.flatten()

        if packed_view.dim() == 2 and packed_view.shape[0] == 1:
            packed_view = packed_view.squeeze(0)

        if packed_view.numel() != NUM_PATCHES:
            logging.warning(f"Shape mismatch: {view_name} {packed_view.shape}")
            return None

        return packed_view.reshape(NUM_PATCHES)

    def _pack_maps_to_tensor(self, map_dict: dict[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not map_dict:
            return None, None

        batch_size, device, source_is_pinned = self._object_map_batch_metadata(next(iter(map_dict.values())))

        packed_tensor = torch.zeros((batch_size, MAX_VIEWS, NUM_PATCHES), dtype=torch.float32, device=device)
        valid_mask = torch.zeros((batch_size, MAX_VIEWS), dtype=torch.bool, device=device)

        for view_name, map_tensor in map_dict.items():
            if view_name not in VIEW_NAME_TO_ID:
                continue

            if self._allowed_object_views_set and view_name not in self._allowed_object_views_set:
                continue

            idx = VIEW_NAME_TO_ID[view_name]
            for batch_index, sample_map in enumerate(self._iter_object_map_samples(map_tensor, batch_size)):
                packed_view = self._pack_single_object_map(sample_map, view_name)
                if packed_view is None:
                    continue
                packed_tensor[batch_index, idx] = packed_view.to(device=device)
                valid_mask[batch_index, idx] = True

        if source_is_pinned:
            packed_tensor = packed_tensor.pin_memory()
            valid_mask = valid_mask.pin_memory()

        return packed_tensor, valid_mask

    @staticmethod
    def _object_target_has_nonzero_supervision(maps: torch.Tensor, masks: torch.Tensor) -> bool:
        return bool(((maps > 0).any(dim=-1) & masks).any())

    def __iter__(self):
        for batch in self._data_loader:
            object_targets = {}

            if self._use_object_loss:
                obj_maps_dict = self._extract_object_maps(batch)
                if not obj_maps_dict:
                    expected_keys = sorted(_attention_map.DATASET_OBJECT_MAP_KEY_TO_VIEW)
                    raise ValueError(
                        "DataConfig(use_object_loss=True) but the batch contains no object-map supervision. "
                        f"Expected at least one of these dataset keys: {expected_keys}. "
                        "If you are using a local LeRobot cache, refresh it from the dataset version that includes "
                        "object mask columns."
                    )

                maps, masks = self._pack_maps_to_tensor(obj_maps_dict)
                if maps is None or masks is None or not bool(masks.any()):
                    raise ValueError(
                        "Object-map keys were found, but none could be packed into valid 16x16 patch maps. "
                        f"Keys found after normalization: {list(obj_maps_dict.keys())}."
                    )

                if not self._object_target_has_nonzero_supervision(maps, masks):
                    raise ValueError(
                        "Object-map supervision is present but every valid packed map is all zero. "
                        f"Keys found after normalization: {list(obj_maps_dict.keys())}."
                    )

                object_targets["object_maps"] = maps
                object_targets["object_masks"] = masks

                if not self._logged_object_map_keys:
                    logging.info(f"Object maps loaded. Keys found: {list(obj_maps_dict.keys())}")
                    self._logged_object_map_keys = True

            final_output = object_targets or None

            yield (
                _model.Observation.from_dict(batch, normalize_torch_images=self._framework != "pytorch"),
                batch["actions"],
                final_output,
            )
