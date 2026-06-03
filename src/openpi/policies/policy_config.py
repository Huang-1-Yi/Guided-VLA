import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """从已训练的 checkpoint 创建 policy。

    Args:
        train_config: 用于创建模型的训练配置。
        checkpoint_dir: 加载模型的目录。
        repack_transforms: 可选 transform，会在其他 transform 之前应用。
        sample_kwargs: 传给 `sample_actions` 方法的 kwargs。未提供时使用默认 kwargs。
        default_prompt: policy 使用的默认 prompt。如果输入数据里还没有 prompt，会自动注入。
        norm_stats: policy 使用的 norm stats。未提供时，会从 checkpoint 目录加载。
        pytorch_device: PyTorch 模型使用的设备，例如 "cpu"、"cuda"、"cuda:0"。
                      若为 None 且 is_pytorch=True，则优先使用 "cuda"，不可用时使用 "cpu"。

    Note:
        该函数会检查 checkpoint 目录中是否存在 "model.safetensors"，从而自动判断模型是否基于 PyTorch。
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # 通过查找 model.safetensors 判断这是否是 PyTorch 模型。
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # 这里从 checkpoint 加载 norm stats，而不是从 config assets 目录加载，
        # 以确保 policy 使用与原始训练过程一致的归一化统计量。
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # 确定 PyTorch 模型使用的设备。
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
