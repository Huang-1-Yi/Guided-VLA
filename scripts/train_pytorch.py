"""
支持多 GPU 和多节点（DDP）的 PI0/PI05 PyTorch 训练入口。
该脚本对齐 JAX 训练器（`scripts/train.py`）的行为，但完全在 PyTorch 中运行，
使用 `PI0Pytorch` 模型以及 `src/openpi/training/config.py`、
`src/openpi/training/data_loader.py` 中已有的 config/data pipeline。

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint

Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

Multi-Node Training:
  torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name>

Dataset Configuration:
  --local_root_dir (str)              : 覆盖本地数据集目录

Supported command-line overrides:
  --control_net_enabled (bool)          : 启用/禁用 ControlNet
  --lambda_object (float)               : object loss 权重（覆盖 config.object_loss_weight）
  --lambda_skill (float)                : skill loss 权重（覆盖 config.skill_loss_weight）
  --local_root_dir (str)                : 数据集本地根目录（覆盖 config）

Dataset Examples:
  # 使用 LIBERO 数据集训练（通常使用 config 中预定义的路径）
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name libero_train

  # 使用显式指定的本地数据集根目录训练
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim \
    --exp_name robotwin_train --local_root_dir=/path/to/lerobot/root

"""

import argparse
import contextlib
import dataclasses
import datetime
import gc
import logging
import os
import pathlib
import platform
import shutil
import sys
import time
from typing import Any
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"The '.*' attribute with value .* was provided to the `Field\(\)` function",
)

import numpy as np
import safetensors
import safetensors.torch
import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data

# ---------------------------------------------------------------------------
# 训练常量
# ---------------------------------------------------------------------------
_DDP_TIMEOUT_MINUTES = 10
_DEFAULT_OBJECT_LOSS_WEIGHT = 0.1
_DEFAULT_SKILL_LOSS_WEIGHT = 0.1
_PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:256,expandable_segments:True"
_COSINE_DECAY_SCALE = 0.5
_COMPILE_WARMUP_STEPS_DEFAULT = 5
_EARLY_MEMORY_LOG_STEPS = 5
_OBJECT_TARGET_KEYS = ("object_maps", "object_masks")


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def make_json_serializable(obj):
    """递归地将不可 JSON 序列化的类型转换为可序列化形式。"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, frozenset | set):
        return list(obj)
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if hasattr(obj, "__dict__"):
        # 处理未被转换的 dataclass-like 对象。
        return str(obj)
    return obj


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """初始化 wandb 日志。"""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    config_dict = make_json_serializable(dataclasses.asdict(config))

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(
            id=run_id,
            resume="must",
            project=config.project_name,
            settings=wandb.Settings(start_method="thread"),
        )
    else:
        wandb.init(
            name=config.exp_name,
            config=config_dict,
            project=config.project_name,
            settings=wandb.Settings(start_method="thread"),
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    # LOCAL_RANK 由 torchrun 设置；未设置时回退到 RANK 或 0。
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))

    # 在任何 distributed init 或模型创建之前尽早设置 device。
    if torch.cuda.is_available():
        # 确保 local_rank 位于 device_count 范围内。
        dev_count = torch.cuda.device_count()
        if local_rank >= dev_count:
            # 防御性检查：输出诊断信息并抛出更有帮助的错误。
            raise RuntimeError(
                f"LOCAL_RANK ({local_rank}) >= torch.cuda.device_count() ({dev_count}). "
                "Check CUDA_VISIBLE_DEVICES and torchrun configuration."
            )
        torch.cuda.set_device(local_rank)

    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        timeout = datetime.timedelta(minutes=_DDP_TIMEOUT_MINUTES)

        # 记录环境信息，便于调试。
        logging.info(
            f"[setup_ddp] Initializing DDP with backend={backend}, "
            f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
            f"MASTER_PORT={os.environ.get('MASTER_PORT')}, "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
            f"RANK={os.environ.get('RANK')}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
        )

        # 减少 NCCL stream management 中的 GIL 竞争；异步检测错误，避免无限挂起。
        os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

        # 不要向 init_process_group 传递 device_id 或 torch.device。
        torch.distributed.init_process_group(backend=backend, init_method="env://", timeout=timeout)

        # NOTE: 不要在这里设置 TORCH_DISTRIBUTED_DEBUG=DETAIL，因为它会创建额外的
        # Gloo process group wrapper，在多节点/容器化环境中可能导致超时问题。

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
    else:
        rank = int(os.environ.get("RANK", str(local_rank)))
        world = int(os.environ.get("WORLD_SIZE", "1"))

    logging.info(
        f"[setup_ddp] local_rank={local_rank} rank={rank} world_size={world} device={device} cuda_count={torch.cuda.device_count()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    if torch.cuda.is_available():
        logging.info(
            f"[setup_ddp] torch.cuda.current_device()={torch.cuda.current_device()} device_index_attr={device.index}"
        )

    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


# 缓存全零 object-supervision tensors，以保持 torch.compile 输入签名稳定。
_empty_object_target_cache: dict[tuple[int, torch.device], dict[str, torch.Tensor]] = {}


def get_empty_object_targets(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """返回固定 `(batch_size, device)` 对应的缓存全零 object-supervision tensors。"""
    key = (batch_size, device)
    if key in _empty_object_target_cache:
        return _empty_object_target_cache[key]

    maps_shape = (batch_size, _data.MAX_VIEWS, _data.NUM_PATCHES)
    masks_shape = (batch_size, _data.MAX_VIEWS)

    cached = {
        "object_maps": torch.zeros(maps_shape, dtype=torch.float32, device=device),
        "object_masks": torch.zeros(masks_shape, dtype=torch.bool, device=device),
    }
    _empty_object_target_cache[key] = cached
    return cached


def prepare_object_targets(
    object_target_batch,
    device: torch.device,
    batch_size: int,
    *,
    use_object_loss: bool,
) -> dict[str, torch.Tensor] | None:
    """返回结构固定的 object-supervision tensors，供 torch.compile 使用。"""
    if not use_object_loss:
        return None

    empty_targets = get_empty_object_targets(batch_size, device)
    if object_target_batch is None:
        raise RuntimeError(
            "Object loss is enabled, but the data loader did not provide object_targets. "
            "Check that DataConfig(use_object_loss=True) is set and that the LeRobot dataset contains "
            "agentview_attention_object_mask/wrist_attention_object_mask columns."
        )

    object_targets = {}
    for key in _OBJECT_TARGET_KEYS:
        batch_tensor = object_target_batch.get(key)
        if batch_tensor is None:
            raise RuntimeError(
                f"Object loss is enabled, but object_targets is missing {key!r}. Expected keys: {_OBJECT_TARGET_KEYS}."
            )
        target_dtype = empty_targets[key].dtype
        if batch_tensor.device == device and batch_tensor.dtype == target_dtype:
            object_targets[key] = batch_tensor
        else:
            object_targets[key] = batch_tensor.to(
                device=device,
                dtype=target_dtype,
                non_blocking=True,
            )
    return object_targets


def build_data_loaders(config: _config.TrainConfig):
    """构建训练和验证 data loaders。"""
    start = time.time()

    train_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True, split="train")
    data_config = train_loader.data_config()
    logging.info(f"Train loader created in {time.time() - start:.2f}s")

    val_start = time.time()
    val_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False, split="val")
    logging.info(f"Val loader created in {time.time() - val_start:.2f}s")

    return train_loader, val_loader, data_config


def should_use_object_loss(model_config: Any, data_config: Any | None = None) -> bool:
    if not getattr(model_config, "use_object_loss", True):
        return False

    object_head_indices = list(getattr(model_config, "object_head_indices", []))
    if not object_head_indices and hasattr(model_config, "num_object_distill_heads"):
        object_head_indices = list(range(max(int(getattr(model_config, "num_object_distill_heads", 0)), 0)))
    if len(object_head_indices) == 0:
        return False

    return data_config is None or bool(getattr(data_config, "use_object_loss", False))


def move_observation_to_device(observation, device: torch.device):
    def move_optional_tensor(value):
        if value is None or not isinstance(value, torch.Tensor):
            return value
        return value.to(device=device, non_blocking=True)

    def move_image_tensor(image: torch.Tensor) -> torch.Tensor:
        if image.dtype == torch.uint8:
            image = image.to(device=device, dtype=torch.float32, non_blocking=True)
            if image.ndim == 4 and image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            return image / 255.0 * 2.0 - 1.0

        if image.ndim == 4:
            image = image.to(device=device, dtype=torch.float32, non_blocking=True)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            return image

        return image.to(device=device, non_blocking=True)

    return _model.Observation(
        images={key: move_image_tensor(image) for key, image in observation.images.items()},
        image_masks={key: move_optional_tensor(mask) for key, mask in observation.image_masks.items()},
        state=move_optional_tensor(observation.state),
        skill_id=move_optional_tensor(observation.skill_id),
        skill_soft=move_optional_tensor(observation.skill_soft),
        tokenized_prompt=move_optional_tensor(observation.tokenized_prompt),
        tokenized_prompt_mask=move_optional_tensor(observation.tokenized_prompt_mask),
        token_ar_mask=move_optional_tensor(observation.token_ar_mask),
        token_loss_mask=move_optional_tensor(observation.token_loss_mask),
    )


def move_actions_to_device(actions: torch.Tensor, device: torch.device) -> torch.Tensor:
    return actions.to(device=device, dtype=torch.float32, non_blocking=True)


def compute_batch_losses(
    model,
    observation,
    actions,
    *,
    object_targets,
    use_object_loss: bool,
    use_skill_loss: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    autocast_enabled = torch.cuda.is_available()
    autocast_device = "cuda" if autocast_enabled else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=autocast_enabled):
        torch.compiler.cudagraph_mark_step_begin()
        main_loss, object_loss, skill_loss = model(
            observation,
            actions,
            object_targets=object_targets,
            use_object_loss=use_object_loss,
            use_skill_loss=use_skill_loss,
        )
        return main_loss, object_loss, skill_loss


def initialize_checkpoint_dir(config: _config.TrainConfig) -> bool:
    """准备 checkpoint 目录，并判断是否处于 resume 状态。"""
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if not exp_checkpoint_dir.exists():
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")

        latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
        if latest_step is None:
            raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")

        logging.info(f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}")
        return True

    if config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created experiment checkpoint directory: {config.checkpoint_dir}")
    return False


def normalize_state_dict_for_loading(
    state_dict: dict[str, torch.Tensor],
    *,
    source_label: str,
) -> dict[str, torch.Tensor]:
    """加载前规范化 compile-wrapper 前缀和 tied weights。"""
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
        logging.info(
            "Normalized %d compiled key(s) from %s for compatibility.",
            normalized_keys,
            source_label,
        )
    state_dict = normalized_state_dict

    if embed_tokens_key not in state_dict and lm_head_key in state_dict:
        logging.info(f"Copying tied weight: {lm_head_key} -> {embed_tokens_key}")
        state_dict[embed_tokens_key] = state_dict[lm_head_key]

    return state_dict


def split_missing_keys(missing_keys: list[str]) -> tuple[list[str], list[str]]:
    """将 missing keys 切分为预期缺失和非预期缺失两组。"""
    expected_missing_keys = []
    unexpected_missing_keys = []
    for key in missing_keys:
        if key.startswith(("depth", "skill_head.")):
            expected_missing_keys.append(key)
        else:
            unexpected_missing_keys.append(key)
    return expected_missing_keys, unexpected_missing_keys


def unwrap_model(model, *, log_compile_unwrap: bool = False):
    """无论嵌套顺序如何，都解开顶层 DDP/torch.compile wrappers。"""
    unwrapped_model = model
    saw_compile_wrapper = False
    while True:
        if isinstance(unwrapped_model, torch.nn.parallel.DistributedDataParallel):
            unwrapped_model = unwrapped_model.module
            continue
        if hasattr(unwrapped_model, "_orig_mod"):
            saw_compile_wrapper = True
            unwrapped_model = unwrapped_model._orig_mod  # noqa: SLF001
            continue
        break

    if saw_compile_wrapper and log_compile_unwrap:
        logging.info("Detected compiled model, unwrapping top-level _orig_mod wrapper(s) for checkpoint I/O.")

    return unwrapped_model


@contextlib.contextmanager
def temporarily_unwrap_compiled_modules(model, *, log_prefix: str | None = None):
    """临时将已编译的子模块替换回其原始模块。"""
    root_model = unwrap_model(model, log_compile_unwrap=log_prefix is not None)
    replaced_children: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def unwrap_children(module: torch.nn.Module) -> None:
        for child_name, child_module in list(module.named_children()):
            child_to_visit = child_module
            if hasattr(child_module, "_orig_mod"):
                replaced_children.append((module, child_name, child_module))
                child_to_visit = child_module._orig_mod  # noqa: SLF001
                setattr(module, child_name, child_to_visit)
            unwrap_children(child_to_visit)

    unwrap_children(root_model)

    if replaced_children and log_prefix is not None:
        logging.info(
            "%s: temporarily unwrapped %d compiled submodule(s) for clean checkpoint I/O.",
            log_prefix,
            len(replaced_children),
        )

    try:
        yield root_model
    finally:
        for parent_module, child_name, compiled_child in reversed(replaced_children):
            setattr(parent_module, child_name, compiled_child)


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """保存包含模型 state、optimizer state 和 metadata 的 checkpoint。"""
    if not is_main:
        return

    # 仅在达到保存间隔或最终 step 时保存。
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # 创建临时目录，用于原子化保存 checkpoint。
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # 移除已有临时目录，并创建新的临时目录。
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 即使训练使用嵌套 torch.compile wrappers，也用干净的 module keys 保存 checkpoint。
        with temporarily_unwrap_compiled_modules(model, log_prefix="Saving checkpoint") as model_to_save:
            safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # 使用 PyTorch 格式保存 optimizer state。
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # 保存训练 metadata，避免保存完整 config 引入 JAX/Flax 兼容性问题。
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # 保存 norm stats。
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # 将临时目录原子化移动到最终位置。
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """加载最新 checkpoint，并返回 global step。"""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # 加载 checkpoint 前清理显存/内存。
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    logging.info("Loading model state...")
    safetensors_path = ckpt_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

    state_dict = safetensors.torch.load_file(safetensors_path, device=str(device))
    state_dict = normalize_state_dict_for_loading(state_dict, source_label="checkpoint")

    with temporarily_unwrap_compiled_modules(model, log_prefix="Loading checkpoint") as model_to_load:
        missing_keys, unexpected_keys = model_to_load.load_state_dict(state_dict, strict=False)
    expected_missing_keys, unexpected_missing_keys = split_missing_keys(missing_keys)

    # 校验 control-attention 结构：config 必须与 checkpoint 匹配。
    # 到这里时，build_model 已经根据 config 注入（或未注入）CA。
    ca_missing = [k for k in missing_keys if ".origin." in k or "object_branch" in k]
    ca_unexpected = [k for k in unexpected_keys if "object_branch" in k]
    if ca_missing:
        raise ValueError(
            f"config.control_attention_enabled=True but resume checkpoint is missing "
            f"{len(ca_missing)} control-attention key(s), e.g. '{ca_missing[0]}'.\n"
            "The checkpoint was trained without ControlAttention. "
            "Set control_attention_enabled=False in your config to match."
        )
    if ca_unexpected:
        raise ValueError(
            f"Resume checkpoint contains {len(ca_unexpected)} control-attention key(s), "
            f"e.g. '{ca_unexpected[0]}', but config.control_attention_enabled=False.\n"
            "The checkpoint was trained with ControlAttention. "
            "Set control_attention_enabled=True in your config to match."
        )

    if unexpected_missing_keys:
        logging.warning(f"Missing keys (unexpected): {unexpected_missing_keys}")
    if expected_missing_keys:
        logging.debug(f"Missing keys (expected, ignored): {expected_missing_keys}")
    other_unexpected = [k for k in unexpected_keys if k not in ca_unexpected]
    if other_unexpected:
        logging.warning(f"Unexpected keys when loading checkpoint: {other_unexpected}")
    logging.info("Loaded model state from safetensors format")

    torch.cuda.empty_cache()
    gc.collect()
    log_memory_usage(device, latest_step, "after_loading_model")

    logging.info("Loading optimizer state...")
    optimizer_path = ckpt_dir / "optimizer.pt"
    if not optimizer_path.exists():
        raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

    optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
    logging.info("Loaded optimizer state from pt format")
    optimizer.load_state_dict(optimizer_state_dict)
    del optimizer_state_dict
    torch.cuda.empty_cache()
    gc.collect()
    log_memory_usage(device, latest_step, "after_loading_optimizer")

    logging.info("Loading metadata...")
    metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
    global_step = metadata.get("global_step", latest_step)
    del metadata
    torch.cuda.empty_cache()
    gc.collect()
    log_memory_usage(device, latest_step, "after_loading_metadata")

    logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
    return global_step


def get_latest_checkpoint_step(checkpoint_dir):
    """从 checkpoint 目录获取最新 checkpoint step。"""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """记录详细的内存/显存使用信息。"""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # 获取更详细的显存信息。
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # 如果可用，获取 DDP 信息。
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


@torch.no_grad()
def run_validation(
    model,
    val_loader,
    device,
    object_loss_weight=0.1,
    skill_loss_weight=0.1,
    max_batches=None,
    *,
    use_object_loss: bool = False,
    use_skill_loss: bool = False,
    use_ddp: bool = False,
):
    """运行验证，并返回平均 loss 指标。"""
    model.eval()

    total_loss = 0.0
    total_main_loss = 0.0
    total_object_loss = 0.0
    total_skill_loss = 0.0
    total_skill_batches = 0.0
    num_batches = 0
    total_skill_correct = 0.0
    total_skill_count = 0.0
    model_core = unwrap_model(model)

    if use_ddp and dist.is_initialized() and max_batches is not None:
        max_batches_tensor = torch.tensor([max_batches], device=device)
        dist.broadcast(max_batches_tensor, src=0)
        max_batches = int(max_batches_tensor.item())

    prefetcher = DevicePrefetcher(val_loader, device)
    for observation, actions, object_target_batch in prefetcher:
        if max_batches is not None and num_batches >= max_batches:
            break

        object_targets = prepare_object_targets(
            object_target_batch,
            device,
            actions.shape[0],
            use_object_loss=use_object_loss,
        )

        main_loss, object_loss, skill_loss = compute_batch_losses(
            model,
            observation,
            actions,
            object_targets=object_targets,
            use_object_loss=use_object_loss,
            use_skill_loss=use_skill_loss,
        )

        if use_skill_loss:
            total_skill_loss += skill_loss.detach().float().item()
            total_skill_batches += 1

        total_batch_loss = main_loss + object_loss_weight * object_loss + skill_loss_weight * skill_loss
        total_loss += total_batch_loss.item()
        total_main_loss += main_loss.item()
        total_object_loss += object_loss.detach().item()
        num_batches += 1

        if torch.cuda.is_available() and use_skill_loss:
            skill_soft = getattr(observation, "skill_soft", None)
            if skill_soft is not None:
                if not isinstance(skill_soft, torch.Tensor):
                    skill_soft = torch.as_tensor(skill_soft)
                targets = skill_soft.argmax(dim=-1) if skill_soft.dim() >= 2 else skill_soft.to(dtype=torch.long)
                targets = targets.to(device=device, dtype=torch.long)
                valid_mask = torch.ones_like(targets, dtype=torch.bool)
            else:
                targets = None
                valid_mask = None

            if targets is not None and valid_mask is not None and valid_mask.any():
                skill_logits = model_core.compute_skill_logits_for_infer(observation, device, deterministic=False)
                predictions = skill_logits.argmax(dim=-1)[valid_mask]
                targets = targets[valid_mask]
                total_skill_correct += (predictions == targets).sum().item()
                total_skill_count += targets.numel()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    model.train()

    if use_ddp and dist.is_initialized():
        metrics = torch.tensor(
            [
                total_loss,
                total_main_loss,
                total_object_loss,
                float(num_batches),
                total_skill_correct,
                total_skill_count,
                total_skill_loss,
                float(total_skill_batches),
            ],
            device=device,
        )

        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        (
            total_loss,
            total_main_loss,
            total_object_loss,
            num_batches,
            total_skill_correct,
            total_skill_count,
            total_skill_loss,
            total_skill_batches,
        ) = metrics.tolist()
        num_batches = int(num_batches)
        total_skill_batches = int(total_skill_batches)

    if num_batches == 0:
        raise RuntimeError("Validation produced zero batches.")

    val_metrics = {
        "val_loss": total_loss / num_batches,
        "val_main_loss": total_main_loss / num_batches,
        "val_object_loss": total_object_loss / num_batches,
    }
    if total_skill_batches > 0:
        val_metrics["val_skill_loss"] = total_skill_loss / total_skill_batches
    val_metrics["val_skill_acc"] = total_skill_correct / max(total_skill_count, 1.0) if total_skill_count > 0 else 0.0

    return val_metrics


class DevicePrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.use_cuda_prefetch = device.type == "cuda"
        self.stream = torch.cuda.Stream() if self.use_cuda_prefetch else None

    def __iter__(self):
        if not self.use_cuda_prefetch:
            for observation, actions, object_target_batch in self.loader:
                yield (
                    move_observation_to_device(observation, self.device),
                    move_actions_to_device(actions, self.device),
                    object_target_batch,
                )
            return

        loader_it = iter(self.loader)
        self.preload(loader_it)
        while self.next_batch is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.next_batch
            self.preload(loader_it)
            yield batch

    def preload(self, iterator):
        self.next_batch = next(iterator, None)
        if self.next_batch is None:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            observation, actions, object_target_batch = self.next_batch

            self.next_batch_obs = move_observation_to_device(observation, self.device)
            self.next_batch_actions = move_actions_to_device(actions, self.device)

            if object_target_batch is not None:
                self.next_batch_object_targets = {}
                for k, v in object_target_batch.items():
                    if isinstance(v, torch.Tensor):
                        self.next_batch_object_targets[k] = v.to(self.device, non_blocking=True)
                    else:
                        self.next_batch_object_targets[k] = v
            else:
                self.next_batch_object_targets = None

            self.next_batch = (self.next_batch_obs, self.next_batch_actions, self.next_batch_object_targets)


def build_model(
    config: _config.TrainConfig,
    device,
    train_model_config,
    *,
    resuming: bool,
    world_size: int,
    use_ddp: bool,
    local_rank: int,
    use_skill_loss: bool,
    use_object_loss: bool,
):
    """构建模型、加载权重、启用 ControlAttention、编译并封装 DDP。"""
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        runtime_model_config = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        runtime_model_config = config.model
        object.__setattr__(runtime_model_config, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(runtime_model_config).to(device)
    del model.paligemma_with_expert.gemma_expert.lm_head

    guided_config_enabled = any(
        [
            getattr(train_model_config, "control_attention_enabled", False),
            getattr(train_model_config, "use_object_loss", False),
            getattr(train_model_config, "use_depth", False),
            getattr(train_model_config, "use_skill_loss", False),
        ]
    )
    if guided_config_enabled and config.pytorch_weight_path is None and not resuming:
        logging.warning(
            "Guided PyTorch training is starting from scratch because pytorch_weight_path=None. "
            "PyTorch training does not use config.weight_loader. "
            "Pass a checkpoint directory containing model.safetensors, e.g. "
            "'gs://openpi-assets/checkpoints/pi0_libero' or a local converted checkpoint."
        )

    # Resume：加载权重前先注入 ControlAttention（inject-then-load）。
    # resume checkpoint 已经包含 .origin./* 和 object_branch./* keys，
    # 与注入后的模型结构匹配。
    control_attention_kwargs = {
        "num_control_heads": getattr(train_model_config, "control_attention_num_heads", None),
        "copy_weights": getattr(train_model_config, "control_attention_copy_weights", None),
        "freeze_origin": getattr(train_model_config, "control_attention_freeze_origin", None),
        "use_headwise_gate": getattr(train_model_config, "control_attention_use_headwise_gate", None),
    }
    if resuming and getattr(train_model_config, "control_attention_enabled", False):
        model.enable_control_attention(**control_attention_kwargs)

    # Pretrained init：先加载 base weights，再注入 CA（load-then-inject）。
    # 先加载可在注入 CA 后保留 .origin 分支中的 pretrained weights。
    if config.pytorch_weight_path is not None and not resuming:
        logging.info(f"Loading pretrained weights from: {config.pytorch_weight_path}")
        checkpoint_dir = download.maybe_download(config.pytorch_weight_path)
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        state_dict = safetensors.torch.load_file(model_path)
        state_dict = normalize_state_dict_for_loading(state_dict, source_label="pretrained weights")
        expert_lm_head_key = "paligemma_with_expert.gemma_expert.lm_head.weight"
        if expert_lm_head_key in state_dict:
            del state_dict[expert_lm_head_key]
        missing_keys, _unexpected_keys = model.load_state_dict(state_dict, strict=False)
        expected_missing_keys, unexpected_missing_keys = split_missing_keys(missing_keys)
        if expected_missing_keys:
            logging.debug(f"Missing keys (expected, ignored): {expected_missing_keys}")
        if unexpected_missing_keys:
            logging.warning(f"Missing keys (unexpected): {len(unexpected_missing_keys)}")
            for k in unexpected_missing_keys:
                logging.warning(f"  - {k}")
    elif resuming:
        logging.info("Skipping pretrained weight loading — will load from checkpoint instead")

    # 带 CA 的 pretrained init：加载 base weights 后再注入（load-then-inject 模式）。
    if not resuming and getattr(train_model_config, "control_attention_enabled", False):
        model.enable_control_attention(**control_attention_kwargs)

    enable_gc = getattr(config, "use_gradient_checkpointing", False)
    if not enable_gc and world_size >= 4:
        logging.info(f"Auto-enabling gradient checkpointing for world_size={world_size}.")
        enable_gc = True
    if hasattr(model, "gradient_checkpointing_enable"):
        if enable_gc:
            model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing")
        else:
            model.gradient_checkpointing_disable()
    else:
        if enable_gc:
            logging.warning("Config requested gradient checkpointing but this model does not support it")
        enable_gc = False

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _PYTORCH_CUDA_ALLOC_CONF
        logging.info("Enabled CUDA optimizations: TF32, cuDNN benchmark")

    # 在 torch.compile 之前封装 DDP。根据 PyTorch 文档，compile(DDP(model)) 允许编译器
    # 将 allreduce 融入编译图，避免在 DDP 边界发生 graph break。
    # 反过来的顺序（DDP(compile(model))）会阻止这种融合。
    if use_ddp:
        os.environ.setdefault("NCCL_P2P_DISABLE", "0")
        num_local_gpus = torch.cuda.device_count()
        if world_size <= num_local_gpus:
            # 单节点：NVLink P2P 是快速 GPU 间通信路径。
            os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
            logging.info("Single-node DDP: enabled NVLink P2P optimization")
        else:
            # 多节点：节点间通信通过 IB/RoCE。
            # NCCL_P2P_LEVEL=NVL 在这里无关；集群会提供 IB 变量。
            logging.info(
                f"Multi-node DDP: {world_size} GPUs across "
                f"{world_size // max(num_local_gpus, 1)} nodes. "
                f"NCCL IB config: IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', 'unset')}, "
                f"IB_HCA={os.environ.get('NCCL_IB_HCA', 'unset')}, "
                f"GDR_LEVEL={os.environ.get('NCCL_GDR_LEVEL', os.environ.get('NCCL_NET_GDR_LEVEL', 'unset'))}"
            )
        # 默认 False：find_unused_parameters=True 会增加约 20% allreduce 开销。
        # 只有当某些参数确实在部分 step 中跳过梯度时，才在 config 中设为 True。
        ddp_find_unused = getattr(config, "ddp_find_unused_parameters", False)
        # static_graph=True 与 gradient checkpointing 不兼容：GC 会在 backward 时重新运行 forward，
        # 这违反了 DDP 对固定图的假设。在多节点 NCCL 中，这会导致死锁，因为 allreduce 操作
        # 是按预设图调度的，但该图在每个 backward step 都会改变形态。
        static_graph = not ddp_find_unused and not enable_gc
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=ddp_find_unused,
            gradient_as_bucket_view=True,
            static_graph=static_graph,
            broadcast_buffers=False,
        )

    # 使用 mode="default" 编译子模块（仅 Inductor fusion，不使用 cudagraphs）。
    # 之前使用 mode="reduce-overhead" 是为了在重计算路径上启用 CUDA graphs，但在多节点 DDP
    #（>1 node）中，cudagraph_trees 会在 backward 触发不变量：
    # "input tensor deallocate during graph recording that did not occur during replay"
    # 在 backward 时，所有 rank 都会以相同方式崩溃。嵌套的 sibling cudagraphs
    #（embed_image -> vision_tower.encoder，paligemma_with_expert -> embed_image）
    # 在 DDP gradient bucketing 下，其 saved-tensor 生命周期在 recording 和 replay 间会分歧。
    # "default" 保留 Inductor kernel fusion（主要收益来源），但不使用 cudagraphs，
    # 因此单节点和多节点行为一致。DDP 使用 static_graph=True，因此 allreduce 会被预调度，
    # 不需要 compile(DDP) 的 allreduce fusion。
    logging.info("Torch compiling the model...")
    _inner = model.module if use_ddp else model
    if hasattr(_inner, "paligemma_with_expert"):
        # 在封装 paligemma_with_expert 自身之前，先将 embedding 入口编译为独立 callable。
        # torch.compile 只拦截 module 的 forward（通过 __call__），因此在 OptimizedModule 上调用
        # .embed_image / .embed_language_tokens 这类 bound methods 会绕过编译图并以 eager 运行。
        # 直接编译 bound methods 可让每次调用进入各自的编译图。必须在外层 compile 前执行，
        # 因为在 OptimizedModule wrapper 上设置属性不会转发到 _orig_mod。
        _pwe = _inner.paligemma_with_expert
        # 直接将 SigLIP vision encoder 作为 nn.Module 编译。
        # 它的 forward 是干净的 encoder layer 循环，没有外层 SiglipVisionModel/Transformer wrappers
        # 上会导致 graph break 的 can_return_tuple decorator。
        # 这样 torch.compile 可以有效融合 vision-tower 的大部分 FLOPs。
        _vt = _pwe.paligemma.model.vision_tower.vision_model
        _vt.encoder = torch.compile(_vt.encoder, mode="default", dynamic=False)
        _pwe.embed_image = torch.compile(_pwe.embed_image, mode="default", dynamic=False)
        _pwe.embed_language_tokens = torch.compile(_pwe.embed_language_tokens, mode="default", dynamic=False)
        _inner.paligemma_with_expert = torch.compile(_inner.paligemma_with_expert, mode="default", dynamic=False)
        logging.info("Torch compiling finished (paligemma_with_expert, default).")
    else:
        model = torch.compile(model, mode="default", dynamic=False)
        logging.info("Torch compiling finished (whole model, default).")

    return model, runtime_model_config, enable_gc


def build_optimizer(model, config: _config.TrainConfig, peak_lr: float):
    """创建 AdamW optimizer，并可选地对 backbone LR 进行缩放。"""
    scale = getattr(config, "backbone_lr_scale", 1.0)
    if scale != 1.0:
        backbone_params, new_head_params = [], []
        for name, param in (model.module if hasattr(model, "module") else model).named_parameters():
            if not param.requires_grad:
                continue
            if any(
                k in name
                for k in [
                    "object_branch",
                    "q_expand_linear",
                    "zero_conv",
                    "skill_head",
                    "depth_token_proj",
                    "token_merging_model",
                ]
            ):
                new_head_params.append(param)
            else:
                backbone_params.append(param)
        param_groups = [
            {"params": backbone_params, "lr": peak_lr * scale},
            {"params": new_head_params, "lr": peak_lr},
        ]
        logging.info(f"Optimizer: 2 param groups — backbone lr={peak_lr * scale:.2e}, heads lr={peak_lr:.2e}")
    else:
        param_groups = model.parameters()

    return torch.optim.AdamW(
        param_groups,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
        fused=True,
    )


def train_loop(
    config: _config.TrainConfig,
    lambda_object: float | None = None,
    lambda_skill: float | None = None,
):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    if is_main and getattr(config, "fsdp_devices", 1) != 1:
        logging.warning(
            "PyTorch trainer does not implement FSDP. `config.fsdp_devices=%s` is ignored here. "
            "PyTorch parallelism is controlled by `torchrun` DDP world size; JAX FSDP exists only in `scripts/train.py`.",
            config.fsdp_devices,
        )

    object_loss_weight = lambda_object
    object_loss_source = "CLI override"
    if object_loss_weight is None:
        object_loss_weight = getattr(config, "object_loss_weight", None)
        object_loss_source = "config/default"
    if object_loss_weight is None:
        object_loss_weight = _DEFAULT_OBJECT_LOSS_WEIGHT
    if is_main:
        logging.info(f"Using object_loss_weight from {object_loss_source} = {object_loss_weight}")

    skill_loss_weight = lambda_skill
    skill_loss_source = "CLI override"
    if skill_loss_weight is None:
        skill_loss_weight = getattr(config, "skill_loss_weight", None)
        skill_loss_source = "config/default"
    if skill_loss_weight is None:
        skill_loss_weight = _DEFAULT_SKILL_LOSS_WEIGHT
    if is_main:
        logging.info(f"Using skill_loss_weight from {skill_loss_source} = {skill_loss_weight}")

    resuming = initialize_checkpoint_dir(config)

    world_size = torch.distributed.get_world_size() if use_ddp else 1
    per_device_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {per_device_batch_size} "
        f"(global batch size across {world_size} GPUs: {config.batch_size})"
    )

    train_loader, val_loader, data_config = build_data_loaders(config)
    use_object_loss = should_use_object_loss(config.model, data_config)
    use_skill_loss = getattr(data_config, "use_skill_loss", False)

    if should_use_object_loss(config.model) and not getattr(data_config, "use_object_loss", False):
        logging.warning(
            "Object loss is enabled on the model, but data_config.use_object_loss=False. "
            "Skipping object loss. Set DataConfig(use_object_loss=True) to load object-map supervision."
        )
    elif getattr(data_config, "use_object_loss", False) and not should_use_object_loss(config.model):
        logging.warning(
            "data_config.use_object_loss=True, but model object supervision is disabled or has no object heads. "
            "Skipping object loss."
        )

    init_wandb(config, resuming=resuming, enabled=is_main and config.wandb_enabled)

    if use_ddp:
        if is_main:
            logging.info("DDP: Rank 0 finished sample logging, synchronizing all ranks...")
        dist.barrier()
        if is_main:
            logging.info("DDP: All ranks synchronized, proceeding to model creation.")

    model, runtime_model_config, enable_gradient_checkpointing = build_model(
        config,
        device,
        config.model,
        resuming=resuming,
        world_size=world_size,
        use_ddp=use_ddp,
        local_rank=local_rank,
        use_skill_loss=use_skill_loss,
        use_object_loss=use_object_loss,
    )

    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    optimizer = build_optimizer(model, config, peak_lr)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optimizer, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # 对齐 JAX 行为：从 peak_lr / (warmup_steps + 1) 开始。
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cosine

    model.train()
    start_time = time.time()
    # 运行时累加器：只在记录日志时调用 .item()，避免每 step 发生 CPU-GPU 同步。
    running_total_loss = torch.zeros(1, device=device)
    running_main_loss = torch.zeros(1, device=device)
    running_object_loss = torch.zeros(1, device=device)
    running_skill_loss = torch.zeros(1, device=device)
    running_grad_norm = torch.zeros(1, device=device)
    running_data_time = 0.0
    running_compute_time = 0.0
    # 进度条缓存值：在记录日志时更新，避免每 step .item() 同步。
    pb_loss = 0.0
    pb_main_loss = 0.0
    pb_object_loss = 0.0
    pb_skill_loss = 0.0
    steps_since_log = 0
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            "Training config: "
            f"batch_size={config.batch_size}, per_gpu_batch_size={per_device_batch_size}, "
            f"num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {runtime_model_config.dtype}")

    logging.info("Warming up DataLoader prefetch queue...")
    prefetch_start = time.time()
    train_iter = iter(train_loader)
    first_batch = next(train_iter)
    if use_ddp:
        dist.barrier()
    logging.info(f"DataLoader prefetch warmup completed in {time.time() - prefetch_start:.2f}s")

    def next_train_batch(train_iterator):
        batch = next(train_iterator, None)
        if batch is not None:
            return batch
        return next(iter(train_loader))

    compile_warmup_steps = int(os.environ.get("COMPILE_WARMUP_STEPS", str(_COMPILE_WARMUP_STEPS_DEFAULT)))

    def run_compile_warmup(train_iterator, initial_batch):
        logging.info(f"Running {compile_warmup_steps} compile warmup iterations...")
        warmup_start = time.time()

        observation, actions, object_target_batch = initial_batch
        object_targets = None
        for warmup_step in range(compile_warmup_steps):
            if warmup_step > 0:
                observation, actions, object_target_batch = next_train_batch(train_iterator)

            observation = move_observation_to_device(observation, device)
            actions = move_actions_to_device(actions, device)
            object_targets = prepare_object_targets(
                object_target_batch,
                device,
                actions.shape[0],
                use_object_loss=use_object_loss,
            )
            main_loss, object_loss, skill_loss = compute_batch_losses(
                model,
                observation,
                actions,
                object_targets=object_targets,
                use_object_loss=use_object_loss,
                use_skill_loss=use_skill_loss,
            )
            warmup_loss = main_loss + object_loss_weight * object_loss + skill_loss_weight * skill_loss
            warmup_loss.backward()
            optimizer.zero_grad(set_to_none=True)

            if is_main:
                logging.info(f"  Warmup step {warmup_step + 1}/{compile_warmup_steps} completed")

        # 预编译 no_grad/eval specialization，避免第一次真实验证时
        # 因 grad_mode guard 改变付出约 2 分钟的 dynamo recompile 开销。
        model.eval()
        with torch.no_grad():
            compute_batch_losses(
                model,
                observation,
                actions,
                object_targets=object_targets,
                use_object_loss=use_object_loss,
                use_skill_loss=use_skill_loss,
            )
        model.train()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if use_ddp:
            dist.barrier()
        if is_main:
            logging.info(f"Compile warmup completed in {time.time() - warmup_start:.2f}s")
            logging.info("All GPUs synchronized and ready for training")

    def run_training_step(observation, actions, object_target_batch, current_step):
        object_targets = prepare_object_targets(
            object_target_batch,
            device,
            actions.shape[0],
            use_object_loss=use_object_loss,
        )

        current_lr = lr_schedule(current_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        main_loss, object_loss, skill_loss = compute_batch_losses(
            model,
            observation,
            actions,
            object_targets=object_targets,
            use_object_loss=use_object_loss,
            use_skill_loss=use_skill_loss,
        )

        total_loss = main_loss + object_loss_weight * object_loss + skill_loss_weight * skill_loss
        total_loss.backward()

        if current_step < _EARLY_MEMORY_LOG_STEPS and is_main and torch.cuda.is_available():
            log_memory_usage(device, current_step, "after_backward")

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        return current_lr, total_loss, main_loss, object_loss, skill_loss, grad_norm

    def log_training_metrics(current_step, current_lr, interval_start_time):
        elapsed = time.time() - interval_start_time
        num_steps = max(steps_since_log, 1)
        avg_loss = running_total_loss.item() / num_steps
        avg_main_loss = running_main_loss.item() / num_steps
        avg_object_loss = running_object_loss.item() / num_steps
        avg_grad_norm = running_grad_norm.item() / num_steps
        avg_data_time = running_data_time / num_steps
        avg_compute_time = running_compute_time / num_steps
        avg_skill_loss = (running_skill_loss.item() / num_steps) if use_skill_loss else None

        log_message = (
            f"step={current_step} loss={avg_loss:.4f} "
            f"(main={avg_main_loss:.4f}, object={avg_object_loss:.4f}) "
            f"lr={current_lr:.2e}"
        )
        if avg_skill_loss is not None:
            log_message += f" skill_loss={avg_skill_loss:.4f}"
        log_message += f" grad_norm={avg_grad_norm:.2f}"
        log_message += f" time={elapsed:.1f}s"
        log_message += f" [data={avg_data_time * 1000:.1f}ms, compute={avg_compute_time * 1000:.1f}ms]"
        logging.info(log_message)

        if config.wandb_enabled:
            log_payload = {
                "loss": avg_loss,
                "main_loss": avg_main_loss,
                "object_loss": avg_object_loss,
                "learning_rate": current_lr,
                "grad_norm": avg_grad_norm,
                "step": current_step,
                "time_per_step": elapsed / config.log_interval,
            }
            if avg_skill_loss is not None:
                log_payload["skill_loss"] = avg_skill_loss
            wandb.log(log_payload, step=current_step)

        running_total_loss.zero_()
        running_main_loss.zero_()
        running_object_loss.zero_()
        running_skill_loss.zero_()
        running_grad_norm.zero_()

        return time.time(), avg_loss, avg_main_loss, avg_object_loss, avg_skill_loss

    def run_validation_step(current_step):
        if is_main:
            logging.info(f"Running validation at step {current_step}...")

        val_start_time = time.time()
        val_metrics = run_validation(
            model,
            val_loader,
            device,
            object_loss_weight=object_loss_weight,
            skill_loss_weight=skill_loss_weight,
            max_batches=val_max_batches,
            use_object_loss=use_object_loss,
            use_skill_loss=use_skill_loss,
            use_ddp=use_ddp,
        )
        if use_ddp:
            dist.barrier()

        if is_main:
            val_time = time.time() - val_start_time
            log_message = (
                f"Validation: val_loss={val_metrics['val_loss']:.4f} "
                f"(val_main={val_metrics['val_main_loss']:.4f}, "
                f"val_object={val_metrics['val_object_loss']:.4f}) "
                f"[took {val_time:.1f}s]"
            )
            if "val_skill_loss" in val_metrics:
                log_message += f" skill_loss={val_metrics['val_skill_loss']:.4f}"
            if "val_skill_acc" in val_metrics:
                log_message += f" skill_acc={val_metrics['val_skill_acc']:.4f}"
            logging.info(log_message)
            if config.wandb_enabled:
                wandb.log(val_metrics, step=current_step)

    if compile_warmup_steps > 0:
        run_compile_warmup(train_iter, first_batch)

    if use_ddp:
        dist.barrier()

    progress_bar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    val_interval = config.val_interval if config.val_interval is not None else config.save_interval
    val_max_batches = config.val_max_batches

    prefetcher = DevicePrefetcher(train_loader, device)

    while global_step < config.num_train_steps:
        if use_ddp and hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(global_step // len(train_loader))

        data_start = time.time()

        for observation, actions, object_target_batch in prefetcher:
            batch_data_time = time.time() - data_start
            compute_start = time.time()

            if global_step >= config.num_train_steps:
                break

            current_lr, total_loss, main_loss, object_loss, skill_loss, grad_norm = run_training_step(
                observation,
                actions,
                object_target_batch,
                global_step,
            )

            batch_compute_time = time.time() - compute_start
            if is_main:
                running_total_loss += total_loss.detach()
                running_main_loss += main_loss.detach()
                running_object_loss += object_loss.detach()
                if use_skill_loss:
                    running_skill_loss += skill_loss.detach()
                if isinstance(grad_norm, torch.Tensor):
                    running_grad_norm += grad_norm.detach()
                else:
                    running_grad_norm += grad_norm
                running_data_time += batch_data_time
                running_compute_time += batch_compute_time
                steps_since_log += 1

            global_step += 1
            save_checkpoint(model, optimizer, global_step, config, is_main, data_config)

            if is_main and (global_step % config.log_interval == 0):
                start_time, avg_loss, avg_main_loss, avg_object_loss, avg_skill_loss = log_training_metrics(
                    global_step,
                    current_lr,
                    start_time,
                )
                # 更新进度条缓存，避免每 step .item() 同步。
                pb_loss, pb_main_loss, pb_object_loss = avg_loss, avg_main_loss, avg_object_loss
                if use_skill_loss and avg_skill_loss is not None:
                    pb_skill_loss = avg_skill_loss

                running_data_time = 0.0
                running_compute_time = 0.0
                steps_since_log = 0

            if (global_step % val_interval == 0) and global_step > 0:
                run_validation_step(global_step)

            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "loss": f"{pb_loss:.4f}",
                        "main_l": f"{pb_main_loss:.4f}",
                        "object_l": f"{pb_object_loss:.4f}",
                        **({"skill_l": f"{pb_skill_loss:.4f}"} if use_skill_loss else {}),
                        "lr": f"{current_lr:.2e}",
                        "step": global_step,
                    }
                )

            data_start = time.time()

    # 关闭进度条。
    if progress_bar is not None:
        progress_bar.close()

    # 结束 wandb run。
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()

    def parse_optional_bool(value: str) -> bool:
        return value.lower() in ("true", "1", "yes")

    # 解析命令行参数，用于动态覆盖 config。
    parser = argparse.ArgumentParser(
        description="从命令行覆盖关键 ControlNet 配置",
        add_help=False,  # 不添加默认 help，避免与 tyro 冲突。
    )

    # ControlNet 相关参数。
    parser.add_argument(
        "--control_net_enabled",
        type=parse_optional_bool,
        default=None,
        help="Enable/disable ControlNet (true/false)",
    )
    parser.add_argument(
        "--lambda_object",
        type=float,
        default=None,
        help="Weight for object loss",
    )
    parser.add_argument(
        "--lambda_skill",
        type=float,
        default=None,
        help="Weight for skill loss",
    )

    # 数据集相关参数。
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="robotwin 数据集的 Repository ID（用于自动生成路径）",
    )
    parser.add_argument(
        "--local_root_dir",
        type=str,
        default=None,
        help="Local root directory for dataset (overrides config)",
    )
    # 只解析已知覆盖项，其余参数留给 tyro。
    overrides, remaining_args = parser.parse_known_args()

    # 恢复 sys.argv，使其仅包含 tyro 需要处理的剩余参数。
    sys.argv = [sys.argv[0], *remaining_args]

    # 使用 tyro 加载 config（标准 config 加载流程）。
    config = _config.cli()

    if overrides.control_net_enabled is not None:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                control_attention_enabled=overrides.control_net_enabled,
            ),
        )
        logging.info(f"Model overrides applied: {{'control_attention_enabled': {overrides.control_net_enabled}}}")

    # 数据集相关覆盖项位于 data.base_config 中，因此需要特殊处理。
    if overrides.repo_id is not None:
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, repo_id=overrides.repo_id),
        )
        logging.info(f"Dataset repo_id override applied: {overrides.repo_id}")

    local_root_dir = overrides.local_root_dir
    if local_root_dir is not None:
        base_config = getattr(config.data, "base_config", None) or _config.DataConfig()
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                base_config=dataclasses.replace(base_config, local_root_dir=local_root_dir),
            ),
        )
        logging.info(f"Dataset base_config overrides applied: {{'local_root_dir': '{local_root_dir}'}}")

    # 将 loss 权重单独传给 train_loop。
    train_loop(config, lambda_object=overrides.lambda_object, lambda_skill=overrides.lambda_skill)


if __name__ == "__main__":
    main()
