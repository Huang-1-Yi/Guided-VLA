from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf
import torch

from padp.data.libero_batch_adapter import move_padp_batch_to_device
from padp.data.openpi_libero_task_loader import OpenPiLiberoTaskPadpDataset
from padp.model.common.normalizer import LinearNormalizer


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Run one task-conditioned PADP-VA LIBERO predict_action smoke test.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train_task.yaml")
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_omegaconf_resolvers()
    cfg = OmegaConf.load(args.config_path)
    if args.local_root_dir is not None:
        cfg.local_root_dir = args.local_root_dir
    if args.device is not None:
        cfg.training.device = args.device
    cfg.dataloader.batch_size = args.batch_size
    cfg.dataloader.num_workers = args.num_workers
    OmegaConf.resolve(cfg)

    device = torch.device(cfg.training.device)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    print(f"checkpoint step: {ckpt.get('step')}")

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(ckpt["model"])
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()

    dataset = make_dataset(cfg, num_batches=1)
    batch = move_padp_batch_to_device(next(iter(dataset)), device)
    task_key = str(cfg.task_condition.obs_key)
    print("PADP task obs keys:", ", ".join(sorted(batch["obs"].keys())))
    print(f"{task_key}: shape={tuple(batch['obs'][task_key].shape)}")
    print("task ids:", batch["obs"][task_key][:, 0, :].argmax(dim=-1).detach().cpu().tolist())

    with torch.inference_mode():
        output = policy.predict_action(batch["obs"])

    for key in ("action", "action_pred"):
        value = output[key].detach().cpu()
        print(
            f"output[{key}]: shape={tuple(value.shape)}, "
            f"min={float(value.min()):.4f}, max={float(value.max()):.4f}"
        )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"output[{key}] contains NaN/Inf")

    expected_action_dim = int(cfg.shape_meta.action.shape[0])
    if output["action"].shape[-1] != expected_action_dim:
        raise RuntimeError(f"Expected action dim {expected_action_dim}, got {output['action'].shape[-1]}")
    print("PADP LIBERO task smoke predict ok")


def make_dataset(cfg: Any, *, num_batches: int | None) -> OpenPiLiberoTaskPadpDataset:
    return OpenPiLiberoTaskPadpDataset(
        openpi_config_name=str(cfg.openpi_config),
        repo_id=str(cfg.repo_id),
        local_root_dir=None if cfg.local_root_dir is None else str(cfg.local_root_dir),
        batch_size=int(cfg.dataloader.batch_size),
        num_workers=int(cfg.dataloader.num_workers),
        shuffle=bool(cfg.dataloader.shuffle),
        split="train",
        num_batches=num_batches,
        skip_norm_stats=True,
        seed=int(cfg.training.seed),
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        action_dim=int(cfg.shape_meta.action.shape[0]),
        state_dim=8,
        task_num_tasks=int(cfg.task_condition.num_tasks),
        task_obs_key=str(cfg.task_condition.obs_key),
        task_id_source=str(cfg.task_condition.id_source),
        strict_task_id=bool(cfg.task_condition.strict_task_id),
        fallback_task_id=None
        if cfg.task_condition.fallback_task_id is None
        else int(cfg.task_condition.fallback_task_id),
    )


def register_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


if __name__ == "__main__":
    main()
