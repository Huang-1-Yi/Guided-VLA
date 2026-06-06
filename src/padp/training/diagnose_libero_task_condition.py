from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from padp.data.openpi_libero_task_loader import OpenPiLiberoTaskPadpDataset


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Inspect task one-hot conditioning in PADP LIBERO batches.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train_task.yaml")
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--print-rows", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    if args.local_root_dir is not None:
        cfg.local_root_dir = args.local_root_dir
    cfg.dataloader.batch_size = args.batch_size
    cfg.dataloader.num_workers = args.num_workers
    OmegaConf.resolve(cfg)

    dataset = make_dataset(cfg, num_batches=args.num_batches)
    counts = torch.zeros(int(cfg.task_condition.num_tasks), dtype=torch.long)
    task_key = str(cfg.task_condition.obs_key)

    for batch_idx, batch in enumerate(dataset):
        onehot = batch["obs"][task_key][:, 0, :].detach().cpu()
        task_ids = onehot.argmax(dim=-1)
        counts += torch.bincount(task_ids, minlength=counts.shape[0])
        if batch_idx == 0:
            print(f"task key: {task_key}")
            print(f"onehot shape: {tuple(batch['obs'][task_key].shape)}")
            print(f"first task ids: {task_ids[: args.print_rows].tolist()}")
            print(f"first onehot rows:\n{onehot[: args.print_rows]}")
        if batch_idx + 1 >= args.num_batches:
            break

    print(f"task id counts over {args.num_batches} batches: {counts.tolist()}")
    print("If ids exceed num_tasks, increase task_condition.num_tasks before training.")


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


if __name__ == "__main__":
    main()
