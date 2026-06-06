from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from padp.data.openpi_libero_task_loader import OpenPiLiberoTaskPadpDataset
from padp.data.openpi_libero_task_loader import save_padp_normalizer


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Fit a task-conditioned PADP LinearNormalizer for LIBERO.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train_task.yaml")
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    apply_overrides(cfg, args)
    OmegaConf.resolve(cfg)

    output_path = Path(args.output_path or cfg.training.normalizer_path)
    dataset = make_dataset(cfg, num_batches=args.num_batches)
    print(f"Fitting task-conditioned PADP normalizer from {args.num_batches} batches...")
    normalizer = dataset.get_normalizer(num_batches=args.num_batches, log_every=args.log_every)
    save_padp_normalizer(
        normalizer,
        output_path,
        metadata={
            "source": "compute_norm_stats_for_padp_task",
            "num_batches": int(args.num_batches),
            "repo_id": str(cfg.repo_id),
            "local_root_dir": None if cfg.local_root_dir is None else str(cfg.local_root_dir),
            "task_num_tasks": int(cfg.task_condition.num_tasks),
            "task_obs_key": str(cfg.task_condition.obs_key),
            "task_id_source": str(cfg.task_condition.id_source),
        },
    )
    print(f"Saved task-conditioned PADP normalizer to: {output_path}")


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


def apply_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.local_root_dir is not None:
        cfg.local_root_dir = args.local_root_dir
    if args.batch_size is not None:
        cfg.dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.dataloader.num_workers = args.num_workers
    if args.seed is not None:
        cfg.training.seed = args.seed


if __name__ == "__main__":
    main()
