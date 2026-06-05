from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from padp.data.openpi_libero_loader import OpenPiLiberoPadpDataset
from padp.data.openpi_libero_loader import save_padp_normalizer


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Compute PADP LinearNormalizer for GuidedVLA LIBERO data.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train.yaml")
    parser.add_argument("--openpi-config", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--output-path", type=Path, default=repo_root / "checkpoints" / "padp_libero_va" / "normalizer.pt")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_omegaconf_resolvers()
    cfg = OmegaConf.load(args.config_path)
    apply_overrides(cfg, args)
    OmegaConf.resolve(cfg)

    dataset = OpenPiLiberoPadpDataset(
        openpi_config_name=str(cfg.openpi_config),
        repo_id=str(cfg.repo_id),
        local_root_dir=None if cfg.local_root_dir is None else str(cfg.local_root_dir),
        batch_size=int(cfg.dataloader.batch_size),
        num_workers=int(cfg.dataloader.num_workers),
        shuffle=False,
        split=args.split,
        num_batches=args.num_batches,
        skip_norm_stats=True,
        seed=int(cfg.training.seed),
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        action_dim=int(cfg.shape_meta.action.shape[0]),
        state_dim=8,
    )

    print(f"Fitting PADP normalizer from {args.num_batches} batches...", flush=True)
    normalizer = dataset.get_normalizer(num_batches=args.num_batches, log_every=args.log_every)
    metadata = {
        "config_path": str(args.config_path),
        "openpi_config": str(cfg.openpi_config),
        "repo_id": str(cfg.repo_id),
        "local_root_dir": None if cfg.local_root_dir is None else str(cfg.local_root_dir),
        "batch_size": int(cfg.dataloader.batch_size),
        "num_batches": int(args.num_batches),
        "horizon": int(cfg.horizon),
        "n_obs_steps": int(cfg.n_obs_steps),
        "action_dim": int(cfg.shape_meta.action.shape[0]),
        "obs_keys": list(cfg.shape_meta.obs.keys()),
    }
    save_padp_normalizer(normalizer, args.output_path, metadata=metadata)
    print(f"Saved PADP normalizer to: {args.output_path}")
    print("input stats keys:", sorted(normalizer.get_input_stats().keys()))


def register_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def apply_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.openpi_config is not None:
        cfg.openpi_config = args.openpi_config
    if args.repo_id is not None:
        cfg.repo_id = args.repo_id
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
