from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf
import torch

from padp.data.libero_batch_adapter import move_padp_batch_to_device
from padp.data.openpi_libero_loader import OpenPiLiberoPadpDataset
from padp.model.common.normalizer import LinearNormalizer


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Run one PADP-VA LIBERO predict_action smoke test.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train.yaml")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/padp_libero_va/train_1000step/last.pt"))
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_omegaconf_resolvers()

    cfg = OmegaConf.load(args.config_path)
    apply_overrides(cfg, args)
    OmegaConf.resolve(cfg)

    seed = int(cfg.training.seed)
    torch.manual_seed(seed)

    device = torch.device(cfg.training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device}, but CUDA is not available.")

    print(f"Loading PADP checkpoint from: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    print("checkpoint step:", ckpt.get("step"))

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])

    print("Instantiating PADP policy...")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(ckpt["model"])
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.eval()
    policy.reset()

    dataset = make_dataset(cfg)
    batch = move_padp_batch_to_device(next(iter(dataset)), device)
    print_obs_summary(batch["obs"])

    with torch.no_grad():
        output = policy.predict_action(batch["obs"])

    print_output_summary(output)
    assert_finite(output)

    action = output["action"]
    expected_action_dim = int(cfg.shape_meta.action.shape[0])
    if action.ndim != 3:
        raise RuntimeError(f"Expected action to have shape (B,T,D), got {tuple(action.shape)}")
    if action.shape[-1] != expected_action_dim:
        raise RuntimeError(f"Expected action dim {expected_action_dim}, got {action.shape[-1]}")

    print("PADP LIBERO smoke predict ok")


def make_dataset(cfg: Any) -> OpenPiLiberoPadpDataset:
    return OpenPiLiberoPadpDataset(
        openpi_config_name=str(cfg.openpi_config),
        repo_id=str(cfg.repo_id),
        local_root_dir=None if cfg.local_root_dir is None else str(cfg.local_root_dir),
        batch_size=int(cfg.dataloader.batch_size),
        num_workers=int(cfg.dataloader.num_workers),
        shuffle=False,
        split="train",
        num_batches=1,
        skip_norm_stats=True,
        seed=int(cfg.training.seed),
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        action_dim=int(cfg.shape_meta.action.shape[0]),
        state_dim=8,
    )


def register_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def apply_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.local_root_dir is not None:
        cfg.local_root_dir = args.local_root_dir
    if args.batch_size is not None:
        cfg.dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.dataloader.num_workers = args.num_workers
    if args.device is not None:
        cfg.training.device = args.device
    if args.seed is not None:
        cfg.training.seed = args.seed


def print_obs_summary(obs: dict[str, torch.Tensor]) -> None:
    print("PADP obs keys:", sorted(obs.keys()))
    for key, value in obs.items():
        print(
            f"obs[{key}]: shape={tuple(value.shape)} dtype={value.dtype} "
            f"min={float(value.min().detach().cpu())} max={float(value.max().detach().cpu())}"
        )


def print_output_summary(output: dict[str, torch.Tensor]) -> None:
    for key, value in output.items():
        print(
            f"output[{key}]: shape={tuple(value.shape)} dtype={value.dtype} "
            f"min={float(value.min().detach().cpu())} max={float(value.max().detach().cpu())}"
        )


def assert_finite(output: dict[str, torch.Tensor]) -> None:
    for key, value in output.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite values found in output[{key}]")


if __name__ == "__main__":
    main()
