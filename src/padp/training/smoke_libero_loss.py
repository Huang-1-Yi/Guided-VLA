from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf
import torch

from padp.data.libero_batch_adapter import move_padp_batch_to_device
from padp.data.openpi_libero_loader import OpenPiLiberoPadpDataset


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_config = repo_root / "src" / "padp" / "config" / "libero_va_smoke.yaml"

    parser = argparse.ArgumentParser(description="Run one PADP-VA LIBERO loss smoke test.")
    parser.add_argument("--config-path", type=Path, default=default_config)
    parser.add_argument("--openpi-config", default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n-obs-steps", type=int, default=None)
    parser.add_argument("--normalizer-batches", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--obs-encoder-model-name", default=None)
    parser.add_argument("--feature-aggregation", default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--backward", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_omegaconf_resolvers()

    cfg = OmegaConf.load(args.config_path)
    apply_overrides(cfg, args)
    OmegaConf.resolve(cfg)

    seed = int(cfg.training.seed)
    torch.manual_seed(seed)

    device = torch.device(args.device or cfg.training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device}, but CUDA is not available.")

    dataset = OpenPiLiberoPadpDataset(
        openpi_config_name=str(cfg.openpi_config),
        repo_id=str(cfg.repo_id),
        local_root_dir=None if cfg.local_root_dir is None else str(cfg.local_root_dir),
        batch_size=int(cfg.dataloader.batch_size),
        num_workers=int(cfg.dataloader.num_workers),
        shuffle=bool(cfg.dataloader.shuffle),
        split=args.split,
        num_batches=None,
        skip_norm_stats=True,
        seed=seed,
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        action_dim=int(cfg.shape_meta.action.shape[0]),
        state_dim=8,
    )

    print("Fitting temporary PADP normalizer...")
    normalizer = dataset.get_normalizer(num_batches=args.normalizer_batches)

    batch = next(iter(dataset))
    print_padp_batch_summary(batch)

    print("Instantiating PADP policy...")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.train()

    batch = move_padp_batch_to_device(batch, device)
    loss_b = policy.compute_loss(batch)
    loss = loss_b.mean()

    print("loss_b shape:", tuple(loss_b.shape))
    print("loss mean:", float(loss.detach().cpu()))

    if args.backward:
        print("Running backward smoke test...")
        loss.backward()
        grad_norm = first_grad_norm(policy)
        print("first grad norm:", grad_norm)

    print("PADP LIBERO smoke loss ok")


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
    if args.horizon is not None:
        cfg.horizon = args.horizon
        cfg.policy.horizon = args.horizon
        cfg.policy.num_inference_steps = args.horizon
        cfg.policy.noise_scheduler.num_train_timesteps = args.horizon
    if args.n_obs_steps is not None:
        cfg.n_obs_steps = args.n_obs_steps
        cfg.policy.n_obs_steps = args.n_obs_steps
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.device is not None:
        cfg.training.device = args.device
    if args.obs_encoder_model_name is not None:
        cfg.policy.obs_encoder.model_name = args.obs_encoder_model_name
    if args.feature_aggregation is not None:
        cfg.policy.obs_encoder.feature_aggregation = args.feature_aggregation
    if args.pretrained is not None:
        cfg.policy.obs_encoder.pretrained = args.pretrained
        if args.pretrained is False:
            cfg.policy.obs_encoder.frozen = False


def print_padp_batch_summary(batch: dict[str, Any]) -> None:
    print("PADP obs keys:", sorted(batch["obs"].keys()))
    for key, value in batch["obs"].items():
        print(
            f"obs[{key}]: shape={tuple(value.shape)} dtype={value.dtype} "
            f"min={float(value.min())} max={float(value.max())}"
        )
    action = batch["action"]
    print(
        f"action: shape={tuple(action.shape)} dtype={action.dtype} "
        f"min={float(action.min())} max={float(action.max())}"
    )


def first_grad_norm(module: torch.nn.Module) -> float | None:
    for param in module.parameters():
        if param.grad is not None:
            return float(param.grad.detach().norm().cpu())
    return None


if __name__ == "__main__":
    main()
