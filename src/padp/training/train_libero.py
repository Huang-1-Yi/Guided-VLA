from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf
import torch

from padp.data.libero_batch_adapter import move_padp_batch_to_device
from padp.data.openpi_libero_loader import OpenPiLiberoPadpDataset
from padp.data.openpi_libero_loader import fit_padp_normalizer
from padp.data.openpi_libero_loader import load_padp_normalizer
from padp.data.openpi_libero_loader import save_padp_normalizer


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Train a PADP-VA policy on GuidedVLA LIBERO data.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train.yaml")
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--normalizer-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Alias for --max-train-steps; PADP train_libero consumes one batch per training step.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--fit-normalizer-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalizer-batches", type=int, default=128)
    parser.add_argument("--backward-only", action=argparse.BooleanOptionalAction, default=False)
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

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer_path = Path(cfg.training.normalizer_path)
    normalizer = prepare_normalizer(cfg, normalizer_path, args)

    print("Instantiating PADP policy...")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    policy.to(device)
    policy.train()

    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())
    optimizer.zero_grad(set_to_none=True)

    dataset = make_dataset(cfg, num_batches=int(cfg.training.max_train_steps))
    iterator = iter(dataset)

    max_steps = int(cfg.training.max_train_steps)
    grad_accum = int(cfg.training.gradient_accumulate_every)
    checkpoint_every = int(cfg.training.checkpoint_every)
    log_every = int(cfg.training.log_every)

    print(f"Starting PADP LIBERO training for {max_steps} steps on {device}...")
    for step in range(max_steps):
        batch = move_padp_batch_to_device(next(iterator), device)
        loss_b = policy.compute_loss(batch)
        batch_loss = loss_b.mean()
        loss = batch_loss / grad_accum
        loss.backward()

        if args.backward_only:
            print(f"backward-only smoke ok, step={step}, loss={float(batch_loss.detach().cpu())}")
            return

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step == 0 or (step + 1) % log_every == 0:
            print(f"step={step + 1:06d} loss={float(batch_loss.detach().cpu()):.6f}")

        if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            save_checkpoint(
                output_dir / f"step_{step + 1:06d}.pt",
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                normalizer=normalizer,
                step=step + 1,
            )

    if bool(cfg.training.save_last_ckpt):
        save_checkpoint(
            output_dir / "last.pt",
            cfg=cfg,
            policy=policy,
            optimizer=optimizer,
            normalizer=normalizer,
            step=max_steps,
        )
    print(f"PADP LIBERO train finished. Output dir: {output_dir}")


def make_dataset(cfg: Any, *, num_batches: int | None) -> OpenPiLiberoPadpDataset:
    return OpenPiLiberoPadpDataset(
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
    )


def prepare_normalizer(cfg: Any, normalizer_path: Path, args: argparse.Namespace):
    if normalizer_path.exists():
        print(f"Loading PADP normalizer from: {normalizer_path}")
        return load_padp_normalizer(normalizer_path)

    if not args.fit_normalizer_if_missing:
        raise FileNotFoundError(
            f"PADP normalizer not found: {normalizer_path}. "
            "Run padp.training.compute_norm_stats_for_padp first."
        )

    print(f"PADP normalizer not found; fitting from {args.normalizer_batches} batches...")
    dataset = make_dataset(cfg, num_batches=args.normalizer_batches)
    normalizer = fit_padp_normalizer(iter(dataset), num_batches=args.normalizer_batches)
    save_padp_normalizer(
        normalizer,
        normalizer_path,
        metadata={
            "source": "train_libero auto-fit",
            "normalizer_batches": int(args.normalizer_batches),
            "repo_id": str(cfg.repo_id),
            "local_root_dir": None if cfg.local_root_dir is None else str(cfg.local_root_dir),
        },
    )
    print(f"Saved auto-fit PADP normalizer to: {normalizer_path}")
    return normalizer


def save_checkpoint(
    path: Path,
    *,
    cfg: Any,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    normalizer: torch.nn.Module,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def register_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def apply_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.local_root_dir is not None:
        cfg.local_root_dir = args.local_root_dir
    if args.normalizer_path is not None:
        cfg.training.normalizer_path = str(args.normalizer_path)
    if args.output_dir is not None:
        cfg.training.output_dir = str(args.output_dir)
    if args.max_train_steps is not None:
        cfg.training.max_train_steps = args.max_train_steps
    if args.num_batches is not None:
        if args.max_train_steps is not None and args.max_train_steps != args.num_batches:
            raise ValueError("--num-batches and --max-train-steps disagree; pass only one value or make them equal.")
        cfg.training.max_train_steps = args.num_batches
    if args.batch_size is not None:
        cfg.dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.dataloader.num_workers = args.num_workers
    if args.device is not None:
        cfg.training.device = args.device
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.checkpoint_every is not None:
        cfg.training.checkpoint_every = args.checkpoint_every
    if args.log_every is not None:
        cfg.training.log_every = args.log_every


if __name__ == "__main__":
    main()
