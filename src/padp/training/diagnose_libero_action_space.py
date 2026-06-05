from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
import warnings

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"moviepy\..*")
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"ml_collections\..*")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning, module=r"pygame\..*")

from omegaconf import OmegaConf
import torch

from padp.data.libero_batch_adapter import unpack_openpi_batch
from padp.data.openpi_libero_loader import OpenPiLiberoLoaderConfig
from padp.data.openpi_libero_loader import create_openpi_libero_loader


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Compare LIBERO training delta actions with env-space absolute actions.",
    )
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train.yaml")
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-rows", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    apply_overrides(cfg, args)
    OmegaConf.resolve(cfg)

    horizon = int(cfg.horizon)
    action_dim = int(cfg.shape_meta.action.shape[0])
    if action_dim < 7:
        raise ValueError(f"Expected action_dim >= 7, got {action_dim}")

    loader_config = OpenPiLiberoLoaderConfig(
        openpi_config_name=str(cfg.openpi_config),
        repo_id=str(cfg.repo_id),
        local_root_dir=None if cfg.local_root_dir is None else str(cfg.local_root_dir),
        batch_size=int(cfg.dataloader.batch_size),
        num_workers=int(cfg.dataloader.num_workers),
        shuffle=False,
        split="train",
        num_batches=int(args.num_batches),
        skip_norm_stats=True,
        seed=int(cfg.training.seed),
        horizon=horizon,
        n_obs_steps=int(cfg.n_obs_steps),
        action_dim=action_dim,
        state_dim=8,
    )
    loader = create_openpi_libero_loader(loader_config)

    state_acc = StreamingStats()
    delta_first_acc = StreamingStats()
    delta_all_acc = StreamingStats()
    abs_first_acc = StreamingStats()
    abs_all_acc = StreamingStats()
    gripper_first_acc = StreamingStats()
    gripper_all_acc = StreamingStats()
    first_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    consumed_batches = 0
    for batch_idx, openpi_batch in enumerate(loader):
        consumed_batches = batch_idx + 1
        obs, actions, _ = unpack_openpi_batch(openpi_batch)
        state = torch.as_tensor(obs.state).detach().cpu().float()
        actions = torch.as_tensor(actions).detach().cpu().float()

        state8 = state[:, :8]
        delta = actions[:, :horizon, :7].contiguous()
        absolute = delta.clone()
        absolute[..., :6] += state[:, None, :6]

        state_acc.update(state8)
        delta_first_acc.update(delta[:, 0, :])
        delta_all_acc.update(delta)
        abs_first_acc.update(absolute[:, 0, :])
        abs_all_acc.update(absolute)
        gripper_first_acc.update(delta[:, 0, 6:7])
        gripper_all_acc.update(delta[..., 6:7])

        if len(first_rows) < args.print_rows:
            remaining = args.print_rows - len(first_rows)
            for row in range(min(remaining, state.shape[0])):
                first_rows.append((state8[row], delta[row, 0], absolute[row, 0]))

    if consumed_batches == 0:
        raise RuntimeError("No LIBERO batches were produced.")

    print("=== PADP/OpenPI LIBERO action-space diagnostic ===")
    print(f"config: {args.config_path}")
    print(f"repo_id: {cfg.repo_id}")
    print(f"local_root_dir: {cfg.local_root_dir}")
    print(f"batches: {consumed_batches}/{args.num_batches}")
    print(f"batch_size: {cfg.dataloader.batch_size}")
    print(f"horizon: {horizon}")
    print()

    print("Assumption used here:")
    print("  openpi training actions[:6] are delta actions because pi0_libero_object uses extra_delta_transform=True.")
    print("  env-space absolute actions[:6] = delta actions[:6] + current state[:6].")
    print("  gripper action[6] is not shifted.")
    print()

    state_acc.print("current state[:8]")
    delta_first_acc.print("train delta action first step [:7]")
    abs_first_acc.print("train absolute action first step [:7]")
    delta_all_acc.print("train delta action all horizon [:7]")
    abs_all_acc.print("train absolute action all horizon [:7]")
    gripper_first_acc.print("train gripper first step action[6]")
    gripper_all_acc.print("train gripper all horizon action[6]")

    print("\n=== First examples ===")
    for idx, (state, delta, absolute) in enumerate(first_rows):
        print(f"row {idx} state[:8]:         {format_vector(state)}")
        print(f"row {idx} delta action[0]:   {format_vector(delta)}")
        print(f"row {idx} abs action[0]:     {format_vector(absolute)}")

    print("\nCompare with serve_libero debug logs:")
    print("  server pred action[0] is already env-space absolute when --output-action-space absolute is used.")
    print("  If server absolute actions are far outside train absolute ranges, check normalizer/action decoding.")
    print("  If server absolute actions are inside train ranges but behavior fails, inspect gripper timing and task conditioning.")


class StreamingStats:
    def __init__(self) -> None:
        self.count = 0
        self.min: torch.Tensor | None = None
        self.max: torch.Tensor | None = None
        self.sum: torch.Tensor | None = None
        self.sumsq: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.reshape(-1, tensor.shape[-1]).to(dtype=torch.float64)
        if self.count == 0:
            self.count = int(flat.shape[0])
            self.min = flat.min(dim=0).values
            self.max = flat.max(dim=0).values
            self.sum = flat.sum(dim=0)
            self.sumsq = (flat * flat).sum(dim=0)
            return

        assert self.min is not None
        assert self.max is not None
        assert self.sum is not None
        assert self.sumsq is not None
        self.count += int(flat.shape[0])
        self.min = torch.minimum(self.min, flat.min(dim=0).values)
        self.max = torch.maximum(self.max, flat.max(dim=0).values)
        self.sum += flat.sum(dim=0)
        self.sumsq += (flat * flat).sum(dim=0)

    def mean(self) -> torch.Tensor:
        assert self.sum is not None
        return self.sum / self.count

    def std(self) -> torch.Tensor:
        assert self.sumsq is not None
        mean = self.mean()
        return torch.sqrt(torch.clamp(self.sumsq / self.count - mean * mean, min=0.0))

    def print(self, name: str) -> None:
        assert self.min is not None
        assert self.max is not None
        print(f"\n{name}: count={self.count}")
        print(f"  min:  {format_vector(self.min)}")
        print(f"  max:  {format_vector(self.max)}")
        print(f"  mean: {format_vector(self.mean())}")
        print(f"  std:  {format_vector(self.std())}")


def apply_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.local_root_dir is not None:
        cfg.local_root_dir = args.local_root_dir
    if args.batch_size is not None:
        cfg.dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.dataloader.num_workers = args.num_workers
    if args.seed is not None:
        cfg.training.seed = args.seed


def format_vector(tensor: torch.Tensor, *, precision: int = 4) -> str:
    values = torch.as_tensor(tensor).detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


if __name__ == "__main__":
    main()
