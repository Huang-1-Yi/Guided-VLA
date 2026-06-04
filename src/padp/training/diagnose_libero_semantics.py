from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import torch

from padp.data.libero_batch_adapter import LiberoPadpBatchAdapter
from padp.data.libero_batch_adapter import unpack_openpi_batch
from padp.data.openpi_libero_loader import OpenPiLiberoLoaderConfig
from padp.data.openpi_libero_loader import create_openpi_libero_loader


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Diagnose PADP/OpenPI LIBERO state and action slicing.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train.yaml")
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--print-rows", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    apply_overrides(cfg, args)
    OmegaConf.resolve(cfg)

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
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        action_dim=int(cfg.shape_meta.action.shape[0]),
        state_dim=8,
    )
    loader = create_openpi_libero_loader(loader_config)
    adapter = LiberoPadpBatchAdapter(
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        action_dim=int(cfg.shape_meta.action.shape[0]),
        state_dim=8,
    )

    state_chunks = []
    action_chunks = []
    first_openpi_obs = None
    first_openpi_actions = None
    first_padp_batch = None

    for idx, openpi_batch in enumerate(loader):
        openpi_obs, openpi_actions, _ = unpack_openpi_batch(openpi_batch)
        state = torch.as_tensor(openpi_obs.state).detach().cpu().float()
        actions = torch.as_tensor(openpi_actions).detach().cpu().float()
        padp_batch = adapter.adapt(openpi_obs, actions)

        state_chunks.append(state[:, :8])
        action_chunks.append(actions[:, : int(cfg.horizon), : int(cfg.shape_meta.action.shape[0])])
        if idx == 0:
            first_openpi_obs = openpi_obs
            first_openpi_actions = actions
            first_padp_batch = padp_batch

    if not state_chunks or first_openpi_obs is None or first_openpi_actions is None or first_padp_batch is None:
        raise RuntimeError("No LIBERO batches were produced.")

    states = torch.cat(state_chunks, dim=0)
    actions = torch.cat(action_chunks, dim=0)

    print("=== OpenPI batch shapes ===")
    print("state:", tuple(first_openpi_obs.state.shape), first_openpi_obs.state.dtype)
    print("actions:", tuple(first_openpi_actions.shape), first_openpi_actions.dtype)
    print("image keys:", sorted(first_openpi_obs.images.keys()))

    print("\n=== State slicing used by PADP ===")
    print("state[:3]  -> robot0_eef_pos")
    print("state[3:7] -> robot0_eef_quat name, but currently just mirrors OpenPI state slice")
    print("state[7:8] -> robot0_gripper_qpos")
    print_stats("state[:8]", states)
    for name, tensor in [
        ("state[0:3]", states[:, 0:3]),
        ("state[3:7]", states[:, 3:7]),
        ("state[7:8]", states[:, 7:8]),
    ]:
        print_stats(name, tensor)

    print("\n=== Action slicing used by PADP ===")
    print("actions[:, :horizon, :7] -> PADP action and env.step action")
    print_stats("actions[:,:,:7]", actions)
    print_stats("actions first step", actions[:, 0, :])

    print("\n=== First rows ===")
    rows = min(int(args.print_rows), states.shape[0])
    for row in range(rows):
        print(f"row {row} state[:8]:", format_vector(states[row]))
        print(f"row {row} action[0,:7]:", format_vector(actions[row, 0]))
        print(f"row {row} action[1,:7]:", format_vector(actions[row, 1]))

    print("\n=== Adapted PADP batch ===")
    for key, value in first_padp_batch["obs"].items():
        print_stats(f"obs[{key}]", value)
    print_stats("batch[action]", first_padp_batch["action"])
    print("\nPADP LIBERO semantics diagnostic finished")


def apply_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.local_root_dir is not None:
        cfg.local_root_dir = args.local_root_dir
    if args.batch_size is not None:
        cfg.dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.dataloader.num_workers = args.num_workers
    if args.seed is not None:
        cfg.training.seed = args.seed


def print_stats(name: str, tensor: torch.Tensor) -> None:
    tensor = torch.as_tensor(tensor).detach().cpu().float()
    flat = tensor.reshape(-1, tensor.shape[-1])
    print(
        f"{name}: shape={tuple(tensor.shape)} min={format_vector(flat.min(dim=0).values)} "
        f"max={format_vector(flat.max(dim=0).values)} mean={format_vector(flat.mean(dim=0))} "
        f"std={format_vector(flat.std(dim=0))}"
    )


def format_vector(tensor: torch.Tensor, *, precision: int = 4) -> str:
    values = torch.as_tensor(tensor).detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


if __name__ == "__main__":
    main()
