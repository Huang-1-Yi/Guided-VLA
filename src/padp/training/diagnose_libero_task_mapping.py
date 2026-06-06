from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from openpi.training import data_loader as openpi_data_loader

from padp.data.openpi_libero_loader import OpenPiLiberoLoaderConfig
from padp.data.openpi_libero_loader import make_openpi_train_config


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Print raw LIBERO task_index to prompt mapping.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train_task.yaml")
    parser.add_argument("--local-root-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config_path)
    if args.local_root_dir is not None:
        cfg.local_root_dir = args.local_root_dir
    OmegaConf.resolve(cfg)

    loader_cfg = OpenPiLiberoLoaderConfig(
        openpi_config_name=str(cfg.openpi_config),
        repo_id=str(cfg.repo_id),
        local_root_dir=None if cfg.local_root_dir is None else str(cfg.local_root_dir),
        batch_size=1,
        num_workers=0,
        shuffle=False,
        split="all",
        num_batches=None,
        skip_norm_stats=True,
        seed=int(cfg.training.seed),
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        action_dim=int(cfg.shape_meta.action.shape[0]),
        state_dim=8,
    )
    train_config = make_openpi_train_config(loader_cfg)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_dataset = openpi_data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
        split="all",
    )

    mapping: dict[int, str] = {}
    limit = min(int(args.max_samples), len(raw_dataset))
    for idx in range(limit):
        sample = raw_dataset[idx]
        if not isinstance(sample, dict) or "task_index" not in sample:
            raise KeyError(f"sample does not contain task_index; keys={sorted(sample.keys())}")
        task_index = int(sample["task_index"])
        prompt = _extract_prompt(sample)
        mapping.setdefault(task_index, prompt)
        if len(mapping) >= int(cfg.task_condition.num_tasks):
            break

    for task_index in sorted(mapping):
        print(f"{task_index}: {mapping[task_index]}")
    print(f"found {len(mapping)} task ids in first {limit} samples")


def _extract_prompt(sample: dict[str, Any]) -> str:
    if "task" in sample:
        return _to_text(sample["task"])
    if "prompt" in sample:
        return _to_text(sample["prompt"])
    if "observation.skill_text" in sample:
        return _to_text(sample["observation.skill_text"])
    return "<prompt not present in raw sample>"


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    return str(value)


if __name__ == "__main__":
    main()
