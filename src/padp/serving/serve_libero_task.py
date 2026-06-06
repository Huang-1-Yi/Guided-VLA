from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch
from openpi.serving import websocket_policy_server

from padp.serving.serve_libero import PadpLiberoPolicy
from padp.serving.serve_libero import register_omegaconf_resolvers


class PadpLiberoTaskPolicy(PadpLiberoPolicy):
    """Websocket PADP policy that injects a preset task one-hot observation."""

    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
        action_chunk_size: int,
        output_action_space: str = "absolute",
        gripper_action_mode: str = "raw",
        debug_log_steps: int = 0,
        default_task_id: int | None = None,
        strict_prompt_task: bool = True,
    ) -> None:
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            action_chunk_size=action_chunk_size,
            output_action_space=output_action_space,
            gripper_action_mode=gripper_action_mode,
            debug_log_steps=debug_log_steps,
        )

        task_cfg = self._cfg.task_condition
        self._task_num_tasks = int(task_cfg.num_tasks)
        self._task_obs_key = str(task_cfg.obs_key)
        self._prompt_to_task_id = {
            _normalize_prompt(str(prompt)): int(task_id)
            for prompt, task_id in OmegaConf.to_container(task_cfg.prompt_to_task_id, resolve=True).items()
        }
        if default_task_id is None and task_cfg.fallback_task_id is not None:
            default_task_id = int(task_cfg.fallback_task_id)
        self._default_task_id = default_task_id
        self._strict_prompt_task = bool(strict_prompt_task)
        self._metadata.update(
            {
                "policy": "padp_libero_va_task",
                "task_num_tasks": self._task_num_tasks,
                "task_obs_key": self._task_obs_key,
                "strict_prompt_task": self._strict_prompt_task,
                "notes": "PADP-VA task-conditioned variant; prompt is mapped to a preset task one-hot vector.",
            }
        )

    def _to_padp_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        obs_dict = super()._to_padp_obs(obs)
        task_id = self._resolve_task_id(obs)
        task_onehot = torch.zeros((1, 1, self._task_num_tasks), dtype=torch.float32, device=self._device)
        task_onehot[..., task_id] = 1.0
        obs_dict[self._task_obs_key] = task_onehot
        return obs_dict

    def _maybe_log_debug(
        self,
        obs: dict,
        obs_dict: dict[str, torch.Tensor],
        model_actions: np.ndarray,
        env_actions: np.ndarray,
    ) -> None:
        if self._infer_count < self._debug_log_steps and self._task_obs_key in obs_dict:
            task_id = int(obs_dict[self._task_obs_key][0, 0].argmax().detach().cpu())
            logging.info("PADP task condition id: %d", task_id)
        super()._maybe_log_debug(obs, obs_dict, model_actions, env_actions)

    def _resolve_task_id(self, obs: dict) -> int:
        for key in ("task_id", "skill_id", "task_index"):
            if key in obs and obs[key] is not None:
                return self._validate_task_id(int(np.asarray(obs[key]).reshape(-1)[0]), source=key)

        prompt = obs.get("prompt")
        if prompt is not None:
            normalized = _normalize_prompt(_to_python_str(prompt))
            if normalized in self._prompt_to_task_id:
                return self._validate_task_id(self._prompt_to_task_id[normalized], source="prompt")

        if self._default_task_id is not None:
            return self._validate_task_id(int(self._default_task_id), source="default_task_id")

        if self._strict_prompt_task:
            known = sorted(self._prompt_to_task_id.keys())
            raise KeyError(
                "Could not map LIBERO prompt to task id. "
                f"prompt={prompt!r}; known prompts={known}. "
                "Update task_condition.prompt_to_task_id or pass --default-task-id."
            )
        return 0

    def _validate_task_id(self, task_id: int, *, source: str) -> int:
        if task_id < 0 or task_id >= self._task_num_tasks:
            raise ValueError(f"{source} produced task_id={task_id}, outside [0, {self._task_num_tasks}).")
        return task_id


def _to_python_str(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        value = array.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().lower().split())


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Serve a task-conditioned PADP-VA LIBERO policy.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train_task.yaml")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/padp_libero_va/train_task_smoke/last.pt"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-chunk-size", type=int, default=1)
    parser.add_argument("--output-action-space", choices=("absolute", "delta"), default="absolute")
    parser.add_argument("--gripper-action-mode", choices=("raw", "invert", "binary", "binary_invert"), default="raw")
    parser.add_argument("--debug-log-steps", type=int, default=0)
    parser.add_argument("--default-task-id", type=int, default=None)
    parser.add_argument("--strict-prompt-task", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_omegaconf_resolvers()
    policy = PadpLiberoTaskPolicy(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        action_chunk_size=args.action_chunk_size,
        output_action_space=args.output_action_space,
        gripper_action_mode=args.gripper_action_mode,
        debug_log_steps=args.debug_log_steps,
        default_task_id=args.default_task_id,
        strict_prompt_task=args.strict_prompt_task,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating task-conditioned PADP LIBERO server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
