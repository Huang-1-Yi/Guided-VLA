from __future__ import annotations

import argparse
import logging
import random
import socket
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from openpi.serving import websocket_policy_server
from openpi_client import base_policy as _base_policy

from padp.model.common.normalizer import LinearNormalizer


class PadpLiberoPolicy(_base_policy.BasePolicy):
    """Websocket-compatible PADP policy for examples/libero/main.py."""

    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        device: str,
        action_chunk_size: int,
        output_action_space: str = "absolute",
        gripper_action_mode: str = "raw",
        seed: int | None = None,
        debug_log_steps: int = 0,
    ) -> None:
        register_omegaconf_resolvers()
        if seed is not None:
            _seed_everything(seed)

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)

        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {self._device}, but CUDA is not available.")

        logging.info("Loading PADP checkpoint: %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        normalizer = LinearNormalizer()
        normalizer.load_state_dict(ckpt["normalizer"])

        policy = hydra.utils.instantiate(cfg.policy)
        policy.load_state_dict(ckpt["model"])
        policy.set_normalizer(normalizer)
        policy.to(self._device)
        policy.eval()
        policy.reset()

        horizon = int(cfg.horizon)
        if action_chunk_size < 1:
            raise ValueError(f"action_chunk_size must be >= 1, got {action_chunk_size}")
        if action_chunk_size > horizon:
            raise ValueError(f"action_chunk_size must be <= horizon ({horizon}), got {action_chunk_size}")
        policy.n_action_steps = int(action_chunk_size)

        self._cfg = cfg
        self._policy = policy
        self._action_chunk_size = int(action_chunk_size)
        if output_action_space not in {"absolute", "delta"}:
            raise ValueError(f"output_action_space must be 'absolute' or 'delta', got {output_action_space!r}")
        if gripper_action_mode not in {"raw", "invert", "binary", "binary_invert"}:
            raise ValueError(
                "gripper_action_mode must be one of "
                "{'raw', 'invert', 'binary', 'binary_invert'}, "
                f"got {gripper_action_mode!r}"
            )
        self._output_action_space = output_action_space
        self._gripper_action_mode = gripper_action_mode
        self._debug_log_steps = int(debug_log_steps)
        self._infer_count = 0
        self._metadata = {
            "policy": "padp_libero_va",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": int(ckpt.get("step", -1)),
            "action_dim": int(cfg.shape_meta.action.shape[0]),
            "action_chunk_size": self._action_chunk_size,
            "output_action_space": self._output_action_space,
            "gripper_action_mode": self._gripper_action_mode,
            "seed": seed,
            "horizon": horizon,
            "n_obs_steps": int(cfg.n_obs_steps),
            "notes": "PADP-VA ignores prompt and reuses the current openpi LIBERO state slicing.",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        obs_dict = self._to_padp_obs(obs)
        with torch.inference_mode():
            output = self._policy.predict_action(obs_dict)

        model_actions = output["action"][0].detach().cpu().numpy().astype(np.float32)
        actions = self._to_env_actions(obs, model_actions)
        actions = self._postprocess_gripper(actions)
        if not np.isfinite(actions).all():
            raise RuntimeError("PADP predicted non-finite actions.")
        self._maybe_log_debug(obs, obs_dict, model_actions, actions)
        return {"actions": actions}

    def reset(self) -> None:
        self._policy.reset()

    def _to_padp_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        image = _image_to_padp_tensor(obs["observation/image"], device=self._device)
        wrist_image = _image_to_padp_tensor(obs["observation/wrist_image"], device=self._device)
        state = _state_to_tensor(obs["observation/state"], device=self._device)

        if state.shape[-1] < 8:
            raise ValueError(f"Expected LIBERO state dim >= 8, got {tuple(state.shape)}")

        # Keep this slicing identical to LiberoPadpBatchAdapter so service inference
        # matches the current training semantics.
        return {
            "agentview_image": image,
            "robot0_eye_in_hand_image": wrist_image,
            "robot0_eef_pos": state[..., 0:3],
            "robot0_eef_quat": state[..., 3:7],
            "robot0_gripper_qpos": state[..., 7:8],
        }

    def _to_env_actions(self, obs: dict, actions: np.ndarray) -> np.ndarray:
        if self._output_action_space == "delta":
            return actions

        raw_state = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)
        if raw_state.shape[0] < 6:
            raise ValueError(f"Expected observation/state dim >= 6 for absolute action conversion, got {raw_state.shape}")

        env_actions = np.array(actions, dtype=np.float32, copy=True)
        env_actions[..., :6] += raw_state[:6]
        return env_actions

    def _postprocess_gripper(self, actions: np.ndarray) -> np.ndarray:
        if self._gripper_action_mode == "raw":
            return actions

        result = np.array(actions, dtype=np.float32, copy=True)
        if self._gripper_action_mode in {"binary", "binary_invert"}:
            result[..., 6] = np.where(result[..., 6] >= 0.0, 1.0, -1.0)
        if self._gripper_action_mode in {"invert", "binary_invert"}:
            result[..., 6] *= -1.0
        return result

    def _maybe_log_debug(
        self,
        obs: dict,
        obs_dict: dict[str, torch.Tensor],
        model_actions: np.ndarray,
        env_actions: np.ndarray,
    ) -> None:
        if self._infer_count >= self._debug_log_steps:
            self._infer_count += 1
            return

        raw_state = np.asarray(obs["observation/state"], dtype=np.float32).reshape(-1)
        logging.info("PADP debug infer %d", self._infer_count)
        logging.info("raw observation/state[:8]: %s", _format_np(raw_state[:8]))
        logging.info("split eef_pos: %s", _format_tensor(obs_dict["robot0_eef_pos"]))
        logging.info("split state[3:7]: %s", _format_tensor(obs_dict["robot0_eef_quat"]))
        logging.info("split gripper: %s", _format_tensor(obs_dict["robot0_gripper_qpos"]))
        logging.info("model actions shape: %s", model_actions.shape)
        logging.info("output action space: %s", self._output_action_space)
        logging.info("gripper action mode: %s", self._gripper_action_mode)
        logging.info("model action[0]: %s", _format_np(model_actions[0]))
        logging.info("env action[0]: %s", _format_np(env_actions[0]))
        logging.info(
            "env action min/max: %s / %s",
            _format_np(env_actions.min(axis=0)),
            _format_np(env_actions.max(axis=0)),
        )
        self._infer_count += 1


def _image_to_padp_tensor(image: Any, *, device: torch.device) -> torch.Tensor:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected image shape H,W,C or C,H,W, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got {array.shape}")

    array = np.ascontiguousarray(array)
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0).unsqueeze(0)


def _state_to_tensor(state: Any, *, device: torch.device) -> torch.Tensor:
    array = np.asarray(state, dtype=np.float32)
    if array.ndim != 1:
        array = array.reshape(-1)
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0).unsqueeze(0)


def _format_tensor(tensor: torch.Tensor) -> str:
    return _format_np(tensor.detach().cpu().numpy().reshape(-1))


def _format_np(array: np.ndarray, *, precision: int = 4) -> str:
    values = np.asarray(array, dtype=np.float32).reshape(-1).tolist()
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _seed_everything(seed: int) -> None:
    logging.info("Setting PADP serving random seed: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def register_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Serve a PADP-VA LIBERO policy over the OpenPI websocket protocol.")
    parser.add_argument("--config-path", type=Path, default=repo_root / "src" / "padp" / "config" / "libero_va_train.yaml")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/padp_libero_va/train_1000step/last.pt"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-chunk-size", type=int, default=1)
    parser.add_argument("--output-action-space", choices=("absolute", "delta"), default="absolute")
    parser.add_argument("--gripper-action-mode", choices=("raw", "invert", "binary", "binary_invert"), default="raw")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug-log-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = PadpLiberoPolicy(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        action_chunk_size=args.action_chunk_size,
        output_action_space=args.output_action_space,
        gripper_action_mode=args.gripper_action_mode,
        seed=args.seed,
        debug_log_steps=args.debug_log_steps,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating PADP LIBERO server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

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
