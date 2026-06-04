from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LiberoPadpBatchAdapter:
    """Convert GuidedVLA/openpi LIBERO batches into PADP-VA batches.

    The openpi loader returns model-shaped tensors:
    images:  B,H,W,C uint8
    state:   B,32
    actions: B,50,32

    PADP only needs the real LIBERO dimensions:
    images:  B,T,C,H,W float32 in [0,1]
    state:   split 8-D state into robomimic-style low-dim keys
    action:  B,horizon,7
    """

    horizon: int = 40
    n_obs_steps: int = 1
    action_dim: int = 7
    state_dim: int = 8
    base_image_key: str = "base_0_rgb"
    wrist_image_key: str = "left_wrist_0_rgb"
    agentview_key: str = "agentview_image"
    wrist_out_key: str = "robot0_eye_in_hand_image"
    eef_pos_key: str = "robot0_eef_pos"
    eef_quat_key: str = "robot0_eef_quat"
    gripper_key: str = "robot0_gripper_qpos"

    def __call__(self, openpi_batch: Any) -> dict[str, Any]:
        obs, actions, extra = unpack_openpi_batch(openpi_batch)
        return self.adapt(obs, actions, extra=extra)

    def adapt(self, obs: Any, actions: torch.Tensor, extra: Any | None = None) -> dict[str, Any]:
        images = getattr(obs, "images", None)
        state = getattr(obs, "state", None)
        if images is None or state is None:
            raise TypeError("Expected an openpi Observation with .images and .state fields.")

        if self.base_image_key not in images:
            raise KeyError(f"Missing image key {self.base_image_key!r}; available keys: {sorted(images.keys())}")
        if self.wrist_image_key not in images:
            raise KeyError(f"Missing image key {self.wrist_image_key!r}; available keys: {sorted(images.keys())}")

        state = ensure_torch(state).float()
        actions = ensure_torch(actions).float()
        if state.shape[-1] < self.state_dim:
            raise ValueError(f"Expected state last dim >= {self.state_dim}, got {tuple(state.shape)}")
        if actions.shape[-1] < self.action_dim:
            raise ValueError(f"Expected action last dim >= {self.action_dim}, got {tuple(actions.shape)}")
        if actions.shape[1] < self.horizon:
            raise ValueError(f"Expected action horizon >= {self.horizon}, got {tuple(actions.shape)}")

        base_image = image_to_padp(images[self.base_image_key], n_obs_steps=self.n_obs_steps)
        wrist_image = image_to_padp(images[self.wrist_image_key], n_obs_steps=self.n_obs_steps)

        padp_obs = {
            self.agentview_key: base_image,
            self.wrist_out_key: wrist_image,
            self.eef_pos_key: repeat_obs_steps(state[..., 0:3], self.n_obs_steps),
            self.eef_quat_key: repeat_obs_steps(state[..., 3:7], self.n_obs_steps),
            self.gripper_key: repeat_obs_steps(state[..., 7:8], self.n_obs_steps),
        }

        padp_batch: dict[str, Any] = {
            "obs": padp_obs,
            "action": actions[:, : self.horizon, : self.action_dim].contiguous(),
        }
        if extra is not None:
            padp_batch["extra"] = extra
        return padp_batch


def unpack_openpi_batch(batch: Any) -> tuple[Any, torch.Tensor, Any | None]:
    if isinstance(batch, Mapping):
        return batch["obs"], batch["actions"], batch.get("extra")
    if not isinstance(batch, tuple | list):
        raise TypeError(f"Unsupported openpi batch type: {type(batch)!r}")
    if len(batch) == 3:
        obs, actions, extra = batch
        return obs, actions, extra
    if len(batch) == 2:
        obs, actions = batch
        return obs, actions, None
    raise ValueError(f"Expected openpi batch of length 2 or 3, got length {len(batch)}")


def ensure_torch(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def image_to_padp(image: Any, *, n_obs_steps: int) -> torch.Tensor:
    image = ensure_torch(image)
    if image.ndim != 4:
        raise ValueError(f"Expected image shape B,H,W,C or B,C,H,W, got {tuple(image.shape)}")

    if image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    elif image.shape[1] != 3:
        raise ValueError(f"Expected image channel dimension of 3, got {tuple(image.shape)}")

    image = image.contiguous().float()
    if image.max() > 1.5:
        image = image / 255.0
    return repeat_obs_steps(image, n_obs_steps)


def repeat_obs_steps(value: torch.Tensor, n_obs_steps: int) -> torch.Tensor:
    value = value.contiguous()
    return value.unsqueeze(1).repeat(1, n_obs_steps, *([1] * (value.ndim - 1)))


def move_padp_batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    device = torch.device(device)
    result = {
        "obs": {key: value.to(device, non_blocking=True) for key, value in batch["obs"].items()},
        "action": batch["action"].to(device, non_blocking=True),
    }
    if "extra" in batch:
        result["extra"] = batch["extra"]
    return result
