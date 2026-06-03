import dataclasses

import einops
import numpy as np

from openpi import transforms
import openpi.models.model as _model


def make_oxe_example() -> dict:
    return {
        "observation": {
            "proprio": np.random.rand(7),
            "image_primary": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "image_secondary": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "action": np.random.rand(7),
        "task": {
            "language_instruction": "perform the task",
        },
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# 目前仅用于 bridge_dataset。
@dataclasses.dataclass(frozen=True)
class OXEInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation"]["proprio"])
        if state.ndim == 2 and state.shape[0] == 1:
            state = state.squeeze(0)

        action = np.asarray(data["action"])
        if action.ndim == 3 and action.shape[0] == 1:
            action = action.squeeze(0)

        top_image = _parse_image(data["observation"]["image_primary"])
        secondary_image = _parse_image(data["observation"]["image_secondary"])
        if top_image.ndim == 4 and top_image.shape[0] == 1:
            top_image = top_image.squeeze(0)
        if secondary_image.ndim == 4 and secondary_image.shape[0] == 1:
            secondary_image = secondary_image.squeeze(0)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (top_image, secondary_image, np.zeros_like(top_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (top_image, np.zeros_like(top_image), secondary_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "action" in data:
            inputs["actions"] = action

        if "task" in data and "language_instruction" in data["task"]:
            prompt = data["task"]["language_instruction"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class OXEOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 只返回前 7 维。
        return {"actions": np.asarray(data["actions"][:, :7])}
