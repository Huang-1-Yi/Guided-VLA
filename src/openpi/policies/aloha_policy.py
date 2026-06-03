import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.shared import attention_map as _attention_map


def make_aloha_example() -> dict:
    """为 Aloha policy 创建随机输入示例。"""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Aloha policy 的输入。

    期望输入：
    - images: dict[name, img]，其中 img 为 [channel, height, width]。name 必须在 EXPECTED_CAMERAS 中。
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # 为 true 时，将 joint 和 gripper 值从标准 Aloha 空间转换到
    # pretrained model 使用的 base PI action normalization 空间。
    adapt_to_pi: bool = True

    # 期望的 camera 名称。所有输入 cameras 都必须在该集合中。缺失 cameras 会被替换为黑图，
    # 对应的 `image_mask` 会被设为 False。
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    @staticmethod
    def _extract_attention_map(data: dict) -> dict[str, np.ndarray] | None:
        """将 Robotwin object-map keys 映射为模型规范的 attention-map 名称。"""
        raw_attention_map = data.get("attention_map")
        if raw_attention_map is None:
            raw_attention_map = data
        if not isinstance(raw_attention_map, dict):
            return None

        normalized_attention_map = _attention_map.to_model_attention_map_keys(raw_attention_map)
        return normalized_attention_map or None

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # 假设 base image 一定存在。
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # 添加额外图像。
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        attention_map = self._extract_attention_map(data)
        if attention_map is not None:
            inputs["attention_map"] = attention_map

        # actions 只在训练期间可用。
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # 如果存在，则透传可选 skill metadata。
        if "skill_id" in data:
            inputs["skill_id"] = data["skill_id"]
        if "skill_soft" in data:
            inputs["skill_soft"] = data["skill_soft"]
        if "skill_text" in data:
            inputs["skill_text"] = data["skill_text"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Aloha policy 的输出。"""

    # 为 true 时，将 joint 和 gripper 值从标准 Aloha 空间转换到
    # pretrained model 使用的 base PI action normalization 空间。
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # 只返回前 14 维。
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """用于在 aloha 与 pi joint angles 之间转换。"""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha 会将 gripper positions 转换到线性空间。以下代码反转该变换，
    # 以便与在角度空间预训练的 pi0 保持一致。
    #
    # 这些值来自 Aloha 代码：
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # 这是 Interbotix 代码中 angular-to-linear 变换的逆变换。
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # 常量取自 Interbotix 代码。
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper 数据在 encoder counts (2405, 3110) 之间归一化到 (0, 1)。
    # encoder counts 总数为 4096，aloha 使用 2048 作为零点。
    # 转换为 radians 后，归一化输入范围为 (0.5476, 1.6296)。
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # 从 pi0 使用的 gripper position 转换为 Aloha 使用的 gripper position。
    # 注意单位仍为角度，但范围不同。

    # 不缩放输出，因为 trossen 模型预测已经是 radians。
    # 常量推导见 _gripper_to_angular 中的注释。
    value = value + 0.5476

    # 这些值来自 Aloha 代码：
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # 直接反转 gripper_from_angular 函数。
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state 为 [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]。
    # 维度大小：[6, 1, 6, 1]。
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # 如果使用 float images，则转换为 uint8。
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # 从 [channel, height, width] 转换为 [height, width, channel]。
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # 翻转 joints。
        state = _joint_flip_mask() * state
        # 反转 Aloha runtime 应用的 gripper transformation。
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # 翻转 joints。
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
