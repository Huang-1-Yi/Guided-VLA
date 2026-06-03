import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import attention_map as _attention_map


def make_libero_example() -> dict:
    """为 Libero policy 创建随机输入示例。"""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    该类用于将模型输入转换为期望格式，同时用于训练和推理。

    对于你自己的数据集，可以复制该类，并根据下面注释修改 key，
    将数据集中的正确元素送入模型。
    """

    # 决定使用哪个模型。为自定义数据集适配时不要修改它。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 可能需要将图像解析为 uint8 (H,W,C)，因为 LeRobot 会自动存为 float32 (C,H,W)，
        # 而 policy inference 会跳过这一步。自定义数据集应保留该逻辑；
        # 如果图像存储在不同于 "observation/image" 或 "observation/wrist_image" 的 key 中，
        # 请在下面修改。Pi0 模型目前支持三路图像输入：一路第三视角，以及左右两路 wrist views。
        # 如果你的数据集没有某类图像，例如 wrist images，可以在这里注释掉，并像下面的 right wrist image
        # 一样用 zeros 替代。
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 创建 inputs dict。不要修改下面 dict 中的 keys。
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # 对不存在的图像，用形状合适的 zero-arrays 进行 padding。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # 只对 pi0 模型 mask padding images，不对 pi0-FAST 这样做。自定义数据集也不要修改这里。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "attention_map" in data and isinstance(data["attention_map"], dict):
            in_attns = _attention_map.to_model_attention_map_keys(data["attention_map"])
            zero_attention_map = np.zeros((16, 16), dtype=np.float32)
            inputs["attention_map"] = {
                "base_0_attn": in_attns.get("base_0_attn", zero_attention_map),
                "left_wrist_0_attn": in_attns.get("left_wrist_0_attn", zero_attention_map),
            }

        # 将 actions padding 到模型动作维度。自定义数据集应保留该逻辑。
        # actions 只在训练期间可用。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 将 prompt（也就是语言指令）传给模型。
        # 自定义数据集应保留该逻辑；如果 instruction 不存储在 "prompt" 中，请修改 key。
        # 输出 dict 始终需要包含 "prompt" key。
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
class LiberoOutputs(transforms.DataTransformFn):
    """
    该类用于将模型输出转换回数据集特定格式，仅用于推理。

    对于你自己的数据集，可以复制该类，并根据下面注释修改 action dimension。
    """

    def __call__(self, data: dict) -> dict:
        # 只返回前 N 个 actions。由于上面为了适配模型动作维度对 actions 做了 padding，
        # 这里需要在返回 dict 中解析出正确数量的 actions。
        # 对 Libero，只返回前 7 个 actions（其余是 padding）。
        # 对你自己的数据集，请将 `7` 替换为对应数据集的 action dimension。
        return {"actions": np.asarray(data["actions"][:, :7])}
