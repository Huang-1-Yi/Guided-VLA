import dataclasses

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


def make_calvin_example() -> dict:
    """为 CALVIN policy 创建一个随机输入示例。"""
    return {
        "state_ee_pos": np.random.rand(3),
        "state_ee_rot": np.random.rand(3),
        "state_gripper": np.random.rand(1),
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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
class CalvinInputs(transforms.DataTransformFn):
    """
    将输入数据转换为模型期望的格式，训练和推理都会使用。

    如果要适配自己的数据集，可以复制这个类，并根据下面注释修改 key，
    把数据集中正确的字段接入模型。
    """

    # 数据契约：model_type 只决定占位图是否被 image_mask 屏蔽；自定义 dataset 通常不用改这个字段。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 数据契约：CALVIN 图像在这里统一转成 uint8 HWC，抹平 LeRobot 常见的 float32 CHW 存储差异。
        # 如果自定义 dataset 的图像 key 不同，只改这里读取的 key，并保持下面 openpi camera slot 名称不变。
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])

        # 数据契约：CALVIN 的 ee_pos、ee_rot、gripper 在这里拼成 openpi 统一的 state 向量。
        # 输出 dict 仍必须使用 state、image、image_mask、actions、prompt 这些 openpi key。
        inputs = {
            "state": np.concatenate(
                [
                    data["state_ee_pos"],
                    data["state_ee_rot"],
                    data["state_gripper"],
                ],
                axis=0,
            ),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # 缺失的右腕视角用零图占位，保持模型侧 camera slot 固定。
                # 如果你的数据确实有右腕图像，替换这里的零图即可。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # pi0/pi05 屏蔽占位图；pi0-FAST 保留占位图参与 tokenization，避免 FAST token 槽位变化。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # CALVIN 训练 action 由 delta ee_pos、delta ee_rot 和 gripper 拼接而成。
        # actions 只会出现在训练样本中，在线推理时不会提供。
        if "action_delta_ee_pos" in data and "action_delta_ee_rot" in data and "action_gripper" in data:
            inputs["actions"] = torch.cat(
                [data["action_delta_ee_pos"], data["action_delta_ee_rot"], data["action_gripper"]], axis=1
            )

        # 语言指令统一放在 prompt，供后续 tokenizer 处理；若源字段不叫 prompt，只改右侧读取 key。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CalvinOutputs(transforms.DataTransformFn):
    """
    将模型输出转换回 CALVIN 数据集/环境需要的格式，仅用于推理。

    如果要适配自己的数据集，可以复制这个类，并根据下面注释修改动作维度。
    """

    def __call__(self, data: dict) -> dict:
        # CALVIN 执行端只消费前 7 维动作：delta ee_pos、delta ee_rot 和 gripper。
        # 自定义机器人需要改成真实 action_dim。
        return {"actions": np.asarray(data["actions"][:, :7])}
