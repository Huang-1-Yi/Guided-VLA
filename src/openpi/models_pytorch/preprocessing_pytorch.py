from collections.abc import Sequence
import logging
import math

import torch

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# 从 model.py 移出的常量。
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)

# 按 (batch_shape, device) 缓存默认的全 True image masks，
# 避免图像缺少显式 mask 时每次 forward 都重新分配 torch.ones。
_DEFAULT_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _default_image_mask(batch_shape: torch.Size, device: torch.device) -> torch.Tensor:
    key = (tuple(batch_shape), device)
    mask = _DEFAULT_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.ones(batch_shape, dtype=torch.bool, device=device)
        _DEFAULT_MASK_CACHE[key] = mask
    return mask


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """简化类型标注、兼容 torch.compile 的 preprocess_observation_pytorch 版本。

    该函数避免使用可能导致 torch.compile 问题的复杂类型标注。
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # 同时处理 [B, C, H, W] 和 [B, H, W, C] 格式。
        is_channels_first = image.shape[1] == 3  # 检查 channel 是否位于第 1 维。

        if is_channels_first:
            # 将 [B, C, H, W] 转为 [B, H, W, C] 以便处理。
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # 为 PyTorch augmentations 从 [-1, 1] 转换到 [0, 1]。
            image = image / 2.0 + 0.5

            # 应用基于 PyTorch 的 augmentations。
            if "wrist" not in key:
                # 非 wrist 相机的几何增强。
                height, width = image.shape[1:3]

                # 随机裁剪并缩放。
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # 随机裁剪。
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # 在 CPU 上采样，使 slice 边界成为普通 Python int。
                    # 使用 GPU 0-d tensor 作为 slice 边界会强制隐式 .item()，
                    # 并触发 cudaStreamSynchronize。该路径是 eager（未编译）路径。
                    start_h = int(torch.randint(0, max_h + 1, (1,)).item())
                    start_w = int(torch.randint(0, max_w + 1, (1,)).item())
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # 缩放回原始尺寸。
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # 随机旋转（小角度）：在 CPU 上采样，使显著性检查成为纯 Python bool，
                # 避免 GPU 到 CPU 同步。
                angle_deg = float(torch.rand(1).item()) * 10.0 - 5.0
                if abs(angle_deg) > 0.1:
                    angle_rad = angle_deg * math.pi / 180.0
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)

                    # 使用 grid_sample 应用旋转。
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # 创建 meshgrid。
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # 扩展到 batch 维度。
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # 应用旋转变换。
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # 为 grid_sample stack 并 reshape。
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # 颜色增强：将三次随机采样合并到一个 kernel 中。
            color_rand = torch.rand(3, device=image.device)

            # [0.7, 1.3] 范围内的随机亮度。
            image = image * (0.7 + color_rand[0] * 0.6)

            # [0.6, 1.4] 范围内的随机对比度。
            contrast_factor = 0.6 + color_rand[1] * 0.8
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # [0.5, 1.5] 范围内的随机饱和度。
            saturation_factor = 0.5 + color_rand[2] * 1.0
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # 将数值 clamp 到 [0, 1]。
            image = torch.clamp(image, 0, 1)

            # 转回 [-1, 1]。
            image = image * 2.0 - 1.0

        # .contiguous() 让 dim-1 stride 在 train/eval 路径间保持稳定；如果没有它，
        # torch.compile embed_image 会在运行中重新编译，并导致 DDP ranks 不同步。
        if is_channels_first:
            image = image.permute(0, 3, 1, 2).contiguous()

        out_images[key] = image

    # 获取 mask。
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 默认不屏蔽：缓存的全 True tensor 可安全共享（只读）。
            out_masks[key] = _default_image_mask(batch_shape, observation.state.device)
        else:
            # 规范化为训练时 mask shape `[*batch]`。对已经正确的 mask 来说这是廉价 view，
            # 也会移除某些推理路径意外带上的 singleton suffix dims，例如 `[B, 1] -> [B]`。
            out_masks[key] = observation.image_masks[key].reshape(batch_shape)

    # 创建一个带所需属性的简单对象，而不是使用复杂的 Observation class。
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        skill_id=getattr(observation, "skill_id", None),
        skill_soft=getattr(observation, "skill_soft", None),
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
