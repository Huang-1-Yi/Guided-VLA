import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """如果图像是 float 类型，则转换为 uint8。

    这对降低网络传输图像时的数据大小很重要。
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """用 PIL 为多张图像复现 tf.image.resize_with_pad，将一批图像缩放到目标高度。

    Args:
        images: 格式为 [..., height, width, channel] 的一批图像。
        height: 图像目标高度。
        width: 图像目标宽度。
        method: 使用的插值方法，默认为 bilinear。

    Returns:
        格式为 [..., height, width, channel] 的缩放后图像。
    """
    # 如果图像已经是正确尺寸，则直接返回。
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """用 PIL 为单张图像复现 tf.image.resize_with_pad。

    通过零填充把图像无失真地缩放到目标高宽。注意，不同于 jax 版本，
    PIL 使用 [width, height, channel] 顺序，而不是 [batch, h, w, c]。
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # 如果图像已经是正确尺寸，则无需缩放。

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
