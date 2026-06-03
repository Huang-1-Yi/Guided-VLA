"""为 msgpack 增加 NumPy array 支持。

msgpack 适合在网络上序列化/反序列化数据，原因包括：
- msgpack 更安全（不同于 pickle/dill 等允许任意代码执行的格式）
- msgpack 使用广泛，并且有良好的跨语言支持
- msgpack 不需要 schema（不同于 protobuf/flatbuffers 等），这对 Python 和 JavaScript
    这样的动态类型语言很方便
- msgpack 快速且高效（不同于 JSON/YAML 等可读格式）；使用下面的策略序列化大 array 时，
    我观察到 msgpack 大约比 pickle 快 4 倍

下面的代码改编自 https://github.com/lebedov/msgpack-numpy。没有直接使用该库的原因是：
它会在处理 object array 时回退到 pickle。
"""

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
