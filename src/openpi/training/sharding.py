import contextlib
import logging

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# 在 FSDP 中，数据会同时沿 batch 和 FSDP 轴分片。
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """在模块树深处传递 mesh 非常繁琐。

    在 JAX 团队提供更好的 API 之前，使用这样的自定义 context manager 是维护全局 mesh 引用的推荐方式。
    这里仅供下面的 `activation_sharding_constraint` 使用。
    """
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """根据 mesh 形状对 array pytree 应用 FSDP 分片。

    Args:
        pytree: 要根据 mesh 应用分片的 pytree。注意，只有 array 类型（例如包含 .shape 属性）
          会被纳入分片考虑。
        mesh: 用于对 pytree 应用分片的 mesh。
        min_size_mbytes: 会被考虑分片的 array 最小大小，单位 MiB；小于该值的 array 会被复制。
        log: 如果为 true，会记录被考虑分片的 array 的分片决策。

    Returns:
        分片后的 pytree。
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # 如果实际上不会使用 FSDP，则复制所有内容以避免多余日志。
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # 复制标量和向量 array。
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # 复制较小 array。
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # 对矩阵和更大的 tensor，沿可被 FSDP 维度整除的最大轴分片。
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # 如果找不到有效分片方式，则复制。
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
