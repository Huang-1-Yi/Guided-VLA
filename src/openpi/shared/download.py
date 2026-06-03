import concurrent.futures
import datetime
import logging
import os
import pathlib
import re
import shutil
import stat
import time
import urllib.parse

import filelock
import fsspec
import fsspec.generic
import tqdm_loggable.auto as tqdm

# 用于控制缓存目录路径的环境变量，默认使用 ~/.cache/openpi。
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"
DEFAULT_CACHE_DIR = "~/.cache/openpi"

logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path:
    cache_dir = pathlib.Path(os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def maybe_download(url: str, *, force_download: bool = False, **kwargs) -> pathlib.Path:
    """从远程文件系统下载文件或目录到本地缓存，并返回本地路径。

    如果本地文件已经存在，则直接返回该路径。

    这个函数可以被多个进程并发安全地调用。缓存目录的更多细节见 `get_cache_dir`。

    Args:
        url: 需要下载的文件 URL。
        force_download: 如果为 True，即使文件已在缓存中也会重新下载。
        **kwargs: 传给 fsspec 的额外参数。

    Returns:
        下载后文件或目录的本地路径。该路径保证存在，并且是绝对路径。
    """
    # 不用 fsspec 解析 url，避免不必要地连接远程文件系统。
    parsed = urllib.parse.urlparse(url)

    # 如果是本地路径，直接返回。
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()

    cache_dir = get_cache_dir()

    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
    local_path = local_path.resolve()

    # 检查缓存是否需要失效。
    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True
        else:
            return local_path

    try:
        lock_path = local_path.with_suffix(".lock")
        with filelock.FileLock(lock_path):
            # 确保 lock 文件权限一致。
            _ensure_permissions(lock_path)
            # 如果已有缓存已过期，先删除它。
            if invalidate_cache:
                logger.info(f"Removing expired cached entry: {local_path}")
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()

            # 将数据下载到本地缓存。
            logger.info(f"Downloading {url} to {local_path}")
            scratch_path = local_path.with_suffix(".partial")
            _download_fsspec(url, scratch_path, **kwargs)

            shutil.move(scratch_path, local_path)
            _ensure_permissions(local_path)

    except PermissionError as e:
        msg = (
            f"Local file permission error was encountered while downloading {url}. "
            f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
        )
        raise PermissionError(msg) from e

    return local_path


def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
    """从远程文件系统下载文件到本地缓存。"""
    fs, _ = fsspec.core.url_to_fs(url, **kwargs)
    info = fs.info(url)
    # 文件夹会表示为大小为 0、并以正斜杠结尾的对象。
    if is_dir := (info["type"] == "directory" or (info["size"] == 0 and info["name"].endswith("/"))):
        total_size = fs.du(url)
    else:
        total_size = info["size"]
    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, url, local_path, recursive=is_dir)
        while not future.done():
            current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
            pbar.update(current_size - pbar.n)
            time.sleep(1)
        pbar.update(total_size - pbar.n)


def _set_permission(path: pathlib.Path, target_permission: int):
    """chmod 需要设置可执行权限；如果权限已经匹配目标值，则跳过。"""
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"Skipping {path} because it already has correct permissions")
        return
    path.chmod(target_permission)
    logger.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """将文件夹权限设置为可读、可写、可搜索。"""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    """缓存目录会被容器化运行时和训练脚本共享，因此需要确保它具有正确权限。
    """

    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """将所有文件设为可读写；如果文件原本是脚本，则保留脚本权限。"""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _get_mtime(year: int, month: int, day: int) -> float:
    """获取给定日期在 UTC 零点的 mtime。"""
    date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)
    return time.mktime(date.timetuple())


# 相对路径正则表达式到过期时间戳（mtime 格式）的映射。
# 会从上到下做部分匹配，并选择第一个匹配项。
# 只有比过期时间戳更新的缓存项才会被保留。
_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): _get_mtime(2025, 2, 17),
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    """如果缓存已过期则令其失效；若缓存需要失效则返回 True。"""

    assert local_path.exists(), f"File not found at {local_path}"

    relative_path = str(local_path.relative_to(cache_dir))
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(relative_path):
            # 如果不比过期时间戳更新，则删除。
            return local_path.stat().st_mtime <= expire_time

    return False
