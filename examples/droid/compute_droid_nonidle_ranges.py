"""
遍历 DROID 数据集，并创建一个 JSON 映射：从 episode 唯一 ID 到训练时应采样的时间步范围
（其余时间步会被过滤）。

过滤逻辑：
寻找最多包含 min_idle_len 个连续静止帧的连续 step 区间（默认 7；因为多数 DROID action-chunking
policy 会执行每个 chunk 里生成的前 8 个动作，这样过滤可以避免 policy 卡在持续输出静止动作）。
此外，只保留长度至少为 min_non_idle_len 的非静止区间（默认 16 帧，约 1 秒），并从每个区间末尾移除
filter_last_n_in_ranges 帧（这些帧通常对应包含大量静止动作的 action chunk）。

这样会留下由连续、显著运动组成的轨迹片段。用这个过滤后的集合训练，得到的 policy 会输出更少静止动作
（也就是更不容易“卡住”）。
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 设置为要使用的 GPU；如果留空则使用 CPU。

builder = tfds.builder_from_directory(
    # `droid` 目录路径（不是它的父目录）。
    builder_dir="<path_to_droid_dataset_tfds_files>",
)
ds = builder.as_dataset(split="train", shuffle_files=False)
tf.data.experimental.ignore_errors(ds)

keep_ranges_path = "<path_to_where_to_save_the_json>"

min_idle_len = 7  # 如果连续静止帧数量超过该值，则全部过滤。
min_non_idle_len = 16  # 如果连续非静止帧数量少于该值，则全部过滤。
filter_last_n_in_ranges = 10  # 使用 filter dict 时，从每个区间末尾移除这么多帧。

keep_ranges_map = {}
if Path(keep_ranges_path).exists():
    with Path(keep_ranges_path).open("r") as f:
        keep_ranges_map = json.load(f)
    print(f"Resuming from {len(keep_ranges_map)} episodes already processed")

for ep_idx, ep in enumerate(tqdm(ds)):
    recording_folderpath = ep["episode_metadata"]["recording_folderpath"].numpy().decode()
    file_path = ep["episode_metadata"]["file_path"].numpy().decode()

    key = f"{recording_folderpath}--{file_path}"
    if key in keep_ranges_map:
        continue

    joint_velocities = [step["action_dict"]["joint_velocity"].numpy() for step in ep["steps"]]
    joint_velocities = np.array(joint_velocities)

    is_idle_array = np.hstack(
        [np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)]
    )

    # 找出从静止到非静止、以及从非静止到静止的 step。
    is_idle_padded = np.concatenate(
        [[False], is_idle_array, [False]]
    )  # 首尾为 False，这样如果第一步是静止，也会被视为一段运动的开始。

    is_idle_diff = np.diff(is_idle_padded.astype(int))
    is_idle_true_starts = np.where(is_idle_diff == 1)[0]  # +1 transition 表示从静止到非静止。
    is_idle_true_ends = np.where(is_idle_diff == -1)[0]  # -1 transition 表示从非静止到静止。

    # 找出长度至少为 min_idle_len 的静止片段对应哪些 step。
    true_segment_masks = (is_idle_true_ends - is_idle_true_starts) >= min_idle_len
    is_idle_true_starts = is_idle_true_starts[true_segment_masks]
    is_idle_true_ends = is_idle_true_ends[true_segment_masks]

    keep_mask = np.ones(len(joint_velocities), dtype=bool)
    for start, end in zip(is_idle_true_starts, is_idle_true_ends, strict=True):
        keep_mask[start:end] = False

    # 获取所有长度至少为 16 的非静止区间。
    # 逻辑同上，但作用于 keep_mask，用来过滤长度小于 min_non_idle_len 的连续区间。
    keep_padded = np.concatenate([[False], keep_mask, [False]])

    keep_diff = np.diff(keep_padded.astype(int))
    keep_true_starts = np.where(keep_diff == 1)[0]  # +1 transition 表示从过滤切换到保留。
    keep_true_ends = np.where(keep_diff == -1)[0]  # -1 transition 表示从保留切换到过滤。

    # 找出长度至少为 min_non_idle_len 的非静止片段对应哪些 step。
    true_segment_masks = (keep_true_ends - keep_true_starts) >= min_non_idle_len
    keep_true_starts = keep_true_starts[true_segment_masks]
    keep_true_ends = keep_true_ends[true_segment_masks]

    # 添加从 episode 唯一 ID key 到待保留非静止区间列表的映射。
    keep_ranges_map[key] = []
    for start, end in zip(keep_true_starts, keep_true_ends, strict=True):
        keep_ranges_map[key].append((int(start), int(end) - filter_last_n_in_ranges))

    if ep_idx % 1000 == 0:
        with Path(keep_ranges_path).open("w") as f:
            json.dump(keep_ranges_map, f)

print("Done!")
with Path(keep_ranges_path).open("w") as f:
    json.dump(keep_ranges_map, f)
