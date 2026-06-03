"""
基于 RLDS 的 DROID data loader。
openpi 通常使用 LeRobot 的 data loader，但它目前还不足以扩展到 DROID 这样的大数据集。
因此这里提供了一个使用 RLDS 数据格式的 data loader 示例。
该 data loader 还会应用一些 DROID 专用的数据过滤和转换。
"""

from enum import Enum
from enum import auto
import json
import logging
from pathlib import Path

import tqdm

import openpi.shared.download as download


class DroidActionSpace(Enum):
    """DROID 数据集的 action space。"""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # 强制后续参数只能通过关键字传入。
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # 默认使用 joint position actions，因为它们支持在仿真中评估 policy。
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # 如果内存不足可以减小该值，但需注意：低于约 100k 时 shuffle 随机性不够。
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE；导入 TensorFlow 后解析。
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE；导入 TensorFlow 后解析。
        filter_dict_path=None,  # 包含训练采样索引的 json 文件路径。
    ):
        # 在这里导入 tensorflow，避免未使用 RLDS data loader 时把它变成强制依赖。
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # 配置 TensorFlow 不使用 GPU 设备，避免与 PyTorch / JAX 冲突。
        tf.config.set_visible_devices([], "GPU")

        builder = tfds.builder("droid", data_dir=data_dir, version="1.0.1")
        dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

        # 过滤掉失败轨迹。这里通过文件名判断。
        dataset = dataset.filter(
            lambda traj: tf.strings.regex_full_match(
                traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
            )
        )

        # 重复 dataset，确保数据不会耗尽。
        dataset = dataset.repeat()

        # 如果提供了 filter dictionary，则加载它。
        # filter dictionary 是一个 JSON 文件，将 episode key 映射到待采样的帧范围：
        # 例如：
        # {
        #     "<episode key>": [[0, 100], [200, 300]]
        # }
        # 表示保留帧 0-99 和 200-299。
        if filter_dict_path is not None:
            cached_filter_dict_path = download.maybe_download(filter_dict_path)
            with Path(cached_filter_dict_path).open("r") as f:
                filter_dict = json.load(f)

            logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

            keys_tensor = []
            values_tensor = []

            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                for start, end in ranges:
                    for t in range(start, end):
                        frame_key = f"{episode_key}--{t}"
                        keys_tensor.append(frame_key)
                        values_tensor.append(True)
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
            )
            logging.info("Filter hash table initialized")
        else:
            self.filter_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
            )

        def restructure(traj):
            """重新组织 observation 和 action key，并采样语言指令。"""
            # 重要：这里使用 joint position action space，更容易仿真。
            actions = tf.concat(
                (
                    (
                        traj["action_dict"]["joint_position"]
                        if action_space == DroidActionSpace.JOINT_POSITION
                        else traj["action_dict"]["joint_velocity"]
                    ),
                    traj["action_dict"]["gripper_position"],
                ),
                axis=-1,
            )
            # 训练时从 DROID 的两张 exterior images 中随机采样一张（每次只训练一张）。
            # 注意："left" 指立体相机对中的左相机，这里只在左相机上训练。
            exterior_img = tf.cond(
                tf.random.uniform(shape=[]) > 0.5,
                lambda: traj["observation"]["exterior_image_1_left"],
                lambda: traj["observation"]["exterior_image_2_left"],
            )
            wrist_img = traj["observation"]["wrist_image_left"]
            # 从三条语言指令中随机采样一条。
            instruction = tf.random.shuffle(
                [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
            )[0]

            traj_len = tf.shape(traj["action"])[0]
            indices = tf.as_string(tf.range(traj_len))

            # 数据过滤：
            # 通过拼接 recording folderpath、file path 和每个 step 的时间步索引，计算唯一 step ID。
            # 该 ID 会索引 filter hash table；如果返回 true，则该帧通过过滤。
            step_id = (
                traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                + "--"
                + traj["traj_metadata"]["episode_metadata"]["file_path"]
                + "--"
                + indices
            )
            passes_filter = self.filter_table.lookup(step_id)

            return {
                "actions": actions,
                "observation": {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                },
                "prompt": instruction,
                "step_id": step_id,
                "passes_filter": passes_filter,
            }

        dataset = dataset.traj_map(restructure, num_parallel_calls)

        def chunk_actions(traj):
            """将 episode 拆分为 action chunks。"""
            traj_len = tf.shape(traj["actions"])[0]

            # 对轨迹中的每个 step，构造后续 n 个 actions 的索引。
            action_chunk_indices = tf.broadcast_to(
                tf.range(action_chunk_size)[None],
                [traj_len, action_chunk_size],
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None],
                [traj_len, action_chunk_size],
            )

            # 截断到序列长度以内，最后的 chunks 会重复最后一个 action。
            # 由于这里使用 absolute joint + gripper position actions，这样处理是合理的。
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

            # 为每个 chunk 收集 actions。
            traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
            return traj

        dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

        # flatten：从 trajectory dataset 映射为单个 action chunk 的 dataset。
        dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

        # 过滤未通过 filter 的数据。
        def filter_from_dict(frame):
            return frame["passes_filter"]

        dataset = dataset.filter(filter_from_dict)

        # 从输出中移除 "passes_filter" key。
        def remove_passes_filter(frame):
            frame.pop("passes_filter")
            return frame

        dataset = dataset.map(remove_passes_filter)

        # 解码图像：RLDS 保存的是编码后图像，为了效率只在此处解码。
        def decode_images(traj):
            traj["observation"]["image"] = tf.io.decode_image(
                traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
            )
            traj["observation"]["wrist_image"] = tf.io.decode_image(
                traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
            )
            return traj

        dataset = dataset.frame_map(decode_images, num_parallel_calls)

        # shuffle 并 batch。
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        # 注意：这似乎可以降低内存占用，且不影响速度。
        dataset = dataset.with_ram_budget(1)

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # 这是 DROID 过滤后的近似样本数。
        # 硬编码比遍历整个数据集计算更简单。
        return 20_000_000
