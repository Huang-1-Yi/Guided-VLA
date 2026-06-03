"""
将 DROID 平台采集的数据集转换为 LeRobot 格式的最小示例脚本。

Usage:
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data

如果想把数据集作为私有数据集推送到 Hugging Face Hub，请使用：
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

生成的数据集会保存到 $LEROBOT_HOME 目录。
"""

from collections import defaultdict
import copy
import glob
import json
from pathlib import Path
import shutil

import cv2
import h5py
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

REPO_NAME = "your_hf_username/my_droid_dataset"  # 输出数据集名称，也用于 Hugging Face Hub。


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def main(data_dir: str, *, push_to_hub: bool = False, private: bool = True):
    # 清理输出目录中已有的数据集。
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    data_dir = Path(data_dir)

    # 创建 LeRobot 数据集，并定义要存储的 features。
    # 这里沿用 DROID 数据命名约定。
    # LeRobot 假设图像数据的 dtype 是 `image`。
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,  # DROID 数据通常以 15fps 录制。
        features={
            # 这里称为 "left"，因为只使用左侧立体相机（遵循 DROID RLDS 约定）。
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),  # DROID RLDS 数据集使用的分辨率。
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # 这里使用 joint velocity actions（7D）+ gripper position（1D）。
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 加载语言标注。
    # 注意：这个示例会加载 DROID 语言标注，但你也可以为自己的数据手动定义语言指令。
    with (data_dir / "aggregated-annotations-030724.json").open() as f:
        language_annotations = json.load(f)

    # 遍历原始 DROID 微调数据集，并将 episode 写入 LeRobot 数据集。
    # 这里假设目录结构如下：
    # RAW_DROID_PATH/
    #   - <...>/
    #     - recordings/
    #        - MP4/
    #          - <camera_id>.mp4  # 左侧立体相机的单视角视频
    #     - trajectory.hdf5
    #   - <...>/
    episode_paths = list(data_dir.glob("**/trajectory.h5"))
    print(f"Found {len(episode_paths)} episodes for conversion")

    # 遍历每个 dataset_name，并将 episode 写入 LeRobot 数据集。
    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        # 加载原始数据。
        recording_folderpath = episode_path.parent / "recordings" / "MP4"
        trajectory = load_trajectory(str(episode_path), recording_folderpath=str(recording_folderpath))

        # 为了加载语言指令，需要从 metadata 文件中解析出 episode_id。
        # 同样，你可以为自己的数据修改这一步，以加载自己的语言指令。
        metadata_filepath = next(iter(episode_path.parent.glob("metadata_*.json")))
        episode_id = metadata_filepath.name.split(".")[0].split("_")[-1]
        language_instruction = language_annotations.get(episode_id, {"language_instruction1": "Do something"})[
            "language_instruction1"
        ]
        print(f"Converting episode with language instruction: {language_instruction}")

        # 写入 LeRobot 数据集。
        for step in trajectory:
            camera_type_dict = step["observation"]["camera_type"]
            wrist_ids = [k for k, v in camera_type_dict.items() if v == 0]
            exterior_ids = [k for k, v in camera_type_dict.items() if v != 0]
            dataset.add_frame(
                {
                    # 注意：加载的图像需要从 BGR 翻转为 RGB。
                    "exterior_image_1_left": resize_image(
                        step["observation"]["image"][exterior_ids[0]][..., ::-1], (320, 180)
                    ),
                    "exterior_image_2_left": resize_image(
                        step["observation"]["image"][exterior_ids[1]][..., ::-1], (320, 180)
                    ),
                    "wrist_image_left": resize_image(step["observation"]["image"][wrist_ids[0]][..., ::-1], (320, 180)),
                    "joint_position": np.asarray(
                        step["observation"]["robot_state"]["joint_positions"], dtype=np.float32
                    ),
                    "gripper_position": np.asarray(
                        step["observation"]["robot_state"]["gripper_position"][None], dtype=np.float32
                    ),
                    # 重要：这里使用 joint velocity actions，因为 pi05-droid 是基于 joint velocity actions 预训练的。
                    "actions": np.concatenate(
                        [step["action"]["joint_velocity"], step["action"]["gripper_position"][None]], dtype=np.float32
                    ),
                    "task": language_instruction,
                }
            )
        dataset.save_episode()

    # 可选推送到 Hugging Face Hub。默认保持 private，避免用户测试转换器时意外发布数据集。
    if push_to_hub:
        dataset.push_to_hub(
            tags=["droid", "panda", "rlds"],
            private=private,
            push_videos=True,
            license="apache-2.0",
        )


##########################################################################################################
################ 用于解析原始 DROID 数据的辅助函数                              #########################
################ 这些工具改编自公开的 R2D2 converter。                           #########################
################ 原始代码来自：https://github.com/JonathanYang0127/r2d2_rlds_dataset_builder/blob/parallel_convert/r2_d2/r2_d2.py
##########################################################################################################


camera_type_dict = {
    "hand_camera_id": 0,
    "varied_camera_1_id": 1,
    "varied_camera_2_id": 1,
}

camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera",
    2: "fixed_camera",
}


def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    return camera_type_to_string_dict[type_int]


class MP4Reader:
    def __init__(self, filepath, serial_number):
        # 保存参数 #
        self.serial_number = serial_number
        self._index = 0

        # 打开视频读取器 #
        self._mp4_reader = cv2.VideoCapture(filepath)
        if not self._mp4_reader.isOpened():
            raise RuntimeError("Corrupted MP4 File")

    def set_reading_parameters(
        self,
        image=True,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        # 保存参数 #
        self.image = image
        self.concatenate_images = concatenate_images
        self.resolution = resolution
        self.resize_func = cv2.resize
        self.skip_reading = not image
        if self.skip_reading:
            return

    def get_frame_resolution(self):
        width = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        return (width, height)

    def get_frame_count(self):
        if self.skip_reading:
            return 0
        return int(self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def set_frame_index(self, index):
        if self.skip_reading:
            return

        if index < self._index:
            self._mp4_reader.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            self._index = index

        while self._index < index:
            self.read_camera(ignore_data=True)

    def _process_frame(self, frame):
        frame = copy.deepcopy(frame)
        if self.resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resolution)

    def read_camera(self, ignore_data=False, correct_timestamp=None):
        # 如果不需要读取，则跳过 #
        if self.skip_reading:
            return {}

        # 读取相机 #
        success, frame = self._mp4_reader.read()

        self._index += 1
        if not success:
            return None
        if ignore_data:
            return None

        # 返回数据 #
        data_dict = {}

        if self.concatenate_images or "stereo" not in self.serial_number:
            data_dict["image"] = {self.serial_number: self._process_frame(frame)}
        else:
            single_width = frame.shape[1] // 2
            data_dict["image"] = {
                self.serial_number + "_left": self._process_frame(frame[:, :single_width, :]),
                self.serial_number + "_right": self._process_frame(frame[:, single_width:, :]),
            }

        return data_dict

    def disable_camera(self):
        if hasattr(self, "_mp4_reader"):
            self._mp4_reader.release()


class RecordedMultiCameraWrapper:
    def __init__(self, recording_folderpath, camera_kwargs={}):  # noqa: B006
        # 保存相机信息 #
        self.camera_kwargs = camera_kwargs

        # 打开相机读取器 #
        mp4_filepaths = glob.glob(recording_folderpath + "/*.mp4")
        all_filepaths = mp4_filepaths

        self.camera_dict = {}
        for f in all_filepaths:
            serial_number = f.split("/")[-1][:-4]
            cam_type = get_camera_type(serial_number)
            camera_kwargs.get(cam_type, {})

            if f.endswith(".mp4"):
                Reader = MP4Reader  # noqa: N806
            else:
                raise ValueError

            self.camera_dict[serial_number] = Reader(f, serial_number)

    def read_cameras(self, index=None, camera_type_dict={}, timestamp_dict={}):  # noqa: B006
        full_obs_dict = defaultdict(dict)

        # 以随机顺序读取相机 #
        all_cam_ids = list(self.camera_dict.keys())
        # random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if "stereo" in cam_id:
                continue
            try:
                cam_type = camera_type_dict[cam_id]
            except KeyError:
                print(f"{self.camera_dict} -- {camera_type_dict}")
                raise ValueError(f"Camera type {cam_id} not found in camera_type_dict")  # noqa: B904
            curr_cam_kwargs = self.camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

            timestamp = timestamp_dict.get(cam_id + "_frame_received", None)
            if index is not None:
                self.camera_dict[cam_id].set_frame_index(index)

            data_dict = self.camera_dict[cam_id].read_camera(correct_timestamp=timestamp)

            # 处理返回数据 #
            if data_dict is None:
                return None
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])

        return full_obs_dict


def get_hdf5_length(hdf5_file, keys_to_ignore=[]):  # noqa: B006
    length = None

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            raise ValueError

        if length is None:
            length = curr_length
        assert curr_length == length

    return length


def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=[]):  # noqa: B006
    data_dict = {}

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict


class TrajectoryReader:
    def __init__(self, filepath, read_images=True):
        self._hdf5_file = h5py.File(filepath, "r")
        is_video_folder = "observations/videos" in self._hdf5_file
        self._read_images = read_images and is_video_folder
        self._length = get_hdf5_length(self._hdf5_file)
        self._video_readers = {}
        self._index = 0

    def length(self):
        return self._length

    def read_timestep(self, index=None, keys_to_ignore=[]):  # noqa: B006
        # 确保读取范围有效 #
        if index is None:
            index = self._index
        else:
            assert not self._read_images
            self._index = index
        assert index < self._length

        # 加载低维数据 #
        keys_to_ignore = [*keys_to_ignore.copy(), "videos"]
        timestep = load_hdf5_to_dict(self._hdf5_file, self._index, keys_to_ignore=keys_to_ignore)

        # 增加读取索引 #
        self._index += 1

        # 返回 timestep #
        return timestep

    def close(self):
        self._hdf5_file.close()


def load_trajectory(
    filepath=None,
    read_cameras=True,
    recording_folderpath=None,
    camera_kwargs={},  # noqa: B006
    remove_skipped_steps=False,
    num_samples_per_traj=None,
    num_samples_per_traj_coeff=1.5,
):
    read_recording_folderpath = read_cameras and (recording_folderpath is not None)

    traj_reader = TrajectoryReader(filepath)
    if read_recording_folderpath:
        camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    horizon = traj_reader.length()
    timestep_list = []

    # 选择要保存的 timestep #
    if num_samples_per_traj:
        num_to_save = num_samples_per_traj
        if remove_skipped_steps:
            num_to_save = int(num_to_save * num_samples_per_traj_coeff)
        max_size = min(num_to_save, horizon)
        indices_to_save = np.sort(np.random.choice(horizon, size=max_size, replace=False))
    else:
        indices_to_save = np.arange(horizon)

    # 遍历轨迹 #
    for i in indices_to_save:
        # 获取 HDF5 数据 #
        timestep = traj_reader.read_timestep(index=i)

        # 如适用，获取录制数据 #
        if read_recording_folderpath:
            timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
            camera_type_dict = {
                k: camera_type_to_string_dict[v] for k, v in timestep["observation"]["camera_type"].items()
            }
            camera_obs = camera_reader.read_cameras(
                index=i, camera_type_dict=camera_type_dict, timestamp_dict=timestamp_dict
            )
            camera_failed = camera_obs is None

            # 如果成功，则将数据添加到 timestep #
            if camera_failed:
                break
            timestep["observation"].update(camera_obs)

        # 过滤 step #
        step_skipped = not timestep["observation"]["controller_info"].get("movement_enabled", True)
        delete_skipped_step = step_skipped and remove_skipped_steps

        # 保存过滤后的 timestep #
        if delete_skipped_step:
            del timestep
        else:
            timestep_list.append(timestep)

    # 移除额外 transition #
    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]

    # 关闭读取器 #
    traj_reader.close()

    # 返回数据 #
    return timestep_list


if __name__ == "__main__":
    tyro.cli(main)
