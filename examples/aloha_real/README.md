# 运行 Aloha（真实机器人）

该示例展示如何在 [ALOHA setup](https://github.com/tonyzhaozh/aloha) 上运行真实机器人。关于如何加载 checkpoint 并运行推理，请参考[这里](../../docs/remote_inference.md)。下面列出了每个已提供微调模型对应的 checkpoint 路径。

## 前置条件

本仓库使用了 ALOHA 仓库的一个 fork，只做了很小的修改，用于支持 Realsense 相机。

1. 按照 ALOHA 仓库中的[硬件安装说明](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation)完成硬件设置。
1. 修改 `third_party/aloha/aloha_scripts/realsense_publisher.py` 文件，使用你的相机 serial number。

## 使用 Docker

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='take the toast out of the toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

## 不使用 Docker

终端窗口 1：

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python -m examples.aloha_real.main
```

终端窗口 2：

```bash
roslaunch aloha ros_nodes.launch
```

终端窗口 3：

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='take the toast out of the toaster'
```

## **ALOHA Checkpoint Guide**

`pi0_base` 模型可以在 ALOHA 平台上以 zero-shot 方式完成一个简单任务；此外我们还提供了两个示例微调 checkpoint，分别对应 “fold the towel” 和 “open the tupperware and put the food on the plate”，可在 ALOHA 上执行更高级的任务。

虽然我们观察到这些 policy 可以在多个 ALOHA 站点的未见条件下工作，但这里仍提供一些场景设置建议，以尽量提高成功率。我们会说明各 policy 使用的 prompt、已经验证过表现较好的物体，以及较有代表性的初始状态分布。zero-shot 运行这些 policy 仍然是非常实验性的功能，并不保证一定能在你的机器人上工作。使用 `pi0_base` 的推荐方式，是使用目标机器人采集的数据进行 finetuning。

---

### **Toast Task**

该任务要求机器人从烤面包机中取出两片吐司，并放到盘子上。

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_base`
- **Prompt**: "take the toast out of the toaster"
- **Objects needed**: 两片吐司、一个盘子和一个标准烤面包机。
- **Object Distribution**:
  - 真实吐司和橡胶仿真吐司均可
  - 兼容标准双片烤面包机
  - 可适配不同颜色的盘子

### **场景设置建议**
<img width="500" alt="Screenshot 2025-01-31 at 10 06 02 PM" src="https://github.com/user-attachments/assets/3d043d95-9d1c-4dda-9991-e63cae61e02e" />

- 烤面包机应放在工作空间的左上象限。
- 两片吐司初始时都应在烤面包机内，并且顶部至少露出 1 cm。
- 盘子应大致放在工作空间的下方中央。
- 自然光和人造光都可以，但应避免场景过暗，例如不要把装置放在封闭空间或帘子下方。

### **Towel Task**

该任务要求机器人将一条小毛巾折成八等分，例如手巾大小的毛巾。

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_aloha_towel`
- **Prompt**: "fold the towel"
- **Object Distribution**:
  - 可处理不同纯色毛巾
  - 对纹理很重或条纹明显的毛巾表现较差

### **场景设置建议**
<img width="500" alt="Screenshot 2025-01-31 at 10 01 15 PM" src="https://github.com/user-attachments/assets/9410090c-467d-4a9c-ac76-96e5b4d00943" />

- 毛巾应摊平，并大致位于桌面中央。
- 请选择不会与桌面颜色混在一起的毛巾。

### **Tupperware Task**

该任务要求机器人打开装有食物的保鲜盒，并将内容物倒到盘子上。

- **Checkpoint path**: `gs://openpi-assets/checkpoints/pi0_aloha_tupperware`
- **Prompt**: "open the tupperware and put the food on the plate"
- **Objects needed**: 保鲜盒、食物（或类似食物的物体）和一个盘子。
- **Object Distribution**:
  - 可处理多种仿真食物，例如仿真鸡块、薯条和炸鸡。
  - 兼容不同盖子颜色和形状的保鲜盒；带角部翻盖的方形保鲜盒效果最好（见下图）。
  - policy 训练中见过多种纯色盘子。

### **场景设置建议**
<img width="500" alt="Screenshot 2025-01-31 at 10 02 27 PM" src="https://github.com/user-attachments/assets/60fc1de0-2d64-4076-b903-f427e5e9d1bf" />

- 当保鲜盒和盘子都大致位于工作空间中央时，通常效果最好。
- 摆放方式：
  - 保鲜盒应位于左侧。
  - 盘子应位于右侧或下方。
  - 保鲜盒翻盖应朝向盘子。

## 在你自己的 Aloha 数据集上训练

1. 将数据集转换为 LeRobot dataset v2.0 格式。

    我们提供了 [convert_aloha_data_to_lerobot.py](./convert_aloha_data_to_lerobot.py) 脚本，用于将数据集转换为 LeRobot dataset v2.0 格式。作为示例，我们已经将 [BiPlay repo](https://huggingface.co/datasets/oier-mees/BiPlay/tree/main/aloha_pen_uncap_diverse_raw) 中的 `aloha_pen_uncap_diverse_raw` 数据集转换并上传到 Hugging Face Hub，地址为 [physical-intelligence/aloha_pen_uncap_diverse](https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse)。

2. 定义一个使用自定义数据集的训练 config。

    我们提供了 [pi0_aloha_pen_uncap config](../../src/openpi/training/config.py) 作为示例。关于如何使用新 config 运行训练，请参考根目录 [README](../../README.md)。

IMPORTANT：我们的 base checkpoint 包含多种常见机器人配置的 normalization stats。当使用这些配置之一采集的自定义数据集微调 base checkpoint 时，推荐使用 base checkpoint 中提供的对应 normalization stats。在示例中，这是通过在 `AssetsConfig` 中指定 `trossen` asset_id，并指向预训练 checkpoint 的 asset 目录来实现的。
