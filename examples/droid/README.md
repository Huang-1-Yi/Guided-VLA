# openpi 中的 DROID Policies

本文档提供以下说明：

- [运行最佳 `$pi_{0.5}$`-DROID policy 推理](./README.md#running-droid-inference)
- [运行其他预训练 DROID policies（`$\pi_0$`、`$\pi_0$`-FAST 等）推理](./README.md#running-roboarena-baseline-policies)
- [在完整 DROID 数据集上预训练 *generalist* policies](./README_train.md#training-on-droid)
- [在你自己的 DROID 数据集上微调 expert `$\pi_{0.5}$`](./README_train.md#fine-tuning-on-custom-droid-datasets)

## 运行 DROID 推理

该示例展示如何在 [DROID robot platform](https://github.com/droid-dataset/droid) 上运行微调后的 `$\pi_{0.5}$`-DROID 模型。根据公开的 [RoboArena benchmark](https://robo-arena.github.io/leaderboard)，这是目前我们最强的 generalist DROID policy。

### Step 1: Start a policy server

由于 DROID 控制笔记本没有强力 GPU，我们会在另一台配备更强 GPU 的机器上启动远程 policy server，然后在推理时从 DROID 控制笔记本查询该服务器。

1. 在一台配备强力 GPU（约 NVIDIA 4090 级别）的机器上，按照 [README](https://github.com/Physical-Intelligence/openpi) 中的说明克隆并安装 `openpi` 仓库。
2. 使用以下命令启动 OpenPI server：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

也可以运行下面的等价命令：

```bash
uv run scripts/serve_policy.py --env=DROID
```

### Step 2: Run the DROID robot

1. 确认 DROID 控制笔记本和 NUC 上都安装了最新版本的 DROID package。
2. 在控制笔记本上激活你的 DROID conda 环境。
3. 克隆 openpi 仓库并安装 openpi client，它用于连接 policy server，依赖很少、安装很快：在已激活的 DROID conda 环境中运行 `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .`。
4. 安装 `tyro`，用于命令行解析：`pip install tyro`。
5. 将本目录下的 `main.py` 文件复制到 `$DROID_ROOT/scripts` 目录。
6. 将 `main.py` 中的 camera ID 替换为你的相机 ID。可以在命令行运行 `ZED_Explorer` 查看所有已连接相机及其 ID，也可以用它确认相机是否摆放到能看清目标场景的位置。
7. 运行 `main.py`。请确保 IP 和 host 地址指向 policy server。可以在 DROID 笔记本上运行 `ping <server_ip>`，确认 server 机器可访问。同时请指定 policy 使用的外部相机；我们只输入一个外部相机，可从 ["left", "right"] 中选择。

```bash
python3 scripts/main.py --remote_host=<server_ip> --remote_port=<server_port> --external_camera="left"
```

脚本会要求你输入一条自由形式的语言指令，让机器人执行。请将相机对准你希望机器人交互的场景。你不需要非常精细地控制相机角度、物体位置等；根据我们的经验，该 policy 具有相当的鲁棒性。祝你 prompt 愉快！

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Cannot reach policy server | 确认 server 已启动，且 IP 和端口正确。可以在 DROID 笔记本上运行 `ping <server_ip>` 检查 server 机器是否可达。 |
| Cannot find cameras | 确认 camera ID 正确，且相机已连接到 DROID 笔记本。有时重新插拔相机会有帮助。可以在命令行运行 `ZED_Explore` 查看所有已连接相机。 |
| Policy inference is slow / inconsistent | 尝试为 DROID 笔记本使用有线网络，以降低延迟；每个 chunk 约 0.5 到 1 秒延迟是正常的。 |
| Policy does not perform the task well | 在我们的实验中，该 policy 可以在多种环境、相机位置和光照条件下执行简单桌面操作任务（pick-and-place）。如果任务表现不好，可以尝试调整场景或物体位置，让任务更容易。也请确认传给 policy 的相机视角能看到场景中的所有相关物体；该 policy 只基于单个外部相机和腕部相机，请确认输入的是期望的相机。可使用 `ZED_Explore` 检查传给 policy 的相机画面是否覆盖所有相关物体。最后，该 policy 并不完美，遇到更复杂的操作任务仍可能失败，但通常会做出相当不错的尝试。 |

## 运行其他 Policies

我们提供了用于运行 [RoboArena](https://robo-arena.github.io/) 论文中 baseline DROID policies 的配置。只需运行下面命令，即可为对应 policy 启动推理服务器。之后按照上面的说明在 DROID 机器人上运行评估。

```
# Train from pi0-FAST, using FAST tokenizer
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid

# Train from pi0, using flow matching
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_droid

# Trained from PaliGemma, using RT-2 / OpenVLA style binning tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_binning_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_binning_droid

# Trained from PaliGemma, using FAST tokenizer (using universal FAST+ tokenizer).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_droid

# Trained from PaliGemma, using FAST tokenizer (tokenizer trained on DROID dataset).
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_fast_specialist_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_fast_specialist_droid

# Trained from PaliGemma, using FSQ tokenizer.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_vq_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_vq_droid

# pi0-style diffusion / flow VLA, trained on DROID from PaliGemma.
uv run scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_diffusion_droid --policy.dir=gs://openpi-assets/checkpoints/roboarena/paligemma_diffusion_droid
```

推理配置可以在 [roboarena_config.py](../../src/openpi/training/misc/roboarena_config.py) 中找到。
