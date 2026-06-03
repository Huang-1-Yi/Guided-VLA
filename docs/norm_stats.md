# Normalization statistics

按照常见做法，我们的模型会在 policy 训练和推理期间对 proprioceptive state inputs 和 action targets 进行归一化。用于归一化的统计量会在训练数据上计算，并与模型 checkpoint 一起保存。

## 重新加载 normalization statistics

当你在新数据集上微调我们的模型时，需要决定是（A）复用已有 normalization statistics，还是（B）在新的训练数据上重新计算统计量。哪种方式更适合，取决于你的机器人和任务与预训练数据集中机器人/任务分布的相似程度。下面列出了每个模型可用的预训练 normalization statistics。

**如果你的目标机器人匹配其中一组预训练统计量，可以考虑重新加载相同的 normalization statistics。** 通过重新加载这些统计量，你的数据集中的 actions 对模型来说会更“熟悉”，这可能带来更好的表现。可以在训练 config 中添加 `AssetsConfig`，让它指向对应 checkpoint 目录和 normalization statistics ID。下面示例使用 `pi0_base` checkpoint 中 `Trossen`（也就是 ALOHA）机器人的统计量：

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

完整训练 config 示例可参考 [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py) 中会重新加载 normalization statistics 的 `pi0_aloha_pen_uncap` config。

**注意：** 要成功重新加载 normalization statistics，你的机器人和数据集必须遵循预训练时使用的 action space 定义。下面会详细说明我们的 action space 定义。

**注意 #2：** 重新加载 normalization statistics 是否有益，取决于你的机器人和任务与预训练数据中机器人/任务分布的相似程度。我们建议始终同时尝试两种方式：一种是重新加载统计量，另一种是使用新数据集重新计算一套统计量进行训练（如何计算新统计量请见 [main README](../README.md)），最后选择对你的任务效果更好的方案。

## 已提供的预训练 Normalization Statistics

下面列出了我们提供的全部预训练 normalization statistics。它们同时支持 `pi0_base` 和 `pi0_fast_base` 模型。对于 `pi0_base`，将 `assets_dir` 设为 `gs://openpi-assets/checkpoints/pi0_base/assets`；对于 `pi0_fast_base`，将 `assets_dir` 设为 `gs://openpi-assets/checkpoints/pi0_fast_base/assets`。

| Robot | Description | Asset ID |
|-------|-------------|----------|
| ALOHA | 6-DoF dual arm robot with parallel grippers | trossen |
| Mobile ALOHA | Mobile version of ALOHA mounted on a Slate base | trossen_mobile |
| Franka Emika (DROID) | 7-DoF arm with parallel gripper based on the DROID setup | droid |
| Franka Emika (non-DROID) | Franka FR3 arm with Robotiq 2F-85 gripper | franka |
| UR5e | 6-DoF UR5e arm with Robotiq 2F-85 gripper | ur5e |
| UR5e bi-manual | Bi-manual UR5e setup with Robotiq 2F-85 grippers | ur5e_dual |
| ARX | Bi-manual ARX-5 robot arm setup with parallel gripper | arx |
| ARX mobile | Mobile version of bi-manual ARX-5 robot arm setup mounted on a Slate base | arx_mobile |
| Fibocom mobile | Fibocom mobile robot with 2x ARX-5 arms | fibocom_mobile |

## Pi0 Model Action Space Definitions

默认情况下，`pi0_base` 和 `pi0_fast_base` 都使用以下 action space 定义；left 和 right 是从机器人后方朝工作空间方向观察时定义的：

```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

proprioceptive state 使用与 action space 相同的定义，但对于移动机器人，不包含 base x-y position，也就是最后两个维度。

对于 7-DoF 机器人（例如 Franka），我们使用 action space 的前 7 个维度表示 joint actions，第 8 个维度表示 gripper action。

Pi 系列机器人的通用说明：

- Joint angles 使用弧度表示，位置零点对应各机器人接口库报告的零位；ALOHA 例外，标准 ALOHA 代码使用略有不同的约定（详情见 [ALOHA example code](../examples/aloha_real/README.md)）。
- Gripper positions 位于 `[0.0, 1.0]`，其中 0.0 表示完全打开，1.0 表示完全闭合。
- Control frequency：UR5e 和 Franka 为 20 Hz，ARX 和 Trossen（ALOHA）机械臂为 50 Hz。

对于 DROID，我们使用原始 DROID action 配置：前 7 个维度为 joint velocity actions，第 8 个维度为 gripper actions，控制频率为 15 Hz。
