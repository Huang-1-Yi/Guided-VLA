# Robomimic 数据集迁移到 GuidedVLA/pi05 训练结构

本文档只规划第一阶段目标：把 `C:\QClaw\PADP` 中使用的 Robomimic/MimicGen HDF5 数据迁移为 LeRobot 数据格式，并让它可以在 `C:\QClaw\GuidedVLA` 当前的 `pi05` 数据、归一化、训练、serve 结构下运行。

这里暂不迁移 PADP 自身的 diffusion policy、workspace、env runner 等算法代码。先让数据被 `pi05` 跑通，再考虑迁移 PADP 算法。

## 目标

最终希望形成这样的训练链路：

```text
PADP Robomimic HDF5
  -> examples/robomimic/convert_robomimic_to_lerobot.py
  -> LeRobot dataset
  -> openpi.training.config.LeRobotRobomimicDataConfig
  -> scripts/compute_norm_stats.py
  -> scripts/train_pytorch.py pi05_robomimic_*
  -> scripts/serve_policy.py policy:checkpoint
```

推荐新增目录：

```text
GuidedVLA/
  examples/robomimic/
    README.md
    convert_robomimic_to_lerobot.py
    inspect_lerobot_sample.py
    main.py                    # 后续再做 Robomimic eval client
  src/openpi/policies/
    robomimic_policy.py
```

数据不要直接放在 `examples/robomimic` 里。建议放在：

```text
GuidedVLA/data/lerobot/
```

例如：

```text
GuidedVLA/data/lerobot/robomimic_stack_d1/
```

后续稳定后，再上传到 Hugging Face Hub。

## 阶段 0：冻结原始 PADP 基线

迁移前先记录原 PADP 的可复现实验，避免后续成功率变化时无法定位问题。

建议记录：

- 原始数据路径，例如 `PADP/data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5`
- 原始训练配置，例如 `PADP/diffusion_policy/config/robomimic_padp_position_wise.yaml`
- `task_name`
- `n_demo`
- `horizon`
- `n_obs_steps`
- `n_action_steps`
- action 是 absolute 还是 delta
- image key、state key、action key
- 原始 normalizer 逻辑
- 原始评估命令和成功率

建议单独保存：

```text
GuidedVLA/examples/robomimic/PADP_BASELINE.md
```

## 阶段 1：确定 LeRobot Schema

第一版只做 pi05 能训练的最小 schema。

推荐 LeRobot 字段：

```text
observation.images.agentview
observation.images.wrist
observation.state
action
task 或 prompt
```

其中：

- `observation.images.agentview`：主视角图像，对应 PADP/Robomimic 的 agentview。
- `observation.images.wrist`：腕部图像。如果原数据没有 wrist，可以先用 agentview 占位或零图占位。
- `observation.state`：机器人状态向量。
- `action`：动作向量。
- `task` 或 `prompt`：语言指令。没有语言时先用固定 prompt，例如 `"complete the task"`。

后续如果 PADP 算法需要阶段、位置、skill 等额外字段，再追加：

```text
observation.stage
observation.position_id
observation.skill_id
```

但第一版不要加太多字段，先让 pi05 训练链路跑通。

## 阶段 2：写 Robomimic 到 LeRobot 的转换脚本

新增：

```text
GuidedVLA/examples/robomimic/convert_robomimic_to_lerobot.py
```

脚本职责：

1. 读取 Robomimic/MimicGen HDF5。
2. 遍历 `data/demo_*` episode。
3. 提取图像、state、action。
4. 写成 LeRobot dataset。
5. 保存 episode/task metadata。

命令形式建议：

```bash
uv run examples/robomimic/convert_robomimic_to_lerobot.py \
  --input data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5 \
  --repo-id robomimic_stack_d1 \
  --output-root data/lerobot \
  --task "stack the blocks"
```

转换完成后的数据路径建议是：

```text
GuidedVLA/data/lerobot/robomimic_stack_d1/
```

注意事项：

- 图像最好统一为 `uint8`、HWC、RGB。
- action 不要在转换时随意改语义；原始是 absolute 就保留 absolute，原始是 delta 就保留 delta。
- 如果原始 PADP 训练用的是已经转换后的 `*_abs.hdf5`，第一版就直接把它视为 absolute action 数据。
- 不要在转换脚本中做训练归一化。归一化交给 `compute_norm_stats.py`。

## 阶段 3：新增 Robomimic Policy Adapter

新增：

```text
GuidedVLA/src/openpi/policies/robomimic_policy.py
```

这个文件不实现 PADP 算法，只做数据适配，类似 `calvin_policy.py`、`libero_policy.py`。

建议包含：

```python
class RobomimicInputs(transforms.DataTransformFn):
    ...

class RobomimicOutputs(transforms.DataTransformFn):
    ...
```

`RobomimicInputs` 负责把 LeRobot 样本转为 openpi/pi05 通用输入：

```python
{
    "state": ...,
    "image": {
        "base_0_rgb": agentview,
        "left_wrist_0_rgb": wrist,
        "right_wrist_0_rgb": zeros,
    },
    "image_mask": {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": False,
    },
    "actions": ...,
    "prompt": ...,
}
```

`RobomimicOutputs` 负责把模型输出转回环境执行需要的 action：

```python
{
    "actions": data["actions"][:, :action_dim]
}
```

第一版要明确 `action_dim`，例如 7、10、14 等，必须和 Robomimic 环境实际动作维度一致。

## 阶段 4：在 config.py 中新增 DataConfig 和 TrainConfig

修改：

```text
GuidedVLA/src/openpi/training/config.py
```

新增 import：

```python
import openpi.policies.robomimic_policy as robomimic_policy
```

新增数据配置：

```python
@dataclasses.dataclass(frozen=True)
class LeRobotRobomimicDataConfig(DataConfigFactory):
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "image": "observation.images.agentview",
                        "wrist_image": "observation.images.wrist",
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    horizon_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[robomimic_policy.RobomimicInputs(model_type=model_config.model_type)],
            outputs=[robomimic_policy.RobomimicOutputs()],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            horizon_sequence_keys=self.horizon_sequence_keys,
        )
```

新增 pi05 训练配置：

```python
TrainConfig(
    name="pi05_robomimic_stack_d1",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotRobomimicDataConfig(
        repo_id="robomimic_stack_d1",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=30_000,
    num_workers=8,
)
```

如果本地数据不在 HF Hub，而是在 `data/lerobot`，训练时通过 `--local-root-dir` 或 `--local_root_dir` 指定。

## 阶段 5：验证数据读取

先写一个样本检查脚本：

```text
GuidedVLA/examples/robomimic/inspect_lerobot_sample.py
```

检查内容：

- dataset 能被 `LeRobotDataset` 打开。
- 单条样本有图像、state、action、prompt。
- 图像 shape 和 dtype 正确。
- action 维度正确。
- episode 数、总帧数与原 HDF5 接近。

建议输出：

```text
repo_id: robomimic_stack_d1
num_frames: ...
num_episodes: ...
fps: ...
agentview: uint8 H W C
wrist: uint8 H W C
state: shape
action: shape
prompt: ...
```

## 阶段 6：计算归一化统计

本地数据命令示例：

```bash
uv run scripts/compute_norm_stats.py pi05_robomimic_stack_d1 \
  --local-root-dir data/lerobot
```

或如果使用参数名形式：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_robomimic_stack_d1 \
  --local-root-dir data/lerobot
```

输出应写入：

```text
GuidedVLA/assets/pi05_robomimic_stack_d1/robomimic_stack_d1/
```

重点检查：

- `state` 的均值、标准差合理。
- `actions` 的均值、标准差合理。
- 不要出现全零 std。
- 不要把图像纳入 norm stats。

## 阶段 7：pi05 小规模训练冒烟测试

先不要跑完整训练，先跑短训练。

建议临时命令：

```bash
uv run scripts/train_pytorch.py pi05_robomimic_stack_d1 \
  --exp_name smoke_test \
  --local_root_dir data/lerobot \
  --num_train_steps 100 \
  --batch_size 8 \
  --num_workers 0 \
  --wandb_enabled false
```

目标：

- dataloader 能启动。
- batch 能进入模型。
- loss 能下降或至少不是 NaN。
- checkpoint 能保存。

通过后再跑正常训练：

```bash
uv run scripts/train_pytorch.py pi05_robomimic_stack_d1 \
  --exp_name first_full_run \
  --local_root_dir data/lerobot
```

## 阶段 8：serve policy

训练完成后：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_robomimic_stack_d1 \
  --policy.dir=checkpoints/pi05_robomimic_stack_d1/first_full_run/<step>
```

第一阶段可以先不做完整 Robomimic 环境评估，只写一个简单 client，拿一条 observation 调 server：

```text
examples/robomimic/simple_client.py
```

验证返回：

```text
actions: [action_horizon, action_dim]
```

## 阶段 9：再迁移 Robomimic 评估客户端

当 pi05 能训练和 serve 后，再做：

```text
GuidedVLA/examples/robomimic/main.py
GuidedVLA/examples/robomimic/env/
GuidedVLA/examples/robomimic/env_runner/
GuidedVLA/examples/robomimic/gym_util/
```

这一步负责启动 Robomimic/MimicGen 环境，并通过 websocket 调用 policy server。

结构类似：

```text
Robomimic env
  -> observation
  -> websocket client
  -> GuidedVLA policy server
  -> action chunk
  -> env.step(action)
```

## 成功率对齐时优先排查的问题

如果 pi05 或后续 PADP 迁移后的成功率和原代码不一致，按这个顺序查：

1. 图像是否一致：camera、颜色通道、resize、crop。
2. state 是否一致：维度、顺序、单位。
3. action 是否一致：absolute/delta、维度、夹爪正负号。
4. normalizer 是否一致：state/action 的统计是否合理。
5. horizon 是否一致：`action_horizon`、`n_action_steps`、实际执行 chunk。
6. prompt 是否一致：固定 prompt 还是 task prompt。
7. reset/seed/max steps 是否一致。
8. eval 环境版本是否一致。

## 现阶段不做的事情

第一阶段不要迁移这些内容：

```text
PADP/diffusion_policy/policy
PADP/diffusion_policy/model
PADP/diffusion_policy/workspace
PADP/diffusion_policy/env_runner
```

这些属于第二阶段：把 PADP 算法本体迁到 GuidedVLA 的代码结构。

第一阶段只追求：

```text
Robomimic 数据 -> LeRobot -> pi05 训练/serve 跑通
```

## 第二阶段预告：迁移 PADP 算法

等第一阶段稳定后，再考虑：

```text
PADP/diffusion_policy/model      -> GuidedVLA/src/openpi/models_pytorch/padp/
PADP/diffusion_policy/policy     -> GuidedVLA/src/openpi/policies/padp_policy.py
PADP/diffusion_policy/common     -> GuidedVLA/src/openpi/models_pytorch/padp/ 或 src/openpi/shared/
PADP/diffusion_policy/env_runner -> GuidedVLA/examples/robomimic/
```

那时可以新增：

```text
padp_robomimic_stack_d1
pi05_padp_robomimic_stack_d1
```

但要注意：如果模型仍然是 PADP 自己的 U-Net/diffusion policy，它只是运行在 GuidedVLA/pi05 风格的数据和训练结构下，不等同于原生 `Pi0Config(pi05=True)` 模型。

