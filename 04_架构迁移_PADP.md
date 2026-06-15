# PADP 架构代码迁移到 GuidedVLA：无需 Robomimic 绑定版

本文档用于规划如何把 `C:\QClaw\PADP` 中的 PADP / diffusion-policy 架构迁移到 `C:\QClaw\GuidedVLA`，但不绑定 Robomimic 环境，也不要求先迁移 Robomimic 评估代码。

核心目标是：

```text
同一个 LeRobot 数据集
  -> pi05 配置训练
  -> PADP 配置训练
  -> 使用同一套 eval client / serve_policy 对比成功率
```

也就是说，先让 PADP 运行在 GuidedVLA 当前的数据、训练、checkpoint、serve 结构下，再和 `pi05` 在相同数据集、相同评估环境下对比成功率。

本文档是独立上下文说明，后续可以直接基于本文档继续问答。

## 一句话结论

建议新建独立包：

```text
GuidedVLA/src/padp/
```

而不是把 PADP 直接塞进 `src/openpi/` 的各个目录。

原因：

- `openpi` 保持 GuidedVLA 原结构，减少污染。
- `padp` 保留 PADP 自己的模型、策略、训练适配逻辑。
- 两者通过 `openpi.training.config.TrainConfig` 和 `scripts/train_pytorch.py` 汇合。
- 后续可以在同一数据集上配置 `pi05_xxx` 和 `padp_xxx` 做公平对比。

## 推荐目标结构

建议新增：

```text
GuidedVLA/
  src/
    openpi/
      ...                         # 保持 GuidedVLA 原结构

    padp/
      __init__.py
      models/
        __init__.py
        padp_config.py             # PADP 的 BaseModelConfig 适配层

      models_pytorch/
        __init__.py
        diffusion_unet_image_policy.py
        conditional_unet1d_padp.py
        schedulers_padp.py
        mask_generator.py
        obs_encoder.py

      policies/
        __init__.py
        padp_policy.py             # 输入输出 adapter，可复用同一数据集 schema

      training/
        __init__.py
        train_adapter.py           # PADP loss / optimizer / checkpoint 适配逻辑
        batch_adapter.py           # openpi batch -> PADP batch

      serving/
        __init__.py
        policy_wrapper.py          # sample_actions / predict_action 兼容层

      shared/
        __init__.py
        normalizer.py
        pytorch_util.py
        shape_util.py
```

这几个目录的职责：

| 目录 | 职责 |
|---|---|
| `src/padp/models` | 配置类，负责和 `openpi.models.model.BaseModelConfig` 对接 |
| `src/padp/models_pytorch` | PADP 神经网络、U-Net、scheduler、采样逻辑 |
| `src/padp/policies` | dataset/env observation 到 PADP 输入输出的适配 |
| `src/padp/training` | 训练 loop 适配、batch 适配、PADP loss 调用 |
| `src/padp/serving` | server 推理 wrapper，统一输出 `actions` |
| `src/padp/shared` | PADP 专属工具函数 |

## pyproject.toml 注意事项

当前 `pyproject.toml` 的项目名是 `openpi`，源代码主要在：

```text
GuidedVLA/src/openpi
```

如果新增：

```text
GuidedVLA/src/padp
```

需要确认 `uv run` / editable install 能导入：

```python
import padp
```

如果导入失败，建议在 `pyproject.toml` 中显式声明 hatchling packages，例如：

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/openpi", "src/padp"]

[tool.hatch.build.targets.editable]
packages = ["src/openpi", "src/padp"]
```

如果项目已有其它 hatch 配置，以实际配置为准。目标是确保 `src/padp` 被安装进当前环境。

## 总体路线

迁移分成 6 个阶段。

```text
阶段 1：定义同一数据集 schema
阶段 2：让 pi05 在该数据集上训练
阶段 3：新增 src/padp 包并迁入 PADP 模型
阶段 4：让 PADP 使用同一数据集训练
阶段 5：让 PADP 使用 serve_policy 推理
阶段 6：同一 eval client 下对比 pi05 和 PADP 成功率
```

重点是：不要先迁 Robomimic。只要有一个 LeRobot 数据集和一个能评估该数据集对应环境的 client，就可以对比。

## 阶段 1：确定共享数据契约

pi05 和 PADP 必须使用同一份 LeRobot 数据集。

建议统一数据字段：

```text
observation.images.base
observation.images.wrist
observation.state
action
prompt
```

如果是双臂或多相机，可以扩展：

```text
observation.images.base
observation.images.left_wrist
observation.images.right_wrist
observation.state
action
prompt
```

最小要求：

- `state` 维度固定。
- `action` 维度固定。
- action 语义固定：absolute 或 delta 必须明确。
- 图像 dtype 和 shape 固定。
- prompt 来源固定。

不要让 pi05 和 PADP 使用两套不同的数据转换逻辑。公平对比的前提是：

```text
同一 dataset
同一 train/val split
同一 normalization stats
同一 eval initial states
同一 action 后处理
```

## 阶段 2：先建立 pi05 基线

先让当前 GuidedVLA 的 `pi05` 在该数据集上跑通。

在 `src/openpi/training/config.py` 中新增类似：

```python
TrainConfig(
    name="pi05_my_dataset",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotMyDatasetConfig(
        repo_id="my_dataset",
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

运行：

```bash
uv run scripts/compute_norm_stats.py pi05_my_dataset \
  --local-root-dir data/lerobot

uv run scripts/train_pytorch.py pi05_my_dataset \
  --exp_name pi05_baseline \
  --local_root_dir data/lerobot
```

评估：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_my_dataset \
  --policy.dir=checkpoints/pi05_my_dataset/pi05_baseline/<step>
```

先记录 pi05 成功率，作为后续 PADP 对比基线。

## 阶段 3：新增 src/padp 包

新建目录：

```text
GuidedVLA/src/padp/
```

每个目录都加 `__init__.py`：

```text
src/padp/__init__.py
src/padp/models/__init__.py
src/padp/models_pytorch/__init__.py
src/padp/policies/__init__.py
src/padp/training/__init__.py
src/padp/serving/__init__.py
src/padp/shared/__init__.py
```

先做最小导入验证：

```bash
uv run python -c "import padp; print(padp.__file__)"
```

如果失败，先修 `pyproject.toml` 包发现问题，不要继续迁代码。

## 阶段 4：迁移 PADP 模型代码

从 `C:\QClaw\PADP` 迁入模型本体。

建议第一批迁移：

```text
PADP/diffusion_policy/model/diffusion/conditional_unet1d_padp.py
PADP/diffusion_policy/policy/schedulers_padp.py
PADP/diffusion_policy/policy/PADP_diffusion_unet_image_policy.py
PADP/diffusion_policy/model/vision/dp_multi_image_obs_encoder.py
PADP/diffusion_policy/model/diffusion/mask_generator.py
```

目标位置：

```text
GuidedVLA/src/padp/models_pytorch/
```

第一版只要求保留：

```python
class PadpDiffusionPolicy(torch.nn.Module):
    def compute_loss(self, batch: dict) -> torch.Tensor:
        ...

    @torch.no_grad()
    def sample_actions(self, observation: dict, **kwargs) -> torch.Tensor:
        ...
```

或者兼容原 PADP：

```python
def predict_action(self, obs_dict: dict) -> dict:
    return {
        "actions": action_chunk,
        "action_pred": action_pred,
    }
```

但最终对外必须能统一成：

```python
{"actions": action_chunk}
```

第一版先不要迁：

- Robomimic env runner。
- Hydra workspace。
- real_world。
- shared_memory。
- 可视化 test 脚本。

## 阶段 5：定义 PadpConfig

新增：

```text
GuidedVLA/src/padp/models/padp_config.py
```

它应该继承或兼容：

```python
openpi.models.model.BaseModelConfig
```

示意：

```python
import dataclasses

import openpi.models.model as _model


@dataclasses.dataclass(frozen=True)
class PadpConfig(_model.BaseModelConfig):
    action_dim: int = 7
    action_horizon: int = 40
    n_obs_steps: int = 1
    n_action_steps: int = 1
    obs_as_global_cond: bool = True
    num_inference_steps: int = 40
    noise_schedule_mode: str = "positionwise"
    window_loss_weights: str = "exponential"
    window_exp_gamma: float = 0.2

    @property
    def model_type(self):
        return _model.ModelType.PI05

    def inputs_spec(self, *, batch_size: int = 1):
        ...

    def load_pytorch(self, train_config, weight_path):
        ...
```

如果 `BaseModelConfig` 的抽象接口较多，第一版可以先做最小可运行实现，再逐步补齐。

重要：`PadpConfig` 不是说 PADP 是 pi05 模型，而是让 PADP 能被当前 openpi 训练/serve 代码识别。

## 阶段 6：定义 PADP 数据 adapter

新增：

```text
GuidedVLA/src/padp/policies/padp_policy.py
```

职责：

- 输入：共享 LeRobot dataset 被 `RepackTransform` 后的字段。
- 输出：PADP 模型需要的 `obs_dict` 和 `actions`。

建议类：

```python
class PadpInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        ...


class PadpOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        ...
```

`PadpInputs` 输出建议：

```python
{
    "obs": {
        "image": ...,
        "wrist_image": ...,
        "state": ...,
    },
    "actions": ...,
    "prompt": ...,
}
```

或者直接输出 openpi 通用字段：

```python
{
    "state": ...,
    "image": {
        "base_0_rgb": ...,
        "left_wrist_0_rgb": ...,
    },
    "image_mask": ...,
    "actions": ...,
    "prompt": ...,
}
```

推荐第二种，因为它更容易复用 GuidedVLA 现有 transform。

## 阶段 7：在 config.py 注册 PADP 配置

修改：

```text
GuidedVLA/src/openpi/training/config.py
```

新增 import：

```python
import padp.models.padp_config as padp_config
import padp.policies.padp_policy as padp_policy
```

新增 DataConfigFactory：

```python
@dataclasses.dataclass(frozen=True)
class LeRobotPadpDataConfig(DataConfigFactory):
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "image": "observation.images.base",
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
            inputs=[padp_policy.PadpInputs(model_config=model_config)],
            outputs=[padp_policy.PadpOutputs()],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            horizon_sequence_keys=self.horizon_sequence_keys,
        )
```

新增 TrainConfig：

```python
TrainConfig(
    name="padp_my_dataset",
    model=padp_config.PadpConfig(
        action_dim=7,
        action_horizon=40,
        n_obs_steps=1,
        n_action_steps=1,
        num_inference_steps=40,
        noise_schedule_mode="positionwise",
        window_loss_weights="exponential",
        window_exp_gamma=0.2,
    ),
    data=LeRobotPadpDataConfig(
        repo_id="my_dataset",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=64,
    num_workers=8,
    num_train_steps=30_000,
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
)
```

这样你会有两组配置：

```text
pi05_my_dataset
padp_my_dataset
```

它们使用同一个 `repo_id="my_dataset"`，就可以做同数据集对比。

## 阶段 8：接入 train_pytorch.py

推荐不要新建长期训练入口，尽量使用：

```text
GuidedVLA/scripts/train_pytorch.py
```

但 PADP 的 `compute_loss(batch)` 可能和 pi05 的 loss 接口不同，因此需要在 `train_pytorch.py` 里增加一个很薄的分支。

示意：

```python
from padp.models.padp_config import PadpConfig


def compute_model_loss(model, batch, config):
    if isinstance(config.model, PadpConfig):
        return model.compute_loss(batch)
    return model.compute_loss(batch, ...)
```

或者在 `padp/training/train_adapter.py` 中写：

```python
def compute_loss(model, batch, config):
    return model.compute_loss(batch)
```

然后 `train_pytorch.py` 只做分发：

```python
if isinstance(config.model, PadpConfig):
    loss = padp_train_adapter.compute_loss(model, batch, config)
else:
    loss = ...
```

这样可以保持主训练入口不变：

```bash
uv run scripts/train_pytorch.py padp_my_dataset --exp_name padp_run
```

## 阶段 9：normalizer 策略

为了公平对比，pi05 和 PADP 应使用同一数据集统计。

建议：

```bash
uv run scripts/compute_norm_stats.py pi05_my_dataset --local-root-dir data/lerobot
uv run scripts/compute_norm_stats.py padp_my_dataset --local-root-dir data/lerobot
```

如果两者 `asset_id` 和 data transform 完全一致，也可以让 PADP 复用 pi05 的 stats：

```python
data=LeRobotPadpDataConfig(
    repo_id="my_dataset",
    assets=AssetsConfig(
        assets_dir="./assets/pi05_my_dataset",
        asset_id="my_dataset",
    ),
    base_config=DataConfig(prompt_from_task=True),
)
```

第一版建议分别算一次，然后比较两个 `norm_stats.json` 是否一致或接近。

注意：

- 不要在 transform 层 normalize 一次，又在 PADP 模型内部 normalize 一次。
- 最推荐第一版让 openpi transform 负责 Normalize/Unnormalize。
- PADP 模型内部吃 normalized tensor。
- PADP 输出 normalized actions，由 openpi output transform 负责 unnormalize。

## 阶段 10：checkpoint 与加载

PADP 训练产物应尽量对齐 GuidedVLA PyTorch checkpoint：

```text
checkpoints/padp_my_dataset/padp_run/<step>/
  model.safetensors
  assets/<asset_id>/norm_stats.json
```

`scripts/serve_policy.py` 会通过：

```python
openpi.policies.policy_config.create_trained_policy(...)
```

加载 checkpoint。它会检查：

```text
model.safetensors
```

所以 `PadpConfig.load_pytorch()` 必须能从 `model.safetensors` 恢复 PADP 模型。

## 阶段 11：接入 serve_policy.py

先不要新增 `--env PADP`。

先用显式 checkpoint：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=padp_my_dataset \
  --policy.dir=checkpoints/padp_my_dataset/padp_run/<step>
```

如果这条命令能启动，再考虑给 `serve_policy.py` 加：

```python
class EnvMode(enum.Enum):
    PADP = "padp"
```

以及默认 checkpoint。

推理时输出必须是：

```python
{"actions": action_chunk}
```

其中 shape 建议是：

```text
[action_horizon, action_dim]
```

## 阶段 12：同一 eval client 对比成功率

最终对比方式：

启动 pi05：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_my_dataset \
  --policy.dir=checkpoints/pi05_my_dataset/pi05_baseline/<step> \
  --port 8000
```

跑 eval client：

```bash
python examples/<env>/main.py --host 127.0.0.1 --port 8000 --save_name pi05
```

启动 PADP：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=padp_my_dataset \
  --policy.dir=checkpoints/padp_my_dataset/padp_run/<step> \
  --port 8000
```

跑同一个 eval client：

```bash
python examples/<env>/main.py --host 127.0.0.1 --port 8000 --save_name padp
```

对比：

```text
same dataset
same eval env
same initial states
same success checker
same max steps
same action postprocess
```

这样得到的成功率才有比较意义。

## 阶段 13：必须保持一致的对比条件

公平比较 `pi05_my_dataset` 和 `padp_my_dataset` 时，至少保持：

- 同一份 LeRobot 数据集。
- 同一 train/val split。
- 同一 eval episodes。
- 同一相机输入。
- 同一 state 输入。
- 同一 action 维度。
- 同一 action 语义：absolute 或 delta。
- 同一 gripper 后处理。
- 同一 action execution frequency。
- 同一 max episode steps。
- 同一 success checker。

可以不同的是：

- 模型结构。
- loss。
- action_horizon。
- 推理采样逻辑。
- batch size。
- 学习率。

但如果 action_horizon 不同，对比时要明确记录。

## 不建议做的事

第一版不要做：

- 不要迁 `env_runner` 到 `src/padp`。
- 不要迁 `real_world` 到 `src/padp`。
- 不要让 PADP 继续依赖 Hydra workspace。
- 不要把 PADP 代码散落到 `src/openpi/models_pytorch`、`src/openpi/policies`、`src/openpi/training` 各处。
- 不要一边改数据 schema，一边迁 PADP 模型。
- 不要用两套不同 normalizer 对比成功率。

环境和真实机器人代码应该放：

```text
GuidedVLA/examples/<env>/
```

核心算法代码才放：

```text
GuidedVLA/src/padp/
```

## 最小实现顺序

推荐按这个顺序实施：

1. 确认 `pi05_my_dataset` 已能训练和评估。
2. 新建 `src/padp` 包和 `__init__.py`。
3. 确认 `uv run python -c "import padp"` 成功。
4. 迁入 PADP 最小模型依赖到 `src/padp/models_pytorch`。
5. 新增 `PadpConfig`。
6. 新增 `PadpInputs` / `PadpOutputs`。
7. 在 `config.py` 注册 `padp_my_dataset`。
8. 在 `train_pytorch.py` 增加 PADP loss 分支。
9. 跑 100 step smoke train。
10. 保存 `model.safetensors`。
11. 用 `serve_policy.py policy:checkpoint` 启动 PADP。
12. 用同一个 eval client 对比 pi05 与 PADP。

## 里程碑

### M1：src/padp 可导入

```bash
uv run python -c "import padp; print(padp.__file__)"
```

### M2：PadpConfig 可被 get_config 找到

```bash
uv run python -c "from openpi.training import config; print(config.get_config('padp_my_dataset'))"
```

### M3：PADP 能吃一个 batch

目标：

```text
LeRobot batch -> openpi transforms -> padp batch -> compute_loss -> scalar loss
```

### M4：PADP 能训练 100 step

```bash
uv run scripts/train_pytorch.py padp_my_dataset \
  --exp_name smoke \
  --num_train_steps 100 \
  --batch_size 8 \
  --num_workers 0 \
  --wandb_enabled false
```

### M5：PADP checkpoint 能 serve

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=padp_my_dataset \
  --policy.dir=checkpoints/padp_my_dataset/smoke/<step>
```

### M6：同一 eval client 成功率对比

记录：

```text
pi05_my_dataset success rate: ...
padp_my_dataset success rate: ...
dataset: ...
eval env: ...
checkpoint step: ...
action_horizon: ...
```

## 成功率不一致时的排查顺序

1. batch 中 `state`、`actions` 是否和 pi05 使用同一数据。
2. PADP 是否重复 normalize。
3. PADP 输出是否被正确 unnormalize。
4. action 是否被截断到正确维度。
5. gripper 后处理是否一致。
6. PADP 的 `reset_buffer()` 是否在 episode 开始时调用。
7. PADP 的 action queue 是否和原算法一致。
8. action_horizon 和 eval 执行步长是否匹配。
9. eval client 是否对 pi05 和 PADP 做了不同后处理。
10. checkpoint 是否加载了正确 step。

## 推荐命名

配置命名建议：

```text
pi05_<dataset>
padp_<dataset>
padp_<dataset>_h40
padp_<dataset>_h10
```

实验命名建议：

```text
pi05_baseline
padp_smoke
padp_h40_positionwise
padp_h10_ablation
```

不要使用含糊名称：

```text
test
new
final
best
```

## 最终状态

完成后，理想状态是：

```text
GuidedVLA
  src/openpi       # GuidedVLA/pi05 原结构
  src/padp         # PADP 算法结构
  scripts          # 统一训练、norm stats、serve 入口
  examples         # 环境 client 和评估入口
```

同一个数据集可以这样训练：

```bash
uv run scripts/train_pytorch.py pi05_my_dataset --exp_name pi05_baseline
uv run scripts/train_pytorch.py padp_my_dataset --exp_name padp_run
```

同一个评估 client 可以这样对比：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_my_dataset --policy.dir=...
python examples/<env>/main.py --save_name pi05

uv run scripts/serve_policy.py policy:checkpoint --policy.config=padp_my_dataset --policy.dir=...
python examples/<env>/main.py --save_name padp
```

这样就能实现：PADP 和 pi05 在相同数据集、相同评估流程下的成功率对比。

