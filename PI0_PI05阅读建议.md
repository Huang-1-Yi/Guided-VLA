# PI0/PI05 阅读建议

本文面向后续继续分析 GuidedVLA 中的 pi0/pi05 训练、推理、数据适配和 GuidedVLA 增强模块。建议把本仓库理解为三层结构：

1. `src/openpi/models`：JAX/NNX 版 pi0/pi05 模型主体，继承 openpi 的原始结构。
2. `src/openpi/models_pytorch`：PyTorch 版 pi0/pi05，以及 GuidedVLA 增强的 object/skill/depth/control-attention 模块。
3. `src/openpi/training`、`scripts`、`examples`、`src/openpi/policies`：训练配置、数据读取、环境适配、服务化推理和评估入口。

如果只想先跑通和理解 pi0/pi05，建议先读 JAX 训练主线；如果目标是理解 GuidedVLA 的优化，则重点读 PyTorch 训练主线和 `models_pytorch`。

## 1. 先读命令入口

建议从两个训练入口开始：

- [scripts/train.py](scripts/train.py)：JAX/NNX 训练入口，接近 openpi 原始 pi0/pi05 训练结构。
- [scripts/train_pytorch.py](scripts/train_pytorch.py)：PyTorch 训练入口，包含 GuidedVLA 的 ControlAttention、object head、skill head、depth head、DDP、混合精度、checkpoint 加载等逻辑。

阅读时先不要陷入模型细节，先回答四个问题：

1. 命令行参数如何变成 `TrainConfig`？
2. `TrainConfig` 如何决定模型、数据集、权重、batch size、训练步数？
3. data loader 最终产出的 `Observation/actions` 长什么样？
4. 每个 train step 如何计算 loss、反传、更新参数、保存 checkpoint？

推荐从以下命令脑内走一遍：

```bash
uv run scripts/train_pytorch.py pi05_libero --exp_name test --num_train_steps 10
uv run scripts/train.py pi05_libero --exp-name=test --num-train-steps=10 --overwrite
```

JAX 入口的核心链路是：

```text
scripts/train.py
-> config.cli()
-> TrainConfig
-> create_data_loader()
-> Pi0Config.create()
-> init TrainState
-> train_step()
-> Pi0.compute_loss()
-> checkpoint
```

PyTorch 入口的核心链路是：

```text
scripts/train_pytorch.py
-> config.cli()
-> create_data_loader(framework="pytorch")
-> PI0Pytorch
-> load pretrained/checkpoint
-> optional ControlAttention / depth / skill / object modules
-> training loop
-> checkpoint / validation / eval hooks
```

## 2. 再读配置系统

核心文件是 [src/openpi/training/config.py](src/openpi/training/config.py)。

优先阅读这些类：

- `TrainConfig`：一次训练实验的总配置。
- `DataConfig`：数据 transform、normalization、repo、assets 的组合。
- `DataConfigFactory`：从配置生成 `DataConfig` 的工厂基类。
- `LeRobotAlohaDataConfig`、`LeRobotLiberoDataConfig`、`LeRobotRobotwinDataConfig`、`LeRobotCalvinDataConfig`：不同环境/数据集的 LeRobot 适配配置。
- `ModelTransformFactory`：根据模型类型 pi0/pi05/pi0-fast 自动生成模型输入 transform。

阅读重点：

1. `pi05_libero`、`pi0_libero_object_depth_skill`、`pi05_calvin` 这些 config 的差别是什么？
2. `model=Pi0Config(...)` 中哪些字段影响模型结构？
3. `data=...DataConfig(...)` 中哪些字段影响数据读取和 key 映射？
4. `weight_loader` 指向哪个预训练 checkpoint？
5. `pytorch_training_precision`、`use_gradient_checkpointing`、`control_attention_enabled` 等参数如何影响 PyTorch 训练？

## 3. 理解数据流：LeRobot 到 Observation

核心文件：

- [src/openpi/training/data_loader.py](src/openpi/training/data_loader.py)
- [src/openpi/transforms.py](src/openpi/transforms.py)
- [src/openpi/policies/libero_policy.py](src/openpi/policies/libero_policy.py)
- [src/openpi/policies/aloha_policy.py](src/openpi/policies/aloha_policy.py)
- [src/openpi/policies/droid_policy.py](src/openpi/policies/droid_policy.py)
- [src/openpi/policies/calvin_policy.py](src/openpi/policies/calvin_policy.py)

一个典型 LeRobot 数据集训练样本会走：

```text
LeRobotDataset
-> PromptFromLeRobotTask
-> RepackTransform
-> environment-specific Inputs
-> Normalize
-> InjectDefaultPrompt
-> ResizeImages
-> TokenizePrompt
-> PadStatesAndActions
-> Observation.from_dict + actions
```

建议重点看每个环境 adapter 做了什么：

- `LiberoInputs`：把 LIBERO 的 state/image/wrist_image/actions/prompt 变成 openpi 的统一字段。
- `AlohaInputs`：把 ALOHA 的多相机和 qpos/action 映射到统一字段。
- `DroidInputs`：处理 DROID 的 exterior/wrist 图像、joint/gripper state 和 action。
- `CalvinInputs`：把 CALVIN 的 ee_pos、ee_rot、gripper、图像和动作拆分字段打包成 openpi 格式。

openpi 统一模型输入的核心字段是：

```text
state
image["base_0_rgb"]
image["left_wrist_0_rgb"]
image["right_wrist_0_rgb"]
image_mask
tokenized_prompt
tokenized_prompt_mask
actions
```

## 4. 理解模型结构：先 JAX，再 PyTorch

JAX 版模型主线：

- [src/openpi/models/model.py](src/openpi/models/model.py)：`Observation`、`Actions`、`BaseModelConfig`、`BaseModel`。
- [src/openpi/models/pi0_config.py](src/openpi/models/pi0_config.py)：pi0/pi05 配置。
- [src/openpi/models/pi0.py](src/openpi/models/pi0.py)：pi0/pi05 flow matching 主体。
- [src/openpi/models/gemma.py](src/openpi/models/gemma.py)：Gemma/PaliGemma backbone。
- [src/openpi/models/siglip.py](src/openpi/models/siglip.py)：视觉 encoder。
- [src/openpi/models/tokenizer.py](src/openpi/models/tokenizer.py)：PaliGemma/FAST/RT-2 风格 tokenizer。

JAX 版 `Pi0.compute_loss()` 的核心逻辑：

```text
actions
-> sample noise
-> sample time
-> x_t = time * noise + (1 - time) * actions
-> u_t = noise - actions
-> embed observation prefix
-> embed action suffix
-> PaliGemma/Gemma forward
-> predict v_t
-> loss = mean((v_t - u_t)^2)
```

PyTorch 版模型主线：

- [src/openpi/models_pytorch/pi0_pytorch.py](src/openpi/models_pytorch/pi0_pytorch.py)：PyTorch 版 pi0/pi05 主体。
- [src/openpi/models_pytorch/gemma_pytorch.py](src/openpi/models_pytorch/gemma_pytorch.py)：PyTorch PaliGemma/Gemma 组合。
- [src/openpi/models_pytorch/preprocessing_pytorch.py](src/openpi/models_pytorch/preprocessing_pytorch.py)：PyTorch 图像和 observation 预处理。
- [src/openpi/models_pytorch/control_attention.py](src/openpi/models_pytorch/control_attention.py)：GuidedVLA ControlNet 风格双分支 attention。
- [src/openpi/models_pytorch/attention/attn_paths.py](src/openpi/models_pytorch/attention/attn_paths.py)：standard/object/depth attention path 路由。
- [src/openpi/models_pytorch/depth/model.py](src/openpi/models_pytorch/depth/model.py)：Depth Anything 3 depth encoder。
- [src/openpi/models_pytorch/depth/depth_attention.py](src/openpi/models_pytorch/depth/depth_attention.py)：depth token cross-attention。
- [src/openpi/models_pytorch/depth/token_merging.py](src/openpi/models_pytorch/depth/token_merging.py)：depth feature token merging。

GuidedVLA 的新增能力主要在 PyTorch 路径：

```text
ControlAttention: origin branch + object/control branch + zero_conv fusion
Object head: 用 attention/object map 做辅助监督
Skill head: 用 skill id 或 soft skill label 做辅助监督
Depth head: 用 Depth Anything 3 生成几何 token，并让部分 attention head 关注 depth token
```

## 5. 理解推理和服务化

先读：

- [scripts/serve_policy.py](scripts/serve_policy.py)：启动 websocket policy server。
- [src/openpi/policies/policy_config.py](src/openpi/policies/policy_config.py)：从 checkpoint/config 创建可推理 policy。
- [src/openpi/policies/policy.py](src/openpi/policies/policy.py)：policy 抽象、推理调用、记录器。
- [src/openpi/serving/websocket_policy_server.py](src/openpi/serving/websocket_policy_server.py)：服务端协议。
- [packages/openpi-client/src/openpi_client/websocket_client_policy.py](packages/openpi-client/src/openpi_client/websocket_client_policy.py)：客户端 policy。

推理主线：

```text
serve_policy.py
-> load TrainConfig
-> create_trained_policy()
-> load model/checkpoint/assets/norm_stats
-> start WebsocketPolicyServer
-> client sends observation
-> policy.infer()
-> outputs actions
```

环境评估入口在 `examples/*/main.py` 中。它们通常做三件事：

1. 创建环境。
2. 创建 websocket client policy。
3. 循环获取 observation、调用 policy、执行 action、记录成功率或视频。

## 6. 建议阅读顺序

第一轮只建立整体地图：

1. [README.md](README.md)
2. [scripts/train_pytorch.py](scripts/train_pytorch.py)
3. [src/openpi/training/config.py](src/openpi/training/config.py)
4. [src/openpi/training/data_loader.py](src/openpi/training/data_loader.py)
5. [src/openpi/transforms.py](src/openpi/transforms.py)
6. [src/openpi/policies/libero_policy.py](src/openpi/policies/libero_policy.py)
7. [src/openpi/models_pytorch/pi0_pytorch.py](src/openpi/models_pytorch/pi0_pytorch.py)

第二轮看 JAX 原始 pi0/pi05：

1. [src/openpi/models/model.py](src/openpi/models/model.py)
2. [src/openpi/models/pi0_config.py](src/openpi/models/pi0_config.py)
3. [src/openpi/models/pi0.py](src/openpi/models/pi0.py)
4. [src/openpi/models/tokenizer.py](src/openpi/models/tokenizer.py)
5. [scripts/train.py](scripts/train.py)

第三轮看 GuidedVLA 增强：

1. [src/openpi/models_pytorch/control_attention.py](src/openpi/models_pytorch/control_attention.py)
2. [src/openpi/models_pytorch/attention/attn_paths.py](src/openpi/models_pytorch/attention/attn_paths.py)
3. [src/openpi/models_pytorch/depth/model.py](src/openpi/models_pytorch/depth/model.py)
4. [src/openpi/models_pytorch/depth/depth_attention.py](src/openpi/models_pytorch/depth/depth_attention.py)
5. [src/openpi/models_pytorch/gemma_pytorch.py](src/openpi/models_pytorch/gemma_pytorch.py)

第四轮看环境和评估：

1. [examples/libero/main.py](examples/libero/main.py)
2. [examples/libero_plus/main.py](examples/libero_plus/main.py)
3. [examples/robotwin/main.py](examples/robotwin/main.py)
4. [examples/calvin/main.py](examples/calvin/main.py)
5. [examples/aloha_sim/main.py](examples/aloha_sim/main.py)
6. [examples/droid/main.py](examples/droid/main.py)

## 7. 调试时的定位方法

如果训练报数据 key 错误，优先看：

```text
config.py -> DataConfigFactory
data_loader.py -> transform_dataset
transforms.py -> RepackTransform
policies/*_policy.py -> Inputs adapter
models/model.py -> Observation.from_dict
```

如果模型 shape 不匹配，优先看：

```text
pi0_config.py -> action_dim / action_horizon / max_token_len
transforms.py -> PadStatesAndActions
policies/*_policy.py -> action slice / state concat
pi0_pytorch.py 或 pi0.py -> action projection
```

如果 checkpoint 加载不匹配，优先看：

```text
config.py -> weight_loader / pytorch_weight_path
weight_loaders.py -> CheckpointWeightLoader
model.py -> load_pytorch / restore_params
train_pytorch.py -> checkpoint restore / ControlAttention inject order
```

如果推理成功连接但动作异常，优先看：

```text
serve_policy.py -> env / config / checkpoint
policy_config.py -> create_trained_policy
policies/*_policy.py -> Outputs adapter
examples/*/main.py -> env action convention
```

## 8. 最关键的一句话

GuidedVLA 的 pi0/pi05 主线可以概括为：

```text
TrainConfig 选择模型和数据
-> LeRobot/RLDS data loader 产出 Observation/actions
-> policy adapter 对齐不同环境字段
-> transform/tokenizer/padding 变成模型输入
-> pi0/pi05 flow matching 学 action chunk
-> PyTorch 路径额外加入 object/skill/depth/control-attention 辅助监督
-> checkpoint 供 serve_policy 和 examples 环境评估复用
```
