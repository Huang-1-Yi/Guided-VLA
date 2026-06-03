# GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization

<div align="center">

<p>
Xiaosong Jia<sup>&#42;&#8224;,1,2</sup>, Bowen Yang<sup>&#42;,3</sup>, Zuhao Ge<sup>&#42;,1,2</sup>, Xian Nie<sup>&#42;,3</sup>, Yuchen Zhou<sup>&#42;,1,2</sup>, Cunxin Fan<sup>&#42;&#8224;,3</sup>, Yufeng Li<sup>3</sup>, Yilin Chai<sup>3</sup>, Chao Jing<sup>1,2</sup>, Zijian Liang<sup>3</sup>, Qingwen Bu<sup>4</sup>, Haidong Cao<sup>1,2</sup>, Chao Wu<sup>1,2</sup>, Qifeng Li<sup>3</sup>, Zhenjie Yang<sup>3</sup>, Chenhe Zhang<sup>1,2</sup>, Hongyang Li<sup>4</sup>, Zuxuan Wu<sup>&#9993;,1,2</sup>, Junchi Yan<sup>&#9993;,3</sup>, Yu-Gang Jiang<sup>&#9993;,1,2</sup>
</p>

<p>
<sup>1</sup>Institute of Trustworthy Embodied AI (TEAI), Fudan University &nbsp;
<sup>2</sup>Shanghai Key Laboratory of Multimodal Embodied AI &nbsp;
<sup>3</sup>Shanghai Jiao Tong University &nbsp;
<sup>4</sup>OpenDriveLab, The University of Hong Kong
</p>

<p><sup>&#42;</sup> Core Contributors &nbsp;&nbsp; <sup>&#8224;</sup> Project Lead &nbsp;&nbsp; <sup>&#9993;</sup> Correspondence Authors</p>

[[Paper]](https://arxiv.org/abs/2605.12369) &nbsp;|&nbsp; [[Project Page]](https://guidedvla.github.io/project_page/) &nbsp;|&nbsp; [[Code]](https://github.com/GuidedVLA/GuidedVLA) &nbsp;|&nbsp; [[Checkpoint]](https://huggingface.co/ybwowen/pi0-libero-object-depth-skill) &nbsp;|&nbsp; [[Dataset]](https://huggingface.co/datasets/ybwowen/libero) &nbsp;|&nbsp; [[Citation]](#citation)

</div>

---

**GuidedVLA** 是一种 VLA 范式：它通过逐头注意力专门化，显式引导 action decoder 捕获任务相关信息，包括 object grounding、空间几何和时间技能逻辑。与依赖端到端监督隐式学习这些特征不同，GuidedVLA 将专用 attention heads 重新用于不同任务相关因素的建模，并通过人工定义的辅助信号进行监督。

本仓库在 [openpi](https://github.com/Physical-Intelligence/openpi)（π0 / π0.5）基础上扩展了 GuidedVLA 框架，并提供完整的 PyTorch 训练流水线。

## 发布状态

- [x] 发布代码
- [x] 发布 LIBERO 训练数据集：[ybwowen/libero](https://huggingface.co/datasets/ybwowen/libero)
- [x] 发布 LIBERO checkpoint：[ybwowen/pi0-libero-object-depth-skill](https://huggingface.co/ybwowen/pi0-libero-object-depth-skill)
- [ ] 发布 RoboTwin 训练数据集

<p align="center">
  <img src="docs/figures/guidedvla-teaser.png" alt="GuidedVLA teaser" width="95%"/>
</p>

## 主要结果

**LIBERO-Plus**（覆盖 7 个扰动维度的鲁棒性 benchmark）：

| Model | Spatial | Object | Goal | Long | **Total** |
|---|---|---|---|---|---|
| π0 baseline | 77.7 | 74.1 | 61.4 | 60.1 | 68.2 |
| w/ object head | 80.6 | **82.5** | 67.1 | 64.0 | 73.4 |
| w/ skill head | 79.8 | 78.9 | 68.9 | 62.7 | 72.5 |
| w/ depth head | 81.4 | **79.0** | 65.4 | 61.8 | 71.7 |
| **GuidedVLA (ours)** | **84.0** | 80.9 | **70.8** | **66.2** | **75.4** |

**RoboTwin 2.0**（8 个操作任务，out-of-domain）：π0 **77.38%** -> GuidedVLA **90.63%**

**Real-world**（ALOHA AgileX + PSI-Bot RealMan，6 个家庭/实验室任务）：

| Generalization | Base Policy | GuidedVLA |
|---|---|---|
| In-domain | 55.8% | **75.8%** |
| Scene | 44.2% | **67.5%** |
| Lighting | 57.5% | **79.2%** |

---

## 方法概览

GuidedVLA 不再将 action decoder 视作单一整体学习器，而是将其视为一组**功能专门化组件**。Attention heads 会通过任务特定的辅助信号进行监督：

<p align="center">
  <img src="docs/figures/guidedvla-model-structure.png" alt="GuidedVLA model structure" width="95%"/>
</p>

1. **Object Head**（视觉定位）：引导一部分 heads H_obj，使其 attention maps 通过加权负对数似然损失 L_object 对齐真实物体区域 mask。该约束促使 action tokens 关注相关物体，并抑制干扰物。

2. **Skill Head**（时间逻辑）：指定 heads H_skill，根据其输出特征分类当前 sub-skill 或任务阶段，并使用 KL-divergence loss L_skill 对 soft skill labels 进行监督。它可以捕获长时程时间结构，而不需要硬性的技能边界。

3. **Depth Head**（几何感知）：从冻结的 depth encoder（[Depth Anything V3](https://github.com/DepthAnything/Depth-Anything-V3)）注入 3D 空间线索，将其作为一部分 heads H_depth 的额外 keys 和 values。这是结构约束，不需要额外 loss。

**通过 ControlNet Adapter 实现即插即用**：专门化 heads 通过轻量 control branch 引入，并借助 zero-initialized projection（ZeroConv）融合进预训练 backbone，遵循 ControlNet residual 策略：

```
Attn_final(x) = Attn_main(x) + ZeroConv(Attn_specified(x))
```

<p align="center">
  <img src="docs/figures/controlnet_style_adapter.png" alt="ControlNet-style adapter" width="70%"/>
</p>

该分支初始贡献为零，随后逐步学习注入与任务因素相关的偏置，从而在训练过程中保留预训练能力。

---

## 安装

克隆仓库并初始化 submodules：

```bash
git clone --recurse-submodules https://github.com/GuidedVLA/GuidedVLA.git
cd GuidedVLA

# Or if already cloned:
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

将必要 patch 应用到 `transformers` 库中；这些 patch 用于支持 AdaRMS、activation precision 和 KV-cache control：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

> **注意**：在默认 uv hardlink 模式下，这会永久 patch transformers cache。若要完全撤销，请运行：`uv cache clean transformers`。

### Depth Encoder 设置

GuidedVLA 使用 [Depth Anything V3 Small](https://github.com/DepthAnything/Depth-Anything-V3) 作为冻结的 depth encoder。请下载 DA3-SMALL checkpoint，并在 config 中将 `depth_model_name` 设置为本地 checkpoint 路径：

```python
depth_model_name = "path/to/da3-small"  # local checkpoint path
```

---

## Checkpoints

已发布的 GuidedVLA checkpoint 托管在 Hugging Face：

| Model | Config | Checkpoint |
|---|---|---|
| GuidedVLA LIBERO object + depth + skill | `pi0_libero_object_depth_skill` | [`ybwowen/pi0-libero-object-depth-skill`](https://huggingface.co/ybwowen/pi0-libero-object-depth-skill) |

该 checkpoint 包含 `model.safetensors`，并在 `assets/ybwowen/libero/norm_stats.json` 下包含 normalization statistics。

GuidedVLA 基于 Physical Intelligence 的 π0 / π0.5 base models 构建：

| Model | Checkpoint |
|---|---|
| π0 base | `gs://openpi-assets/checkpoints/pi0_base` |
| π0.5 base | `gs://openpi-assets/checkpoints/pi05_base` |

训练前请先将 JAX checkpoint 转换为 PyTorch 格式：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name pi0_libero \
    --output_path /path/to/pytorch/checkpoint \
    --precision float32
```

---

## 训练

### 1. 准备数据集

对于 GuidedVLA LIBERO 训练，我们在 Hugging Face 上发布了 LeRobot 格式数据集：
[`ybwowen/libero`](https://huggingface.co/datasets/ybwowen/libero)。[src/openpi/training/config.py](src/openpi/training/config.py) 中默认的 GuidedVLA LIBERO configs 使用该数据集，其中也包括已发布 checkpoint 对应的 config `pi0_libero_object_depth_skill`。

如果使用你自己的数据，请先将其转换为 [LeRobot](https://github.com/huggingface/lerobot) 格式。

对于 GuidedVLA 的辅助监督，数据集还应包含：

- **Object head**：`agentview_attention_object_mask` 和 `wrist_attention_object_mask`
- **Skill head**：`observation.skill_id`，用于在线构造 soft skill label
- **Depth head**：RGB 图像；depth 会由冻结的 encoder 在线计算

如果使用不同的数据 schema，请更新 [src/openpi/training/config.py](src/openpi/training/config.py) 中的数据 transforms，使 object 和 skill targets 能被 repack 到 PyTorch trainer 消费的 keys 中。

### 2. 配置训练任务

在 [src/openpi/training/config.py](src/openpi/training/config.py) 中编辑你的 config。关键 GuidedVLA configs 如下：

| Config | Description |
|---|---|
| `pi0_libero_object_depth_skill` | 完整 GuidedVLA：object + depth + skill heads |
| `pi0_libero_object` | 仅启用 Object head |
| `pi0_libero_depth` | 仅启用 Depth head |
| `pi0_libero_skill` | 仅启用 Skill head |
| `pi0_libero` | π0 baseline（无 guided heads） |

`Pi0Config` 中的关键字段：

```python
# ControlNet-style attention branch
control_attention_enabled: bool = True
control_attention_target: str = "expert"   # "expert", "paligemma", or "both"
control_attention_num_heads: int | None = 8  # heads in control branch
control_attention_use_headwise_gate: bool = True

# Depth
use_depth: bool = True
depth_model_name: str = "path/to/da3-small"
guided_layer_indices: list = [9, 10, 11, 12]
depth_head_indices: list = [4, 5]

# Skill
use_skill_loss: bool = True
skill_num_classes: int = 4  # 3 effective skills + 1 null/background class
skill_head_indices: list = [6, 7]
```

默认 LIBERO full GuidedVLA config 中，辅助 loss 权重为：

```python
object_loss_weight: float = 0.001
skill_loss_weight: float = 0.001
```

### 3. 计算 normalization statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_libero_object_depth_skill
```

### 4. 启动训练

```bash
# Single GPU
uv run scripts/train_pytorch.py pi0_libero_object_depth_skill \
    --exp_name my_run --save_interval 2000

# Multi-GPU (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi0_libero_object_depth_skill \
    --exp_name my_run --save_interval 2000

# Resume from latest checkpoint
uv run scripts/train_pytorch.py pi0_libero_object_depth_skill \
    --exp_name my_run --resume

# Multi-node (e.g., 2 nodes x 8 GPUs)
uv run torchrun \
    --nnodes=2 --nproc_per_node=8 \
    --node_rank=<rank> --master_addr=<ip> --master_port=<port> \
    scripts/train_pytorch.py pi0_libero_object_depth_skill \
    --exp_name my_run --save_interval 2000
```

Checkpoints 会保存到 `./checkpoints/<config_name>/<exp_name>/`。

### 精度

GuidedVLA 使用 **float32 master weights** 训练，并通过 `torch.autocast(bfloat16)` 进行混合精度计算。

---

## 评估

### LIBERO-Plus

当前 LIBERO-Plus 流程包含：

- `scripts/serve_policy.py`：PyTorch policy server
- `examples/libero_plus/main.py`：单个评估任务
- `examples/libero_plus/eval_libero_plus.py`：多 GPU / 多进程批量评估

#### 1. 准备 LIBERO-Plus 模拟器环境

`examples/libero_plus/main.py` 运行在 LIBERO-Plus simulator environment 中，该环境与主训练环境分离：

```bash
uv venv --python 3.8 examples/libero_plus/.venv
source examples/libero_plus/.venv/bin/activate

uv pip sync examples/libero_plus/requirements.txt third_party/LIBERO-plus/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/LIBERO-plus
uv pip install -r third_party/LIBERO-plus/extra_requirements.txt

export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/LIBERO-plus
```

#### 2. 启动 policy server

在一个终端中，从仓库根目录启动 checkpoint server：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --no-sync scripts/serve_policy.py \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi0_libero_object_depth_skill \
    --policy.dir hf://models/ybwowen/pi0-libero-object-depth-skill
```

#### 3. 运行单个 LIBERO-Plus 任务

在第二个终端中，使用 simulator environment 评估一个 suite 或一个扰动类别：

```bash
source examples/libero_plus/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/LIBERO-plus

python examples/libero_plus/main.py \
    --host 127.0.0.1 \
    --port 8000 \
    --task-suite-name libero_object \
    --category "Objects Layout" \
    --num-trials-per-task 1 \
    --video-out-path data/libero_plus/videos \
    --results-json-path data/libero_plus/libero_object.json
```

常用 `main.py` 参数：

- `--task-suite-name`：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10` 或 `all`
- `--category`：`Objects Layout`、`Camera Viewpoints`、`Robot Initial States`、`Language Instructions`、`Light Conditions`、`Background Textures`、`Sensor Noise`
- `--task-ids`：例如 `0`、`0,3,7` 或 `10-19`
- `--replan-steps`：从 server 请求的 action chunk 大小
- `--results-json-path`：滚动写入的 JSON 汇总；设置 `--category` 时会自动追加类别后缀

#### 4. 跨 GPU 并行评估

`examples/libero_plus/eval_libero_plus.py` 会为每张 GPU 启动一个 policy server，并自动分发评估任务：

```bash
.venv/bin/python examples/libero_plus/eval_libero_plus.py \
    --checkpoint-dir hf://models/ybwowen/pi0-libero-object-depth-skill \
    --policy-config pi0_libero_object_depth_skill \
    --gpu-ids 0,1,2,3 \
    --client-python examples/libero_plus/.venv/bin/python \
    --libero-plus-path third_party/LIBERO-plus
```

常用 `eval_libero_plus.py` 参数：

- `--task-suites`：用逗号分隔的 suite 列表，默认是 `libero_spatial,libero_object,libero_goal,libero_10`
- `--categories`：用逗号分隔的扰动类别
- `--task-ids`：限制只评估部分任务
- `--num-trials-per-task`：每个任务的 rollout 次数

输出会写入：

- `data/libero_plus/`：JSON 结果和 rollout 视频
- `logs/libero_plus/`：每个 worker 的日志

### RoboTwin 2.0

RoboTwin 2.0 的评估脚本位于 `examples/robotwin/`。
运行 RoboTwin pipeline 前，请先通过 `git submodule update --init --recursive third_party/RoboTwin` 初始化 RoboTwin。
RoboTwin 评估遵循相同的 policy-server 流程：先使用 `scripts/serve_policy.py` 服务 checkpoint，然后针对目标任务运行 `examples/robotwin/main.py`。

完整 RoboTwin 设置、数据转换、训练配置和评估流程请见 [examples/robotwin/README.md](examples/robotwin/README.md)。

---

## Citation

如果你觉得本工作有帮助，请引用：

```bibtex
@misc{jia2026guidedvla,
  title         = {GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization},
  author        = {Xiaosong Jia and Bowen Yang and Zuhao Ge and Xian Nie and Yuchen Zhou and Cunxin Fan and Yufeng Li and Yilin Chai and Chao Jing and Zijian Liang and Qingwen Bu and Haidong Cao and Chao Wu and Qifeng Li and Zhenjie Yang and Chenhe Zhang and Hongyang Li and Zuxuan Wu and Junchi Yan and Yu-Gang Jiang},
  year          = {2026},
  eprint        = {2605.12369},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2605.12369},
}
```

---

## 致谢

GuidedVLA 是基于 Physical Intelligence 的 [openpi](https://github.com/Physical-Intelligence/openpi) 扩展而来。感谢 openpi 团队开源代码库和预训练模型。

---

## License and Third-Party Notices

本仓库中的 GuidedVLA 源代码遵循 [Apache License 2.0](LICENSE)，除非文件中另有说明。

Gemma、PaliGemma 相关组件和模型权重受 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt) 中的 Gemma 条款约束。本仓库还通过 submodules 和依赖使用了多个第三方项目，包括 openpi、Depth Anything V3、LIBERO、LIBERO-Plus、RoboTwin 和 ALOHA。这些项目仍遵循其各自的许可证、模型条款或数据集条款。

---

<details>
<summary><b>openpi 文档（原始）</b></summary>

openpi 包含由 [Physical Intelligence team](https://www.physicalintelligence.company/) 发布的开源机器人模型和软件包。

当前，该仓库包含三类模型：

- [π0 model](https://www.physicalintelligence.company/blog/pi0)：基于 flow 的 vision-language-action model（VLA）。
- [π0-FAST model](https://www.physicalintelligence.company/research/fast)：基于 FAST action tokenizer 的自回归 VLA。
- [π0.5 model](https://www.physicalintelligence.company/blog/pi05)：π0 的升级版本，具有更好的 open-world 泛化能力。

### 更新

- [Sept 2025] openpi 发布 PyTorch 支持。
- [Sept 2025] 发布 π0.5，具备更好的 open-world 泛化能力。
- [Jun 2025] 发布在完整 [DROID dataset](https://droid-dataset.github.io/) 上训练的[说明](examples/droid/README_train.md)。

### Base Model Checkpoints

| Model | Checkpoint |
|---|---|
| π0 | `gs://openpi-assets/checkpoints/pi0_base` |
| π0-FAST | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| π0.5 | `gs://openpi-assets/checkpoints/pi05_base` |

### Fine-Tuned Checkpoints

| Model | Checkpoint |
|---|---|
| π0-FAST-DROID | `gs://openpi-assets/checkpoints/pi0_fast_droid` |
| π0-DROID | `gs://openpi-assets/checkpoints/pi0_droid` |
| π0-ALOHA-towel | `gs://openpi-assets/checkpoints/pi0_aloha_towel` |
| π0-ALOHA-tupperware | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| π0-ALOHA-pen-uncap | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap` |
| π0.5-LIBERO | `gs://openpi-assets/checkpoints/pi05_libero` |
| π0.5-DROID | `gs://openpi-assets/checkpoints/pi05_droid` |

Checkpoints 会自动下载并缓存到 `~/.cache/openpi`。可通过 `OPENPI_DATA_HOME` 覆盖缓存位置。

### 运行推理

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
policy = policy_config.create_trained_policy(config, checkpoint_dir)

action_chunk = policy.infer({
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    "prompt": "pick up the fork",
})["actions"]
```

分步骤示例见：[DROID](examples/droid/README.md) | [ALOHA](examples/aloha_real/README.md) | [Remote Inference](docs/remote_inference.md)

### Troubleshooting

| Issue | Resolution |
|---|---|
| `uv sync` fails | 删除 `.venv` 后重试。更新 uv：`uv self update`。 |
| Out of GPU memory | 使用多 GPU DDP（`--nproc_per_node=N`）或减小 batch size。 |
| Missing norm stats | 先运行 `scripts/compute_norm_stats.py --config-name <name>`。 |
| Dataset download fails | 检查网络或 HuggingFace 登录状态：`huggingface-cli login`。 |
| CUDA errors | 尝试卸载系统 CUDA；uv 会安装正确版本。 |
| Diverging training loss | 检查 `norm_stats.json` 中是否存在接近 0 的 `std`/`q01`/`q99` 值，并手动调整。 |

</details>
