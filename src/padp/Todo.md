# 已完成

## 2026-06-03：PADP 核心代码已复制

当前 `C:\QClaw\GuidedVLA\src\padp` 已经完成第一批 PADP 核心文件复制，迁移来源以这份配置为根节点：

```text
C:\QClaw\PADP\diffusion_policy\config\robomimic_padp_position_wise_v3.yaml
```

已复制的主要链路包括：

```text
config:
  src/padp/config/robomimic_padp_position_wise_v3.yaml
  src/padp/config/task/mimicgen_abs_padp.yaml

workspace:
  src/padp/workspace/base_workspace.py
  src/padp/workspace/workspace_util.py
  src/padp/workspace/robomimic/train_padp_workspace_v3.py

policy:
  src/padp/policy/base_image_policy.py
  src/padp/policy/schedulers_padp.py
  src/padp/policy/robomimic/diffusion_unet_hybrid_padp.py

model:
  src/padp/model/diffusion/conditional_unet1d_padp.py
  src/padp/model/diffusion/conv1d_components.py
  src/padp/model/diffusion/ema_model.py
  src/padp/model/diffusion/mask_generator.py
  src/padp/model/vision/crop_randomizer.py
  src/padp/model/vision/robomimic_obs_encoder.py
  src/padp/model/common/*.py

common:
  src/padp/common/*.py

dataset:
  src/padp/dataset/base_dataset.py
  src/padp/dataset/robomimic/replay_image_dataset_padp.py
```

## 2026-06-03：前 4 项检查通过

以下四项已经完成并检查：

```text
1. 新建 Python package 初始化文件
2. 修复已搬 PADP 文件中的旧 import
3. 修改 PADP yaml 的 _target_
4. 把 workspace 评估依赖改成延后导入
```

检查结果：

```text
src/padp/**/__init__.py 已齐全。
已搬 Python 文件中的 diffusion_policy.* 主 import 已替换为 padp.*。
src/padp/config/robomimic_padp_position_wise_v3.yaml 已改为 padp.* target。
train_padp_workspace_v3.py 已使用 TYPE_CHECKING 和函数内 lazy import 延后 env_runner 依赖。
```

当前仍保留两个可接受残留：

```text
src/padp/model/diffusion/conditional_unet1d_padp.py
  只有注释中残留 diffusion_policy 路径，不影响运行。

src/padp/config/task/mimicgen_abs_padp.yaml
  env_runner target 仍是 diffusion_policy.env_runner...
  这是原 PADP robomimic rollout 评估入口；当前目标是 LIBERO service/client，所以暂时不改也不使用。
```

## 2026-06-03：当前路线

当前目标不是复刻 PADP 原始 `env_runner` 评估，而是：

```text
PADP 训练：复用 GuidedVLA/openpi 的 LIBERO 数据入口
PADP 推理：启动 padp 自己的 websocket policy server
PADP 评估：复用 GuidedVLA/examples/libero/main.py 和 openpi-client websocket client
```

因此不搬：

```text
src/padp/env_runner
src/padp/env
src/padp/gym_util
```

# 服务器 git pull 后测试流程

以下命令参考了你之前在 `C:\QClaw\FASTER_hy\安装.txt` 和 `C:\QClaw\FASTER_hy\训练与测试.md` 中已经跑通的服务器流程。

## 0. 更新代码和基础环境

在服务器进入 GuidedVLA 仓库：

```bash
cd ~/Desktop/GuidedVLA
git pull
git submodule update --init --recursive
```

激活你之前跑通 LIBERO 的环境：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
```

如果 `uv` 不在 PATH，用你服务器上已跑通的路径：

```bash
which uv || export PATH="$HOME/.local/bin:$PATH"
```

同步依赖：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

当前 `pyproject.toml` 还没有显式把 `src/padp` 纳入安装包，所以本阶段测试请先加：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

后续如果要长期使用 `padp` 包，再把 `src/padp` 写进打包配置。

## 1. 检查 PADP 包能否导入

先做最小导入：

```bash
uv run python -c "import padp; print(padp.__file__)"
```

再检查 PADP 核心模块：

```bash
uv run python -c "from padp.policy.robomimic.diffusion_unet_hybrid_padp import SlidingWindowDiffusionPolicy; print(SlidingWindowDiffusionPolicy)"
```

检查 workspace 是否不再因为没搬 env_runner 而顶层导入失败：

```bash
uv run python -c "from padp.workspace.robomimic.train_padp_workspace_v3 import TrainDiffusionUnetHybridWorkspace; print(TrainDiffusionUnetHybridWorkspace)"
```

如果这里报缺依赖，优先记录错误。常见可能缺：

```text
diffusers
hydra-core
omegaconf
dill
termcolor
zarr
numcodecs
imagecodecs
numba
h5py
threadpoolctl
robomimic
pytorch3d
```

这些是 PADP 原代码带来的依赖，不一定都属于 GuidedVLA 原依赖。不要一口气乱装，先按报错逐个补。

## 2. 检查 LIBERO 数据是否可读

如果服务器已经按 FASTER 流程下载到默认 Hugging Face / LeRobot cache，可以先只覆盖 `repo_id`，不传 `local_root_dir`：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero \
  --repo-id physical-intelligence/libero \
  --max-frames 256 \
  --asset-id pi05_libero
```

如果你之前训练时使用了自定义本地 LeRobot root，则加上当时跑通的 root：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero \
  --repo-id physical-intelligence/libero \
  --local-root-dir /path/to/your/lerobot/root \
  --max-frames 256 \
  --asset-id pi05_libero
```

说明：

```text
--max-frames 256 是 smoke test，确认能读 batch 即可。
--asset-id pi05_libero 用于把 norm stats 写到 assets/pi05_libero。
如果 --config-name 在当前 tyro 版本下报参数错误，改用位置参数：
uv run scripts/compute_norm_stats.py pi05_libero --repo-id physical-intelligence/libero --max-frames 256 --asset-id pi05_libero
```

烟测成功后，再跑完整统计：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero \
  --repo-id physical-intelligence/libero \
  --asset-id pi05_libero
```

如果你的数据不在默认 cache，完整统计也要带上：

```bash
--local-root-dir /path/to/your/lerobot/root
```

## 3. 先验证 pi05 LIBERO service/client 仍能跑

这一步不是 PADP 评估，而是确认服务器环境、LIBERO、websocket client、官方 checkpoint 这条基线仍然正常。

终端 1：启动 `pi05_libero` policy server。

```bash
cd ~/Desktop/GuidedVLA
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
```

终端 2：运行 LIBERO client。先用 1 个 trial 做烟测。

```bash
cd ~/Desktop/GuidedVLA
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export LIBERO_CONFIG_PATH="$PWD/third_party/libero"
export PYTHONPATH="$PWD/src:$PWD/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl

examples/libero/.venv/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.save-name pi05_libero_pull_smoke \
  --args.host 127.0.0.1 \
  --args.port 8000
```

烟测通过后，再跑你之前的完整设置：

```bash
examples/libero/.venv/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 \
  --args.save-name pi05_libero_official \
  --args.host 127.0.0.1 \
  --args.port 8000
```

结果通常写到：

```text
data/libero_eval/results/pi05_libero_official_results.txt
```

## 4. 如果需要重新下载 LIBERO 数据

如果服务器默认 cache 里没有数据，按你之前跑通过的方式下载：

```bash
uv run hf download physical-intelligence/libero \
  --repo-type dataset \
  --revision v2.0 \
  --local-dir ~/.cache/huggingface/lerobot/physical-intelligence/libero \
  --max-workers 4
```

如果代理报错 `Unknown scheme for proxy URL URL('socks://127.0.0.1:7897/')`，按之前记录修正：

```bash
export ALL_PROXY=http://127.0.0.1:7897
export all_proxy=http://127.0.0.1:7897
```

或者直接临时取消：

```bash
unset ALL_PROXY all_proxy
```

# 待完成

## 1. 新增 LIBERO 数据适配层

新建：

```text
src/padp/data/openpi_libero_loader.py
src/padp/data/libero_batch_adapter.py
src/padp/data/__init__.py
```

职责：

```text
openpi LeRobot LIBERO batch
-> 提取 observation / image / state / action
-> 转成 PADP 的 batch:
   {
     "obs": {
       "agentview_image": ...,
       "robot0_eye_in_hand_image": ...,
       "robot0_eef_pos": ...,
       "robot0_eef_quat": ...,
       "robot0_gripper_qpos": ...
     },
     "action": ...
   }
-> 调用 SlidingWindowDiffusionPolicy.compute_loss(batch)
```

## 2. 新增 PADP LIBERO 训练入口

新建：

```text
src/padp/training/train_libero.py
src/padp/training/checkpoint.py
src/padp/training/__init__.py
```

目标命令：

```bash
uv run python -m padp.training.train_libero \
  --openpi-config pi05_libero \
  --repo-id physical-intelligence/libero \
  --exp-name padp_libero_smoke \
  --max-steps 100
```

## 3. 新增 PADP websocket 推理服务

新建：

```text
src/padp/serving/policy_factory.py
src/padp/serving/serve_libero.py
src/padp/serving/__init__.py
```

目标命令：

```bash
uv run python -m padp.serving.serve_libero \
  --checkpoint-dir checkpoints/padp_libero/padp_libero_smoke/<step> \
  --port 8000
```

输出接口必须和 pi05 一致：

```python
{"actions": action_chunk}
```

## 4. 复用 LIBERO client 评估 PADP

PADP server 启动后，直接复用：

```bash
examples/libero/.venv/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 \
  --args.save-name padp_libero \
  --args.host 127.0.0.1 \
  --args.port 8000
```

# 可选项

## A. 迁移原 PADP env_runner

当前目标是 LIBERO service/client 评估，所以这一项先不做。

只有当你想复现 PADP 原始 robomimic rollout 评估时，才考虑复制：

```text
PADP/diffusion_policy/env_runner
PADP/diffusion_policy/env
PADP/diffusion_policy/gym_util
```

## B. 后续接入统一命令

独立链路跑通后，可选地让 PADP 也支持：

```bash
uv run scripts/train_pytorch.py padp_libero --exp_name padp_libero
uv run scripts/serve_policy.py policy:checkpoint --policy.config padp_libero --policy.dir ...
```

但这不是第一版目标。第一版目标是：

```text
PADP 能吃 pi05_libero 同源数据训练
PADP 能通过 websocket 输出 actions
PADP 能被 examples/libero/main.py 评估
```
