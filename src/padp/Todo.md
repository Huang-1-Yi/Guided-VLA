# PADP 迁移阶段总结

## 2026-06-04 当前最新状态

本文件保留历史迁移记录；后续执行以本节和 `Todo_v2.md` 为准。

已确认：

```text
1. GuidedVLA 当前使用 LeRobot v3.0，LIBERO 数据源优先使用 ybwowen/libero。
2. ybwowen/libero 已能通过本地 root 打开，并且 openpi data loader 已能读取一批数据。
3. openpi batch 中的 state/actions 形状是 (B,32) 和 (B,50,32)，这是 openpi 模型 padding 后的格式，不是 PADP 的真实状态/动作维度。
4. PADP-VA 第一版使用 LIBERO 真实语义：state=8，action=7，horizon=40。
5. 已新增顶层配置 C:\QClaw\GuidedVLA\src\padp\config\libero_va.yaml。
6. 已新增 smoke 专用配置 C:\QClaw\GuidedVLA\src\padp\config\libero_va_smoke.yaml。
7. 已新增 src/padp/data/libero_batch_adapter.py，把 openpi LIBERO batch 裁剪为 PADP batch。
8. 已新增 src/padp/data/openpi_libero_loader.py，复用 openpi data loader 读取 ybwowen/libero。
9. 已新增 src/padp/training/smoke_libero_loss.py，并在服务器上跑通。
10. 当前 smoke 使用 224x224 图像、state=8、action=7、horizon=40。
11. 已新增 src/padp/training/compute_norm_stats_for_padp.py，用于保存 PADP LinearNormalizer。
12. 已新增 src/padp/config/libero_va_train.yaml，作为不含 Hydra runtime 字段的 PADP-VA 训练配置。
13. 已新增 src/padp/training/train_libero.py，参考 PADP 原训练循环实现最小 LIBERO smoke train。
14. 原始 scripts/compute_norm_stats.py 只服务 openpi 训练，不能直接生成 PADP LinearNormalizer。
15. 服务器上已生成 PADP normalizer：checkpoints/padp_libero_va/normalizer.pt。
16. 服务器上已跑通 PADP LIBERO 10 step smoke train，并保存 checkpoint。
17. 当前 smoke train 输出目录：checkpoints/padp_libero_va/train_smoke。
18. 当前 checkpoint 包括 step_000010.pt 和 last.pt，只证明训练链路可运行，不代表策略成功率。
19. 服务器上已跑通 PADP LIBERO 100 step 训练测试，num_workers=2 可用。
20. 100 step 输出目录：checkpoints/padp_libero_va/train_100step。
21. PyTorch 2.6 起 torch.load 默认 weights_only=True，读取包含旧 OmegaConf 对象的 smoke checkpoint 时可能报 ListConfig 安全加载错误；本地自训 checkpoint 检查可用 weights_only=False。
22. 服务器上已跑通 PADP LIBERO 1000 step 小规模训练，loss 没有 NaN/Inf，并持续保存 checkpoint。
23. 1000 step 输出目录：checkpoints/padp_libero_va/train_1000step。
24. 1000 step checkpoint 包括 step_000250.pt、step_000500.pt、step_000750.pt、step_001000.pt 和 last.pt。
25. PADP policy 已有 predict_action(obs_dict)，后续 serving 只需要做 LIBERO client observation 到 PADP obs_dict 的适配。
26. 已确认 train_1000step/last.pt 可用 weights_only=False 正常读取，checkpoint keys 为 cfg、model、normalizer、optimizer、step，step=1000。
27. 已新增 src/padp/training/smoke_libero_predict.py，用于加载 checkpoint 后做单批 predict_action 推理烟测。
28. 服务器上已跑通 smoke_libero_predict.py，输出 action=(2,1,7)、action_pred=(2,40,7)，没有 NaN/Inf。
29. 已新增 src/padp/serving/serve_libero.py，复用 openpi.websocket server 协议，向 examples/libero/main.py 返回 {"actions": ...}。
30. 已修改 examples/libero/main.py，新增 selected_task_ids，可先只评估单个 task。
31. 当前服务端 state 适配保持和 LiberoPadpBatchAdapter 一致：state[0:3]、state[3:7]、state[7:8]。不要在第一版 service 中额外把 axis-angle 转 quaternion，否则会和已训练 checkpoint 的输入分布不一致。
32. 服务器上已成功启动 PADP websocket server，监听 0.0.0.0:8000。
33. LIBERO client 端报错 `examples/libero/.venv/bin/activate: 没有那个文件或目录` 和 `ModuleNotFoundError: imageio`，原因是 examples/libero 专用 Python 3.8 虚拟环境尚未创建；这不是 PADP server 问题。
34. 已创建 examples/libero/.venv 后，最小 service/client 闭环已跑通：libero_object task 0、1 trial、replan_steps=1。
35. 最小评估结果为 0/1 success。该结果说明当前 1000 step PADP-VA checkpoint 尚未形成可用成功率，但接口、仿真、websocket、action 返回链路已经打通。
36. 末尾 `EGL_NOT_INITIALIZED` traceback 出现在 MuJoCo/EGL context 析构阶段，当前不影响本次评估结果读取；若频繁干扰日志，可尝试 `MUJOCO_GL=glx`。
37. 当前 client 日志中的 assets/datasets path warning 暂不阻塞评估；但后续建议用干净 `PYTHONPATH` 启动 examples/libero client，避免继承其他 conda 环境中的 LIBERO 路径。
38. 小批量稳定性评估已跑通：libero_object task 0/1/2，每个 task 2 个 trial，总 6 个 episode。
39. 小批量评估结果为 0/6 success。该结果继续说明当前 1000 step checkpoint 只能证明链路，不代表可用策略。
40. 小批量评估期间 websocket、仿真、action 返回和结果记录均未中断，说明 service/client 链路具备继续扩展评估的稳定性。
41. 已检查 `data/libero/padp_results_small.json`：记录 6 个 episode，total_successes=0，success_rate=0.0。
42. 已检查 `data/libero/padp_videos_small`：6 个 mp4 均已写出，说明视频保存链路正常。
43. 10k 训练首次启动失败，原因是训练命令仍处在 `examples/libero/.venv` 客户端环境影响下，且未设置 `OPENPI_PALIGEMMA_TOKENIZER_PATH`，openpi loader 试图从 GCS 下载 `paligemma_tokenizer.model` 并报 FileNotFoundError。
44. 该 10k 报错发生在第一个 batch 读取前，不是 PADP loss、backward、optimizer 或 checkpoint 保存问题。
45. 已新增 `src/padp/training/diagnose_libero_semantics.py`，用于打印 openpi/PADP 训练 loader 中 state[:8]、actions[:7] 的统计和样例。
46. 已给 `src/padp/serving/serve_libero.py` 新增 `--debug-log-steps`，用于记录评估侧前 N 次 observation/state 切片和 PADP action 输出。
```

服务器 smoke 成功输出要点：

```text
PADP obs keys: agentview_image, robot0_eye_in_hand_image, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos
agentview_image: (2,1,3,224,224)
robot0_eye_in_hand_image: (2,1,3,224,224)
action: (2,40,7)
Obs encoder output shape: (392,)
Obs feature dim: 392
loss_b shape: (2,)
loss mean: 3.1426055431365967
PADP LIBERO smoke loss ok
```

下一步：

```text
1. 重新用干净训练环境启动 10k 训练：退出 examples/libero/.venv，激活 lerobot，设置 `OPENPI_PALIGEMMA_TOKENIZER_PATH`。
2. 若 PADP server 还占用 cuda:1，先停止 server 或把训练切到其他 GPU，避免显存冲突。
3. 同时运行 `diagnose_libero_semantics.py`，核对训练数据中的 `state[:8]`、`actions[:7]` 与评估侧 `observation/state`、`env.step(action)` 是否语义一致。
4. 在语义核对前，不建议把完整 LIBERO 评估当作有效成功率对比。
5. 真正和 pi05 对比前，需要固定 task_suite、selected_task_ids、num_trials_per_task、seed，并使用充分训练后的 PADP checkpoint。
```

## 当前结论

现在已经确认一件关键事情：

```text
pi05 是 VLA 模型：Vision + Language + Action
PADP 当前是 VA / diffusion policy：Vision + Action
```

也就是说，PADP 原始实现并没有语言指令接口。它可以吃图像、状态、动作，可以做 diffusion action prediction，但它不会像 pi05 一样理解 prompt，也不会天然根据语言任务描述切换行为。

所以后续不能简单地说“把 PADP 接入 LIBERO 就等于做了 pi05 风格训练”。更准确地说：

```text
短期目标：让 PADP 在 GuidedVLA 的数据、训练、serve/client 结构下运行。
长期目标：再决定是否给 PADP 增加语言条件，变成真正的 VLA-like PADP。
```

## 已完成

已经完成的事情：

```text
1. 从 C:\QClaw\PADP 复制了 robomimic_padp_position_wise_v3.yaml 对应的 PADP 核心代码。
2. 在 C:\QClaw\GuidedVLA\src\padp 下保留 PADP 自己的结构。
3. 新建了 __init__.py，使 padp 可以被 Python 导入。
4. 将已搬文件中的 diffusion_policy.* import 改为 padp.*。
5. 将 PADP yaml 中主要 _target_ 改为 padp.*。
6. 将 workspace 中 env_runner 相关依赖改成延后导入。
7. 服务器上已经验证：
   - import padp 成功
   - SlidingWindowDiffusionPolicy 导入成功
   - TrainDiffusionUnetHybridWorkspace 导入成功
8. 服务器临时补了 PADP 导入依赖：
   - zarr
   - hydra-core
   - robomimic pointW fork
9. 服务器已经确认 openpi tokenizer 可用：
   - `/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model`
   - `PaligemmaTokenizer(48)` 测试通过
10. LIBERO 数据读取当前卡在 LeRobot 版本兼容：
   - 本地 LIBERO 数据集 `meta/info.json` 中 `codebase_version` 是 `v2.0`
   - 当前 GuidedVLA `.venv` 中 `lerobot` 的 `CODEBASE_VERSION` 是 `v3.0`
   - 因此 `BackwardCompatibilityError` 不是路径问题，而是数据集版本和 loader 版本不匹配
11. 已尝试把旧本地缓存移动到 `libero_v2_backup` 后重新打开远端 `physical-intelligence/libero`：
   - GuidedVLA 当前 `lerobot v3.0` 仍然报 `BackwardCompatibilityError`
   - 说明远端 `physical-intelligence/libero` 当前可用版本本身也不兼容 GuidedVLA 的 `lerobot v3.0`
   - 重新下载同一个 repo 不能解决
12. 已重新阅读 `C:\QClaw\GuidedVLA\README.md`，确认 GuidedVLA 的复现路线应优先使用：
   - 训练数据集：`ybwowen/libero`
   - checkpoint：`ybwowen/pi0-libero-object-depth-skill`
   - 复现 config：`pi0_libero_object_depth_skill`
   - 轻量数据读取测试可先用：`pi0_libero_object`
13. `physical-intelligence/libero` 是 openpi/pi0.5 旧示例数据源，不是 GuidedVLA README 的首选复现数据源。后续 PADP-VA baseline 应先测试 `ybwowen/libero`。
14. 测试 `ybwowen/libero` 时出现新报错：
   - `RevisionNotFoundError: Your dataset must be tagged with a codebase version`
   - 这不是 `physical-intelligence/libero` 那种 v2/v3 数据格式不兼容报错
   - 这是 LeRobot v3 默认按 codebase version tag 选择 dataset revision，但 `ybwowen/libero` 目前没有对应 tag
   - 下一步应直接用 Hugging Face Hub 检查 `ybwowen/libero` 的 `main` 分支 `meta/info.json`
15. 已直接检查 `ybwowen/libero` 的 `main` 分支：
   - branches: `main`
   - tags: 空
   - `meta/info.json` 中 `codebase_version` 是 `v3.0`
   - features 包含 `image`、`wrist_image`、`state`、`joint_state`、`actions`、`observation.skill_id`、`observation.skill_text`、`agentview_attention_object_mask`、`wrist_attention_object_mask` 等
   - 结论：`ybwowen/libero` 数据内容是 GuidedVLA/LeRobot v3.0 格式，只是 HF repo 没有打 LeRobot 版本 tag
16. 已开始用 `snapshot_download(..., revision="main")` 下载到：
   - `/home/hy/.cache/huggingface/lerobot/ybwowen/libero`
   - `local_dir_use_symlinks` 的 warning 是 huggingface_hub 的弃用提示，不影响下载
17. `ybwowen/libero` 本地 root 已经可以被 GuidedVLA 当前 `lerobot v3.0` 打开：
   - fps: 10
   - tasks: 40
   - root: `/home/hy/.cache/huggingface/lerobot/ybwowen/libero`
18. openpi data loader 已经能读取一批 `pi0_libero_object` 数据：
   - obs type: `openpi.models.model.Observation`
   - image keys: `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`
   - `base_0_rgb`: `(2, 224, 224, 3)`, `torch.uint8`, range `[0,255]`
   - `left_wrist_0_rgb`: `(2, 224, 224, 3)`, `torch.uint8`, range `[0,255]`
   - `right_wrist_0_rgb`: zero padding
   - state: `(2, 32)`, `torch.float32`
   - actions: `(2, 50, 32)`, `torch.float32`
   - object_targets: `object_maps`, `object_masks`
   - 结论：现在正式进入 PADP adapter 阶段
19. 需要修正一个关键理解：
   - `scripts/compute_norm_stats.py --config-name pi0_libero_object_depth_skill` 是给 openpi 模型训练准备 `state/actions` 的 normalization stats。
   - 它统计的是 openpi data transforms 之后、Normalize/ModelTransformFactory 之前的数据。
   - openpi 正式训练时会先 Normalize，再经过 `ModelTransformFactory`，例如 resize、tokenize、`PadStatesAndActions(32)`。
   - PADP 不应该把 `PadStatesAndActions` 后的 32 维 state/action 当作自己的真实动作空间。
   - PADP 应复用 openpi 的 LeRobot 数据读取和 LIBERO 字段适配，但在进入 openpi 模型专用 transforms 前接出，转成 PADP 自己的 batch。
20. 已将你手动新增的两个 LIBERO 配置合并为：
   - `C:\QClaw\GuidedVLA\src\padp\config\libero_va.yaml`
   - 原文件 `src/padp/config/libero.yaml` 和 `src/padp/config/task/libero.yaml` 暂时保留，避免误删你的对照版本。
21. `state/action` 不改成 32 维：
   - 32 不是时序长度。
   - 32 是 openpi 模型内部的 padded action/state 维度，由 `PadStatesAndActions(model_config.action_dim)` 产生。
   - PADP 的时序长度是 `horizon=40`。
   - PADP-VA 第一版使用 LIBERO 真实语义：`state=8`，`action=7`。
22. 已修正 DINOv3 encoder 配置路径：
   - 实际文件是 `padp.model.vision.dinov3_timm_obs_encoder.TimmObsEncoder`
   - 不是 `padp.model.task_padp.dinov3_timm_obs_encoder.TimmObsEncoder`
```

当前服务器临时安装命令：

```bash
cd ~/Desktop/Guided-VLA
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

uv pip install --python .venv/bin/python zarr hydra-core

uv pip install --python .venv/bin/python --no-deps "robomimic @ https://github.com/pointW/robomimic/archive/8aad5b3caaaac9289b1504438a7f5d3a76d06c07.tar.gz"
```

当前决定：保持 GuidedVLA 当前依赖版本，不降级 `lerobot`。

也就是：

```text
不更新 FASTER_hy。
不把 GuidedVLA 的 lerobot 降到 FASTER_hy 的旧版本。
优先使用 README 发布的 ybwowen/libero。
只有 ybwowen/libero 也不兼容时，才考虑重新转换 LIBERO 数据集。
```

当前旧缓存已被移动到：

```bash
~/.cache/huggingface/lerobot/physical-intelligence/libero_v2_backup
```

如果后续还要继续用 `FASTER_hy` 跑旧数据，可以恢复：

```bash
mv ~/.cache/huggingface/lerobot/physical-intelligence/libero_v2_backup \
   ~/.cache/huggingface/lerobot/physical-intelligence/libero
```

但恢复后，GuidedVLA 仍然会因为 `v2.0 -> v3.0` 不兼容而报错。

当前应测试 README 数据源：

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model

uv run python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id = "ybwowen/libero"
meta = LeRobotDatasetMetadata(repo_id)
print("download/open ok")
print("repo_id:", meta.repo_id)
print("fps:", meta.fps)
print("tasks:", len(meta.tasks))
print("root:", meta.root)
PY
```

上面当前报错：

```text
RevisionNotFoundError: Your dataset must be tagged with a codebase version.
```

因此下一步先不要继续 `LeRobotDatasetMetadata(repo_id)`，改用 Hugging Face Hub 直接检查 `main` 分支：

```bash
uv run python - <<'PY'
import json
from huggingface_hub import HfApi, hf_hub_download

repo_id = "ybwowen/libero"
api = HfApi()
refs = api.list_repo_refs(repo_id, repo_type="dataset")
print("branches:", [b.name for b in refs.branches])
print("tags:", [t.name for t in refs.tags])

info_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="meta/info.json",
    revision="main",
)
print("info_path:", info_path)
with open(info_path) as f:
    info = json.load(f)
print("codebase_version:", info.get("codebase_version"))
print("features:", list(info.get("features", {}).keys()))
PY
```

如果 `codebase_version` 兼容 GuidedVLA 当前 `lerobot v3.0`，再把 `main` 分支下载到本地 root 后测试：

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ybwowen/libero",
    repo_type="dataset",
    revision="main",
    local_dir="/home/hy/.cache/huggingface/lerobot/ybwowen/libero",
    local_dir_use_symlinks=False,
)
print("snapshot downloaded")
PY
```

当前已确认：

```text
ybwowen/libero main 分支 codebase_version = v3.0
snapshot_download 已开始正常下载
```

下载完成后，继续测试本地 root：

```bash
uv run python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

meta = LeRobotDatasetMetadata(
    "ybwowen/libero",
    root="/home/hy/.cache/huggingface/lerobot/ybwowen/libero",
)
print("local open ok")
print("fps:", meta.fps)
print("tasks:", len(meta.tasks))
print("root:", meta.root)
PY
```

如果本地 root 可打开，再测试 data loader：

```bash
uv run python - <<'PY'
import dataclasses

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

local_root = "/home/hy/.cache/huggingface/lerobot/ybwowen/libero"

cfg = _config.get_config("pi0_libero_object")
cfg = dataclasses.replace(
    cfg,
    batch_size=2,
    num_workers=0,
    data=dataclasses.replace(
        cfg.data,
        repo_id="ybwowen/libero",
        base_config=dataclasses.replace(
            cfg.data.base_config,
            local_root_dir=local_root,
        ),
    ),
)

loader = _data_loader.create_data_loader(
    cfg,
    framework="pytorch",
    split="train",
    shuffle=False,
    num_batches=1,
    skip_norm_stats=True,
)

batch = next(iter(loader))
if len(batch) == 3:
    obs, actions, object_targets = batch
else:
    obs, actions = batch
    object_targets = None

print("obs type:", type(obs))
print("image keys:", list(obs.images.keys()))
for key, value in obs.images.items():
    print("image", key, tuple(value.shape), value.dtype, float(value.min()), float(value.max()))
print("state:", tuple(obs.state.shape), obs.state.dtype, float(obs.state.min()), float(obs.state.max()))
print("actions:", tuple(actions.shape), actions.dtype, float(actions.min()), float(actions.max()))
print("object_targets:", None if object_targets is None else object_targets.keys())
print("LIBERO batch read ok")
PY
```

然后测试本地 root：

```bash
uv run python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

meta = LeRobotDatasetMetadata(
    "ybwowen/libero",
    root="/home/hy/.cache/huggingface/lerobot/ybwowen/libero",
)
print("local open ok")
print("fps:", meta.fps)
print("tasks:", len(meta.tasks))
print("root:", meta.root)
PY
```

如果上面成功，再测试：

```bash
uv run python scripts/test_data_loader.py \
  --config-name pi0_libero_object \
  --framework pytorch \
  --split train \
  --num-batches 1 \
  --num-workers 0
```

验证命令：

```bash
uv run python -c "import padp; print(padp.__file__)"
uv run python -c "from padp.workspace.robomimic.train_padp_workspace_v3 import TrainDiffusionUnetHybridWorkspace; print(TrainDiffusionUnetHybridWorkspace)"
uv run python -c "from padp.policy.robomimic.diffusion_unet_hybrid_padp import SlidingWindowDiffusionPolicy; print(SlidingWindowDiffusionPolicy)"
```

## 当前卡点

当前卡点分两层：

```text
第一层：LIBERO 数据读取已经跑通。
第二层：需要把 openpi 的 LIBERO batch 转成 PADP 的 obs/action batch。
第三层：即使 LIBERO batch 能读，PADP 仍然没有 language condition。
```

因此现在进入 PADP-VA baseline 的 adapter 阶段。

## 下一步：PADP adapter

需要新增：

```text
C:\QClaw\GuidedVLA\src\padp\data\__init__.py
C:\QClaw\GuidedVLA\src\padp\data\libero_batch_adapter.py
C:\QClaw\GuidedVLA\src\padp\data\openpi_libero_loader.py
C:\QClaw\GuidedVLA\src\padp\config\libero_va.yaml
C:\QClaw\GuidedVLA\src\padp\training\smoke_libero_loss.py
```

第一版 adapter 只做 VA，不使用 prompt。

openpi batch 到 PADP batch 的映射：

```text
obs.images["base_0_rgb"]        -> obs["agentview_image"]
obs.images["left_wrist_0_rgb"]  -> obs["robot0_eye_in_hand_image"]
obs.state[..., 0:3]             -> obs["robot0_eef_pos"]
obs.state[..., 3:7]             -> obs["robot0_eef_quat"]
obs.state[..., 7:8]             -> obs["robot0_gripper_qpos"]
actions[..., :7]                -> action
```

形状处理：

```text
image:  (B,224,224,3) uint8 -> (B,1,3,224,224) float32, range [0,1]
state:  (B,32) -> 只取前 8 维，再拆成 (B,1,D)
action: (B,50,32) -> (B,40,7)，先取 PADP horizon=40，再取 LIBERO 真实动作前 7 维
```

需要新建 PADP LIBERO shape_meta，不要继续沿用 mimicgen 的 action 10 维：

```yaml
shape_meta:
  obs:
    agentview_image:
      shape: [3, 224, 224]
      type: rgb
    robot0_eye_in_hand_image:
      shape: [3, 224, 224]
      type: rgb
    robot0_eef_pos:
      shape: [3]
    robot0_eef_quat:
      shape: [4]
    robot0_gripper_qpos:
      shape: [1]
  action:
    shape: [7]
```

`smoke_libero_loss.py` 的目标不是正式训练，而是验证：

```text
openpi loader -> libero_batch_adapter -> PADP policy.compute_loss()
```

能打印：

```text
PADP obs keys
PADP image/state/action shapes
loss mean
```

## openpi norm stats 与 PADP normalizer 的关系

README 中的命令：

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_libero_object_depth_skill
```

作用是：

```text
为 openpi / GuidedVLA 模型计算 state/actions 的 norm_stats。
结果写入 assets/<repo_id 或 asset_id>/norm_stats.json。
后续 openpi data_loader 会用这些 stats 做 Normalize。
```

它不是 PADP 的 normalizer。

PADP 使用的是：

```text
padp.model.common.normalizer.LinearNormalizer
```

因此 PADP 也需要自己的统计流程。最小路线：

```text
1. 先用当前 batch fit 一个临时 PADP normalizer，只为了跑通 smoke_libero_loss。
2. smoke loss 跑通后，再写 padp/training/compute_norm_stats_for_padp.py。
3. 正式训练时保存 PADP normalizer 到 checkpoint，并在 serve 时恢复。
```

正式 PADP 数据接口不应该直接使用 openpi 模型态输出：

```text
Observation.state  = (B,32)
actions            = (B,50,32)
```

这只是 openpi `PadStatesAndActions` 后的模型输入格式。

PADP 第一版应使用未 padding 的 LIBERO 真实维度：

```text
state  = 8 维
action = 7 维
```

所以 `openpi_libero_loader.py` 有两种实现层级：

```text
临时 smoke：复用 create_data_loader(..., skip_norm_stats=True)，再在 adapter 里裁掉 padding。
正式训练：复用 openpi 的 create_torch_dataset + repack_transforms + data_transforms，跳过 Normalize 和 ModelTransformFactory。
```

当前建议先做临时 smoke，跑通后再改成正式 loader。

## PADP norm stats 计划

需要参考：

```text
C:\QClaw\GuidedVLA\scripts\compute_norm_stats.py
```

新写：

```text
C:\QClaw\GuidedVLA\src\padp\training\compute_norm_stats_for_padp.py
```

但第一版不需要马上做。

原因：

```text
openpi 的 compute_norm_stats.py 计算的是 openpi Normalize 需要的 NormStats。
PADP 需要的是 padp.model.common.normalizer.LinearNormalizer。
两者保存格式、使用位置、数据维度都不同。
```

第一阶段：

```text
smoke_libero_loss.py 里直接用一个 batch 临时 fit PADP normalizer。
只验证 adapter + policy.compute_loss 能跑通。
```

第二阶段：

```text
参考 compute_norm_stats.py 的数据遍历方式。
使用 openpi create_torch_dataset + repack_transforms + data_transforms。
跳过 Normalize 和 ModelTransformFactory。
经过 padp libero_batch_adapter 后，对 PADP batch 计算 LinearNormalizer。
保存到 checkpoints 或 assets/padp/libero_va/normalizer.pt。
```

不能直接用原始 `compute_norm_stats.py` 自动完成 PADP normalizer，除非改代码增加 PADP 输出格式和 LinearNormalizer 保存逻辑。

模型层面的真正问题仍然是：

真正的问题是：

```text
PADP 没有 language condition。
LIBERO / pi05 的标准流程包含 prompt / language instruction。
```

因此，如果直接让 PADP 读取 LIBERO 数据，有两种含义：

```text
1. 只训练 Vision-Action policy：
   PADP 使用图像、状态和动作训练，不使用语言。

2. 改造 PADP 为语言条件 policy：
   在 PADP 的 obs condition 中加入 language embedding，再训练真正的 Vision-Language-Action policy。
```

第一种更容易，第二种才更接近 pi05。

## 两条路线

### 路线 A：PADP 接 LIBERO 数据，先做 VA baseline

目标：

```text
让 PADP 使用 LIBERO 的图像、状态、动作训练。
不使用语言 prompt。
评估时通过同一个 websocket client 输出 actions。
```

优点：

```text
改动最少。
最容易先跑通。
可以先比较“PADP action diffusion”和 pi05 在同一环境中的表现。
```

缺点：

```text
这不是严格意义上的 VLA。
多任务 LIBERO 中，如果不同任务需要语言区分，PADP 无法利用 prompt。
更适合单任务、单 suite、或每个任务单独训练一个 PADP checkpoint。
```

适用场景：

```text
先验证 PADP 能不能在 GuidedVLA 的 LIBERO service/client 流程中跑起来。
先做最小闭环。
```

### 路线 B：把 robomimic 数据和环境改造成 pi05 风格

目标：

```text
把 robomimic / MimicGen 数据转换成 LeRobot / openpi 能读取的格式。
让 pi05 和 PADP 都吃同一份转换后的数据。
评估时也尽量走统一 service/client。
```

优点：

```text
更适合公平对比。
pi05 继续是 VLA。
PADP 可以先作为 VA baseline，再逐步加语言条件。
数据、归一化、serve/client 结构更统一。
```

缺点：

```text
前期工程量更大。
需要转换数据字段、图像、状态、动作、任务描述。
如果还要做 robomimic 环境评估，需要额外制作 client 或 example。
```

适用场景：

```text
最终目标是严肃比较 pi05 与 PADP。
希望复用 GuidedVLA/pi05 的数据结构、norm stats、checkpoint、serve 风格。
```

## 当前建议

建议不要急着把 PADP 改成语言模型。

更稳的顺序是：

```text
第一阶段：PADP 作为 VA baseline，先接入 LIBERO 图像/状态/动作，跑通训练和 websocket 推理。
第二阶段：如果 VA baseline 能跑，再决定是否加入语言条件。
第三阶段：如果目标是严格对比，再做 robomimic -> LeRobot/openpi 格式转换。
```

原因：

```text
先证明 PADP 能在 GuidedVLA 的训练/serve/client 框架下跑起来。
再讨论是否值得做语言条件。
否则会同时面对数据格式、语言条件、模型结构、serve 接口四个问题。
```

## 下一步文档

新的技术路线已经单独写入：

```text
C:\QClaw\GuidedVLA\src\padp\Todo_v2.md
```

后续以 `Todo_v2.md` 为主继续推进。
