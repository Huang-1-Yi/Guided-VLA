# PADP 迁移技术路线 v2

## 2026-06-04 当前执行口径

当前路线是 PADP-VA baseline，不是把 PADP 直接改成 pi05/VLA。

```text
数据源：ybwowen/libero，本地 root 为 /home/hy/.cache/huggingface/lerobot/ybwowen/libero
读取链路：优先复用 openpi data loader，确认数据字段与 GuidedVLA 一致
PADP 输入：裁掉 openpi padding，只保留 LIBERO 真实 state/action
图像尺寸：当前 DINOv3 encoder 使用 224x224
动作维度：7
状态维度：8
时间长度：PADP horizon=40
```

PADP adapter 和最小 smoke loss 已完成：

```text
src/padp/data/libero_batch_adapter.py
src/padp/data/openpi_libero_loader.py
src/padp/training/smoke_libero_loss.py
```

smoke 测试已改用专门配置：

```text
src/padp/config/libero_va_smoke.yaml
```

这个配置不包含 `hydra.run`、`hydra.sweep`、时间插值或 `hydra.job.num`，避免直接用 `OmegaConf.load()` 时触发 Hydra runtime resolver 问题。正式训练配置仍保留 `src/padp/config/libero_va.yaml`。

服务器已跑通：

```text
uv run python -m padp.training.smoke_libero_loss \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --device cuda:1
```

成功标志：

```text
PADP obs keys 正确
action shape = (2,40,7)
Obs encoder output shape = (392,)
loss_b shape = (2,)
loss mean = 3.1426055431365967
PADP LIBERO smoke loss ok
```

当前不要再重复修 adapter。normalizer 和最小训练链路也已经跑通：

```text
uv run python -m padp.training.compute_norm_stats_for_padp \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --batch-size 8 \
  --num-batches 128 \
  --output-path checkpoints/padp_libero_va/normalizer.pt

uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer.pt \
  --output-dir checkpoints/padp_libero_va/train_smoke \
  --batch-size 2 \
  --num-workers 0 \
  --max-train-steps 10 \
  --checkpoint-every 10 \
  --device cuda:1
```

成功标志：

```text
Saved PADP normalizer to: checkpoints/padp_libero_va/normalizer.pt
input stats keys: action, agentview_image, robot0_eef_pos, robot0_eef_quat, robot0_eye_in_hand_image, robot0_gripper_qpos
step=000001 loss=1.909037
step=000010 loss=2.254615
Saved checkpoint: checkpoints/padp_libero_va/train_smoke/step_000010.pt
Saved checkpoint: checkpoints/padp_libero_va/train_smoke/last.pt
PADP LIBERO train finished
```

注意：这个 10 step checkpoint 只是功能烟测结果，只能说明 loader、adapter、normalizer、policy、loss、backward、optimizer、checkpoint 这一整条链路已经连通，不能作为成功率或收敛效果结论。

100 step 训练测试也已经跑通：

```text
uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer.pt \
  --output-dir checkpoints/padp_libero_va/train_100step \
  --batch-size 4 \
  --num-workers 2 \
  --max-train-steps 100 \
  --checkpoint-every 50 \
  --device cuda:1
```

成功标志：

```text
step=000001 loss=1.858434
step=000050 loss=0.603832
Saved checkpoint: checkpoints/padp_libero_va/train_100step/step_000050.pt
step=000100 loss=1.007593
Saved checkpoint: checkpoints/padp_libero_va/train_100step/step_000100.pt
Saved checkpoint: checkpoints/padp_libero_va/train_100step/last.pt
PADP LIBERO train finished
```

当前已经确认 `num_workers=2` 在服务器上可用。后续训练可以继续优先用 `num_workers=2`；如果出现 dataloader 卡住、worker 异常或内存压力，再退回 `num_workers=0`。

checkpoint 检查注意事项：PyTorch 2.6 以后 `torch.load` 默认 `weights_only=True`。旧 smoke checkpoint 里如果包含 OmegaConf 的 `ListConfig`，会触发安全加载错误。自训本地 checkpoint 可以用：

```python
torch.load(path, map_location="cpu", weights_only=False)
```

只对自己刚训练生成的 checkpoint 这样做，不要对不可信来源的 checkpoint 使用 `weights_only=False`。

1000 step 小规模训练已经跑通：

```text
uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer.pt \
  --output-dir checkpoints/padp_libero_va/train_1000step \
  --batch-size 4 \
  --num-workers 2 \
  --max-train-steps 1000 \
  --checkpoint-every 250 \
  --device cuda:1
```

成功标志：

```text
step=000001 loss=1.858434
step=000250 loss=0.573068
Saved checkpoint: checkpoints/padp_libero_va/train_1000step/step_000250.pt
step=000500 loss=0.206781
Saved checkpoint: checkpoints/padp_libero_va/train_1000step/step_000500.pt
step=000750 loss=0.107386
Saved checkpoint: checkpoints/padp_libero_va/train_1000step/step_000750.pt
step=001000 loss=0.061737
Saved checkpoint: checkpoints/padp_libero_va/train_1000step/step_001000.pt
Saved checkpoint: checkpoints/padp_libero_va/train_1000step/last.pt
PADP LIBERO train finished
```

结论：当前 PADP-VA 的训练链路已经从 smoke loss 推进到可持续训练。现在不再优先改训练循环，下一阶段转向 checkpoint 推理和 service/client 评估。

当前训练准备状态：

```text
1. src/padp/training/compute_norm_stats_for_padp.py 已新增
   - 参考 scripts/compute_norm_stats.py 的数据遍历方式
   - 保存 PADP 的 LinearNormalizer，不保存 openpi NormStats
   - 服务器已跑通

2. src/padp/config/libero_va_train.yaml 已新增
   - 用于 standalone PyTorch smoke train
   - 不包含 hydra.run / hydra.sweep / now resolver

3. src/padp/training/train_libero.py 已新增
   - 参考 PADP 原 train_padp_workspace_v3.py 的训练循环
   - 使用 openpi LIBERO loader + PADP adapter
   - 加载 PADP normalizer
   - 保存 checkpoint
   - 服务器已跑通 10 step smoke train

4. 100 step 训练测试已完成
   - num_workers=2 已验证可用
   - checkpoint 已保存到 checkpoints/padp_libero_va/train_100step

5. 1000 step 小规模训练已完成
   - loss 没有 NaN/Inf
   - checkpoint 已保存到 checkpoints/padp_libero_va/train_1000step
   - num_workers=2 继续可用

6. checkpoint 单批推理 smoke 脚本已新增
   - 文件：src/padp/training/smoke_libero_predict.py
   - 加载 train_1000step/last.pt
   - 读取一批 LIBERO 数据
   - 调用 policy.predict_action(obs_dict)
   - 确认输出 action 形状为 (B,n_action_steps,7)
   - 确认输出 action_pred 形状为 (B,horizon,7)
   - 服务器已跑通：action=(2,1,7)，action_pred=(2,40,7)，没有 NaN/Inf

7. src/padp/serving/serve_libero.py 已新增
   - 复用 openpi.serving.websocket_policy_server.WebsocketPolicyServer
   - 实现 openpi_client BasePolicy 的 infer(obs) 接口
   - 加载 PADP checkpoint 和 normalizer
   - 返回 {"actions": action_chunk}

8. examples/libero/main.py 已新增 selected_task_ids
   - 便于先运行单个 task、单个 trial 的最小闭环评估

9. 当前 service state 适配规则
   - examples/libero/main.py 发来的字段是 observation/image、observation/wrist_image、observation/state、prompt
   - PADP-VA 第一版忽略 prompt
   - 当前 checkpoint 训练时使用 LiberoPadpBatchAdapter 的切片：state[0:3]、state[3:7]、state[7:8]
   - smoke_libero_predict 输出显示 state[3:7] 并不是严格单位 quaternion 分布
   - 因此第一版 service 必须保持同样切片，不额外把 axis-angle 转 quaternion
   - 如果后续要修正为严格 pos+quat+gripper 语义，需要重新生成 normalizer 并重新训练 PADP checkpoint

10. 下一步复用 examples/libero/main.py 做 LIBERO client 评估
   - 先运行 1 个 task、1 个 trial 的最小评估
   - 再扩大到完整 task suite

11. 最小 service/client 闭环已完成
   - PADP server 已启动并监听 0.0.0.0:8000
   - LIBERO client 已连接 ws://127.0.0.1:8000
   - 已运行 libero_object task 0、num_trials_per_task=1、replan_steps=1
   - episode 正常跑完，结果为 Success: False，Total episodes: 1
   - 这说明接口闭环成功，但当前 1000 step PADP-VA checkpoint 尚不能说明成功率
   - 末尾 EGL_NOT_INITIALIZED 是 MuJoCo/EGL 析构阶段报错，当前不影响结果文件和 episode 统计
   - assets/datasets path warning 暂不阻塞评估；后续 client 启动前建议清理 PYTHONPATH，避免继承其他 conda 环境中的 LIBERO 路径

12. 小批量稳定性评估已完成
   - 运行 libero_object task 0、1、2
   - 每个 task 2 个 trial
   - 总计 6 个 episode
   - websocket、仿真、action 返回、结果写入均稳定
   - 结果为 0/6 success
   - `data/libero/padp_results_small.json` 已确认写出 total_episodes=6、total_successes=0、success_rate=0.0
   - `data/libero/padp_videos_small` 已确认写出 6 个 mp4
   - 结论：service/client 链路稳定；下一阶段重点不是继续扩完整评估，而是提升 PADP checkpoint 策略质量

13. 10k 训练首次启动失败
   - 报错位置：openpi loader 创建 tokenizer 时
   - 报错内容：FileNotFoundError: openpi-assets/checkpoints/paligemma_tokenizer.model
   - 原因：训练命令仍受 examples/libero/.venv 客户端环境影响，且未设置 OPENPI_PALIGEMMA_TOKENIZER_PATH
   - 该错误发生在读取第一个 batch 前，不是 PADP 训练循环问题

14. 已新增策略质量诊断工具
   - `src/padp/training/diagnose_libero_semantics.py`
   - 打印 openpi loader 的 state/actions shape、state[:8]、actions[:7] 统计和样例
   - 打印 PADP adapter 后的 obs/action 统计
   - `src/padp/serving/serve_libero.py` 已新增 `--debug-log-steps`
   - service 可记录前 N 次 evaluation observation/state 切片和 PADP action 输出
```

`scripts/compute_norm_stats.py` 可以作为参考，但它生成的是 openpi 的 norm_stats。PADP 需要单独的 `src/padp/training/compute_norm_stats_for_padp.py`，用于保存 `padp.model.common.normalizer.LinearNormalizer`。

服务器已确认 1000 step checkpoint 可读：

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

uv run python - <<'PY'
import torch

path = "checkpoints/padp_libero_va/train_1000step/last.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("loaded:", path)
print("keys:", sorted(ckpt.keys()))
print("step:", ckpt.get("step"))
PY
```

已新增单批推理 smoke 脚本：

```text
src/padp/training/smoke_libero_predict.py
```

脚本目标：

```text
1. 加载 src/padp/config/libero_va_train.yaml。
2. 加载 checkpoints/padp_libero_va/train_1000step/last.pt。
3. hydra.utils.instantiate(cfg.policy)。
4. policy.load_state_dict(ckpt["model"])。
5. normalizer.load_state_dict(ckpt["normalizer"]) 并 policy.set_normalizer(normalizer)。
6. 从 OpenPiLiberoPadpDataset 读取一个 batch。
7. 调用 policy.predict_action(batch["obs"])。
8. 打印 action 和 action_pred 的 shape、dtype、min/max。
```

通过标准：

```text
action shape = (B,n_action_steps,7)
action_pred shape = (B,horizon,7)
没有 NaN/Inf
```

服务器下一步命令。运行 checkpoint 单批推理 smoke：

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

uv run python -m padp.training.smoke_libero_predict \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --checkpoint-path checkpoints/padp_libero_va/train_1000step/last.pt \
  --batch-size 2 \
  --num-workers 0 \
  --device cuda:1
```

预期输出要点：

```text
checkpoint step: 1000
PADP obs keys: ...
output[action]: shape=(2,1,7)
output[action_pred]: shape=(2,40,7)
PADP LIBERO smoke predict ok
```

checkpoint 单批推理已通过。现在进入 websocket service。

终端 1：启动 PADP policy server。

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

uv run python -m padp.serving.serve_libero \
  --checkpoint-path checkpoints/padp_libero_va/train_1000step/last.pt \
  --device cuda:1 \
  --host 0.0.0.0 \
  --port 8000 \
  --action-chunk-size 1
```

终端 2：启动 LIBERO client，先跑单个 task、单个 trial。

如果 `examples/libero/.venv/bin/activate` 不存在，先创建客户端环境：

```bash
cd ~/Desktop/Guided-VLA

uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate

uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

如果 `uv venv --python 3.8` 提示找不到 Python 3.8，先运行：

```bash
uv python install 3.8
```

环境创建完成后再运行 client：

```bash
cd ~/Desktop/Guided-VLA
source examples/libero/.venv/bin/activate
export PYTHONPATH="$PWD/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"

MUJOCO_GL=egl python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_object \
  --args.selected-task-ids 0 \
  --args.num-trials-per-task 1 \
  --args.replan-steps 1 \
  --args.video-out-path data/libero/padp_videos_smoke \
  --args.results-json-path data/libero/padp_results_smoke.json
```

说明：

```text
server 当前默认 action_chunk_size=1，所以 client 也要设置 replan_steps=1。
如果希望保持 examples/libero/main.py 默认 replan_steps=5，则 server 改为 --action-chunk-size 5。
但第一轮建议先使用 replan_steps=1，保证和当前训练配置 n_action_steps=1 更一致。
如果 client 端报 `ModuleNotFoundError: imageio`，说明当前没有进入 examples/libero/.venv，而是在 base 或其他环境中运行。
```

最小闭环已跑通后的检查命令：

```bash
cd ~/Desktop/Guided-VLA
source examples/libero/.venv/bin/activate

python -m json.tool data/libero/padp_results_smoke.json
ls -lh data/libero/padp_videos_smoke
```

下一步做小批量稳定性评估。保持 PADP server 运行，client 使用干净 PYTHONPATH：

```bash
cd ~/Desktop/Guided-VLA
source examples/libero/.venv/bin/activate
unset PYTHONPATH
export PYTHONPATH="$PWD/third_party/libero"

python - <<'PY'
import libero
import openpi_client
print("libero:", libero.__file__)
print("openpi_client:", openpi_client.__file__)
PY

MUJOCO_GL=egl python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_object \
  --args.selected-task-ids 0 1 2 \
  --args.num-trials-per-task 2 \
  --args.replan-steps 1 \
  --args.video-out-path data/libero/padp_videos_small \
  --args.results-json-path data/libero/padp_results_small.json
```

如果 EGL 析构报错频繁干扰日志，可以改成：

```bash
MUJOCO_GL=glx python examples/libero/main.py ...
```

小批量评估结束后检查：

```bash
python -m json.tool data/libero/padp_results_small.json
ls -lh data/libero/padp_videos_small
```

当前小批量评估结论：

```text
total_episodes = 6
total_successes = 0
success_rate = 0.0
videos = 6 mp4 files in data/libero/padp_videos_small
```

这不是接口失败。它说明当前 `train_1000step/last.pt` 还只是迁移链路 checkpoint，不是可用于成功率对比的 checkpoint。

注意：当前 1000 step checkpoint 的 `Success: False` 不应被解释为 PADP 失败定论。它只说明当前极短训练的模型还没有形成可用策略。真正比较 PADP 与 pi05 成功率前，至少需要：

```text
1. 更长训练。
2. 明确 state/action 语义是否与 LIBERO client 完全一致。
3. 固定 task_suite、selected_task_ids、num_trials_per_task、seed 后再横向对比。
```

下一步建议从“继续扩大评估”切换为“策略质量排查”：

```text
1. 核对训练 batch 的 state[0:8] 与 examples/libero/main.py 中 observation/state 的语义是否完全一致。
2. 核对训练 action[:7] 与 env.step(action.tolist()) 所需动作语义是否一致。
3. 先训练更长步数，例如 10k、50k，再用同一套 smoke/small eval 命令测试。
4. 如果希望使用 replan_steps=5，则 server 用 --action-chunk-size 5，并重新跑 1 个 episode 闭环。
5. 决定是否修正 state 表示为严格 pos+axis-angle+gripper 或 pos+quat+gripper；一旦修改，就必须重新生成 normalizer 并重新训练。
```

建议下一轮先做两个诊断，而不是直接跑完整评估：

```text
诊断 A：打印训练 loader 中 state[:8]、actions[:7] 的统计和样例，确认 state[3:7] 到底是什么表示。
诊断 B：在 service 端记录前几个 env observation/state 和 PADP actions，确认动作数值范围是否异常。
```

如果只是想先看更长训练是否改善，可保留现有配置直接训练更久：

```bash
cd ~/Desktop/Guided-VLA
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model

test -f "$OPENPI_PALIGEMMA_TOKENIZER_PATH"
uv run python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer(48); print('tokenizer ok')"

uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer.pt \
  --output-dir checkpoints/padp_libero_va/train_10kstep \
  --batch-size 4 \
  --num-workers 2 \
  --max-train-steps 10000 \
  --checkpoint-every 2500 \
  --device cuda:1
```

注意：如果 PADP websocket server 仍在 cuda:1 上运行，先在 server 终端按 Ctrl-C 停掉，或者把训练改到其他 GPU，例如 `--device cuda:0`。否则 tokenizer 修好后，下一步可能遇到显存冲突。

训练语义诊断命令：

```bash
cd ~/Desktop/Guided-VLA
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model

uv run python -m padp.training.diagnose_libero_semantics \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --batch-size 4 \
  --num-workers 0 \
  --num-batches 4 \
  --print-rows 3
```

评估侧 action/state 调试命令：

```bash
uv run python -m padp.serving.serve_libero \
  --checkpoint-path checkpoints/padp_libero_va/train_1000step/last.pt \
  --device cuda:1 \
  --host 0.0.0.0 \
  --port 8000 \
  --action-chunk-size 1 \
  --debug-log-steps 5
```

但更推荐先完成 state/action 语义诊断，再决定是否继续用当前 normalizer 和 checkpoint 路线。

建议先保留当前能跑通的链路 checkpoint，不要覆盖：

```text
checkpoints/padp_libero_va/train_1000step/last.pt
```

## 目标重新定义

目标不是把 PADP 直接改成 pi05。

更准确的目标是：

```text
在 GuidedVLA 的外部接口下运行 PADP。
训练和推理命令可以不同。
但评估接口尽量和 pi05 一致：
  service -> websocket -> client -> actions
```

## 服务器下一步测试路线：先测 PADP-VA 能不能闭环

### README 对复现的实际假设

`C:\QClaw\GuidedVLA\README.md` 中的 GuidedVLA LIBERO 复现路线不是优先使用 `physical-intelligence/libero`。

README 明确说明：

```text
GuidedVLA LIBERO 训练数据集：ybwowen/libero
GuidedVLA LIBERO checkpoint：ybwowen/pi0-libero-object-depth-skill
默认复现 config：pi0_libero_object_depth_skill
```

对应代码位置：

```text
src/openpi/training/config.py
```

其中：

```text
pi0_libero_object_depth_skill -> repo_id="ybwowen/libero"
pi0_libero_object             -> repo_id="ybwowen/libero"
pi0_libero_depth              -> repo_id="ybwowen/libero"
pi0_libero_skill              -> repo_id="ybwowen/libero"
```

而 `pi05_libero` 当前仍是模板/自定义数据 config：

```text
pi05_libero -> repo_id="your-hf-username/your-dataset"
```

因此，后续 PADP-VA smoke test 应先复用 README 的数据源：

```text
ybwowen/libero
```

不要再用已经确认不兼容 GuidedVLA 当前 `lerobot v3.0` 的：

```text
physical-intelligence/libero
```

当前服务器已经找到本地 tokenizer：

```bash
/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model
```

后续每次测试先执行：

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model
```

先不要手动设置 `LEROBOT_ROOT` 到 `physical-intelligence/libero`。让 `lerobot` 按 `repo_id="ybwowen/libero"` 自动下载到默认 Hugging Face cache。若后续要使用本地路径，再使用：

```bash
export LEROBOT_ROOT=/home/hy/.cache/huggingface/lerobot/ybwowen/libero
```

注意：`LEROBOT_ROOT` 指向数据集根目录，不是 `meta` 目录。

### 第 1 步：确认 openpi 能读 LIBERO batch

先不要训练 PADP，先确认 README 推荐的 `ybwowen/libero` 能被 GuidedVLA 当前 loader 读出来。

```bash
uv run python scripts/test_data_loader.py \
  --config-name pi0_libero_object_depth_skill \
  --framework pytorch \
  --split train \
  --num-batches 1 \
  --num-workers 0
```

如果 full GuidedVLA config 因为 object/skill/depth 辅助项导致排查不方便，可以先用更轻的 `pi0_libero_object` 测试同一个 `ybwowen/libero` 数据源：

```bash
uv run python scripts/test_data_loader.py \
  --config-name pi0_libero_object \
  --framework pytorch \
  --split train \
  --num-batches 1 \
  --num-workers 0
```

如果仍然报 tokenizer 下载错误，说明 `OPENPI_PALIGEMMA_TOKENIZER_PATH` 没有生效；先运行：

```bash
uv run python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer(48); print('tokenizer ok')"
```

### 第 2 步：新增 PADP LIBERO batch adapter

新增文件：

```text
src/padp/data/__init__.py
src/padp/data/openpi_libero_loader.py
src/padp/data/libero_batch_adapter.py
```

第一版只做 VA，不使用 prompt。目标是把 openpi loader 返回的 batch 转成 PADP 的 batch：

```python
{
    "obs": {
        "agentview_image": image,
        "robot0_eye_in_hand_image": wrist_image,
        "robot0_eef_pos": state[..., 0:3],
        "robot0_eef_quat": state[..., 3:7],
        "robot0_gripper_qpos": state[..., 7:8],
    },
    "action": actions[..., :7],
}
```

如果 LIBERO state 维度不是上述切法，以实际 batch 打印结果为准。第一版重点是跑通，不追求完美语义。

### 第 3 步：新增最小 loss smoke test

新增文件：

```text
src/padp/training/smoke_libero_loss.py
```

目标：

```text
openpi LIBERO batch
-> PADP adapter
-> SlidingWindowDiffusionPolicy
-> compute_loss
-> 打印 loss
```

建议命令：

```bash
uv run python -m padp.training.smoke_libero_loss \
  --openpi-config pi0_libero_object \
  --repo-id ybwowen/libero \
  --batch-size 2 \
  --device cuda:0
```

这一步成功的标志是能打印类似：

```text
padp batch obs keys: ...
action shape: ...
loss shape: ...
loss mean: ...
```

这一步比直接训练重要。它能验证图像 shape、state/action 切片、PADP normalizer、obs_encoder 和 diffusion loss 是否能连起来。

### 第 4 步：新增 PADP LIBERO smoke train

新增文件：

```text
src/padp/training/train_libero.py
src/padp/training/checkpoint.py
```

第一版只训练很少步数：

```bash
uv run python -m padp.training.train_libero \
  --openpi-config pi0_libero_object \
  --repo-id ybwowen/libero \
  --exp-name padp_libero_smoke \
  --batch-size 8 \
  --max-steps 100 \
  --device cuda:0
```

成功标志：

```text
loss 能下降或至少稳定反向传播
checkpoint 能保存
训练中没有 shape / dtype / normalizer 报错
```

### 第 5 步：新增 PADP websocket service

新增文件：

```text
src/padp/serving/policy_factory.py
src/padp/serving/serve_libero.py
```

目标输出必须和 GuidedVLA LIBERO client 对齐：

```python
{"actions": action_chunk}
```

建议命令：

```bash
uv run python -m padp.serving.serve_libero \
  --checkpoint-path checkpoints/padp_libero_smoke/latest.pt \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda:0
```

### 第 6 步：用 LIBERO client 评估成功率

service 跑起来后，再开另一个终端：

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

examples/libero/.venv/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.save-name padp_libero_va_smoke \
  --args.host 127.0.0.1 \
  --args.port 8000
```

先只跑 `1 trial`。能完整 rollout 后，再逐步增加：

```bash
--args.num-trials-per-task 5
--args.num-trials-per-task 10
```

### 当前判断标准

第一阶段不要用 pi05 成功率要求 PADP。PADP-VA baseline 的目标是回答：

```text
在同一个 LIBERO 数据和同一个 websocket 评估链路下，
纯视觉/状态/动作的 PADP 能不能训练、能不能闭环推理、能达到多少成功率。
```

如果这个 baseline 成功率明显低，下一步再判断是：

```text
1. 数据适配问题：图像 key、state 切片、action 语义、归一化不对；
2. 模型能力问题：PADP 没有语言条件，无法区分多任务 prompt；
3. 评估接口问题：action chunk、控制频率、reset、normalizer 或 action 后处理不一致。
```

只有排除第 1 和第 3 类问题后，才值得进入 PADP-VLA，也就是给 PADP 加 prompt encoder / language condition。

### 补充修正：如果需要内联覆盖 repo/local root

`scripts/test_data_loader.py` 不能覆盖 `local_root_dir`。如果需要显式指定 repo 或本地 root，使用下面的内联脚本。

默认优先测试 README 数据源 `ybwowen/libero`，并让 `lerobot` 自动下载：

```bash
uv run python - <<'PY'
import dataclasses
import torch

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

cfg = _config.get_config("pi0_libero_object")
cfg = dataclasses.replace(
    cfg,
    batch_size=2,
    num_workers=0,
    data=dataclasses.replace(
        cfg.data,
        repo_id="ybwowen/libero",
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

obs, actions, object_targets = next(iter(loader))
print("obs type:", type(obs))
print("image keys:", list(obs.images.keys()))
for key, value in obs.images.items():
    print("image", key, tuple(value.shape), value.dtype, float(value.min()), float(value.max()))
print("state:", tuple(obs.state.shape), obs.state.dtype, float(obs.state.min()), float(obs.state.max()))
print("actions:", tuple(actions.shape), actions.dtype, float(actions.min()), float(actions.max()))
print("object_targets:", None if object_targets is None else object_targets.keys())
PY
```

这条命令能跑通后，再去写 PADP adapter。否则先不要进入 PADP 训练。

如果你已经下载到本地，也可以显式传本地 root。注意 root 仍然是数据集根目录，不是 `meta`：

```python
base_config=dataclasses.replace(
    cfg.data.base_config,
    local_root_dir="/home/hy/.cache/huggingface/lerobot/ybwowen/libero",
)
```

### 当前新报错：LeRobot 数据集版本不兼容

如果 LIBERO batch 读取报：

```text
BackwardCompatibilityError
NotImplementedError: Contact the maintainer on Discord
```

说明已经越过 tokenizer，当前卡在 LeRobot 数据集版本兼容性。

目前已知：

```text
GuidedVLA 使用 lerobot rev d9e74a9d374a8f26582ad326c699740a227b483c
FASTER_hy 使用 lerobot rev 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

而你之前在 `FASTER_hy` 中跑通过 LIBERO，所以很可能是：

```text
本地缓存的 physical-intelligence/libero 数据格式
与 GuidedVLA 当前锁定的 lerobot 代码版本不兼容。
```

先在服务器上定位版本：

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LEROBOT_ROOT=/home/hy/.cache/huggingface/lerobot/physical-intelligence/libero

find "$LEROBOT_ROOT/meta" -maxdepth 2 -type f | sort | head -50
cat "$LEROBOT_ROOT/meta/info.json" | python -m json.tool | head -80

uv run python - <<'PY'
import lerobot
import lerobot.datasets.lerobot_dataset as ld
import lerobot.datasets.utils as utils

print("lerobot file:", lerobot.__file__)
print("lerobot_dataset file:", ld.__file__)
print("CODEBASE_VERSION in lerobot_dataset:", getattr(ld, "CODEBASE_VERSION", None))
print("CODEBASE_VERSION in utils:", getattr(utils, "CODEBASE_VERSION", None))
PY
```

服务器输出已经确认：

```text
LIBERO 数据集 meta/info.json:
  codebase_version = v2.0

GuidedVLA 当前 .venv 中的 lerobot:
  CODEBASE_VERSION = v3.0
```

因此这不是路径问题，而是明确的版本不匹配：

```text
数据集是 LeRobot v2.0 格式。
当前 GuidedVLA 的 lerobot 按 v3.0 代码读取。
```

### 当前决定：保持 GuidedVLA 的 lerobot v3.0

现在决定不降级 `Guided-VLA/.venv`，而是使用 GuidedVLA 当前锁定的 `lerobot v3.0`。

因此：

```text
不更新 FASTER_hy。
不把 GuidedVLA 的 lerobot 降到 FASTER_hy 的旧版本。
后续要更新/重建的是 LIBERO 数据集，让它兼容 GuidedVLA 当前 lerobot v3.0。
```

结合 README，当前更准确的选择是：

```text
优先使用 ybwowen/libero。
它是 GuidedVLA 作者为本仓库复现发布的 LIBERO 训练数据集。
physical-intelligence/libero 属于 openpi/pi0.5 原始示例数据源，不是 GuidedVLA README 的首选复现数据源。
```

### 重新下载测试结果

你已经执行：

```bash
mv ~/.cache/huggingface/lerobot/physical-intelligence/libero \
   ~/.cache/huggingface/lerobot/physical-intelligence/libero_v2_backup
```

然后用 GuidedVLA 当前 `lerobot v3.0` 重新打开远端数据：

```bash
uv run python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id = "physical-intelligence/libero"
meta = LeRobotDatasetMetadata(repo_id)
print("download/open ok")
print("repo_id:", meta.repo_id)
print("fps:", meta.fps)
print("tasks:", len(meta.tasks))
PY
```

仍然报：

```text
BackwardCompatibilityError
NotImplementedError: Contact the maintainer on Discord
```

这说明：

```text
不是本地缓存旧导致的。
physical-intelligence/libero 远端当前可用版本本身不兼容 GuidedVLA 的 lerobot v3.0。
重新下载同一个 repo 不能解决。
```

但这不等于 GuidedVLA 的 README 复现路线不可用。README 的复现路线应测试：

```text
ybwowen/libero
```

而不是继续测试：

```text
physical-intelligence/libero
```

当前旧缓存已经被你移动到：

```bash
~/.cache/huggingface/lerobot/physical-intelligence/libero_v2_backup
```

如果后续还要在 `FASTER_hy` 中继续使用旧数据，可以恢复：

```bash
mv ~/.cache/huggingface/lerobot/physical-intelligence/libero_v2_backup \
   ~/.cache/huggingface/lerobot/physical-intelligence/libero
```

但恢复后，GuidedVLA 仍然会因为 `v2.0 -> v3.0` 不兼容而报错。

### 接下来的正确路线

保持 GuidedVLA 当前环境不变，下一步先测试 README 推荐数据源：

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

当前服务器已经报：

```text
RevisionNotFoundError: Your dataset must be tagged with a codebase version.
```

含义：

```text
LeRobot v3 默认会按 codebase version tag 选择 Hugging Face dataset revision。
ybwowen/libero 当前没有对应 tag。
这不等于 ybwowen/libero 的数据内容一定不兼容，只是默认 revision 解析失败。
```

下一步先绕开 LeRobot 的自动 revision 解析，用 Hugging Face Hub 直接检查 `main` 分支的 `meta/info.json`：

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

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

判断：

```text
如果 codebase_version 是 v3.0 或能被当前 GuidedVLA lerobot 接受：
  手动 snapshot_download 到本地 root，再让 data loader 用 local_root_dir。

如果 codebase_version 仍是 v2.0：
  ybwowen/libero 也需要迁移/重转，不能直接用于 GuidedVLA 当前 lerobot v3.0。
```

当前服务器输出已经确认：

```text
branches: ['main']
tags: []
codebase_version: v3.0
features:
  image
  wrist_image
  state
  joint_state
  actions
  observation.skill_id
  observation.skill_text
  agentview_attention_object_mask
  wrist_attention_object_mask
  timestamp
  frame_index
  episode_index
  index
  task_index
```

结论：

```text
ybwowen/libero 的 main 分支数据内容是 LeRobot v3.0。
它只是没有 Hugging Face dataset tag，所以 LeRobotDatasetMetadata(repo_id) 默认自动解析 revision 会失败。
当前正确做法是 snapshot_download(..., revision="main") 到本地 root，再显式 local_root_dir 读取。
```

当前已经开始正常下载。`local_dir_use_symlinks` warning 是 huggingface_hub 的弃用提示，不影响下载。

手动下载到本地 root 的命令：

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

如果本地 root 可以打开，再测试 openpi data loader。因为 `scripts/test_data_loader.py` 不能传 `local_root_dir`，优先用内联脚本：

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

当前服务器已经验证通过：

```text
LeRobotDatasetMetadata local open ok
fps: 10
tasks: 40
root: /home/hy/.cache/huggingface/lerobot/ybwowen/libero

openpi data loader read ok
obs type: openpi.models.model.Observation
image keys:
  base_0_rgb
  left_wrist_0_rgb
  right_wrist_0_rgb
base_0_rgb: (2,224,224,3), torch.uint8, range [0,255]
left_wrist_0_rgb: (2,224,224,3), torch.uint8, range [0,255]
right_wrist_0_rgb: zero padding
state: (2,32), torch.float32
actions: (2,50,32), torch.float32
object_targets: object_maps, object_masks
```

结论：

```text
LIBERO 数据读取问题已经解决。
现在正式进入 PADP adapter 阶段。
```

### PADP adapter 阶段：新增文件

新增：

```text
src/padp/data/__init__.py
src/padp/data/libero_batch_adapter.py
src/padp/data/openpi_libero_loader.py
src/padp/config/libero_va.yaml
src/padp/training/smoke_libero_loss.py
```

已完成：

```text
src/padp/config/libero_va.yaml
```

该文件合并自：

```text
src/padp/config/libero.yaml
src/padp/config/task/libero.yaml
```

并修正了：

```text
DINOv3 encoder target:
  padp.model.vision.dinov3_timm_obs_encoder.TimmObsEncoder

action shape:
  [7]

state split:
  3 + 4 + 1
```

暂时不做：

```text
src/padp/training/train_libero.py
src/padp/serving/serve_libero.py
```

先完成 `compute_loss` smoke test，再进入正式训练和 service。

### PADP adapter 阶段：字段映射

当前已经验证通过的 openpi loader 返回的是模型态 batch：

```text
Observation.images:
  base_0_rgb
  left_wrist_0_rgb
  right_wrist_0_rgb
Observation.state:
  shape = (B,32)
actions:
  shape = (B,50,32)
```

注意：

```text
这个 batch 已经经过 openpi 的 ModelTransformFactory。
其中 state/actions 的 32 维来自 PadStatesAndActions(model_config.action_dim)。
这不是 LIBERO 的真实 state/action 维度。
```

PADP 第一版只使用 LIBERO 真实字段：

```text
obs.images["base_0_rgb"]        -> obs["agentview_image"]
obs.images["left_wrist_0_rgb"]  -> obs["robot0_eye_in_hand_image"]
obs.state[..., 0:3]             -> obs["robot0_eef_pos"]
obs.state[..., 3:7]             -> obs["robot0_eef_quat"]
obs.state[..., 7:8]             -> obs["robot0_gripper_qpos"]
actions[..., :7]                -> action
```

忽略：

```text
right_wrist_0_rgb
prompt
object_targets
skill labels
attention maps
actions[..., 7:32] padding
state[..., 8:32] padding
```

### openpi norm stats 与 PADP normalizer

README 中的命令：

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_libero_object_depth_skill
```

是给 openpi / GuidedVLA 训练用的。

它做的是：

```text
LeRobotDataset
-> repack_transforms
-> data_transforms
-> 统计 state/actions
-> 写入 assets/<repo_id 或 asset_id>/norm_stats.json
```

随后 openpi 训练时会：

```text
Normalize(state/actions)
-> ModelTransformFactory
   - ResizeImages
   - TokenizePrompt
   - PadStatesAndActions(32)
```

所以：

```text
openpi norm_stats 不能直接等价于 PADP normalizer。
openpi 的 32 维 actions 也不能直接作为 PADP action_dim。
```

PADP 使用自己的：

```text
padp.model.common.normalizer.LinearNormalizer
```

当前分两阶段：

```text
阶段 A：smoke_libero_loss
  直接用一个 batch fit 临时 PADP normalizer。
  只验证 shape、forward、compute_loss 能跑通。

阶段 B：正式训练
  写 padp/training/compute_norm_stats_for_padp.py 或在 train_libero.py 初始化阶段扫描数据。
  对 adapter 后的 PADP batch 计算 normalizer。
  保存到 PADP checkpoint。
```

### adapter 接口层级

我们的目标不是复刻 `src/openpi` 的模型内部。

更准确地说，PADP 的接口只需要等价于 openpi 的数据/训练外部接口：

```text
同一个 LeRobot 数据集
同一个 LIBERO 字段语义
同一个 service/client 评估入口
不同的模型内部 batch
```

因此有两个 loader 层级：

```text
临时 smoke loader：
  使用 openpi.training.data_loader.create_data_loader(...)
  得到 Observation/Actions
  在 adapter 中裁掉 32 维 padding

正式 PADP loader：
  使用 openpi.training.data_loader.create_torch_dataset(...)
  应用 repack_transforms + data_transforms
  跳过 Normalize
  跳过 ModelTransformFactory
  直接得到未 padding 的 LIBERO state/actions
  再转成 PADP batch
```

当前先做临时 smoke loader，因为它已经被服务器验证能读 batch。
跑通后再改正式 PADP loader。

### PADP adapter 阶段：形状约定

PADP 当前 `RobomimicObsEncoder` 和 robomimic dataset 约定图像为：

```text
(B, S, C, H, W)
float32
range [0,1]
```

因此 adapter 要做：

```text
image:
  (B,224,224,3) uint8
  -> float32 / 255.0
  -> permute to (B,3,224,224)
  -> keep or resize to (B,3,224,224)
  -> unsqueeze time dim -> (B,1,3,224,224)

state:
  (B,32)
  -> 前 8 维
  -> 拆为 (B,1,3), (B,1,4), (B,1,1)

action:
  (B,50,32)
  -> 前 horizon=40
  -> 前 action_dim=7
  -> (B,40,7)
```

### PADP adapter 阶段：新建 LIBERO shape_meta

不要继续沿用 `mimicgen_abs_padp.yaml` 的 action 10 维，也不要改成 openpi 内部的 32 维。

已新增：

```text
src/padp/config/libero_va.yaml
```

内容第一版：

```yaml
name: libero_va

shape_meta: &shape_meta
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

abs_action: &abs_action False
```

说明：

```text
LIBERO/openpi 输出的 action 是 7 维 delta action。
PADP-VA 第一版按 7 维训练，避免为了复用 mimicgen 的 10 维 action 做无意义 padding。
openpi batch 中的 32 维不是时序长度，而是模型 action/state padding 维度。
PADP 的时序长度是 horizon=40。
```

当前 `libero_va.yaml` 使用 DINOv3 encoder，所以图像 shape 保持：

```text
[3, 224, 224]
```

如果后续切回 `RobomimicObsEncoder`，再改为：

```text
[3, 224, 224]
```

### PADP norm stats 脚本计划

需要参考 openpi：

```text
scripts/compute_norm_stats.py
```

但不能直接使用它自动得到 PADP normalizer。

原因：

```text
compute_norm_stats.py 生成 openpi Normalize 使用的 NormStats。
PADP 使用 padp.model.common.normalizer.LinearNormalizer。
openpi 统计的是 openpi data_transforms 后的 state/actions。
PADP 需要统计 adapter 后的 obs/action。
```

后续新增：

```text
src/padp/training/compute_norm_stats_for_padp.py
```

推荐实现：

```text
1. 读取 openpi config，例如 pi0_libero_object。
2. 覆盖 repo_id=ybwowen/libero。
3. 覆盖 local_root_dir=/home/hy/.cache/huggingface/lerobot/ybwowen/libero。
4. 使用 openpi.training.data_loader.create_torch_dataset。
5. 应用 repack_transforms + data_transforms。
6. 不应用 openpi Normalize。
7. 不应用 ModelTransformFactory。
8. 调用 padp.data.libero_batch_adapter 转成 PADP batch。
9. 对 PADP action 和 lowdim obs fit LinearNormalizer。
10. image 使用 get_image_range_normalizer。
11. 保存 normalizer，例如 checkpoints/padp_libero_va/normalizer.pt。
```

当前第一版先不做正式统计，先在 `smoke_libero_loss.py` 里用一个 batch 临时 fit normalizer。等 smoke loss 跑通后再补正式脚本。

### PADP adapter 阶段：smoke loss

新增：

```text
src/padp/training/smoke_libero_loss.py
```

目标链路：

```text
openpi data loader
-> libero_batch_adapter
-> SlidingWindowDiffusionPolicy
-> compute_loss
-> print loss mean
```

建议命令：

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model

uv run python -m padp.training.smoke_libero_loss \
  --openpi-config pi0_libero_object \
  --repo-id ybwowen/libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --batch-size 2 \
  --horizon 40 \
  --device cuda:0
```

这一步成功后，才进入：

```text
src/padp/training/train_libero.py
src/padp/serving/serve_libero.py
```

如果 `ybwowen/libero` 可以打开，也可以再测试 openpi data loader 的默认下载路径：

```bash
uv run python scripts/test_data_loader.py \
  --config-name pi0_libero_object \
  --framework pytorch \
  --split train \
  --num-batches 1 \
  --num-workers 0
```

如果 `ybwowen/libero` 也确认数据内容版本不兼容，再选下面两条之一：

```text
路线 A：检查是否需要 Hugging Face 登录或指定 dataset revision。
路线 B：使用 GuidedVLA 当前 lerobot v3.0，从原始 LIBERO/RLDS 数据重新转换生成一个 v3.0 LeRobot 数据集。
```

当前建议先测 `ybwowen/libero`，因为它是 README 明确发布的数据集。只有它也失败时，才进入重新转换。

需要新增或迁移的转换脚本：

```text
GuidedVLA/examples/libero/convert_libero_data_to_lerobot.py
```

注意：`GuidedVLA/examples/libero` 当前没有这个转换脚本；`FASTER_hy/examples/libero/convert_libero_data_to_lerobot.py` 有旧版脚本，但它是为旧 LeRobot API 写的，不能直接假设兼容 v3.0。可以参考字段映射，但要按 GuidedVLA 当前 `lerobot.datasets.lerobot_dataset.LeRobotDataset.create(...)` API 改写。

转换目标字段保持和当前 `LeRobotLiberoDataConfig` 一致：

```text
image
wrist_image
state
actions
task / prompt metadata
```

转换后的数据集必须满足：

```text
meta/info.json 中 codebase_version 与 GuidedVLA 当前 lerobot 兼容。
GuidedVLA 当前 LeRobotDatasetMetadata 能直接打开。
openpi.training.data_loader.create_data_loader 能读出 batch。
```

转换完成后，再重新测试 LIBERO batch：

```bash
cd ~/Desktop/Guided-VLA
conda activate lerobot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model

uv run python - <<'PY'
import dataclasses

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

cfg = _config.get_config("pi0_libero_object")
cfg = dataclasses.replace(
    cfg,
    batch_size=2,
    num_workers=0,
    data=dataclasses.replace(
        cfg.data,
        repo_id="ybwowen/libero",
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

如果这一步通过，下一步才开始新增 PADP adapter。

如果没有 v3.0 LIBERO 数据集，这一步不会通过。因此在 PADP adapter 前，先完成：

```text
1. 找到 v3.0 兼容的 LIBERO 数据；
2. 或用 GuidedVLA 当前 lerobot 重新转换 LIBERO 原始数据。
```

外部接口一致即可：

```text
输入：环境 observation
输出：{"actions": action_chunk}
评估：复用 examples/libero/main.py 或类似 client
```

## 关键认识

### pi05 是 VLA

pi05 使用：

```text
图像
状态
语言 prompt
动作
```

它的训练和推理流程中，`libero_policy.py` 负责把 LIBERO 数据/环境输入适配成 openpi 模型需要的字段。

### PADP 当前是 VA

PADP 使用：

```text
图像
状态
动作
```

PADP 当前没有：

```text
语言 tokenizer
语言 encoder
prompt condition
VLM backbone
```

因此，PADP 不能天然等价于 pi05。

## libero_policy.py 和 PADP policy 的区别

`src/openpi/policies/libero_policy.py` 不是神经网络策略主体。

它更像是：

```text
LIBERO 数据/环境适配器
```

作用：

```text
把 LIBERO 的字段：
  observation/image
  observation/wrist_image
  observation/state
  prompt
  actions

转换成 openpi 模型需要的：
  image.base_0_rgb
  image.left_wrist_0_rgb
  state
  prompt tokens
  padded actions
```

而 `padp.policy.robomimic.diffusion_unet_hybrid_padp.SlidingWindowDiffusionPolicy` 是真正的模型策略：

```text
obs_encoder
ConditionalUnet1D
diffusion loss
predict_action
normalizer
```

所以后续 PADP 需要的是：

```text
仿照 libero_policy.py 的思想，写一个 padp 的 LIBERO adapter。
不是把 PADP policy 改成 openpi policy。
```

## 推荐路线：先做 PADP-VA LIBERO baseline

这是第一版最小闭环。

### 阶段 1：验证 openpi LIBERO 数据读取

目的：

```text
确认 GuidedVLA 当前环境可以读取 LIBERO LeRobot 数据。
```

注意：

```text
优先使用 README 发布的 ybwowen/libero。
如果需要指定 LEROBOT_ROOT，它应该指向数据集根目录，不是 meta 目录。
```

正确示例：

```bash
export LEROBOT_ROOT=/home/hy/.cache/huggingface/lerobot/ybwowen/libero
```

不是：

```bash
export LEROBOT_ROOT=/home/hy/.cache/huggingface/lerobot/ybwowen/libero/meta
```

如果创建 transform 时下载 tokenizer 失败，需要先设置本地 tokenizer：

```bash
find ~/.cache -name "paligemma_tokenizer.model" -type f 2>/dev/null
find ~/Desktop -name "paligemma_tokenizer.model" -type f 2>/dev/null
```

找到后：

```bash
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model
```

然后测试：

```bash
uv run python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer(48); print('tokenizer ok')"
```

### 阶段 2：新增 PADP LIBERO batch adapter

新增文件：

```text
src/padp/data/__init__.py
src/padp/data/openpi_libero_loader.py
src/padp/data/libero_batch_adapter.py
```

职责：

```text
读取 openpi 的 LIBERO batch
转成 PADP batch
```

目标格式：

```python
{
    "obs": {
        "agentview_image": ...,
        "robot0_eye_in_hand_image": ...,
        "robot0_eef_pos": ...,
        "robot0_eef_quat": ...,
        "robot0_gripper_qpos": ...,
    },
    "action": ...,
}
```

注意：

```text
第一版可以不使用 prompt。
prompt 可以保留在 batch 里做日志，但不输入 PADP。
```

### 阶段 3：写一个最小 compute_loss 测试

新增测试脚本或临时命令，目标是：

```text
openpi LIBERO batch
-> padp libero_batch_adapter
-> SlidingWindowDiffusionPolicy.compute_loss()
-> 得到 scalar / per-sample loss
```

这一步比直接训练更重要。

因为它能验证：

```text
字段映射是否正确
图像 shape 是否正确
action dim 是否正确
PADP normalizer 是否可用
obs_encoder 是否能处理 LIBERO 图像 key
```

### 阶段 4：新增 PADP LIBERO 训练入口

新增文件：

```text
src/padp/training/__init__.py
src/padp/training/train_libero.py
src/padp/training/checkpoint.py
```

目标命令：

```bash
uv run python -m padp.training.train_libero \
  --openpi-config pi0_libero_object \
  --repo-id ybwowen/libero \
  --exp-name padp_libero_smoke \
  --max-steps 100
```

第一版只做 smoke train。

### 阶段 5：新增 PADP websocket serve

新增文件：

```text
src/padp/serving/__init__.py
src/padp/serving/policy_factory.py
src/padp/serving/serve_libero.py
```

目标命令：

```bash
uv run python -m padp.serving.serve_libero \
  --checkpoint-dir checkpoints/padp_libero/padp_libero_smoke/<step> \
  --port 8000
```

输出必须是：

```python
{"actions": action_chunk}
```

这样才能复用 LIBERO client。

### 阶段 6：复用 LIBERO client 评估

命令：

```bash
examples/libero/.venv/bin/python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 1 \
  --args.save-name padp_libero_smoke \
  --args.host 127.0.0.1 \
  --args.port 8000
```

先 1 trial，确认接口能跑。

## 语言条件的后续路线

如果 PADP-VA baseline 跑通，再考虑 PADP-VLA。

可选做法：

```text
1. 使用 frozen text encoder 把 prompt 编码成 language embedding。
2. 将 language embedding 拼到 global_cond。
3. 修改 ConditionalUnet1D 的 global_cond_dim。
4. 训练时输入 prompt embedding。
5. 推理时从 LIBERO task description 生成同样 embedding。
```

这才是“让 PADP 具备语言条件”的路线。

但这不是第一版。

## robomimic -> pi05 格式转换路线

这是另一条路线，不建议和第一版 PADP-LIBERO 同时做。

目标：

```text
把 robomimic/MimicGen 数据转换成 LeRobot/openpi 格式。
让 pi05 和 PADP 都吃同一份转换后的数据。
```

需要做：

```text
1. HDF5 -> LeRobot dataset
2. 图像 key 映射
3. state/action 映射
4. task prompt 生成
5. norm stats
6. pi05 config
7. PADP adapter
8. service/client
```

优点：

```text
更适合最终公平比较。
```

缺点：

```text
工程量更大。
不适合作为当前第一步。
```

## 当前建议

当前建议选择：

```text
先走 PADP-VA LIBERO baseline。
```

也就是：

```text
PADP 不使用语言。
先接 LIBERO 图像、状态、动作。
先跑通训练和 websocket 评估。
```

理由：

```text
这是最小闭环。
它能验证 PADP 是否能在 GuidedVLA 的外部接口下运行。
等 VA baseline 能跑，再讨论语言条件或 robomimic 格式转换。
```

## 2026-06-05：对齐 pi05 的诊断与训练测试口径

当前决定：学习 pi05 的数据和训练外部口径，先诊断，再测试训练。

对齐规则：

```text
pi05_libero batch_size = 256
pi05_libero num_train_steps = 30000
本项目本轮指定 num_workers = 32
PADP train_libero.py 当前一批数据对应一步训练，所以新增 `--num-batches` 作为 `--max-train-steps` 的别名。
openpi 的 compute_norm_stats.py 不直接暴露 `--num-batches`，而是按 `len(dataset) // batch_size` 自动计算。
ybwowen/libero 约 273465 frames；batch_size=256 时，全量统计约为 1068 batches。
```

已完成调整：

```text
src/padp/config/libero_va_train.yaml
  - dataloader.batch_size: 256
  - dataloader.num_workers: 32
  - training.device: cuda:1

src/padp/training/diagnose_libero_semantics.py
  - 默认 batch_size: 256
  - 默认 num_workers: 32

src/padp/training/train_libero.py
  - 新增 `--num-batches`
  - 若同时传入 `--num-batches` 和 `--max-train-steps` 且不一致，则直接报错，避免语义混乱。
```

服务器测试顺序：

```bash
cd ~/Desktop/Guided-VLA
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model
test -f "$OPENPI_PALIGEMMA_TOKENIZER_PATH"
uv run python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer(48); print('tokenizer ok')"
```

多 worker 诊断/训练前，额外设置：

```bash
ulimit -n
ulimit -n 65535 || true
export DATALOADER_PREFETCH_FACTOR=1
```

已知问题记录：

```text
`--batch-size 256 --num-workers 32` 诊断首次运行时触发 `OSError: [Errno 24] Too many open files`。
原因是 PyTorch DataLoader 多 worker 传输 LIBERO 图像 batch 时占用大量文件句柄。
这不是数据语义错误，也不是 CUDA/OOM。
```

已修复：

```text
`src/padp/data/openpi_libero_loader.py` 在创建 openpi LIBERO loader 前会：
1. 对 num_workers > 0 设置 `torch.multiprocessing.set_sharing_strategy("file_system")`。
2. 默认设置 `DATALOADER_PREFETCH_FACTOR=1`，减少 worker 预取带来的文件句柄压力。
```

1. 先诊断训练 loader 的 state/action 语义：

```bash
uv run python -m padp.training.diagnose_libero_semantics \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 4 \
  --print-rows 3
```

这里 `--num-batches 4` 是抽样诊断，不是 pi05 的训练步数。诊断目标是确认：

```text
state[:8] 的数值范围是否正常。
state[3:7] 是否只是沿用 OpenPI state slice，还是确实应解释为 quat。
actions[:,:,:7] 的范围是否和 LIBERO env.step(action) 侧一致。
PADP adapter 后的 obs/action shape 是否仍为 state=8、action=7、horizon=40。
```

2. 再测试 pi05 batch_size 是否能在 PADP 单卡上 backward：

```bash
uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer.pt \
  --output-dir checkpoints/padp_libero_va/train_pi05_batch_backward_only \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 1 \
  --checkpoint-every 0 \
  --device cuda:1 \
  --backward-only
```

3. backward-only 成功后，再做短训练：

```bash
uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer.pt \
  --output-dir checkpoints/padp_libero_va/train_pi05_batch_100step \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 100 \
  --checkpoint-every 50 \
  --device cuda:1
```

4. 如果短训练成功，再考虑长训练：

```bash
uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer.pt \
  --output-dir checkpoints/padp_libero_va/train_pi05_batch_10kstep \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 10000 \
  --checkpoint-every 2500 \
  --device cuda:1
```

如果 batch_size=256 在 PADP 上 OOM，不要把它视为数据或接口错误。pi05 的 batch_size=256 是 VLA 训练配置口径，PADP 当前是单卡 PyTorch diffusion policy，显存曲线不同。下一步应增加 gradient accumulation，用较小 micro batch 近似 effective batch size=256。

诊断执行结果：

```text
batch_size=256
num_workers=32
num_batches=4
OpenPI state shape = (256, 32)
OpenPI actions shape = (256, 50, 32)
PADP action shape = (256, 40, 7)
PADP LIBERO semantics diagnostic finished
```

结论：

```text
pi05 风格 batch_size=256 的 LIBERO 数据读取已经跑通。
PADP adapter 在 batch_size=256 下可以正常转换数据。
下一步可以进入 backward-only 训练显存测试。
```

warning 处理：

```text
worker 启动时会 import moviepy / pygame / ml_collections 等第三方库，因此 warning 可能按 worker 数重复出现。
这不是每个 batch 的训练输出，也不是 PADP 逻辑错误。
`diagnose_libero_semantics.py` 已加入 warning 过滤和紧凑图像统计，后续诊断输出会更干净。
```

backward-only 显存测试执行结果：

```text
batch_size=256
num_workers=32
device=cuda:1
num_batches=1
backward-only smoke ok, step=0, loss=1.976599931716919
```

结论：

```text
pi05 风格 batch_size=256 在当前 PADP-VA 单卡训练中至少可以完成一次 forward/backward。
下一步进入 100 step 短训练测试。
```

短训练命令：

```bash
cd ~/Desktop/Guided-VLA
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model
export DATALOADER_PREFETCH_FACTOR=1
export SDL_AUDIODRIVER=dummy
ulimit -n 65535 || true

uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer.pt \
  --output-dir checkpoints/padp_libero_va/train_pi05_batch_100step \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 100 \
  --checkpoint-every 50 \
  --device cuda:1
```

注意：

```text
100 step 的目标是验证 batch_size=256 的持续训练稳定性和 checkpoint 保存，不用于评估成功率。
如果 100 step 稳定，再进入 10k 或 30k。
正式长期训练前，建议再用 batch_size=256 重新计算更充分的 PADP normalizer。
```

100 step 短训练执行结果：

```text
batch_size=256
num_workers=32
device=cuda:1
num_batches=100
step=000001 loss=1.976600
step=000050 loss=0.384287
step=000100 loss=0.156336
checkpoint step=100
```

结论：

```text
pi05 风格 batch_size=256 的 PADP-VA 持续训练测试已通过。
loss 没有 NaN/Inf，checkpoint 保存和读取正常。
可以进入长训练前准备。
```

长训练前先做 full-ish PADP normalizer：

```bash
cd ~/Desktop/Guided-VLA
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model
export DATALOADER_PREFETCH_FACTOR=1
export SDL_AUDIODRIVER=dummy
ulimit -n 65535 || true

uv run python -m padp.training.compute_norm_stats_for_padp \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 1068 \
  --output-path checkpoints/padp_libero_va/normalizer_pi05_batch_full.pt
```

然后先跑 10k：

```bash
uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer_pi05_batch_full.pt \
  --output-dir checkpoints/padp_libero_va/train_pi05_batch_10kstep \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 10000 \
  --checkpoint-every 2500 \
  --device cuda:1
```

暂不建议直接 30k 的原因：

```text
10k 可以先验证长时间 DataLoader、checkpoint、loss 曲线和 small eval 是否稳定。
如果 10k 已经无法改善 0/6 small eval，再直接 30k 可能只是浪费时间。
如果 10k 链路稳定且初步成功率有改善，再扩到 pi05 的 30000 steps 更合理。
```

10k 训练执行结果：

```text
batch_size=256
num_workers=32
num_batches=10000
device=cuda:1

step=000001 loss=1.747137
step=002500 loss=0.022580
step=005000 loss=0.020152
step=007500 loss=0.016854
step=010000 loss=0.015840
checkpoint: checkpoints/padp_libero_va/train_pi05_batch_10kstep/last.pt
```

结论：

```text
10k 训练已跑通。
loss 曲线是下降并趋稳，不是异常偏大。
```

重要备注：

```text
预期的 1068-batch full normalizer 没有实际保存成功。
train_libero 检测到 normalizer_pi05_batch_full.pt 不存在后，自动 fallback 到 128 batches fit normalizer。
因此本次 10k checkpoint 使用的是 128-batch auto-fit normalizer。
```

当前优先级：

```text
1. 先验证 10k checkpoint 可读。
2. 跑 smoke_libero_predict，确认 predict_action 正常。
3. 启动 serve_libero.py 做 LIBERO small eval。
4. 根据 small eval 决定是继续 30k，还是先修正 normalizer / state-action 语义。
```

10k checkpoint 推理 smoke 结果：

```text
checkpoint step = 10000
output[action] shape = (2, 1, 7)
output[action_pred] shape = (2, 40, 7)
action range roughly = [-2.87, 1.90]
PADP LIBERO smoke predict ok
```

结论：

```text
10k checkpoint 的单批推理链路正常。
现在可以进入 service/client 评估，但先做 small eval，不直接做完整 suite。
```

下一步：10k small eval。

终端 1：

```bash
cd ~/Desktop/Guided-VLA
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SDL_AUDIODRIVER=dummy

uv run python -m padp.serving.serve_libero \
  --checkpoint-path checkpoints/padp_libero_va/train_pi05_batch_10kstep/last.pt \
  --device cuda:1 \
  --host 0.0.0.0 \
  --port 8000 \
  --action-chunk-size 1 \
  --debug-log-steps 5
```

终端 2：

```bash
cd ~/Desktop/Guided-VLA
source examples/libero/.venv/bin/activate
unset PYTHONPATH
export PYTHONPATH="$PWD/third_party/libero"
export SDL_AUDIODRIVER=dummy

MUJOCO_GL=egl python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_object \
  --args.selected-task-ids 0 1 2 \
  --args.num-trials-per-task 2 \
  --args.replan-steps 1 \
  --args.video-out-path data/libero/padp_videos_10k_small \
  --args.results-json-path data/libero/padp_results_10k_small.json
```

判断标准：

```text
如果 10k small eval > 0/6，说明训练步数确实开始带来行为改善，可以继续扩大 trial 或训练到 30k。
如果仍是 0/6，优先看视频和 server debug log，检查 action 是否方向/尺度/语义不匹配，而不是直接盲目 30k。
```

10k small eval 结果：

```text
task_suite = libero_object
selected_task_ids = [0, 1, 2]
num_trials_per_task = 2
total_episodes = 6
total_successes = 0
success_rate = 0.0
```

根因候选更新：

```text
最优先怀疑 action 空间转换，而不是训练步数。
```

原因：

```text
`pi0_libero_object` 使用 `LeRobotLiberoDataConfig(extra_delta_transform=True)`。
训练输入侧会执行 `DeltaActions(make_bool_mask(6, -1))`：
  actions[..., :6] -= state[..., :6]
这意味着 PADP 当前学到的是前 6 维 delta action。

LIBERO env.step(action) 需要的动作与 openpi 输出侧一致。
openpi 的输出侧会通过 `AbsoluteActions(make_bool_mask(6, -1))`：
  actions[..., :6] += state[..., :6]

此前 PADP server 少了这一步，直接把 delta action 返回给 env.step，可能导致 0/6。
```

已修复：

```text
src/padp/serving/serve_libero.py
  - 新增 `--output-action-space {absolute,delta}`
  - 默认 `absolute`
  - `absolute` 模式下对前 6 维执行 delta -> absolute 转换
  - 第 7 维 gripper 不加 state
```

下一步重新评估，不要继续训练：

终端 1：

```bash
cd ~/Desktop/Guided-VLA
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SDL_AUDIODRIVER=dummy

uv run python -m padp.serving.serve_libero \
  --checkpoint-path checkpoints/padp_libero_va/train_pi05_batch_10kstep/last.pt \
  --device cuda:1 \
  --host 0.0.0.0 \
  --port 8000 \
  --action-chunk-size 1 \
  --output-action-space absolute \
  --debug-log-steps 5
```

终端 2：

```bash
cd ~/Desktop/Guided-VLA
source examples/libero/.venv/bin/activate
unset PYTHONPATH
export PYTHONPATH="$PWD/third_party/libero"
export SDL_AUDIODRIVER=dummy

MUJOCO_GL=egl python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_object \
  --args.selected-task-ids 0 1 2 \
  --args.num-trials-per-task 2 \
  --args.replan-steps 1 \
  --args.video-out-path data/libero/padp_videos_10k_small_abs \
  --args.results-json-path data/libero/padp_results_10k_small_abs.json
```

对比：

```text
旧结果：padp_results_10k_small.json = 0/6，使用 raw delta action。
新结果：padp_results_10k_small_abs.json，使用 absolute action。
```

## 2026-06-05：当前优先路线改为 full normalizer 重跑

当前不是继续直接扩大评估，也不是马上训练 30k。优先怀疑点是上一轮 10k 训练没有真正使用 full normalizer。

已确认：

```text
1. 10k 训练链路是通的。
2. checkpoint 可读，step=10000。
3. smoke_libero_predict 正常。
4. service/client small eval 能跑完。
5. raw delta small eval = 0/6。
6. absolute action small eval = 0/6。
7. 但这轮 10k 实际使用的是 128-batch fallback normalizer，不是预期的 1068-batch normalizer。
```

根因：

```text
旧版 fit_padp_normalizer 会把所有 batch 连同图像一起缓存到内存里。
batch_size=256、num_batches=1068 时，图像缓存量极大，full normalizer 很容易失败或被系统杀掉。
```

已修正：

```text
src/padp/data/openpi_libero_loader.py
  - fit_padp_normalizer 改成 streaming 统计。
  - 图像不缓存，只登记 image key 并使用 get_image_range_normalizer()。
  - action 和低维 obs 逐 batch 累计 min/max/sum/sumsq/count。

src/padp/training/compute_norm_stats_for_padp.py
  - 新增 --log-every，便于观察 1068 batch 统计进度。
```

下一步服务器执行顺序：

```bash
cd ~/Desktop/Guided-VLA
git pull

deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model
export DATALOADER_PREFETCH_FACTOR=1
export SDL_AUDIODRIVER=dummy
ulimit -n 65535 || true
```

1. 重新生成 full normalizer：

```bash
uv run python -m padp.training.compute_norm_stats_for_padp \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 1068 \
  --log-every 50 \
  --output-path checkpoints/padp_libero_va/normalizer_pi05_batch_stream_1068.pt
```

2. 检查 normalizer 已保存：

```bash
uv run python - <<'PY'
import torch

path = "checkpoints/padp_libero_va/normalizer_pi05_batch_stream_1068.pt"
payload = torch.load(path, map_location="cpu", weights_only=False)
print("loaded:", path)
print("keys:", sorted(payload.keys()))
print("metadata:", payload.get("metadata", {}))
PY
```

3. 使用 full normalizer 重跑 10k：

```bash
uv run python -m padp.training.train_libero \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --normalizer-path checkpoints/padp_libero_va/normalizer_pi05_batch_stream_1068.pt \
  --output-dir checkpoints/padp_libero_va/train_pi05_batch_10kstep_fullnorm \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 10000 \
  --checkpoint-every 2500 \
  --device cuda:1 \
  --no-fit-normalizer-if-missing
```

这里 `--no-fit-normalizer-if-missing` 很重要。如果 normalizer 文件不存在，应直接失败，而不是再次退回 128-batch fallback。

4. 训练后检查 checkpoint 和 predict：

```bash
uv run python - <<'PY'
import torch

path = "checkpoints/padp_libero_va/train_pi05_batch_10kstep_fullnorm/last.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("loaded:", path)
print("keys:", sorted(ckpt.keys()))
print("step:", ckpt.get("step"))
PY

uv run python -m padp.training.smoke_libero_predict \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --checkpoint-path checkpoints/padp_libero_va/train_pi05_batch_10kstep_fullnorm/last.pt \
  --batch-size 2 \
  --num-workers 0 \
  --device cuda:1
```

5. 再做 small eval：

server：

```bash
uv run python -m padp.serving.serve_libero \
  --checkpoint-path checkpoints/padp_libero_va/train_pi05_batch_10kstep_fullnorm/last.pt \
  --device cuda:1 \
  --host 0.0.0.0 \
  --port 8000 \
  --action-chunk-size 1 \
  --output-action-space absolute \
  --debug-log-steps 5
```

client：

```bash
cd ~/Desktop/Guided-VLA
source examples/libero/.venv/bin/activate
unset PYTHONPATH
export PYTHONPATH="$PWD/third_party/libero"
export SDL_AUDIODRIVER=dummy

MUJOCO_GL=egl python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_object \
  --args.selected-task-ids 0 1 2 \
  --args.num-trials-per-task 2 \
  --args.replan-steps 1 \
  --args.video-out-path data/libero/padp_videos_10k_fullnorm_abs \
  --args.results-json-path data/libero/padp_results_10k_fullnorm_abs.json
```

判断：

```text
如果 fullnorm small eval > 0/6：
  说明 normalizer 质量确实影响明显，可继续训练到 30k 或扩大评估。

如果 fullnorm small eval 仍然 0/6：
  不再优先怀疑 normalizer。
  下一步看视频和 server debug log，重点排查 state/action 语义、任务条件缺失、以及 PADP-VA 在多任务 LIBERO object 上是否需要 task/language condition。
```

## 2026-06-05：full normalizer 10k 已完成，下一步先测试而不是先补全训练框架

本轮训练输出已确认：

```text
normalizer: checkpoints/padp_libero_va/normalizer_pi05_batch_stream_1068.pt
checkpoint: checkpoints/padp_libero_va/train_pi05_batch_10kstep_fullnorm/last.pt
step: 10000
loss: 1.747026 -> 0.015536
smoke_libero_predict: 通过
output[action]: (2, 1, 7)
output[action_pred]: (2, 40, 7)
无 NaN/Inf
```

当前判断：

```text
先测试 small eval，不要先补全 val/rollout。
```

原因：

```text
1. 训练链路已经稳定，loss 正常下降。
2. checkpoint 可加载，predict_action 已通过单批推理 smoke。
3. 当前最关键的问题不是训练代码是否完整，而是 full normalizer + absolute action 后真实 LIBERO rollout 是否仍然 0/6。
4. val loss 只能说明离线 imitation loss，不能替代仿真成功率。
5. 如果 small eval 仍然 0/6，再补 val/diagnostics 才能更有针对性。
```

下一步执行 small eval。

终端 1：启动 PADP server。

```bash
cd ~/Desktop/Guided-VLA
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SDL_AUDIODRIVER=dummy

uv run python -m padp.serving.serve_libero \
  --checkpoint-path checkpoints/padp_libero_va/train_pi05_batch_10kstep_fullnorm/last.pt \
  --device cuda:1 \
  --host 0.0.0.0 \
  --port 8000 \
  --action-chunk-size 1 \
  --output-action-space absolute \
  --debug-log-steps 10
```

终端 2：启动 LIBERO client small eval。

```bash
cd ~/Desktop/Guided-VLA
source examples/libero/.venv/bin/activate
unset PYTHONPATH
export PYTHONPATH="$PWD/third_party/libero"
export SDL_AUDIODRIVER=dummy

MUJOCO_GL=egl python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_object \
  --args.selected-task-ids 0 1 2 \
  --args.num-trials-per-task 2 \
  --args.replan-steps 1 \
  --args.video-out-path data/libero/padp_videos_10k_fullnorm_abs \
  --args.results-json-path data/libero/padp_results_10k_fullnorm_abs.json
```

测试后检查：

```bash
cd ~/Desktop/Guided-VLA
source examples/libero/.venv/bin/activate
python -m json.tool data/libero/padp_results_10k_fullnorm_abs.json
ls -lh data/libero/padp_videos_10k_fullnorm_abs
```

根据结果决策：

```text
如果 > 0/6：
  说明 full normalizer 和 absolute action 路线有效。下一步可以扩大 trial 或训练到 30k。

如果仍然 0/6：
  不再优先怀疑 normalizer。下一步先看视频和 server debug log，再补离线 val / state-action 诊断。

如果动作明显发散、方向错误或几乎不动：
  优先排查 state/action 语义、delta->absolute、gripper 语义。

如果动作合理但任务对象错误：
  优先怀疑 PADP-VA 无 task/language condition，不适合直接混训多任务 LIBERO object。
```

暂缓补全项：

```text
1. 训练中 val loss。
2. 训练中 rollout eval。
3. top-k checkpoint。
4. 自动 service/client eval。
```

这些应在 small eval 结果出来后再做。当前不要先补工程框架，否则可能在真正问题尚未定位前增加复杂度。

## 2026-06-06：fullnorm small eval 仍为 0/6，进入 action-space 诊断

full normalizer + 10k + absolute action 的 small eval 结果：

```text
libero_object task 0/1/2
num_trials_per_task = 2
total_episodes = 6
total_successes = 0
success_rate = 0.0
videos: data/libero/padp_videos_10k_fullnorm_abs
results: data/libero/padp_results_10k_fullnorm_abs.json
```

视频观察：

```text
提前闭合 2 个
撞到盒子闭合 1 个
抓取失败 3 个
随后逐渐伸直并向上抬
```

当前判断：

```text
不是接口失败，也不是 normalizer 首要问题。
优先排查 action/state/gripper 语义，以及 PADP-VA 无 task/language condition 导致多任务混训不清的问题。
```

已新增诊断脚本：

```text
src/padp/training/diagnose_libero_action_space.py
```

用途：

```text
读取训练集 openpi/PADP batch。
打印训练集 delta action[:7] 的范围。
按 openpi 的 AbsoluteActions 逻辑将前 6 维加回 state[:6]。
打印训练集 env-space absolute action[:7] 的范围。
单独打印 gripper action[6] 的范围。
用于和 serve_libero debug log 中的推理动作范围对比。
```

服务器命令：

```bash
cd ~/Desktop/Guided-VLA
git pull

deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/home/hy/.cache/openpi/big_vision/paligemma_tokenizer.model
export DATALOADER_PREFETCH_FACTOR=1
export SDL_AUDIODRIVER=dummy
ulimit -n 65535 || true

uv run python -m padp.training.diagnose_libero_action_space \
  --local-root-dir /home/hy/.cache/huggingface/lerobot/ybwowen/libero \
  --batch-size 256 \
  --num-workers 32 \
  --num-batches 1068 \
  --print-rows 8
```

重点对比：

```text
server absolute action roughly:
x: -0.031 到 0.210
y: -0.000 到 0.064
z: -0.245 到 0.131
gripper: 0.969 到 0.983
```

判断规则：

```text
如果 server absolute xyz 超出训练集 absolute 范围：优先查 normalizer / action decode / delta->absolute。
如果 server gripper 一开始接近 0.98，但训练集中成功抓取前不应如此：优先查 gripper 开合语义或 gripper loss 偏置。
如果动作范围都合理但仍抓错物体或提前闭合：优先加入 task/language condition，或先按单 task 训练。
```

## 2026-06-06：action-space 诊断结论与 gripper A/B 测试

训练集 action-space 诊断结果：

```text
train absolute action first step [:7]
  min:  [-0.9375, -0.9375, -0.9375, -0.2421, -0.3750, -0.3643, -1.0000]
  max:  [ 0.9375,  0.9375,  0.9375,  0.3557,  0.3750,  0.3750,  1.0000]

train absolute action all horizon [:7]
  min:  [-0.9375, -0.9375, -0.9375, -0.2421, -0.3750, -0.3643, -1.0000]
  max:  [ 0.9375,  0.9375,  0.9375,  0.3557,  0.3750,  0.3750,  1.0000]

grunner/server debug absolute action roughly:
  x: -0.031 到 0.210
  y: -0.000 到 0.064
  z: -0.245 到 0.131
  gripper: 0.969 到 0.983
```

结论：

```text
1. xyz / rotation 的 server 输出基本在训练集 absolute action 范围内，不像是 delta->absolute 整体错误。
2. gripper 是最可疑项：训练集 gripper 为二值 -1 / +1，但当前 server 几乎一直输出 +1。
3. 视频里出现提前闭合、撞盒子闭合、抓取失败，和 gripper 输出偏向 +1 的现象一致。
4. 下一步先做 gripper 方向 A/B 测试，不先重训。
```

已修改：

```text
src/padp/serving/serve_libero.py
  - 新增 --gripper-action-mode。
  - 可选 raw / invert / binary / binary_invert。
  - 默认 raw，保持旧行为。
  - debug log 现在同时打印 model action 和 env action。
```

先测试 gripper invert：

server：

```bash
cd ~/Desktop/Guided-VLA
git pull

deactivate 2>/dev/null || true
unset VIRTUAL_ENV
conda activate lerobot

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export SDL_AUDIODRIVER=dummy

uv run python -m padp.serving.serve_libero \
  --checkpoint-path checkpoints/padp_libero_va/train_pi05_batch_10kstep_fullnorm/last.pt \
  --device cuda:1 \
  --host 0.0.0.0 \
  --port 8000 \
  --action-chunk-size 1 \
  --output-action-space absolute \
  --gripper-action-mode invert \
  --debug-log-steps 10
```

client：

```bash
cd ~/Desktop/Guided-VLA
source examples/libero/.venv/bin/activate
unset PYTHONPATH
export PYTHONPATH="$PWD/third_party/libero"
export SDL_AUDIODRIVER=dummy

MUJOCO_GL=egl python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port 8000 \
  --args.task-suite-name libero_object \
  --args.selected-task-ids 0 1 2 \
  --args.num-trials-per-task 2 \
  --args.replan-steps 1 \
  --args.video-out-path data/libero/padp_videos_10k_fullnorm_abs_gripper_invert \
  --args.results-json-path data/libero/padp_results_10k_fullnorm_abs_gripper_invert.json
```

判断：

```text
如果 invert 后提前闭合明显减少或成功率上升：说明 gripper 符号/开合方向需要修正。
如果 invert 后一直张开、抓不到：说明 +1 可能原本就是打开，问题在时机或任务条件。
如果两者都 0/6 但行为不同：保留 gripper 开关，下一步加入 task condition 或做单 task 训练。
```
