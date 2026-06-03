# LIBERO-Plus 评估

[LIBERO-Plus](https://arxiv.org/abs/2510.13626) 是基于 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 构建的鲁棒性 benchmark。它引入了 **7 个扰动维度**，用于评估模型在分布内评估之外的泛化能力：

| 维度 | 说明 |
|---|---|
| Objects Layout | 干扰物体和目标物体位置变化 |
| Camera Viewpoints | 相机位置、朝向和视场变化 |
| Robot Initial States | 机械臂初始位姿变化 |
| Language Instructions | 基于 LLM 的指令改写 |
| Light Conditions | 光照强度、方向、颜色和阴影变化 |
| Background Textures | 场景和表面外观变化 |
| Sensor Noise | 光度扰动和图像退化 |

GuidedVLA 在 LIBERO-Plus 上达到 **75.4% 平均成功率**，相比之下，蟺鈧€ baseline 为 68.2%。

## 依赖要求

该示例需要 LIBERO-Plus submodule。请确认它已经初始化：

```bash
git submodule update --init --recursive
```

## 设置（不使用 Docker）

为 LIBERO-Plus 模拟器创建 Python 3.8 环境：

```bash
# System dependencies
sudo apt install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev

# Create virtual environment and install dependencies
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

## 运行评估

### 步骤 1：启动 policy server

在一个终端中启动 policy server，并指向你训练好的 checkpoint：

```bash
uv run --no-sync scripts/serve_policy.py \
    --env LIBERO \
    --port 8000 \
    policy:checkpoint \
    --policy.config pi0_libero_object_depth_skill \
    --policy.dir checkpoints/pi0_libero_object_depth_skill/<exp_name>/<step>
```

### 步骤 2：运行单个扰动类别

在第二个终端中运行：

```bash
source examples/libero_plus/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/LIBERO-plus

python examples/libero_plus/main.py \
    --host 127.0.0.1 \
    --port 8000 \
    --task-suite-name libero_object \
    --category "Objects Layout" \
    --video-out-path data/libero_plus/videos \
    --num-trials-per-task 1 \
    --results-json-path data/libero_plus/libero_object.json
```

可用的 `--task-suite-name` 取值：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`、`all`

可用的 `--category` 取值：`"Objects Layout"`、`"Camera Viewpoints"`、`"Robot Initial States"`、`"Language Instructions"`、`"Light Conditions"`、`"Background Textures"`、`"Sensor Noise"`

常用 `main.py` 参数：
- `--task-ids`：例如 `0`、`0,3,7` 或 `10-19`
- `--replan-steps`：从服务器请求的 action chunk 大小
- `--results-json-path`：滚动写入的 JSON 汇总；设置 `--category` 时会自动追加类别后缀

### 步骤 3：一次性运行所有 task suite 和扰动

```bash
uv run examples/libero_plus/eval_libero_plus.py \
    --checkpoint-dir checkpoints/pi0_libero_object_depth_skill/<exp_name>/<step> \
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
