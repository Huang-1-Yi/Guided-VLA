# LIBERO Benchmark

该示例用于运行 LIBERO benchmark：https://github.com/Lifelong-Robot-Learning/LIBERO

注意：如果需要更新本目录下的 `requirements.txt`，在执行 `uv pip compile` 时必须额外添加 `--extra-index-url https://download.pytorch.org/whl/cu113`。

该示例依赖已初始化的 git submodule。请先运行：

```bash
git submodule update --init --recursive
```

## 使用 Docker（推荐）

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

可以通过额外的 `SERVER_ARGS` 自定义要加载的 checkpoint（见 `scripts/serve_policy.py`），也可以通过额外的 `CLIENT_ARGS` 自定义 LIBERO task suite（见 `examples/libero/main.py`）。
例如：

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## 不使用 Docker（不推荐）

终端窗口 1：

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

终端窗口 2：

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

如果希望复现下表结果，可以评估 `gs://openpi-assets/checkpoints/pi05_libero/` 处的 checkpoint。该 checkpoint 使用 openpi 中的 `pi05_libero` 配置训练得到。

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
