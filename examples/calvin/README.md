# CALVIN Benchmark

本目录包含 [CALVIN benchmark](https://github.com/mees/calvin) 的客户端评估入口。
对于 policy 训练，可以使用 LeRobot 数据集 [InternRobotics/InternData-Calvin_ABC](https://huggingface.co/datasets/InternRobotics/InternData-Calvin_ABC)。本仓库提供了对应配置：

- [`CalvinInputs` and `CalvinOutputs`](../../src/openpi/policies/calvin_policy.py)
- [`LeRobotCalvinDataConfig`](../../src/openpi/training/config.py)
- [`pi05_calvin` and `pi05_faster_calvin`](../../src/openpi/training/config.py)

## 设置

克隆 CALVIN 仓库，并在 [main.py](main.py) 中更新路径：

```python
CALVIN_ROOT = "/path/to/calvin"
```

按照[官方说明](https://github.com/mees/calvin#computer--quick-start)创建 conda 环境 `calvin_venv` 并安装依赖。然后在 GuidedVLA 仓库根目录安装客户端包：

```bash
cd /path/to/GuidedVLA
pip install -e packages/openpi-client
pip install draccus
```

请确认数据集已经下载到 `dataset/task_ABC_D/validation/`。

## 评估

评估使用两个进程：

1. 在主 openpi 环境中运行的 policy server。
2. 在 `calvin_venv` 环境中运行的 CALVIN 客户端。

### 启动 policy server

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_faster_calvin \
  --policy.dir=checkpoints/pi05_faster_calvin/my_experiment/29999
```

### 运行评估

```bash
conda activate calvin_venv

export PYTHONPATH=$PYTHONPATH:$PWD

python examples/calvin/main.py \
  --save_name pi05_faster \
  --host 0.0.0.0 \
  --port 8000
```

常用选项：

- `--out_path`：输出目录。
- `--save_name`：本次评估运行的标识符。日志会写入 `{out_path}/{save_name}/`。

完整 CLI 选项列表请见 [main.py](main.py)。

结果会写入：

```text
{out_path}/{save_name}/result.json
{out_path}/{save_name}/success_rate.txt
```

`result.json` 包含：

- `avg_seq_len`：每条 5 指令链中的平均成功序列长度。
- `chain_sr`：连续完成 1、2、3、4 和 5 条指令的成功率。
- `task_info`：每个任务的成功次数和总次数。

## 结果

以下结果使用标准 CALVIN ABC->D benchmark 进行评估：

- 1,000 条评估序列
- 每条序列包含 5 条链式指令
- 每条指令最多 720 个环境步

| Model       | Config               | 1    | 2    | 3    | 4    | 5    | Average Len |
| ----------- | -------------------- | ---- | ---- | ---- | ---- | ---- | ----------- |
| 蟺0.5        | `pi05_calvin`        | 94.2 | 88.7 | 85.7 | 83.2 | 79.5 | 4.313       |
| 蟺0.5+FASTER | `pi05_faster_calvin` | 95.1 | 89.1 | 85.0 | 81.9 | 78.1 | 4.292       |
