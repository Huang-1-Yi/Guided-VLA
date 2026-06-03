# RoboTwin 评估

该示例用于在 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 上评估 OpenPI / GuidedVLA policy，并且不需要修改 RoboTwin 代码库。

推荐设置：

```bash
git submodule update --init --recursive third_party/RoboTwin
```

该集成与 RoboTwin 的 `policy/pi05` 分支保持一致：

- observation/state 顺序为 `[left_arm(6), left_gripper, right_arm(6), right_gripper]`
- action 是相同 14 维顺序下的 **absolute** joint target
- gripper 已经归一化到 `[0, 1]`
- 训练时仍然保持 `use_delta_joint_actions=True`，因此 action 会先转换为模型使用的 delta，并在推理时转换回 absolute qpos
- RoboTwin 必须使用 `adapt_to_pi=False`，该设置已经写入 RoboTwin configs

## 1. 准备 LeRobot 数据

假设你已经在本地或 Hugging Face Hub 上准备好了 LeRobot 格式的 RoboTwin 数据集。

## 2. 计算 norm stats

RoboTwin configs 使用固定的 asset id `robotwin`，因此 norm stats 应写入该 asset id 下。

```bash
uv run scripts/compute_norm_stats.py pi0_base_aloha_robotwin_full \
  --repo-id robotwin_grab_roller_demo_randomized \
  --local-root-dir /path/to/lerobot/root \
  --asset-id robotwin
```

下面命令中的 `pi0_base_aloha_robotwin_full` 可以替换为任意 RoboTwin config。

## 3. 训练

可用的 RoboTwin configs：

- `pi05_aloha_robotwin_full`
- `pi05_aloha_robotwin_lora`
- `pi0_base_aloha_robotwin_full`
- `pi0_base_aloha_robotwin_lora`
- `pi0_fast_aloha_robotwin_full`
- `pi0_fast_aloha_robotwin_lora`
- `pi0_base_aloha_robotwin_object_depth_skill`

说明：

- `scripts/train_pytorch.py` 通过 `torchrun` 使用 DDP；`--nproc_per_node` 控制 PyTorch 并行度。
- `fsdp_devices` 只被 JAX trainer（`scripts/train.py`）使用，PyTorch trainer 会忽略它。
- `pi0_base_aloha_robotwin_object_depth_skill` 还要求数据集中包含 `observation.skill_id`；加载数据时会在线构造 `skill_soft`。

RoboTwin `policy/pi05` 分支中的向后兼容别名也可用：

- `pi05_aloha_full_base`
- `pi05_base_aloha_lora`

示例：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py \
  pi0_base_aloha_robotwin_full \
  --exp_name robotwin_grab_roller \
  --repo_id robotwin_grab_roller_demo_randomized \
  --local_root_dir /path/to/lerobot/root
```

## 4. 启动 checkpoint 服务

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_base_aloha_robotwin_full \
  --policy.dir=checkpoints/pi0_base_aloha_robotwin_full/<exp_name>/<step>
```

## 5. 运行 RoboTwin 评估

请在已经通过 `third_party/RoboTwin` submodule 安装好 RoboTwin 的环境中运行：

```bash
bash examples/robotwin/run.sh \
  --args.task-name adjust_bottle \
  --args.task-config demo_randomized \
  --args.instruction-type unseen \
  --args.action-horizon 50 \
  --args.record-videos
```

结果会写入 `data/robotwin/eval/<task>/<task_config>/`。
