from typing import Optional, Tuple
import math
import torch

"""
PADP 位置感知加噪模块。

核心理念：
1. 标准扩散模型把噪声强度绑定到扩散时间步 t；PADP 把噪声强度绑定到动作在预测窗口中的位置 h。
2. 越靠近当前观测的动作越可信，保留更多真实动作信号；越远的未来动作越不确定，注入更强噪声。
3. 当 horizon=H 时，本文件显式构造长度 H+1 的噪声表，索引 0 表示纯净动作，索引 H 表示纯高斯噪声。

统一加噪公式：
    x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon

模式区别：
- positionwise: 固定按位置查余弦表，位置 h 使用表索引 h+1；最后一个动作使用索引 H，严格纯高斯。
- linear: 不查余弦表，直接构造线性 alpha_bar；最后一个动作 alpha_bar=0，严格纯高斯。
- constant: 每个 batch 样本随机抽一个表索引，并让该样本的整条轨迹共享同一噪声强度。
- random: 每个 batch 样本、每个位置独立随机抽表索引，测试时域噪声连续性被破坏后的鲁棒性。
- chunkwise: 每个 batch 样本按 chunk 分块查余弦表；最后一个 chunk 强制表索引 H，保持远端纯高斯边界。
"""


def build_horizon_alpha_bar(
    horizon: int,
    beta_schedule: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    预计算基于 Horizon 的扩散调度表。
    
    【理论对齐】PADP 核心理念：设定 T = H，扩散步数等价于位置索引。
    调度表长度为 H + 1，覆盖从纯净动作 (h=0 -> k=1) 到纯高斯噪声 (h=H-1 -> k=H) 的所有阶段。
    """
    if beta_schedule != "squaredcos_cap_v2":
        raise NotImplementedError(f"仅支持 squaredcos_cap_v2 调度: {beta_schedule}")

    if horizon <= 0:
        raise ValueError(f"horizon 必须为正数, 当前为 {horizon}")

    # 显式保留两个端点：t=0 对应纯净动作，t=horizon 对应纯噪声。
    # positionwise / constant / random / chunkwise 都通过这张余弦表取系数。
    s = 0.008
    t = torch.arange(0, horizon + 1, dtype=torch.float32)
    alpha_bar = torch.cos((t / horizon + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    alpha_bar[0] = 1.0
    alpha_bar[-1] = 0.0

    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)
    return alpha_bar, sqrt_alpha_bar, sqrt_one_minus_alpha_bar


def build_position_index(
    horizon: int,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """生成窗口内的位置索引 [0, 1, ..., H-1]，可按 batch 复制成 [B, H]。"""
    position_index = torch.arange(horizon, device=device, dtype=torch.long)
    if batch_size is None:
        return position_index
    return position_index.unsqueeze(0).repeat(batch_size, 1)


def build_constant_position_timesteps(
    horizon: int,
    batch_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Constant mode: 每个 batch 样本随机抽一个在 [0, H] 范围内的余弦表索引。

    返回形状为 [B, H]：
    - 同一个样本内部的 H 个位置共享同一个噪声强度。
    - 不同 batch 样本可以抽到不同噪声强度。
    - 若某个样本抽到 H，则该样本整条轨迹都是纯高斯级别。
    """
    timesteps = torch.randint(0, horizon + 1, (batch_size, 1), device=device, dtype=torch.long)
    return timesteps.repeat(1, horizon)



def build_random_position_timesteps(
    horizon: int,
    batch_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Random mode: 每个 batch 样本、每个位置都独立随机抽 [0, H] 范围内的余弦表索引。

    这是最强的时域破坏对照组：相邻动作位置可能对应完全不同的噪声强度。
    """
    return torch.randint(0, horizon + 1, (batch_size, horizon), device=device, dtype=torch.long)


def build_chunkwise_position_timesteps(
    horizon: int,
    batch_size: int,
    chunk_size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Chunk-wise mode: 基于 horizon 的分块噪声级分配。

    设计目的：
    - 介于 constant 和 positionwise 之间，用来验证“位置感知粒度”的影响。
    - 同一 chunk 内共享近似的噪声层级，chunk 之间整体向远端增强。
    - 最后一个 chunk 强制为表索引 H，保证远端动作满足纯高斯边界条件。
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正数, 当前为 {chunk_size}")

    position_index = build_position_index(horizon, batch_size=batch_size, device=device)
    num_complete_chunks = max(1, math.floor(horizon / chunk_size))
    chunk_indices = torch.div(position_index, chunk_size, rounding_mode="floor")
    chunk_indices = torch.clamp(chunk_indices, max=num_complete_chunks - 1)

    upper_limit = max(1, math.floor((horizon + 1) / num_complete_chunks))
    j = torch.randint(0, upper_limit, (batch_size, 1), device=device, dtype=torch.long).repeat(1, horizon)

    timesteps = (
        torch.floor(((horizon + 1) * (chunk_indices + 1)) / num_complete_chunks) - 1 - j
    ).long()
    timesteps = torch.clamp(timesteps, 0, horizon)

    # 保持 PADP 的远端边界条件：最后一个 chunk 必须退化为纯高斯噪声。
    final_chunk_mask = chunk_indices == (num_complete_chunks - 1)
    timesteps = torch.where(final_chunk_mask, torch.full_like(timesteps, horizon), timesteps)
    return timesteps


def add_horizon_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_bar_h: torch.Tensor,
    sqrt_one_minus_alpha_bar_h: torch.Tensor,
) -> torch.Tensor:
    """
    【极速路径】直接接收 Policy 层传来的 (1, H, 1) 预对齐静态 Buffer 进行广播加噪。
    消除了旧版中在此处执行的 reshape 和 to(device) 操作。

    positionwise 模式专用：
    - Policy 侧已经把余弦表索引 1..H 预先 reshape 成 [1, H, 1]。
    - 第 h 个动作使用表索引 h+1。
    - 第 H-1 个动作使用表索引 H，因此动作系数为 0、噪声系数为 1。
    """
    return sqrt_alpha_bar_h * original_samples + sqrt_one_minus_alpha_bar_h * noise


def add_position_noise(
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    mode: str,
    chunk_size: int = 1,
    sqrt_alpha_bar_h: Optional[torch.Tensor] = None,
    sqrt_one_minus_alpha_bar_h: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    按指定模式生成噪声调度并加噪。

    输入：
    - original_samples: 真实动作 x_0，形状 [B, H, D]。
    - noise: 标准高斯 epsilon，形状 [B, H, D]。
    - sqrt_alpha_bar / sqrt_one_minus_alpha_bar: 余弦表，供 constant/random/chunkwise 查表。
    - sqrt_alpha_bar_h / sqrt_one_minus_alpha_bar_h: positionwise 的预对齐极速表。

    输出：
    - noisy trajectory，作为扩散模型训练或推理时的输入。
    """
    horizon = original_samples.shape[1]
    batch_size = original_samples.shape[0]

    if mode in ("gaussian", "pure_gaussian"):
        # 极端对照：完全丢弃动作信号，只输入纯高斯噪声。
        return noise
        
    elif mode in ("horizon", "positionwise") and sqrt_alpha_bar_h is not None:
        # 最高效路径：完全静态的位置感知 O(1) 广播
        return add_horizon_noise(original_samples, noise, sqrt_alpha_bar_h, sqrt_one_minus_alpha_bar_h)
        
    elif mode == "linear":
        # ========================================================
        # 【真正的线性加噪】：跳过 cosine 查表，直接进行数学线性插值
        # ========================================================
        target_dtype = original_samples.dtype
        device = original_samples.device
        
        # 按照你的思路：alpha_bar 严格线性地从 1 衰减到 0
        # h=0 时取 1 - 1/H (保留极少噪声)，h=H-1 时取 0.0 (纯噪声)
        alpha_bar_linear = torch.linspace(
            1.0 - 1.0 / horizon, 0.0, horizon, device=device, dtype=target_dtype
        )
        
        # 根据平方和为 1 的原则，计算对应的动作系数和噪声系数
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_linear).view(1, horizon, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_linear).view(1, horizon, 1)
        
        # 像 positionwise 一样，通过 O(1) 广播极速返回
        return sqrt_alpha_bar_t * original_samples + sqrt_one_minus_alpha_bar_t * noise

    elif mode == "constant":
        # 样本内恒定：每条轨迹共享一个随机余弦表索引。
        timesteps = build_constant_position_timesteps(
            horizon=horizon, batch_size=batch_size, device=original_samples.device
        )
    elif mode == "random":
        # 样本内逐位置随机：每个动作位置独立抽余弦表索引。
        timesteps = build_random_position_timesteps(
            horizon=horizon, batch_size=batch_size, device=original_samples.device
        )
    elif mode in ("chunkwise", "chunk_wise"):
        # 分块粒度位置感知：chunk 内共享粗粒度噪声等级，最后 chunk 强制纯高斯。
        timesteps = build_chunkwise_position_timesteps(
            horizon=horizon, batch_size=batch_size, chunk_size=chunk_size, device=original_samples.device
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # 【动态模式（查表法）的维度扩展】
    # timesteps 形状为 [B, H]，查表后扩展为 [B, H, 1]，再广播到动作维度 D。
    target_dtype = original_samples.dtype
    
    sqrt_alpha_bar_t = sqrt_alpha_bar[timesteps]
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[timesteps]
    # 快路径：查表张量通常已经和输入样本在同一设备上，
    # 因为缓冲区和时间步张量都跟随策略模块所在设备。
    # 只有数据类型真的不一致时才转换，避免无意义的类型转换算子。
    if sqrt_alpha_bar_t.dtype != target_dtype:
        sqrt_alpha_bar_t = sqrt_alpha_bar_t.to(dtype=target_dtype)
    if sqrt_one_minus_alpha_bar_t.dtype != target_dtype:
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.to(dtype=target_dtype)
    sqrt_alpha_bar_t = sqrt_alpha_bar_t.unsqueeze(-1)
    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.unsqueeze(-1)

    return sqrt_alpha_bar_t * original_samples + sqrt_one_minus_alpha_bar_t * noise
