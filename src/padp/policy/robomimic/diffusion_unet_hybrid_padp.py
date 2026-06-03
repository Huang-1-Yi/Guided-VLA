from typing import Dict, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from padp.model.common.normalizer import LinearNormalizer
from padp.policy.base_image_policy import BaseImagePolicy
# from padp.model.diffusion.conditional_unet1d import ConditionalUnet1D
from padp.model.diffusion.conditional_unet1d_padp import ConditionalUnet1D

from padp.model.diffusion.mask_generator import LowdimMaskGenerator
from padp.common.pytorch_util import dict_apply
from padp.model.vision.robomimic_obs_encoder import RobomimicObsEncoder
from padp.policy.schedulers_padp import add_position_noise

class SlidingWindowDiffusionPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_encoder: Optional[nn.Module] = None,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            window_min_weight: float = 0.02,
            # 滑动窗口相关参数
            window_loss_weights="exponential",  # 窗口内不同位置的损失权重
            window_exp_gamma: float = 0.2,       # 指数权重衰减系数
            noise_schedule_mode: str = "positionwise",
            noise_chunk_size: int = 1,
            pred_type = 'sample',      # self.kwargs.get('pred_type', 'sample')  # 'epsilon' or 'sample'
            **kwargs):
        """
        pred_type: 预测类型，'epsilon'表示预测噪声，'sample'表示预测去噪后的动作
        """
        super().__init__()

        self.pred_type = pred_type
        self.noise_schedule_mode = noise_schedule_mode
        self.noise_chunk_size = noise_chunk_size

        # 解析shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        if obs_encoder is None:
            # 自动从policy参数创建编码器，用户只需配置policy的crop_shape等
            obs_encoder = RobomimicObsEncoder(
                shape_meta=shape_meta,
                crop_shape=crop_shape,
                obs_encoder_group_norm=obs_encoder_group_norm,
                eval_fixed_crop=eval_fixed_crop,
            )
        # Create diffusion model.
        obs_feature_dim = obs_encoder.output_shape()[0]
        print(f"Obs encoder output shape: {obs_encoder.output_shape()}")
        print(f"Obs feature dim: {obs_feature_dim}")

        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps
        
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        
        # 初始化noise_scheduler
        self.noise_scheduler = noise_scheduler

        # ========================================================
        # 核心优化 1：预计算并注册预对齐的 Buffer (1, H, 1)
        # ========================================================
        max_horizon = horizon  # 使用horizon作为最大时间步
        beta_schedule = self.noise_scheduler.config.beta_schedule
        if beta_schedule == "squaredcos_cap_v2":
            s = 0.008
            t = torch.arange(0, max_horizon + 1, dtype=torch.float32)   # (0, max_horizon + 1)
            alpha_bar = torch.cos((t / max_horizon + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            # 显式确保端点绝对精确
            alpha_bar[0] = 1.0
            alpha_bar[-1] = 0.0
        else:
            raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")
        
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        print("PADP cosine noise table before register_buffer:")
        print("idx | alpha_bar | sqrt_alpha_bar | sqrt_one_minus_alpha_bar")
        for idx in range(max_horizon + 1):
            print(
                f"{idx:03d} | "
                f"{alpha_bar[idx].item():.8f} | "
                f"{sqrt_alpha_bar[idx].item():.8f} | "
                f"{sqrt_one_minus_alpha_bar[idx].item():.8f}"
            )
        
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('sqrt_alpha_bar', sqrt_alpha_bar)
        self.register_buffer('sqrt_one_minus_alpha_bar', sqrt_one_minus_alpha_bar)
        
        # 【新增优化】预对齐形状为 [1, H, 1] 供零开销广播使用
        self.register_buffer('sqrt_alpha_bar_h', sqrt_alpha_bar[1:max_horizon+1].view(1, max_horizon, 1))
        self.register_buffer('sqrt_one_minus_alpha_bar_h', sqrt_one_minus_alpha_bar[1:max_horizon+1].view(1, max_horizon, 1))
        
        # 【新增优化】预注册末位噪声系数（严格为 1.0 的高斯噪声）
        self.register_buffer('sqrt_one_minus_alpha_bar_tail', sqrt_one_minus_alpha_bar[max_horizon:max_horizon+1].view(1, 1, 1))

        self.normalizer = LinearNormalizer()

        self.horizon = horizon                  # 滑动窗口参数
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        # ========================================================
        # 核心优化 2：损失权重作为 Buffer 预注册
        # ========================================================
        if window_loss_weights == "linear":
            window_weights = 1.0 - torch.arange(horizon, dtype=torch.float32) / horizon
        elif window_loss_weights == "exponential":
            window_weights = torch.exp(-torch.arange(horizon, dtype=torch.float32) * float(window_exp_gamma))
        elif window_loss_weights == "constant":
            window_weights = torch.ones(horizon)
        else:
            raise ValueError(f"Unknown window loss weights: {window_loss_weights}")
        
        if window_min_weight > 0.0:
            window_weights = torch.clamp(window_weights, min=float(window_min_weight))
            
        # 注册并预对齐为 [1, H, 1]
        self.register_buffer('window_weights_h', window_weights.view(1, horizon, 1))
        
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # 推理状态
        self._inference_buffer = None
        self._inference_t_buffer = None
        self._inference_global_cond = None

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        print(f"PADP noise schedule mode: {self.noise_schedule_mode}, chunk size: {self.noise_chunk_size}")

    def _apply_position_noise(self, original_samples: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return add_position_noise(
            original_samples=original_samples,
            noise=noise,
            sqrt_alpha_bar=self.sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar=self.sqrt_one_minus_alpha_bar,
            mode=self.noise_schedule_mode,
            chunk_size=self.noise_chunk_size,
            sqrt_alpha_bar_h=self.sqrt_alpha_bar_h,
            sqrt_one_minus_alpha_bar_h=self.sqrt_one_minus_alpha_bar_h,
        )
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. 检查 batch 必备键
        assert 'obs' in batch
        assert 'action' in batch
        pred_type = self.pred_type

        # 2. 归一化 obs / action，并设置轨迹为归一化后的动作
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])  # [B,H,D]
        trajectory = nactions

        # 3. 获取形状并检查窗口长度
        B, H, D = nactions.shape

        # 4. 观测编码作为全局条件编码（只取前 n_obs_steps 帧）
        local_cond = None
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        global_cond = nobs_features.reshape(B, -1)

        # 5. 按命令行指定的 noise_schedule_mode 加噪，用于真实消融实验。
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)
        noisy_trajectory = self._apply_position_noise(trajectory, noise)  # [B,H,D]

        # 6. 前向预测
        pred_actions = self.model(noisy_trajectory, local_cond=local_cond, global_cond=global_cond)  # [B,H,D]

        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = nactions
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        # 7. MSE（不归约）,目标使用原始轨迹（nactions）
        loss_mse = F.mse_loss(pred_actions, target, reduction='none')  # [B,H,D]或 [B,P,H,D]

        # 8. 按时间加权并归约：使用注册好的 window_weights_h Buffer
        loss_weighted = (loss_mse * self.window_weights_h).sum(dim=1)  # [B,D]

        # 9. 按特征归约
        loss_b = loss_weighted.mean(dim=-1)      # [B]
        return loss_b

    def _initialize_inference_buffer(self, obs_dict: Dict[str, torch.Tensor]):
        """
        初始化推理缓冲区。
        """
        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        dtype = self.dtype
        H = self.horizon
        D = self.action_dim
        pred_type = self.pred_type

        # 生成基础高斯噪声
        base_noise = torch.randn(B, H, D, device=device, dtype=dtype)

        # 以零动作为 x0 初始化，使用同一套加噪逻辑生成初始缓冲区。
        initialized_buffer = self._apply_position_noise(torch.zeros_like(base_noise), base_noise)
        self._inference_buffer = initialized_buffer

        # 编码当前观测作为全局条件
        nobs = self.normalizer.normalize(obs_dict)
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        self._inference_global_cond = nobs_features.reshape(B, -1)
        
        try:
            num_warm = int(self.horizon + self.n_obs_steps - 1)
        except Exception:
            num_warm = H + max(0, getattr(self, 'n_obs_steps', 0) - 1)

        if num_warm > 0:
            print(f"使用预测类型: {pred_type}, 预热 {num_warm} 步, 执行{self.n_action_steps}步")
            for _ in range(num_warm):
                x0 = self._inference_buffer
                noise = torch.randn_like(x0)
                noisy_trajectory = self._apply_position_noise(x0, noise)

                global_cond = self._inference_global_cond
                model_output = self.model(noisy_trajectory, local_cond=None, global_cond=global_cond)

                if pred_type == 'epsilon':
                    x0_pred = noisy_trajectory - model_output   # x0 = xt - *ε
                    pred_actions = x0_pred
                elif pred_type == 'sample':
                    pred_actions = model_output
                else:
                    raise ValueError(f"Unsupported prediction type {pred_type}")

                # 左移
                self._inference_buffer[:, :H-1, :] = pred_actions[:, 1:, :]

                # 注入新的最大噪声级别的噪声：直接使用 self.sqrt_one_minus_alpha_bar_tail
                new_noise_mean = torch.randn(B, 1, D, device=device, dtype=self._inference_buffer.dtype)
                self._inference_buffer[:, H-1:, :] = self.sqrt_one_minus_alpha_bar_tail * new_noise_mean
                

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        推理与训练对齐。
        """
        B = next(iter(obs_dict.values())).shape[0]
        device = self.device
        pred_type = self.pred_type

        # 初始化或更新推理缓冲区与全局条件
        if self._inference_buffer is None or self._inference_buffer.shape[0] != B:
            self._initialize_inference_buffer(obs_dict)
        else:
            nobs = self.normalizer.normalize(obs_dict)
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            self._inference_global_cond = nobs_features.reshape(B, -1)

        H = self.horizon
        D = self.action_dim

        # 构造 noisy_trajectory（以缓冲区作为 x0 估计）
        x0 = self._inference_buffer  # [B,H,D]
        noise = torch.randn_like(x0)

        noisy_trajectory = self._apply_position_noise(x0, noise)

        global_cond = self._inference_global_cond
        model_output = self.model(noisy_trajectory, local_cond=None, global_cond=global_cond)  # [B,H,D]
        
        if pred_type == 'epsilon':
            pred_actions = noisy_trajectory - model_output
        elif pred_type == 'sample':
            pred_actions = model_output
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 取第一步动作并更新缓冲区（左移）
        action_to_execute = pred_actions[:, 0:self.n_action_steps, :]  # [B,n_action_steps,D]
        self._inference_buffer[:, :H-1, :] = pred_actions[:, 1:, :]
        
        # 为下一次推理准备新的噪声输入到最后一位：直接使用预对齐 Buffer
        new_noise_mean = torch.randn(B, 1, D, device=device, dtype=self.dtype)
        self._inference_buffer[:, H-1:, :] = self.sqrt_one_minus_alpha_bar_tail * new_noise_mean
        
        # 反归一化输出
        action_out = self.normalizer['action'].unnormalize(action_to_execute)
        full_pred_out = self.normalizer['action'].unnormalize(pred_actions)

        return {
            'action': action_out.reshape(B, self.n_action_steps, self.action_dim),
            'action_pred': full_pred_out
        }

    def reset(self):
        """重置推理状态"""
        self._inference_buffer = None
        self._inference_global_cond = None
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
