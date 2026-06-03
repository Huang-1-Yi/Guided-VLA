# GuidedVLA 新增：ControlNet 风格双分支 attention，用于 plug-and-play action head specialization。
# 论文："GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization" (RSS 2026)
"""
受 ControlNet 启发的 Pi0 模型 Attention 模块。

架构：
- Origin branch：预训练 PaliGemma/Gemma attention（可选冻结）
- Control branch：带 num_control_heads 个 head 的轻量分支，以及可选 headwise gate
- Fusion：zero_conv（ControlNet 设计），即 origin + zero_conv(branch_output)

headwise gate 由 control branch 的 Q projection 预测，并在 o_proj 之前应用到
control branch 的 SDPA 输出。
"""

from collections.abc import Sequence
import copy
import logging

import torch
from torch import nn


class ControlAwareAttention(nn.Module):
    """
    带可训练 origin/control 分支的 ControlNet 风格 attention wrapper。

    融合模式：zero_conv，即 output = origin_output + zero_conv(branch_output)。
    zero_conv 采用零初始化，因此该分支初始贡献为零，与标准 ControlNet 初始化策略一致。
    """

    @staticmethod
    def _init_linear(linear: nn.Linear, *, std: float):
        nn.init.normal_(linear.weight, mean=0.0, std=std)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    @classmethod
    def _replace_q_proj_with_headwise_gate(
        cls,
        object_branch,
        *,
        hidden_size: int,
        query_size: int,
        gate_num_heads: int,
        device,
        dtype,
        initializer_std: float,
        copy_query_from: nn.Linear | None = None,
    ):
        old_q_proj = object_branch.q_proj
        gated_q_proj = nn.Linear(
            hidden_size,
            query_size + gate_num_heads,
            bias=old_q_proj.bias is not None,
            device=device,
            dtype=dtype,
        )

        if copy_query_from is None:
            cls._init_linear(gated_q_proj, std=initializer_std)
        else:
            with torch.no_grad():
                gated_q_proj.weight[:query_size].copy_(copy_query_from.weight[:query_size])
                nn.init.normal_(gated_q_proj.weight[query_size:], mean=0.0, std=initializer_std)
                if gated_q_proj.bias is not None:
                    gated_q_proj.bias[:query_size].copy_(copy_query_from.bias[:query_size])
                    nn.init.zeros_(gated_q_proj.bias[query_size:])

        object_branch.q_proj = gated_q_proj

    def __init__(
        self,
        original_attn,
        hidden_size: int,
        *,
        num_control_heads: int = 2,
        copy_weights: bool = False,
        freeze_origin: bool = False,
        use_headwise_gate: bool | None = None,
    ):
        super().__init__()

        # 从原始 attention layer 获取 device/dtype。
        device = next(original_attn.parameters()).device
        dtype = next(original_attn.parameters()).dtype

        # Origin branch：参与训练并接收 main_loss 梯度。
        self.origin = original_attn

        # 保存配置。
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.num_control_heads = num_control_heads
        self.copy_weights = copy_weights
        self.freeze_origin = freeze_origin

        # 获取原始 attention 维度。
        num_heads = self.config.num_attention_heads
        head_dim = original_attn.head_dim
        initializer_std = float(getattr(self.config, "initializer_range", 0.02))
        if use_headwise_gate is None:
            use_headwise_gate = True
        self.use_headwise_gate = bool(use_headwise_gate)
        self.gate_num_heads = num_heads

        # 确定 control branch 维度。
        if num_control_heads is None or copy_weights:
            # full copy 模式：从 original 复制所有权重。
            self.object_branch = copy.deepcopy(original_attn).to(device=device, dtype=dtype)
            self.num_control_heads = num_heads
            control_hidden_size = self.num_control_heads * head_dim
            if self.use_headwise_gate:
                self._replace_q_proj_with_headwise_gate(
                    self.object_branch,
                    hidden_size=hidden_size,
                    query_size=control_hidden_size,
                    gate_num_heads=self.gate_num_heads,
                    device=device,
                    dtype=dtype,
                    initializer_std=initializer_std,
                    copy_query_from=self.object_branch.q_proj,
                )
        else:
            # 克隆 config，并减少 head 数量。
            control_config = copy.deepcopy(self.config)
            control_config.num_attention_heads = num_control_heads

            # 使用与 original 相同的 attention class。
            attention_class = type(original_attn)
            self.object_branch = attention_class(control_config, layer_idx=self.layer_idx).to(
                device=device, dtype=dtype
            )

            control_hidden_size = num_control_heads * head_dim  # e.g., 2 * 256 = 512

            # control branch Q projection 上的可选 headwise gate。
            # gate 维度等于 origin num_heads（不是 control heads），因此每个 origin head
            # 都有自己的 gate scalar。
            if self.use_headwise_gate:
                self._replace_q_proj_with_headwise_gate(
                    self.object_branch,
                    hidden_size=hidden_size,
                    query_size=control_hidden_size,
                    gate_num_heads=self.gate_num_heads,
                    device=device,
                    dtype=dtype,
                    initializer_std=initializer_std,
                )

            # 使用较小随机值重新初始化 control branch。
            # projection biases 为零，因此 gate logits 以 0 附近为中心。
            for name, param in self.object_branch.named_parameters():
                if "proj" in name and param.ndim == 2:
                    nn.init.normal_(param, mean=0.0, std=initializer_std)
                elif "proj" in name and param.ndim == 1:
                    nn.init.zeros_(param)

        # 确保 control branch 参数可训练。
        for param in self.object_branch.parameters():
            param.requires_grad = True
        if self.freeze_origin:
            for param in self.origin.parameters():
                param.requires_grad = False

        # 用于 ControlNet 风格融合的零初始化 linear projection。
        self.zero_conv = nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

        # Q-head expansion layer：扩展 control branch 的 Q heads，使其匹配 origin heads。
        # 仅当 control heads 少于 origin heads 时需要（例如 2 heads -> 8 heads）。
        origin_num_heads = num_heads
        control_q_heads = self.num_control_heads if self.num_control_heads is not None else num_heads

        if control_q_heads != origin_num_heads:
            control_hidden_dim = control_q_heads * self.head_dim
            origin_hidden_dim = origin_num_heads * self.head_dim
            self.q_expand_linear = nn.Linear(
                control_hidden_dim, origin_hidden_dim, bias=True, device=device, dtype=dtype
            )
            nn.init.normal_(self.q_expand_linear.weight, mean=0.0, std=initializer_std)
            nn.init.zeros_(self.q_expand_linear.bias)

            # 重新初始化 control branch 的 o_proj，使其匹配扩展后的 head 数量。
            self.object_branch.o_proj = nn.Linear(
                origin_hidden_dim,
                self.config.hidden_size,
                bias=self.object_branch.o_proj.bias is not None,
                device=device,
                dtype=dtype,
            )
            nn.init.normal_(self.object_branch.o_proj.weight, mean=0.0, std=initializer_std)
            if self.object_branch.o_proj.bias is not None:
                nn.init.zeros_(self.object_branch.o_proj.bias)

            self.has_q_expansion = True
        else:
            self.q_expand_linear = None
            self.has_q_expansion = False

        mode_str = "copy" if copy_weights or num_control_heads is None else f"{num_control_heads}heads"
        freeze_str = ", frozen_origin" if self.freeze_origin else ""
        logging.info(f"Created ControlAwareAttention for layer {self.layer_idx} [{mode_str}, zero_conv{freeze_str}]")

    # transformers library 兼容属性（委托给 control branch）。
    @property
    def q_proj(self):
        return self.object_branch.q_proj

    @property
    def k_proj(self):
        return self.object_branch.k_proj

    @property
    def v_proj(self):
        return self.object_branch.v_proj

    @property
    def o_proj(self):
        return self.object_branch.o_proj

    @property
    def head_dim(self):
        return self.origin.head_dim

    @property
    def scaling(self):
        return self.object_branch.scaling

    def get_q_expand_linear(self):
        """获取可用的 Q expansion linear layer（用于 2->8 heads 扩展）。"""
        return self.q_expand_linear if self.has_q_expansion else None

    # 注意：dual-path ControlNet 模式不会调用 forward()。
    def forward(self, *args, **kwargs):
        """为了兼容性，回退到 origin attention。"""
        return self.origin(*args, **kwargs)

    def compute_dual_path_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor | None,  # q_branch_gate
    ]:
        """
        为 Origin 和 Branch 两条路径计算 Q/K/V（真正的 ControlNet 架构）。

        两条路径都可以独立关注共享上下文（例如 PaliGemma KV），随后融合输出：
        Final = origin_out + zero_conv(branch_out)

        Args:
            hidden_states: 输入 hidden states [batch, seq, hidden_dim]

        Returns:
            tuple: (
                (Q_origin, K_origin, V_origin),
                (Q_branch, K_branch, V_branch),
                q_branch_gate,  # [batch, gated_heads, seq, 1] or None
            )
            所有 Q/K/V tensor 的形状为 [batch, heads, seq, head_dim]
        """
        input_shape = hidden_states.shape[:-1]

        # === Origin Path（预训练权重，可选冻结）===
        q_origin_proj = self.origin.q_proj(hidden_states)
        k_origin_proj = self.origin.k_proj(hidden_states)
        v_origin_proj = self.origin.v_proj(hidden_states)

        q_origin = q_origin_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)
        k_origin = k_origin_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)
        v_origin = v_origin_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)

        # === Control Branch Path（可训练、轻量）===
        q_branch_proj = self.object_branch.q_proj(hidden_states)
        k_branch_proj = self.object_branch.k_proj(hidden_states)
        v_branch_proj = self.object_branch.v_proj(hidden_states)

        # 如果存在 headwise gates，则拆分出来。
        has_headwise_gate = hasattr(self, "use_headwise_gate") and self.use_headwise_gate
        expected_q_size = self.num_control_heads * self.head_dim

        q_branch_gate = None
        if has_headwise_gate and q_branch_proj.shape[-1] > expected_q_size:
            # q_branch_proj: [batch, seq, control_heads*head_dim + gate_heads]
            q_branch_proj_query = q_branch_proj[..., :expected_q_size]
            q_branch_proj_gate = q_branch_proj[..., expected_q_size:]

            q_branch = q_branch_proj_query.view(*input_shape, self.num_control_heads, self.head_dim).transpose(1, 2)
            # Gate: [batch, seq, gate_heads] -> [batch, gate_heads, seq, 1]
            q_branch_gate = q_branch_proj_gate.view(*input_shape, self.gate_num_heads, 1).transpose(1, 2)
        else:
            q_branch = q_branch_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)

        k_branch = k_branch_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)
        v_branch = v_branch_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)

        return (q_origin, k_origin, v_origin), (q_branch, k_branch, v_branch), q_branch_gate

    def fuse_outputs(
        self,
        origin_output: torch.Tensor,
        branch_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用 zero_conv fusion 融合 Origin 和 Branch 路径输出。

        y = origin_output + zero_conv(branch_output)

        zero_conv 采用零初始化，因此该分支初始贡献为零，并会随训练推进逐渐学习贡献。

        Args:
            origin_output: origin attention path 的输出 [batch, seq, hidden_dim]
            branch_output: branch attention path 的输出 [batch, seq, hidden_dim]

        Returns:
            融合后的输出 tensor [batch, seq, hidden_dim]
        """
        return origin_output + self.zero_conv(branch_output)


def inject_control_attention(
    model,
    *,
    num_control_heads: int = 2,
    copy_weights: bool = False,
    freeze_origin: bool = False,
    layer_indices: Sequence[int] | None = None,
    use_headwise_gate: bool | None = None,
):
    """
    将 action expert attention layers 替换为 ControlAwareAttention。

    应在加载预训练 checkpoint 之后调用，这样 origin branch 可以保留原始预训练权重不变
    （load-then-inject 模式）。

    Args:
        model: PI0Pytorch 模型实例
        num_control_heads: control branch 的 attention head 数量（默认 2）。
                          使用 None 表示 full copy 模式（head 数与 original 相同）
        copy_weights: 若为 True，则从 original 复制权重；若为 False，则随机初始化
        freeze_origin: 若为 True，则冻结原始 action-expert attention branch
        layer_indices: 可选的待替换层索引子集（None 表示所有层）
        use_headwise_gate: 是否向 control branch Q proj 添加逐 head sigmoid gate

    Returns:
        int: 被替换的层数
    """
    mode_str = "copy" if copy_weights or num_control_heads is None else f"{num_control_heads}heads"
    freeze_str = ", frozen_origin" if freeze_origin else ""
    logging.info(f"Injecting ControlAwareAttention into action expert [{mode_str}, zero_conv{freeze_str}]")

    replaced_count = 0
    layer_indices_set = set(layer_indices) if layer_indices is not None else None

    expert_model = model.paligemma_with_expert.gemma_expert.model
    hidden_size = expert_model.config.hidden_size

    for _layer_idx, layer in enumerate(expert_model.layers):
        if layer_indices_set is not None and _layer_idx not in layer_indices_set:
            continue
        layer.self_attn = ControlAwareAttention(
            original_attn=layer.self_attn,
            hidden_size=hidden_size,
            num_control_heads=num_control_heads,
            copy_weights=copy_weights,
            freeze_origin=freeze_origin,
            use_headwise_gate=use_headwise_gate,
        )
        replaced_count += 1

    logging.info(f"Total {replaced_count} expert layers replaced [{mode_str}, zero_conv{freeze_str}]")
    return replaced_count


def get_trainable_control_params(model):
    """
    获取 ControlAwareAttention 模块中的可训练参数。

    Returns:
        list: control branch 参数 + zero_conv fusion 参数
    """
    control_params = []

    for _name, module in model.named_modules():
        if isinstance(module, ControlAwareAttention):
            control_params.extend(module.object_branch.parameters())
            if module.q_expand_linear is not None:
                control_params.extend(module.q_expand_linear.parameters())
            control_params.append(module.zero_conv.weight)
            control_params.append(module.zero_conv.bias)

    return control_params
