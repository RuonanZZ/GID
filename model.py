"""Fusion denoiser architecture used by the retained GID checkpoints."""

import math

import torch
from torch import nn

from Common.models import Inertial_PoseTransformer_Upper


def logit(p: float, eps: float = 1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class RawTokenizerNxD(nn.Module):
    """Map flattened IMU features to one token per IMU."""

    def __init__(self, n_imu=4, D_per=12, init_eps=0.05):
        super().__init__()
        self.n_imu = n_imu
        self.D = D_per
        channels = n_imu * D_per
        self.proj = nn.Linear(channels, channels, bias=True)
        self.out_norm = nn.LayerNorm(channels)
        with torch.no_grad():
            weight = torch.zeros(channels, channels)
            for imu_id in range(n_imu):
                start, end = imu_id * D_per, (imu_id + 1) * D_per
                weight[start:end, start:end] = torch.eye(D_per)
            weight += init_eps * torch.randn_like(weight) / math.sqrt(D_per)
            self.proj.weight.copy_(weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x_flat):
        tokens = self.out_norm(self.proj(x_flat))
        batch_size, num_frames, channels = tokens.shape
        assert channels == self.n_imu * self.D
        return tokens.view(batch_size, num_frames, self.n_imu, self.D)


class GatedBlend(nn.Module):
    """Blend two IMU token tensors with a learnable gate."""

    def __init__(self, D_per=12, n_imu=4, gate_mode='per_imu_chan', init_ratio=0.1):
        super().__init__()
        self.mode = gate_mode
        if gate_mode == 'scalar':
            self.g = nn.Parameter(torch.tensor(logit(init_ratio)))
        elif gate_mode == 'per_imu':
            self.g = nn.Parameter(torch.full((1, 1, n_imu, 1), logit(init_ratio)))
        elif gate_mode == 'per_chan':
            self.g = nn.Parameter(torch.full((1, 1, 1, D_per), logit(init_ratio)))
        elif gate_mode == 'per_imu_chan':
            self.g = nn.Parameter(torch.full((1, 1, n_imu, D_per), logit(init_ratio)))
        elif gate_mode == 'mlp':
            hidden = max(32, 2 * D_per)
            self.mlp = nn.Sequential(
                nn.Linear(2 * D_per, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, D_per),
            )
        else:
            raise ValueError(f'Unsupported gate mode: {gate_mode}')

    def forward(self, first, second):
        if self.mode == 'mlp':
            gate = torch.sigmoid(self.mlp(torch.cat((first, second), dim=-1)))
        else:
            gate = torch.sigmoid(self.g)
        return first * (1.0 - gate) + second * gate


class TemporalDepthwiseSmooth(nn.Module):
    """Depthwise temporal low-pass filter for IMU tokens."""

    def __init__(self, D_per=12, n_imu=4, k=5):
        super().__init__()
        channels = n_imu * D_per
        self.n_imu = n_imu
        self.D_per = D_per
        self.dw = nn.Conv1d(
            channels, channels, kernel_size=k, padding=k // 2,
            groups=channels, bias=False,
        )
        with torch.no_grad():
            self.dw.weight.zero_()
            self.dw.weight[:, :, k // 2] = 1.0

    def forward(self, tokens):
        batch_size, num_frames, num_imus, channels = tokens.shape
        assert num_imus == self.n_imu and channels == self.D_per
        values = tokens.view(batch_size, num_frames, -1).permute(0, 2, 1)
        values = self.dw(values)
        return values.permute(0, 2, 1).view(batch_size, num_frames, num_imus, channels)


class TemporalHPF(nn.Module):
    """High-pass residual: input minus its depthwise low-pass version."""

    def __init__(self, D_per=12, n_imu=4, k=5):
        super().__init__()
        self.lpf = TemporalDepthwiseSmooth(D_per=D_per, n_imu=n_imu, k=k)

    def forward(self, tokens):
        return tokens - self.lpf(tokens)


class CrossAttnSpatialPerT(nn.Module):
    """Apply spatial cross-attention among IMUs for every time step."""

    def __init__(self, D_in=12, d_model=64, num_heads=8, dropout=0.0):
        super().__init__()
        self.q_proj = nn.Linear(D_in, d_model)
        self.k_proj = nn.Linear(D_in, d_model)
        self.v_proj = nn.Linear(D_in, d_model)
        self.mha = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.o_proj = nn.Linear(d_model, D_in)
        self.ln = nn.LayerNorm(D_in)

    def forward(self, denoised, raw_tokens):
        batch_size, num_frames, num_imus, channels = denoised.shape
        query = self.q_proj(denoised.reshape(batch_size * num_frames, num_imus, channels))
        key = self.k_proj(raw_tokens.reshape(batch_size * num_frames, num_imus, channels))
        value = self.v_proj(raw_tokens.reshape(batch_size * num_frames, num_imus, channels))
        attended, _ = self.mha(query, key, value)
        attended = self.o_proj(attended).reshape(
            batch_size, num_frames, num_imus, channels,
        )
        return self.ln(denoised + attended)


class FourHead_ResFusion(nn.Module):
    """Fusion denoiser for the retained 4- and 6-IMU GID checkpoints."""

    def __init__(
        self,
        num_frame=30,
        in_num_joints=4,
        in_chans=12,
        out_num_joints=4,
        out_chans=12,
        embed_dim_ratio=32,
        depth=4,
        num_heads=8,
        mlp_ratio=2.0,
        with_spatial_block=True,
        with_spatial_pos_embed=True,
        with_temporal_pos_embed=True,
        with_ssms=True,
        with_ssmt=True,
        weight_path='checkpoint/spm/ses_real.pt',
        tok_init_eps=0.05,
        use_early=True,
        use_cross=False,
        use_late=False,
        gate_mode='per_imu_chan',
        gate_init_ratio_early=0.10,
        gate_init_ratio_cross=0.10,
        gate_init_ratio_late=0.05,
        cross_d_model=64,
        cross_heads=8,
        late_mode='hpf',
        upper_res_scale=0.10,
    ):
        super().__init__()
        assert in_num_joints == out_num_joints
        assert in_chans == out_chans

        self.n_imu = out_num_joints
        self.out_chans = out_chans
        self.upper_res_scale = upper_res_scale
        self.use_early = use_early
        self.use_cross = use_cross
        self.use_late = use_late
        self.late_mode = late_mode

        transformer_kwargs = {
            'num_frame': num_frame,
            'in_num_joints': in_num_joints,
            'in_chans': in_chans,
            'out_chans': out_chans,
            'with_spatial_block': with_spatial_block,
            'with_spatial_pos_embed': with_spatial_pos_embed,
            'with_temporal_pos_embed': with_temporal_pos_embed,
            'with_ssms': with_ssms,
            'with_ssmt': with_ssmt,
            'weight_path': weight_path,
            'embed_dim_ratio': embed_dim_ratio,
            'depth': depth,
            'num_heads': num_heads,
            'mlp_ratio': mlp_ratio,
        }
        self.heads = nn.ModuleList([
            Inertial_PoseTransformer_Upper(out_num_joints=1, **transformer_kwargs)
            for _ in range(self.n_imu)
        ])

        self.raw_tok = RawTokenizerNxD(
            n_imu=self.n_imu, D_per=out_chans, init_eps=tok_init_eps,
        )
        # Retained for strict compatibility with the saved fusion checkpoints.
        self.denoised_tok = RawTokenizerNxD(
            n_imu=self.n_imu, D_per=out_chans, init_eps=tok_init_eps,
        )

        if use_early:
            self.early_gate = GatedBlend(
                D_per=out_chans, n_imu=self.n_imu, gate_mode=gate_mode,
                init_ratio=gate_init_ratio_early,
            )
        if use_cross:
            self.cross = CrossAttnSpatialPerT(
                D_in=out_chans, d_model=cross_d_model,
                num_heads=cross_heads, dropout=0.0,
            )
            self.cross_gate = GatedBlend(
                D_per=out_chans, n_imu=self.n_imu, gate_mode=gate_mode,
                init_ratio=gate_init_ratio_cross,
            )
        if use_late:
            self.late_gate = GatedBlend(
                D_per=out_chans, n_imu=self.n_imu, gate_mode=gate_mode,
                init_ratio=gate_init_ratio_late,
            )
            self.lpf = TemporalDepthwiseSmooth(D_per=out_chans, n_imu=self.n_imu, k=5)
            self.hpf = TemporalHPF(D_per=out_chans, n_imu=self.n_imu, k=5)

        self.upper = Inertial_PoseTransformer_Upper(
            out_num_joints=out_num_joints, **transformer_kwargs,
        )

    def forward(self, x_loose):
        batch_size, num_frames, channels = x_loose.shape
        assert channels == self.n_imu * self.out_chans

        denoised = torch.stack([head(x_loose) for head in self.heads], dim=2)
        raw_tokens = self.raw_tok(x_loose)

        fused = self.early_gate(denoised, raw_tokens) if self.use_early else denoised
        if self.use_cross:
            fused = self.cross_gate(fused, self.cross(fused, raw_tokens))

        fused_flat = fused.view(batch_size, num_frames, -1)
        output = fused_flat + self.upper_res_scale * self.upper(fused_flat)

        if self.use_late:
            late_tokens = raw_tokens
            if self.late_mode == 'lpf':
                late_tokens = self.lpf(late_tokens)
            elif self.late_mode == 'hpf':
                late_tokens = self.hpf(late_tokens)
            output = self.late_gate(
                output.view(batch_size, num_frames, self.n_imu, self.out_chans),
                late_tokens,
            ).view(batch_size, num_frames, -1)
        return output

    def load_heads(
        self,
        ckpt_map: dict,
        order=None,
        strict=True,
        freeze=True,
        finetune_last=False,
        finetune_patterns=('head.', 'proj_out', 'out', 'mlp_head'),
        verbose=True,
    ):
        """Load one retained per-IMU checkpoint into each denoising head."""
        if order is None:
            order = list(ckpt_map.keys())
        assert len(order) == len(self.heads)

        summary = {}
        for index, name in enumerate(order):
            source = ckpt_map[name]
            state = torch.load(source, map_location='cpu') if isinstance(source, str) else source
            if isinstance(state, dict) and isinstance(state.get('model'), dict):
                state = state['model']
            elif isinstance(state, dict) and isinstance(state.get('state_dict'), dict):
                state = state['state_dict']
            if any(isinstance(key, str) and key.startswith('module.') for key in state):
                state = {key.replace('module.', '', 1): value for key, value in state.items()}

            try:
                self.heads[index].load_state_dict(state, strict=strict)
                missing, unexpected = [], []
            except RuntimeError as error:
                incompatibility = self.heads[index].load_state_dict(state, strict=False)
                missing = incompatibility.missing_keys
                unexpected = incompatibility.unexpected_keys
                if verbose:
                    print(f'[WARN][{name}] strict loading failed: {error}')

            if freeze:
                for parameter in self.heads[index].parameters():
                    parameter.requires_grad = False
            if finetune_last:
                for parameter_name, parameter in self.heads[index].named_parameters():
                    if any(
                        parameter_name.startswith(pattern) or f'.{pattern}' in parameter_name
                        for pattern in finetune_patterns
                    ):
                        parameter.requires_grad = True

            if verbose:
                trainable = sum(parameter.requires_grad for parameter in self.heads[index].parameters())
                total = sum(1 for _ in self.heads[index].parameters())
                print(f'[OK][{name}] head[{index}] trainable {trainable}/{total}')
            summary[name] = {'missing': missing, 'unexpected': unexpected}
        return summary
