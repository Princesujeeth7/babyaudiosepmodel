import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

from .configuration_bs_roformer import BSRoformerConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim,
        theta=10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class BSRoformerMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        out_size: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.out_size = out_size if out_size is not None else hidden_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.out_size, bias=bias)
        self.act_fn = ACT2FN["gelu"]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class BSRoformerAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_dropout: float,
        head_dim: int,
    ):
        super().__init__()
        self.is_causal = False

        self.head_dim = head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=True)

    def forward(
        self,
        hidden_states,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask=None,
    ):
        input_shape = hidden_states.size()[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)  # b, n, d -> b, n, -1, d'

        # proj
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # positional embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # mask
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]

        if key_states.shape[1] != query_states.shape[1]:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
        )
        attn_weights = None

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class BSRoformerLayer(nn.Module):
    def __init__(self, config: BSRoformerConfig):
        super().__init__()
        self.self_attn = BSRoformerAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.attention_dropout,
            config.head_dim,
        )
        self.mlp = BSRoformerMLP(
            config.hidden_size,
            config.intermediate_size,
        )

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class BSRoformerAxialTransformer(nn.Module):
    def __init__(
        self,
        config: BSRoformerConfig,
        transformer_depth: int,
        is_time_transformer: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList([BSRoformerLayer(config) for _ in range(transformer_depth)])
        self.is_time_transformer = is_time_transformer

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
    ):
        if self.is_time_transformer:
            hidden_states = rearrange(hidden_states, 'b t f d -> b f t d')

        # merge batch
        b, seq_len_1, seq_len_2, d = hidden_states.shape
        hidden_states = rearrange(hidden_states, 'b n m d -> (b n) m d')

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                attention_mask,
            )

        # unpack batch
        hidden_states = rearrange(hidden_states, '(b n) m d -> b n m d', b=b)

        if self.is_time_transformer:
            hidden_states = rearrange(hidden_states, 'b f t d -> b t f d')

        return hidden_states


class BandSplit(nn.Module):
    def __init__(self, config: BSRoformerConfig):
        super().__init__()
        self.dim_inputs = tuple(2 * f * config.num_input_channels for f in config.freqs_per_bands)
        self.to_features = nn.ModuleList(
            [
                nn.Sequential(nn.RMSNorm(dim_in, eps=config.rms_norm_eps), nn.Linear(dim_in, config.band_proj_size))
                for dim_in in self.dim_inputs
            ]
        )

    def forward(self, x):
        x_split = x.split(self.dim_inputs, dim=-1)
        outs = [to_feature(split_input) for split_input, to_feature in zip(x_split, self.to_features)]
        return torch.stack(outs, dim=-2)


class MaskEstimator(nn.Module):
    def __init__(self, config: BSRoformerConfig):
        super().__init__()

        dim_inputs = tuple(f * config.num_input_channels * 2 for f in config.freqs_per_bands_out)
        self.to_freq_mlps = nn.ModuleList([nn.Linear(config.band_proj_size, dim) for dim in dim_inputs])
        self.to_gate_mlps = nn.ModuleList([nn.Linear(config.band_proj_size, dim // 2) for dim in dim_inputs])

    def forward(self, x):
        """

        Args:
            x: (batch, time, bands, band_proj_size)

        Returns:
            (batch, time, freq * channel * 2)
        """
        x_unbind = x.unbind(dim=-2)
        outs = []
        for band_features, freq_mlp, gate_mlp in zip(x_unbind, self.to_freq_mlps, self.to_gate_mlps):
            mask = freq_mlp(band_features)
            gate = gate_mlp(band_features)
            gate = gate.repeat_interleave(2, dim=-1)
            outs.append(mask * torch.sigmoid(gate))
        return torch.cat(outs, dim=-1)


class BSRoformerPreTrainedModel(PreTrainedModel):
    config_class = BSRoformerConfig
    base_model_prefix = "freq_domain_model"
    _no_split_modules = ["BSRoformerLayer"]
    _tied_weights_keys = []
    all_tied_weights_keys = {}


class BSRoformerModel(BSRoformerPreTrainedModel):
    """BS-RoFormer 模型的核心，在频域上对音频进行建模。"""

    def __init__(self, config: BSRoformerConfig):
        super().__init__(config)
        self.config = config

        # 主要模块
        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_base)
        self.band_split = BandSplit(config)
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    BSRoformerAxialTransformer(config, config.time_transformer_depth, is_time_transformer=True),
                    BSRoformerAxialTransformer(config, config.freq_transformer_depth, is_time_transformer=False),
                ]
            )
            for _ in range(config.num_hidden_layers)
        )
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mask_estimators = nn.ModuleList([MaskEstimator(config) for _ in range(config.num_stems)])

        # 时域压缩
        self.time_conv_length = config.time_conv_length
        if self.time_conv_length is not None:
            self.time_conv = nn.Sequential(
                nn.RMSNorm(config.band_proj_size * self.time_conv_length, eps=config.rms_norm_eps),
                BSRoformerMLP(
                    hidden_size=config.band_proj_size * self.time_conv_length,
                    intermediate_size=config.hidden_size * self.time_conv_length,
                    out_size=config.hidden_size,
                    bias=True,
                ),
            )
            self.time_deconv = nn.Sequential(
                BSRoformerMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.hidden_size * self.time_conv_length,
                    out_size=config.band_proj_size * self.time_conv_length,
                    bias=True,
                ),
                nn.RMSNorm(config.band_proj_size * self.time_conv_length, eps=config.rms_norm_eps),
            )

        # 寄存器
        rn = config.register_token_num
        self.register_tokens = nn.Parameter(torch.normal(0, 0.02, size=(rn, rn, config.hidden_size)))

        self.post_init()

    def forward(
        self,
        x,
        position_ids=None,
    ):
        """
        Args:
            x: (batch, time, freq_bins)，其中 freq_bins = freq * channel * 2
            position_ids: (batch, time)
        Returns:
            mask: (batch, num_stems, time, freq_bins)
        """
        origin_dtype = x.dtype
        target_dtype = next(self.parameters()).dtype
        x = x.to(dtype=target_dtype)
        t_origin = x.shape[1]

        # 1. band split
        if self.time_conv_length is not None:
            pad_t = (self.time_conv_length - (t_origin % self.time_conv_length)) % self.time_conv_length
            if pad_t > 0:
                x = F.pad(x, (0, 0, 0, pad_t), value=0.0)
        hidden_states = self.band_split(x)
        if self.time_conv_length is not None:
            hidden_states = rearrange(hidden_states, "b (t t_c) n d -> b t n (d t_c)", t_c=self.time_conv_length)
            hidden_states = self.time_conv(hidden_states)
        b, t, n, h = hidden_states.shape  # [batch, t, n, hidden_size]

        # 2. RoPE
        if position_ids is None:
            position_ids = torch.arange(t, device=hidden_states.device).unsqueeze(0)
        pos_embeds = self.rotary_emb(hidden_states, position_ids)
        pos_embeds_for_freq = self.rotary_emb(
            hidden_states,
            torch.arange(n, device=hidden_states.device).unsqueeze(0),
        )

        # 3. add register tokens
        rn = self.config.register_token_num
        hidden_states = F.pad(hidden_states, (0, 0, 0, rn, 0, rn))
        hidden_states[:, t:, n:, :] = self.register_tokens

        def pad_rope(cos, sin):
            cos_padded = F.pad(cos, (0, 0, 0, rn), value=1.0)
            sin_padded = F.pad(sin, (0, 0, 0, rn), value=0.0)
            return cos_padded, sin_padded

        pos_embeds = pad_rope(*pos_embeds)
        pos_embeds_for_freq = pad_rope(*pos_embeds_for_freq)

        # 4. axial transformer layers
        for time_transformer, freq_transformer in self.layers:
            hidden_states = time_transformer(
                hidden_states,
                position_embeddings=pos_embeds,
                attention_mask=None,
            )
            hidden_states = freq_transformer(
                hidden_states,
                position_embeddings=pos_embeds_for_freq,
                attention_mask=None,
            )

        # 5. remove register tokens, and final norm
        hidden_states = hidden_states[:, :t, :n, :]
        hidden_states = self.final_norm(hidden_states)

        # 6. mask estimation
        if self.time_conv_length is not None:
            hidden_states = self.time_deconv(hidden_states)
            hidden_states = rearrange(hidden_states, "b t n (d t_c) -> b (t t_c) n d", t_c=self.time_conv_length)
        hidden_states = hidden_states[:, :t_origin, :, :]
        mask = torch.stack([fn(hidden_states) for fn in self.mask_estimators], dim=1)

        return mask.to(dtype=origin_dtype)


class BSRoformerForMaskedEstimation(BSRoformerPreTrainedModel):
    """包含 STFT/iSTFT 和频域处理的完整模型。"""

    def __init__(
        self,
        config: BSRoformerConfig,
    ):
        super().__init__(config)
        self.freq_domain_model = BSRoformerModel(config)
        self.config = config

        self.register_buffer("stft_window", torch.hann_window(config.stft_n_fft), persistent=False)
        self.register_buffer("stft_out_window", torch.hann_window(config.stft_n_fft_out), persistent=False)

        self.stft_kwargs = dict(
            n_fft=config.stft_n_fft,
            hop_length=config.stft_hop_length,
            win_length=config.stft_n_fft,
            normalized=False,
        )
        self.stft_out_kwargs = dict(
            n_fft=config.stft_n_fft_out,
            hop_length=config.stft_hop_length,
            win_length=config.stft_n_fft_out,
            normalized=False,
        )

        freqs = config.stft_n_fft // 2 + 1
        assert sum(config.freqs_per_bands) == freqs, f"Sum of freqs_per_bands must be {freqs}"
        self.wave_channels = config.num_input_channels

    def forward(
        self,
        raw_audio: torch.Tensor,
        target: torch.Tensor | None = None,
    ):
        """
        Args:
            raw_audio (`torch.Tensor` of shape `(batch, channels, time)`):
                The raw audio waveform. `time` must be `config.wave_chunk_size`.
            target (`torch.Tensor`, *optional*, shape `(batch, num_stems, channels, time)`):
                The target audio waveform for loss calculation.

        Returns:
            torch.Tensor (`torch.Tensor` of shape `(batch, num_stems, channels, time)`):
            The reconstructed audio waveform.
        """

        device = raw_audio.device
        dtype = raw_audio.dtype
        b, c, t = raw_audio.shape  # batch, channel, time

        # 1. STFT: Convert audio to spectrogram
        with torch.autocast(device_type=device.type, enabled=False):
            raw_audio = raw_audio.to(dtype=torch.float32)

            raw_audio_packed = rearrange(raw_audio, "b c t -> (b c) t")
            stft_repr = torch.stft(
                raw_audio_packed,
                **self.stft_kwargs,
                window=self.stft_window,
                return_complex=True,
            )
            stft_repr = torch.view_as_real(stft_repr)
            stft_repr = rearrange(stft_repr, "(b c) f t T -> b c f t T", c=c)
            # Merge frequency, channel, and complex dimensions
            stft_repr_merged = rearrange(stft_repr, "b c f t T -> b t (f c T)")

            stft_repr_merged = stft_repr_merged.to(dtype=dtype)

        # 2. Mask Estimation
        mask = self.freq_domain_model(stft_repr_merged)
        mask = rearrange(mask, "b n t (f c T) -> b n c f t T", T=2, c=c)
        mask = mask.to(dtype=torch.float32)

        # 3. Mask Application
        with torch.autocast(device_type=device.type, enabled=False):
            stft_repr = torch.stft(
                raw_audio_packed,
                **self.stft_out_kwargs,
                window=self.stft_out_window,
                return_complex=True,
            )
            stft_repr = torch.view_as_real(stft_repr)
            stft_repr_expanded = rearrange(stft_repr, "(b c) f t T -> b 1 c f t T", c=c)
            stft_repr_complex = torch.view_as_complex(stft_repr_expanded)
            mask_complex = torch.view_as_complex(mask)
            masked_stft = stft_repr_complex * mask_complex

            # 4. iSTFT: Convert masked spectrogram back to audio
            masked_stft = rearrange(masked_stft, "b n c f t -> (b n c) f t")
            recon_audio = torch.istft(
                masked_stft,
                **self.stft_out_kwargs,
                window=self.stft_out_window,
                return_complex=False,
                length=raw_audio.shape[-1],
            )
            recon_audio = rearrange(recon_audio, "(b n c) t -> b n c t", c=self.wave_channels, n=self.config.num_stems)

        if target is None:  # return recon_audio
            return recon_audio

        # 5. Loss Calculation
        target = target[..., : recon_audio.shape[-1]]
        loss = F.l1_loss(recon_audio, target)
        return loss

    @torch.inference_mode()
    def separate(
        self,
        mixed_wave,
        chunk_size=None,
        overlap_size=None,
        batch_size=1,
        gap_size=44100 * 0,
        verbose=True,
    ):
        """
        输入一段 (C, wave_length) 音频张量，使用模型推理，输出 (num_stems, C, wave_length) 音频张量。

        其中 C 是音频通道数，num_stems 是分轨数量。

        Separates a full audio waveform into its constituent stems.

        Args:
            mixed_wave (`torch.Tensor` of shape `(channels, time)`):
                The raw audio waveform of the mixture.
            chunk_size (`int`, *optional*, defaults to model.config.wave_chunk_size):
                The size of each audio chunk for processing.
            overlap_size (`int`, *optional*, defaults to `chunk_size // 2`):
                The size of the overlap between consecutive chunks.
            batch_size (`int`, *optional*, defaults to `4`):
                The number of chunks to process in a single batch.
            gap_size (`int`, *optional*, defaults to `44100` (1 second at 44.1kHz)):
                The size of the gap for the fade-in/fade-out window.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether to print progress information during processing.
        Returns:
            torch.Tensor (`torch.Tensor` of shape `(num_stems, channels, time)`):
            The separated audio waveforms.
        """
        assert mixed_wave.dim() == 2, "mixed_wave must be a 2D tensor of shape (channels, time)"
        assert (
            mixed_wave.size(0) == self.config.num_input_channels
        ), f"mixed_wave must have {self.config.num_input_channels} channels, but got {mixed_wave.size(0)}"

        chunk_size = chunk_size or self.config.wave_chunk_size
        overlap_size = overlap_size or (chunk_size // 2)

        # 淡入淡出窗口
        fade_size = chunk_size // 10
        window = torch.ones(chunk_size - 2 * gap_size)
        window[:fade_size] = torch.linspace(0, 1, fade_size)
        window[-fade_size:] = torch.linspace(1, 0, fade_size)
        window = F.pad(window, (gap_size, gap_size), value=0.0)
        window = window.to(mixed_wave.device)

        # 分块准备
        wave_length = mixed_wave.shape[-1]
        n = math.ceil(max(wave_length - chunk_size, 0) / overlap_size) + 1  # 分块数量
        required_length = (n - 1) * overlap_size + chunk_size

        if verbose:
            print(f"Input wave shape: {mixed_wave.shape}")
            print(f"Padded wave length: {required_length}")
            print(f"Batch size: {batch_size}")

        # pad 与分块
        padded_wave = F.pad(mixed_wave, (0, required_length - wave_length), mode="constant")
        unfolded_chunks = padded_wave.unfold(dimension=-1, size=chunk_size, step=overlap_size)  # (C, n, chunk_size)
        batch = unfolded_chunks.permute(1, 0, 2)  # (n, C, chunk_size)

        # 模型推理
        outputs = []
        for i, chunk_batch in enumerate(batch.split(batch_size, dim=0)):
            if verbose:
                print(f"\rProcessing: {i * batch_size + chunk_batch.shape[0]} / {n}")
            outputs.append(self(chunk_batch))
        batch = torch.cat(outputs, dim=0)  # (n, num_stems, C, chunk_size)

        # 加窗
        _, num_stems, C, _ = batch.shape
        batch = batch * window

        # 还原波形
        batch = batch.view(n, -1, chunk_size).permute(1, 2, 0)  # (num_stems * C, chunk_size, n)
        output_result_buffer = F.fold(
            batch,
            output_size=(1, required_length),
            kernel_size=(1, chunk_size),
            stride=(1, overlap_size),
        )  # (num_stems * C, 1, 1, required_length)
        output_result_buffer = output_result_buffer.view(num_stems, C, -1)  # (num_stems, C, required_length)

        # 获得权重和
        window_for_fold = window.expand(1, 1, -1).repeat(1, n, 1)
        weighted_sum_counter = F.fold(
            window_for_fold.permute(0, 2, 1),
            output_size=(1, required_length),
            kernel_size=(1, chunk_size),
            stride=(1, overlap_size),
        )  # (1, 1, 1, required_length)
        weighted_sum_counter = weighted_sum_counter.view(1, 1, -1)  # (1, 1, required_length)
        weighted_sum_counter.clamp_min_(1e-8)

        # 归一化
        final_output = (output_result_buffer / weighted_sum_counter)[:, :, :wave_length]

        return final_output
