import torch
import torch.nn.functional as F
from einops import rearrange


def _hann_window(stft_kwargs: dict, device: torch.device) -> torch.Tensor:
    # Fresh float32 windows avoid the NaN window issue observed on Kaggle with
    # loaded teacher buffers while keeping the original STFT configuration.
    win_length = stft_kwargs.get("win_length") or stft_kwargs["n_fft"]
    return torch.hann_window(win_length, periodic=True, device=device, dtype=torch.float32)


def waveform_to_teacher_features(model, raw_audio: torch.Tensor) -> torch.Tensor:
    """
    Compute the exact feature tensor consumed by the teacher RoPE stack.

    Args:
        model: BSRoformerForMaskedEstimation.
        raw_audio: [batch, channels, time].

    Returns:
        [batch, time_compressed, bands, hidden].
    """
    device = raw_audio.device
    b, c, _ = raw_audio.shape
    freq_model = model.freq_domain_model

    with torch.autocast(device_type=device.type, enabled=False):
        raw_audio = raw_audio.float()
        packed = rearrange(raw_audio, "b c t -> (b c) t")
        stft = torch.stft(
            packed,
            **model.stft_kwargs,
            window=_hann_window(model.stft_kwargs, device),
            return_complex=True,
        )
        stft = torch.view_as_real(stft)
        stft = rearrange(stft, "(b c) f t z -> b c f t z", b=b, c=c)
        features = rearrange(stft, "b c f t z -> b t (f c z)")

    features = features.to(dtype=next(model.parameters()).dtype)
    t_origin = features.shape[1]

    if freq_model.time_conv_length is not None:
        pad_t = (freq_model.time_conv_length - (t_origin % freq_model.time_conv_length)) % freq_model.time_conv_length
        if pad_t > 0:
            features = F.pad(features, (0, 0, 0, pad_t), value=0.0)

    hidden = freq_model.band_split(features)
    if freq_model.time_conv_length is not None:
        hidden = rearrange(hidden, "b (t tc) n d -> b t n (d tc)", tc=freq_model.time_conv_length)
        hidden = freq_model.time_conv(hidden)

    return hidden


def teacher_rope_target(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Run the frozen teacher RoPE stack and return the tensor the student should
    imitate. Register tokens are used internally and removed before returning.
    """
    freq_model = model.freq_domain_model
    b, t, n, _ = hidden_states.shape

    pos_time = torch.arange(t, device=hidden_states.device).unsqueeze(0)
    pos_freq = torch.arange(n, device=hidden_states.device).unsqueeze(0)
    pos_embeds = freq_model.rotary_emb(hidden_states, pos_time)
    pos_embeds_for_freq = freq_model.rotary_emb(hidden_states, pos_freq)

    rn = freq_model.config.register_token_num
    hidden = F.pad(hidden_states, (0, 0, 0, rn, 0, rn))
    hidden[:, t:, n:, :] = freq_model.register_tokens

    def pad_rope(cos, sin):
        return F.pad(cos, (0, 0, 0, rn), value=1.0), F.pad(sin, (0, 0, 0, rn), value=0.0)

    pos_embeds = pad_rope(*pos_embeds)
    pos_embeds_for_freq = pad_rope(*pos_embeds_for_freq)

    for time_transformer, freq_transformer in freq_model.layers:
        hidden = time_transformer(hidden, position_embeddings=pos_embeds, attention_mask=None)
        hidden = freq_transformer(hidden, position_embeddings=pos_embeds_for_freq, attention_mask=None)

    return hidden[:, :t, :n, :]


def student_features_to_stems(model, student_hidden: torch.Tensor, output_length: int, raw_audio: torch.Tensor) -> torch.Tensor:
    """
    Decode student hidden features through the teacher final norm, mask
    estimators, mask application, and iSTFT. Useful for later fine-tuning.
    """
    freq_model = model.freq_domain_model
    device = raw_audio.device
    b, c, _ = raw_audio.shape

    hidden = freq_model.final_norm(student_hidden)
    if freq_model.time_conv_length is not None:
        hidden = freq_model.time_deconv(hidden)
        hidden = rearrange(hidden, "b t n (d tc) -> b (t tc) n d", tc=freq_model.time_conv_length)

    with torch.autocast(device_type=device.type, enabled=False):
        raw_audio = raw_audio.float()
        packed = rearrange(raw_audio, "b c t -> (b c) t")
        stft = torch.stft(
            packed,
            **model.stft_out_kwargs,
            window=_hann_window(model.stft_out_kwargs, device),
            return_complex=True,
        )
        stft_real = torch.view_as_real(stft)

    t_frames = stft_real.shape[-2]
    hidden = hidden[:, :t_frames, :, :]
    mask = torch.stack([fn(hidden) for fn in freq_model.mask_estimators], dim=1)
    mask = rearrange(mask, "b n t (f c z) -> b n c f t z", z=2, c=c).float()

    with torch.autocast(device_type=device.type, enabled=False):
        stft_expanded = rearrange(stft_real, "(b c) f t z -> b 1 c f t z", b=b, c=c)
        masked = torch.view_as_complex(stft_expanded) * torch.view_as_complex(mask)
        masked = rearrange(masked, "b n c f t -> (b n c) f t")
        audio = torch.istft(
            masked,
            **model.stft_out_kwargs,
            window=_hann_window(model.stft_out_kwargs, device),
            return_complex=False,
            length=output_length,
        )
        audio = rearrange(audio, "(b n c) t -> b n c t", b=b, n=model.config.num_stems, c=c)
    return audio
