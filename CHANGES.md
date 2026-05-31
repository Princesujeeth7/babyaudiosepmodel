# Change Summary

This file records what was changed after downloading the Hugging Face teacher model and what was added for student training.

## 1. Teacher Model Compatibility Changes

The original teacher files were downloaded from:

```text
HiDolen/Mini-BS-RoFormer-V2-46.8M
```

Files kept in `teacher_model/`:

```text
config.json
configuration_bs_roformer.py
modeling_bs_roformer.py
model.safetensors
README_teacher.md
```

The teacher architecture was not intentionally changed. The following safe runtime/compatibility edits were made in `modeling_bs_roformer.py`.

### Attention Dispatch

The original code expected newer Transformers attention helper APIs. For older or different Transformers installations, `ALL_ATTENTION_FUNCTIONS` can be unavailable or incompatible.

Change:

```text
Use torch.nn.functional.scaled_dot_product_attention directly.
```

Reason:

```text
Keeps optimized PyTorch SDPA attention and avoids very slow manual attention.
```

### Torch 2.x SDPA Scale Argument

Older PyTorch versions do not support:

```python
scale=
```

inside `F.scaled_dot_product_attention`.

Change:

```text
Do not pass scale=.
Do not manually scale Q.
Let PyTorch apply 1 / sqrt(head_dim) internally.
```

### Grouped Query Attention

The config uses:

```text
num_attention_heads = 8
num_key_value_heads = 4
```

So K/V heads must be expanded to match Q heads.

Change:

```python
key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
```

### Transformers Tied Weight Metadata

Some newer Transformers versions expect `all_tied_weights_keys`.

Change:

```text
Add tied-weight metadata compatibility fields.
```

Reason:

```text
Prevents AutoModel.from_pretrained from failing during final load-state handling.
```

### STFT Window Safety

On Kaggle we observed NaN outputs when relying on loaded STFT window buffers in some situations.

Change in student/evaluation helper code:

```text
Create fresh finite float32 Hann windows before torch.stft / torch.istft.
```

Important:

```text
STFT/iSTFT configuration itself was not changed.
```

## 2. What Was Not Changed In The Teacher

The following teacher behavior was preserved:

```text
model dimensions
RoPE logic
BandSplit layout
mask estimator structure
STFT/iSTFT n_fft and hop settings
stem order
teacher weight file
```

## 3. Student Model Additions

Implemented:

```text
code/student/rope_unet.py
```

This is the selected lightweight replacement for the teacher RoPE Transformer stack.

Shape contract:

```text
input:  [batch, time, bands, hidden]
output: [batch, time, bands, hidden]
```

The model uses:

```text
depthwise time-axis convolution
depthwise frequency-axis convolution
1x1 pointwise convolution
U-Net encoder-decoder skips
residual output
```

Selected model:

```text
RopeReplacementUNet v1
5,762,304 parameters
```

Also implemented but not selected:

```text
code/student/axial_rope_unet_v2.py
```

v2 was larger and did not outperform v1 during available experiments.

## 4. Final Training Losses

Final Phase 8 training used:

```text
ground-truth waveform L1 loss
multi-resolution STFT loss
stem-weighted SI-SDR loss
hidden RoPE distillation loss
teacher-stem distillation loss
mask-level distillation loss
mixture consistency loss
```

Final training script:

```text
code/scripts/train_v1_mask_distill.py
```

Final evaluation script:

```text
code/scripts/eval_student_overlap_mc_sdr.py
```

## 5. Final Result

Selected checkpoint:

```text
student_model/final_student_phase8_weights_only.pt
```

Final SDR:

```text
bass:  3.760 dB
drums: 5.778 dB
other: 3.314 dB
vocal: 5.055 dB
mean:  4.477 dB
```

## 6. Rejected Experiments

The following were tested but not selected:

```text
v2 axial U-Net, because it was larger and had lower SDR
aggressive bass/other stem weighting, because mean SDR did not improve
continuing v1 without mask-level distillation, because it saturated near 4.47 SDR
```

## 7. Final Submission Decision

The final selected model is v1 Phase 8 because it gave the best measured SDR while preserving the original proposal:

```text
Replace BS-RoFormer RoPE Transformer block with a lightweight U-Net/CNN module.
```
