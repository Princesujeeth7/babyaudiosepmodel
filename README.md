# Lightweight BS-RoFormer RoPE-Replacement Student

This repository contains the final office-submission package for a lightweight music source-separation prototype.

The main idea is to keep the BS-RoFormer teacher pipeline as a reference and replace its heavy RoPE Transformer stack with a small CNN/U-Net student block.

## Goal

- Task: 4-stem music separation.
- Stems: bass, drums, other, vocal.
- Audio: stereo, 44.1 kHz.
- Teacher: `HiDolen/Mini-BS-RoFormer-V2-46.8M`.
- Student: lightweight U-Net replacement for the RoPE Transformer block.
- Deployment direction: TV/on-device inference, with STFT/iSTFT expected to be handled by hardware later.

Stem order is fixed:

```text
0 = bass
1 = drums
2 = other
3 = vocal
```

## Final Selected Model

Use this checkpoint for the selected final student:

```text
student_model/final_student_phase8_weights_only.pt
```

The full training checkpoint, including optimizer state, is also kept:

```text
student_model/final_student_phase8_mask_distill.pt
```

Final selected student:

```text
RopeReplacementUNet v1
5,762,304 trainable parameters
```

## Student U-Net Architecture

The final selected student is:

```text
code/student/rope_unet.py
```

It replaces the teacher RoPE Transformer stack. The input and output are both teacher hidden features:

```text
input:  [batch, time, bands, hidden]
output: [batch, time, bands, hidden]
```

For this teacher:

```text
hidden = 384
bands = 80
```

The model first converts the hidden dimension into convolution channels:

```text
[B, T, Bands, Hidden] -> [B, Hidden, T, Bands]
```

Then it applies a lightweight U-Net:

```text
Input projection:
  1x1 conv: 384 -> 96

Encoder widths:
  96 -> 160 -> 256 -> 384

Decoder widths:
  384 -> 256 -> 160 -> 96

Output projection:
  1x1 conv: 96 -> 384

Final residual:
  output = predicted_delta + input
```

Each block is a depthwise time-frequency convolution block:

```text
GroupNorm
time-axis depthwise conv
frequency-axis depthwise conv
1x1 pointwise channel mixing
GELU
residual connection
```

The exact kernels are:

```text
time-axis conv:      Conv2d kernel = (7, 1)
frequency-axis conv: Conv2d kernel = (1, 7)
```

So yes, the U-Net has separate time-axis and frequency-axis convolutions. They are implemented as **2D depthwise convolutions with one axis fixed to 1**, which is equivalent to applying a 1D convolution along only one axis:

```text
(7, 1) sees time context only.
(1, 7) sees frequency/band context only.
```

This was chosen to keep the model CNN/U-Net based while still modeling the two dependencies that RoPE attention originally handled:

```text
temporal dependency across frames
spectral dependency across frequency bands
```

## What Is v2?

The file:

```text
code/student/axial_rope_unet_v2.py
```

contains an experimental larger axial time-frequency U-Net. It uses:

```text
time-axis branch
frequency-axis branch
gated fusion
dilated time convolutions
more blocks/channels
```

It was tested as a stronger replacement for the RoPE Transformer, but it was **not selected** because it had more parameters and lower SDR than v1 within the available training time.

Final selected model remains:

```text
code/student/rope_unet.py
```

Approximate checkpoint sizes:

```text
weights-only checkpoint: 22 MB, fp32
full training checkpoint: 66 MB, includes optimizer state
expected fp16 student size: about 11 MB
expected int8 student size: about 6 MB
```

## Final SDR

Evaluation setup:

```text
Dataset: MUSDB18-HQ test split
Chunk length: 8 seconds
Inference: 50% overlap-add
Post-processing: mixture consistency
Metric: simple SDR per stem
```

Final result:

```text
bass:  3.760 dB
drums: 5.778 dB
other: 3.314 dB
vocal: 5.055 dB
mean:  4.477 dB
```

Full result JSON:

```text
student_model/final_sdr_phase8_mask_distill.json
```

## Folder Layout

```text
office_submission_minimal/
|-- README.md
|-- CHANGES.md
|-- GITHUB_UPLOAD_NOTES.md
|-- requirements.txt
|-- teacher_model/
|   |-- config.json
|   |-- configuration_bs_roformer.py
|   |-- modeling_bs_roformer.py
|   |-- model.safetensors
|   `-- README_teacher.md
|-- student_model/
|   |-- CHECKPOINT_INFO.txt
|   |-- final_student_phase8_weights_only.pt
|   |-- final_student_phase8_mask_distill.pt
|   `-- final_sdr_phase8_mask_distill.json
|-- code/
|   |-- student/
|   |   |-- rope_unet.py
|   |   |-- axial_rope_unet_v2.py
|   |   |-- teacher_features.py
|   |   |-- baby_separator.py
|   |   `-- train_rope_unet.py
|   `-- scripts/
|       |-- train_v1_mask_distill.py
|       |-- eval_student_overlap_mc_sdr.py
|       |-- train_audio_phases.py
|       |-- eval_audio_student_sdr.py
|       |-- infer_baby.py
|       |-- eval_baby_sdr.py
|       |-- precompute_rope_features.py
|       |-- verify_musdb_hq_layout.py
|       `-- count_baby_params.py
|-- docs/
|   `-- bs_roformer_researchpaper.pdf
`-- notebooks/
    `-- kaggle_training_process.ipynb
```

## Required Libraries

Recommended environment:

```text
Python 3.10 or newer
PyTorch 2.x
Transformers 4.x
CUDA GPU for training/evaluation
```

Install:

```bash
pip install -r requirements.txt
```

The code was tested primarily on Kaggle GPU notebooks.

## Dataset Format

The training/evaluation scripts expect MUSDB-style folders:

```text
dataset_root/
|-- train/
|   `-- SongName/
|       |-- mixture.wav
|       |-- bass.wav
|       |-- drums.wav
|       |-- other.wav
|       `-- vocals.wav
`-- test/
    `-- SongName/
        |-- mixture.wav
        |-- bass.wav
        |-- drums.wav
        |-- other.wav
        `-- vocals.wav
```

`vocal.wav` is also accepted as an alias for `vocals.wav`.

## Full Pipeline

### 1. Verify Dataset

```bash
python code/scripts/verify_musdb_hq_layout.py --dataset-root /path/to/musdb18-hq
```

### 2. Count Student Parameters

```bash
python code/scripts/count_baby_params.py
```

Expected student parameter count:

```text
5,762,304
```

### 3. Train Final Phase 8 Model

The final selected run used mask-level distillation plus audio losses:

```bash
python code/scripts/train_v1_mask_distill.py \
  --project . \
  --dataset-root /path/to/musdb18-hq/train \
  --init-checkpoint student_model/final_student_phase8_weights_only.pt \
  --run-dir runs/phase8_v1_mask_distill \
  --train-songs 90 \
  --val-songs 10 \
  --epochs 6 \
  --steps-per-epoch 300 \
  --val-steps 30 \
  --batch-size 1 \
  --lr 1e-5 \
  --remix-prob 0.2 \
  --mask-weight 0.2 \
  --sisdr-weight 0.03 \
  --device cuda
```

For Kaggle, copy this folder to `/kaggle/working/project`, then use:

```bash
--project /kaggle/working/project
--dataset-root /kaggle/input/.../musdb18-hq/train
```

### 4. Evaluate Final Model

```bash
python code/scripts/eval_student_overlap_mc_sdr.py \
  --project . \
  --dataset-root /path/to/musdb18-hq/test \
  --checkpoint student_model/final_student_phase8_weights_only.pt \
  --out-dir outputs/final_sdr \
  --device cuda \
  --overlap 0.5 \
  --mc-strength 1.0
```

This performs:

```text
8-second chunking
50% overlap-add
mixture-consistency post-processing
simple SDR calculation
```

## Losses Used In Final Training

The final Phase 8 training combined the following losses:

```text
1. Ground-truth waveform L1 loss
   L1(student_stems, ground_truth_stems)

2. Multi-resolution STFT magnitude loss
   STFT loss at n_fft = 1024, 2048, 4096

3. Stem-weighted SI-SDR loss
   Extra weight for bass and other because they were weakest stems

4. Hidden RoPE distillation loss
   L1(student_hidden, teacher_rope_hidden)

5. Teacher-stem distillation loss
   L1(student_stems, teacher_stems)

6. Mask-level distillation loss
   L1(student_masks, teacher_masks)

7. Mixture consistency loss
   L1(sum(student_stems), mixture)
```

The final stem weights were:

```text
bass:  1.5
drums: 1.0
other: 1.7
vocal: 1.1
```

## Training History Summary

The selected model came from the following experimental path:

```text
1. Teacher loading and Torch compatibility fixes.
2. v1 U-Net hidden distillation from teacher RoPE output.
3. Audio fine-tuning with ground-truth waveform and STFT losses.
4. Teacher-stem distillation.
5. Remix augmentation.
6. Stem-weighted SI-SDR fine-tuning.
7. Overlap-add inference and mixture consistency.
8. Mask-level distillation, selected as final.
```

The best observed mean SDR after each meaningful step was approximately:

```text
early v1 audio fine-tune: 4.16 dB
v1 remix + overlap:       4.36 dB
weighted SI-SDR:          4.47 dB
mask distillation final:  4.477 dB
```

## Teacher Model Notes

The `teacher_model/` folder contains the local downloaded Hugging Face teacher files. The teacher was kept for:

```text
reference inference
feature extraction
teacher hidden distillation
teacher stem distillation
mask-level distillation
reproducible evaluation
```

The student does not use the teacher RoPE Transformer as its deployment target; it replaces that heavy block with the U-Net.

Teacher compatibility changes are documented in `CHANGES.md`.


