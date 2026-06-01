---
library_name: transformers
license: cc-by-nc-4.0
datasets:
- CLAPv2/MUSDB18-HQ
pipeline_tag: audio-to-audio
tags:
- music
---

# Mini-BS-RoFormer-V2

Model for the Music source separation task.
Made a bunch of improvements to [the existing BS-RoFormer open-source implementation](https://github.com/lucidrains/BS-RoFormer).

针对音乐音频分离任务的模型。对 [现有的 BS-RoFormer 开源实现](https://github.com/lucidrains/BS-RoFormer) 做出了一些改进。

[demo 示例试听](https://demo-pages.20241107.xyz/#/Mini-BS-RoFormer-V2-46.8M)

## Model Details

模型总参数量 46.8M，权重精度 BF16。

在 MUSDB18HQ 数据的 val 集上的性能（单位 SDR，越高越好）：

| tracks  | Mini-BS-RoFormer-V2-46.8M | [Mini-BS-RoFormer-18M](https://huggingface.co/HiDolen/Mini-BS-RoFormer-18M) | [Mini-BS-RoFormer](https://huggingface.co/HiDolen/Mini-BS-RoFormer) |
| ------- | ------------------------- | --- | --- |
| overall | **10.03**                 | 9.01                                                                        | 6.48                                                                |
| bass    | **9.68**                  | 8.31                                                                        | 5.66                                                                |
| drums   | **10.58**                 | 9.55                                                                        | 6.77                                                                |
| other   | **8.99**                  | 8.14                                                                        | 6.06                                                                |
| vocal   | **10.86**                 | 10.03                                                                       | 7.44                                                                |

使用时间维度下采样极大减轻了资源消耗。推理 30 秒音频所需运算量：

| model                                                                       | GFLOPs      |
| --------------------------------------------------------------------------- | ----------- |
| Mini-BS-RoFormer-V2-46.8M                                                   | **8343.55** |
| [Mini-BS-RoFormer-18M](https://huggingface.co/HiDolen/Mini-BS-RoFormer-18M) | 10115.77    |
| [Mini-BS-RoFormer](https://huggingface.co/HiDolen/Mini-BS-RoFormer)         | 3068.64     |

## Uses

使用的 transformers 库版本为 4.55.4。为了正常运行模型还需要安装库 soudfile、einops 和 librosa。

GPU 推理：

```python
from transformers import AutoModel
import soundfile
import torch
import librosa

model_name = "HiDolen/Mini-BS-RoFormer-V2-46.8M"
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
)
model.to("cuda")

# 加载音频
file = "./Bruno Mars - Runaway Baby.mp3"
waveform, sr = librosa.load(file, sr=44100, mono=False)
waveform = torch.tensor(waveform).float()
waveform = waveform.to("cuda")

# 进行推理
result = model.separate(
    waveform,
    batch_size=2,
    verbose=True,
)

# 保存处理结果
for i in range(result.shape[0]):
    soundfile.write(f"separated_stem_{i}.wav", result[i].cpu().numpy().T, 44100)
```

以上代码会分离出 bass、drums、other 和 vocal 四个轨道。若想只分离人声和伴奏两轨，在最后保存音频时合并即可：

```python
···

result = model.separate(
    waveform,
    batch_size=2,
    verbose=True,
)

# 合并 bass、drums、other 作为伴奏
instrumental = result[0] + result[1] + result[2]
vocals = result[3]
result = torch.stack([instrumental, vocals], dim=0)
for i in range(result.shape[0]):
    soundfile.write(f"separated_stem_{i}.wav", result[i].cpu().numpy().T, 44100)
```

## Differences from the previous version

Mini-BS-RoFormer-V2 相比于之前版本的主要改进：

1. 使用 bf16-true 精度进行训练
2. 使用 muon 优化器训练 transformer 层，加速收敛
3. 音频输入的 stft 运算，使用的 n_fft 从 2048 变为 4096。音频输出维持在 n_fft=2048
4. freq_band 从原来的 62 段分频更换为基于梅尔频率的 80 段分频
5. 时间维度下采样，采样步长为 4。大大减少了训练和推理的运算量
6. MaskEstimator 额外预测 gate 门控，输出音频的空白部分会更加安静
7. 其他若干代码修改

## Training Details

使用 MUSDB18HQ 数据集的 train 和 test 集进行训练。

学习率恒定 5e-4，以 batch_size=16 训练 310k 步。

对 transformer 层使用 Muon 优化器，其他网络层使用 AdamW 优化器。

## Acknowledgments

- https://github.com/lucidrains/BS-RoFormer
- https://arxiv.org/abs/2309.02612 (Music Source Separation with Band-Split RoPE Transformer)