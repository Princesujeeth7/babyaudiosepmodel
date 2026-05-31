import os
from pathlib import Path
from typing import Union

os.environ.setdefault("HF_MODULES_CACHE", str((Path.cwd() / ".hf_modules").resolve()))

import torch
import torch.nn as nn
from transformers import AutoModel

from .rope_unet import RopeReplacementUNet
from .teacher_features import student_features_to_stems, waveform_to_teacher_features


def _torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class BabySeparator(nn.Module):
    """
    Complete runnable student separator.

    The model keeps the pretrained BS-RoFormer STFT, BandSplit, time
    compression, final norm, mask estimators, and iSTFT. The expensive RoPE
    Transformer stack is replaced by the trained U-Net student.
    """

    def __init__(self, teacher_model, student):
        super().__init__()
        self.teacher = teacher_model
        self.student = student

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    @property
    def config(self):
        return self.teacher.config

    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_audio: [batch, 2, time]

        Returns:
            separated stems: [batch, 4, 2, time]
        """
        student_input = waveform_to_teacher_features(self.teacher, raw_audio)
        student_hidden = self.student(student_input)
        return student_features_to_stems(self.teacher, student_hidden, raw_audio.shape[-1], raw_audio)


def load_baby_separator(
    bundle_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    checkpoint_name: str = "student_model/final_student_phase8_weights_only.pt",
):
    bundle_dir = Path(bundle_dir)
    teacher_dir = bundle_dir / "teacher_model"
    ckpt_path = bundle_dir / checkpoint_name
    teacher = AutoModel.from_pretrained(
        str(teacher_dir),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    teacher = teacher.float().eval()

    ckpt = _torch_load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    hidden_size = int(ckpt.get("hidden_size", teacher.config.hidden_size)) if isinstance(ckpt, dict) else teacher.config.hidden_size

    student = RopeReplacementUNet(hidden_size=hidden_size).to(device)
    student.load_state_dict(state)
    student.eval()

    model = BabySeparator(teacher, student).to(device)
    model.eval()
    return model
