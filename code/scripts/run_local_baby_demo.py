import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student.baby_separator import load_baby_separator


STEMS = ("bass", "drums", "other", "vocal")


def make_demo_mixture(seconds: float, sample_rate: int) -> torch.Tensor:
    """Create a deterministic stereo demo mixture when no local song is provided."""
    n = int(seconds * sample_rate)
    t = torch.linspace(0.0, seconds, n, dtype=torch.float32)

    bass = 0.18 * torch.sin(2 * math.pi * 55.0 * t)
    bass += 0.08 * torch.sin(2 * math.pi * 110.0 * t)

    kick_times = torch.arange(0.0, seconds, 0.5)
    drums = torch.zeros_like(t)
    for kt in kick_times:
        env = torch.exp(-70.0 * torch.clamp(t - kt, min=0.0))
        drums += 0.20 * env * torch.sin(2 * math.pi * (95.0 - 35.0 * torch.clamp(t - kt, min=0.0)) * t)

    chord = (
        torch.sin(2 * math.pi * 220.0 * t)
        + 0.7 * torch.sin(2 * math.pi * 277.18 * t)
        + 0.5 * torch.sin(2 * math.pi * 329.63 * t)
    )
    other = 0.08 * chord * (0.65 + 0.35 * torch.sin(2 * math.pi * 0.35 * t))

    vocal = 0.12 * torch.sin(2 * math.pi * (330.0 + 45.0 * torch.sin(2 * math.pi * 0.8 * t)) * t)
    vocal *= 0.5 + 0.5 * torch.sin(2 * math.pi * 0.22 * t + 0.4)

    left = bass + drums + other + vocal
    right = 0.95 * bass + 0.85 * drums + 1.10 * other + 0.9 * vocal
    mixture = torch.stack([left, right], dim=0)
    mixture = mixture / mixture.abs().max().clamp_min(1e-6) * 0.8
    return mixture


def read_audio(path: Path, seconds: float, sample_rate: int) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(audio.T).contiguous()
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    elif x.shape[0] > 2:
        x = x[:2]

    if sr != sample_rate:
        new_len = round(x.shape[-1] * sample_rate / sr)
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)

    wanted = int(seconds * sample_rate)
    if x.shape[-1] < wanted:
        x = torch.nn.functional.pad(x, (0, wanted - x.shape[-1]))
    return x[:, :wanted]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default=".")
    parser.add_argument("--audio", default="")
    parser.add_argument("--out-dir", default="outputs/baby_demo_30sec")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_baby_separator(bundle_dir, device=args.device)
    sample_rate = int(model.config.wave_sample_rate)

    if args.audio:
        mixture = read_audio(Path(args.audio), args.seconds, sample_rate)
        source_note = f"source audio: {Path(args.audio).resolve()}"
    else:
        mixture = make_demo_mixture(args.seconds, sample_rate)
        source_note = "source audio: generated 30-second stereo demo mixture"

    sf.write(out_dir / "input_mixture_30sec.wav", mixture.numpy().T, sample_rate)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        stems = model(mixture.unsqueeze(0).to(args.device)).cpu()[0]
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    for index, stem in enumerate(STEMS):
        sf.write(out_dir / f"{stem}.wav", stems[index].numpy().T, sample_rate)

    audio_seconds = mixture.shape[-1] / sample_rate
    realtime_factor = elapsed / audio_seconds
    speed = audio_seconds / elapsed if elapsed > 0 else float("inf")

    report = [
        "Baby Audio Separation Demo",
        "==========================",
        source_note,
        f"device: {args.device}",
        f"sample_rate: {sample_rate}",
        f"audio_duration_seconds: {audio_seconds:.3f}",
        f"total_inference_seconds: {elapsed:.3f}",
        f"real_time_factor: {realtime_factor:.3f}x",
        f"speed: {speed:.3f} audio_seconds_per_wall_second",
        "",
        "Output stems:",
        "bass.wav",
        "drums.wav",
        "other.wav",
        "vocal.wav",
    ]
    (out_dir / "latency_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "This folder contains one 30-second baby-model source-separation demo.\n"
        "input_mixture_30sec.wav is the model input. The four stem WAV files are model outputs.\n"
        "Latency and real-time factor are written in latency_report.txt.\n",
        encoding="utf-8",
    )

    print(f"saved demo outputs to {out_dir}")
    print(f"elapsed {elapsed:.3f}s for {audio_seconds:.3f}s audio, RTF {realtime_factor:.3f}x")


if __name__ == "__main__":
    main()
