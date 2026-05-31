import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student.axial_rope_unet_v2 import AxialRopeUNetV2
from student.rope_unet import RopeReplacementUNet
from student.teacher_features import student_features_to_stems, waveform_to_teacher_features


STEMS = ("bass", "drums", "other", "vocal")


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_audio(song_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        for ext in (".wav", ".flac"):
            path = song_dir / f"{name}{ext}"
            if path.exists():
                return path
    raise FileNotFoundError(f"Missing {names} in {song_dir}")


def find_songs(root: Path) -> list[Path]:
    if (root / "mixture.wav").exists() or (root / "mixture.flac").exists():
        return [root]
    return sorted(
        p for p in root.rglob("*")
        if p.is_dir() and ((p / "mixture.wav").exists() or (p / "mixture.flac").exists())
    )


def read_audio(path: Path, sample_rate: int) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(audio).transpose(0, 1).contiguous()
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    elif x.shape[0] > 2:
        x = x[:2]
    if sr != sample_rate:
        new_len = round(x.shape[-1] * sample_rate / sr)
        x = F.interpolate(x.unsqueeze(0), size=new_len, mode="linear", align_corners=False).squeeze(0)
    return x


def make_student(variant: str, hidden_size: int):
    if variant == "v1":
        return RopeReplacementUNet(hidden_size=hidden_size)
    if variant == "v2":
        return AxialRopeUNetV2(hidden_size=hidden_size)
    raise ValueError(variant)


def simple_sdr(pred: torch.Tensor, target: torch.Tensor) -> float:
    n = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :n].float()
    target = target[..., :n].float()
    return float(10.0 * torch.log10((target.pow(2).sum() + 1e-8) / ((target - pred).pow(2).sum() + 1e-8)))


@torch.no_grad()
def separate(teacher, student, mixture: torch.Tensor, chunk_size: int, device: str) -> torch.Tensor:
    outs = []
    for start in range(0, mixture.shape[-1], chunk_size):
        chunk = mixture[:, start:start + chunk_size]
        length = chunk.shape[-1]
        raw = chunk.unsqueeze(0).to(device)
        x = waveform_to_teacher_features(teacher, raw)
        h = student(x)
        stems = student_features_to_stems(teacher, h, length, raw).cpu()[0]
        outs.append(stems[..., :length])
    return torch.cat(outs, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-variant", choices=("v1", "v2"), default="")
    parser.add_argument("--out-dir", default="/kaggle/working/student_audio_sdr")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--max-songs", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher = AutoModel.from_pretrained(
        str(Path(args.project) / "teacher_model"),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(args.device).float().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ckpt = torch_load(Path(args.checkpoint), args.device)
    variant = args.model_variant or ckpt.get("model_variant", "v1")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    student = make_student(variant, int(teacher.config.hidden_size)).to(args.device).eval()
    student.load_state_dict(state)

    sr = int(teacher.config.wave_sample_rate)
    chunk_size = int(args.chunk_seconds * sr)
    songs = find_songs(Path(args.dataset_root))
    if args.max_songs:
        songs = songs[:args.max_songs]

    results = []
    totals = {stem: [] for stem in STEMS}
    for song_dir in tqdm(songs, desc="songs"):
        mix = read_audio(find_audio(song_dir, ("mixture",)), sr)
        targets = {
            "bass": read_audio(find_audio(song_dir, ("bass", "target_bass")), sr),
            "drums": read_audio(find_audio(song_dir, ("drums", "target_drums")), sr),
            "other": read_audio(find_audio(song_dir, ("other", "target_other")), sr),
            "vocal": read_audio(find_audio(song_dir, ("vocals", "vocal", "target_vocals", "target_vocal")), sr),
        }
        pred = separate(teacher, student, mix, chunk_size, args.device)
        row = {"song": song_dir.name}
        for i, stem in enumerate(STEMS):
            row[stem] = simple_sdr(pred[i], targets[stem])
            totals[stem].append(row[stem])
        row["mean"] = sum(row[s] for s in STEMS) / len(STEMS)
        results.append(row)
        print(
            f"{song_dir.name}: bass {row['bass']:.3f} | drums {row['drums']:.3f} | "
            f"other {row['other']:.3f} | vocal {row['vocal']:.3f} | mean {row['mean']:.3f}"
        )

    summary = {stem: sum(vals) / max(1, len(vals)) for stem, vals in totals.items()}
    summary["mean"] = sum(summary.values()) / len(STEMS)
    print("SUMMARY")
    for stem in STEMS:
        print(f"{stem}: {summary[stem]:.3f} dB")
    print(f"mean: {summary['mean']:.3f} dB")

    out_path = out_dir / "student_sdr_results.json"
    out_path.write_text(json.dumps({"summary": summary, "songs": results}, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
