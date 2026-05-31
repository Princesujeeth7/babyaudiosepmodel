import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student.baby_separator import load_baby_separator


STEMS = ("bass", "drums", "other", "vocal")


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


def find_audio(song_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        for ext in (".wav", ".flac"):
            path = song_dir / f"{name}{ext}"
            if path.exists():
                return path
    raise FileNotFoundError(f"Missing one of {names} in {song_dir}")


def find_songs(root: Path) -> list[Path]:
    if (root / "mixture.wav").exists() or (root / "mixture.flac").exists():
        return [root]
    songs = [
        p for p in root.rglob("*")
        if p.is_dir() and ((p / "mixture.wav").exists() or (p / "mixture.flac").exists())
    ]
    return sorted(songs)


def simple_sdr(pred: torch.Tensor, target: torch.Tensor) -> float:
    n = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :n].float()
    target = target[..., :n].float()
    noise = target - pred
    return float(10.0 * torch.log10((target.pow(2).sum() + 1e-8) / (noise.pow(2).sum() + 1e-8)))


def separate_chunked(model, mixture: torch.Tensor, chunk_size: int, device: str) -> torch.Tensor:
    outs = []
    with torch.inference_mode():
        for start in range(0, mixture.shape[-1], chunk_size):
            chunk = mixture[:, start:start + chunk_size]
            length = chunk.shape[-1]
            stems = model(chunk.unsqueeze(0).to(device)).cpu()[0]
            outs.append(stems[..., :length])
    return torch.cat(outs, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default=".")
    parser.add_argument("--dataset-root", required=True, help="MUSDB-HQ test folder or one song folder.")
    parser.add_argument("--checkpoint", default="student_best_audio_weights_only.pt")
    parser.add_argument("--out-dir", default="outputs/baby_sdr")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--max-songs", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_baby_separator(args.bundle_dir, device=args.device, checkpoint_name=args.checkpoint)
    sr = int(model.config.wave_sample_rate)
    chunk_size = int(args.chunk_seconds * sr)

    songs = find_songs(Path(args.dataset_root))
    if args.max_songs:
        songs = songs[:args.max_songs]
    if not songs:
        raise RuntimeError(f"No MUSDB-style songs found under {args.dataset_root}")

    results = []
    totals = {stem: [] for stem in STEMS}
    for song_dir in tqdm(songs, desc="songs"):
        mixture = read_audio(find_audio(song_dir, ("mixture",)), sr)
        targets = {
            "bass": read_audio(find_audio(song_dir, ("bass", "target_bass")), sr),
            "drums": read_audio(find_audio(song_dir, ("drums", "target_drums")), sr),
            "other": read_audio(find_audio(song_dir, ("other", "target_other")), sr),
            "vocal": read_audio(find_audio(song_dir, ("vocals", "vocal", "target_vocals", "target_vocal")), sr),
        }
        pred = separate_chunked(model, mixture, chunk_size, args.device)

        song_scores = {}
        for i, stem in enumerate(STEMS):
            score = simple_sdr(pred[i], targets[stem])
            song_scores[stem] = score
            totals[stem].append(score)
        song_scores["mean"] = sum(song_scores.values()) / len(STEMS)
        results.append({"song": song_dir.name, **song_scores})
        print(
            f"{song_dir.name}: "
            f"bass {song_scores['bass']:.3f} | drums {song_scores['drums']:.3f} | "
            f"other {song_scores['other']:.3f} | vocal {song_scores['vocal']:.3f} | "
            f"mean {song_scores['mean']:.3f}"
        )

    summary = {stem: sum(vals) / max(1, len(vals)) for stem, vals in totals.items()}
    summary["mean"] = sum(summary.values()) / len(STEMS)
    print("SUMMARY")
    for stem in STEMS:
        print(f"{stem}: {summary[stem]:.3f} dB")
    print(f"mean: {summary['mean']:.3f} dB")

    output = {"summary": summary, "songs": results}
    out_path = out_dir / "baby_sdr_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
