import argparse, json, sys
from pathlib import Path
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

# Allow the script to run both on Kaggle and from this repository layout.
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
for path in (CODE_DIR, PROJECT_ROOT, Path("/kaggle/working/project")):
    if path.exists():
        sys.path.insert(0, str(path))

from student.rope_unet import RopeReplacementUNet
from student.teacher_features import waveform_to_teacher_features, student_features_to_stems

STEMS = ("bass", "drums", "other", "vocal")

def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def read_audio(path, sample_rate):
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

def find_audio(song_dir, names):
    for name in names:
        for ext in (".wav", ".flac"):
            p = song_dir / f"{name}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Missing {names} in {song_dir}")

def find_songs(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_dir() and ((p/"mixture.wav").exists() or (p/"mixture.flac").exists()))

def simple_sdr(pred, target):
    n = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :n].float()
    target = target[..., :n].float()
    return float(10 * torch.log10((target.pow(2).sum() + 1e-8) / ((target - pred).pow(2).sum() + 1e-8)))

def mixture_consistency(pred, mix, strength=1.0):
    # Project predictions so the four stems add back to the input mixture.
    # Shapes: pred [4, 2, T], mix [2, T].
    residual = mix - pred.sum(dim=0)
    return pred + strength * residual.unsqueeze(0) / pred.shape[0]

@torch.no_grad()
def separate_overlap(teacher, student, mixture, chunk_size, hop_size, device):
    """Run 8-second chunk inference with 50% overlap-add by default."""
    total_len = mixture.shape[-1]
    output = torch.zeros(4, 2, total_len)
    weight_sum = torch.zeros(total_len)

    weight = torch.hann_window(chunk_size, periodic=False).float().clamp_min(0.05)
    starts = list(range(0, max(1, total_len - chunk_size + 1), hop_size))
    if not starts or starts[-1] + chunk_size < total_len:
        starts.append(max(0, total_len - chunk_size))

    for start in starts:
        end = min(start + chunk_size, total_len)
        chunk = mixture[:, start:end]
        real_len = chunk.shape[-1]
        if real_len < chunk_size:
            chunk = F.pad(chunk, (0, chunk_size - real_len))

        raw = chunk.unsqueeze(0).to(device)
        x = waveform_to_teacher_features(teacher, raw)
        h = student(x)
        pred = student_features_to_stems(teacher, h, chunk_size, raw).cpu()[0, :, :, :real_len]

        w = weight[:real_len]
        output[:, :, start:end] += pred * w.view(1, 1, -1)
        weight_sum[start:end] += w

    return output / weight_sum.clamp_min(1e-8).view(1, 1, -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="/kaggle/working/sdr_mc")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--mc-strength", type=float, default=1.0)
    ap.add_argument("--chunk-seconds", type=float, default=8.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher = AutoModel.from_pretrained(
        str(Path(args.project) / "teacher_model"),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(args.device).float().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ckpt = torch_load(args.checkpoint, args.device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    student = RopeReplacementUNet(hidden_size=int(teacher.config.hidden_size)).to(args.device).eval()
    student.load_state_dict(state)

    sr = int(teacher.config.wave_sample_rate)
    chunk_size = int(args.chunk_seconds * sr)
    hop_size = int(chunk_size * (1 - args.overlap))

    totals = {s: [] for s in STEMS}
    rows = []

    for song_dir in tqdm(find_songs(args.dataset_root), desc="songs"):
        mix = read_audio(find_audio(song_dir, ("mixture",)), sr)
        targets = {
            "bass": read_audio(find_audio(song_dir, ("bass", "target_bass")), sr),
            "drums": read_audio(find_audio(song_dir, ("drums", "target_drums")), sr),
            "other": read_audio(find_audio(song_dir, ("other", "target_other")), sr),
            "vocal": read_audio(find_audio(song_dir, ("vocals", "vocal", "target_vocals", "target_vocal")), sr),
        }

        pred = separate_overlap(teacher, student, mix, chunk_size, hop_size, args.device)
        pred = mixture_consistency(pred, mix, strength=args.mc_strength)

        row = {"song": song_dir.name}
        for i, stem in enumerate(STEMS):
            row[stem] = simple_sdr(pred[i], targets[stem])
            totals[stem].append(row[stem])
        row["mean"] = sum(row[s] for s in STEMS) / 4
        rows.append(row)

        print(f"{song_dir.name}: bass {row['bass']:.3f} | drums {row['drums']:.3f} | other {row['other']:.3f} | vocal {row['vocal']:.3f} | mean {row['mean']:.3f}")

    summary = {stem: sum(vals)/len(vals) for stem, vals in totals.items()}
    summary["mean"] = sum(summary.values()) / 4

    print("SUMMARY")
    for stem in STEMS:
        print(f"{stem}: {summary[stem]:.3f} dB")
    print(f"mean: {summary['mean']:.3f} dB")

    out_path = out_dir / "student_overlap_mc_sdr_results.json"
    out_path.write_text(json.dumps({"summary": summary, "songs": rows}, indent=2))
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
