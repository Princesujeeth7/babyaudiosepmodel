import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", default=".")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out-dir", default="outputs/baby_infer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seconds", type=float, default=0.0)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_baby_separator(bundle_dir, device=args.device)
    sr = int(model.config.wave_sample_rate)
    audio = read_audio(Path(args.audio), sr)
    if args.seconds > 0:
        audio = audio[:, : int(args.seconds * sr)]

    with torch.inference_mode():
        stems = model(audio.unsqueeze(0).to(args.device)).cpu()[0]

    for idx, stem in enumerate(STEMS):
        sf.write(out_dir / f"{stem}.wav", stems[idx].numpy().T, sr)
    sf.write(out_dir / "mixture.wav", audio.numpy().T, sr)
    print(f"saved stems to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
