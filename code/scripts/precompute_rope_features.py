import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student.teacher_features import teacher_rope_target, waveform_to_teacher_features


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


def find_stem(song_dir: Path, stem: str) -> Path:
    names = [stem]
    if stem == "vocal":
        names = ["vocals", "vocal"]
    for name in names:
        prefixed = f"target_{name}"
        for ext in (".wav", ".flac"):
            for candidate in (song_dir / f"{name}{ext}", song_dir / f"{prefixed}{ext}"):
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"Missing {stem} stem in {song_dir}")


def find_songs(root: Path) -> list[Path]:
    songs = []
    if (root / "mixture.wav").exists() or (root / "mixture.flac").exists():
        songs.append(root)
    for song_dir in root.rglob("*"):
        if not song_dir.is_dir():
            continue
        if (song_dir / "mixture.wav").exists() or (song_dir / "mixture.flac").exists():
            songs.append(song_dir)
    return sorted(songs)


def load_song(song_dir: Path, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
    mix_path = (song_dir / "mixture.wav") if (song_dir / "mixture.wav").exists() else (song_dir / "mixture.flac")
    mix = read_audio(mix_path, sample_rate)
    targets = [read_audio(find_stem(song_dir, stem), sample_rate) for stem in STEMS]
    min_len = min([mix.shape[-1], *(x.shape[-1] for x in targets)])
    mix = mix[:, :min_len]
    targets = torch.stack([x[:, :min_len] for x in targets], dim=0)
    return mix, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--dataset-root", required=True, help="MUSDB-HQ folder containing song folders with mixture/stems.")
    parser.add_argument("--out-dir", default="outputs/rope_feature_cache")
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--max-songs", type=int, default=0)
    parser.add_argument("--max-chunks-per-song", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(
        str(Path(args.model_dir).resolve()),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sample_rate = int(model.config.wave_sample_rate)
    chunk_size = int(args.chunk_seconds * sample_rate)

    songs = find_songs(Path(args.dataset_root))
    if args.max_songs:
        songs = songs[: args.max_songs]
    if not songs:
        raise RuntimeError(f"No MUSDB-style songs found under {args.dataset_root}")

    saved = 0
    for song_dir in tqdm(songs, desc="songs"):
        mix, stems = load_song(song_dir, sample_rate)
        chunks = max(1, mix.shape[-1] // chunk_size)
        if args.max_chunks_per_song:
            chunks = min(chunks, args.max_chunks_per_song)

        for chunk_idx in range(chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, mix.shape[-1])
            chunk = mix[:, start:end]
            stem_chunk = stems[:, :, start:end]
            if chunk.shape[-1] < chunk_size:
                pad = chunk_size - chunk.shape[-1]
                chunk = F.pad(chunk, (0, pad))
                stem_chunk = F.pad(stem_chunk, (0, pad))

            with torch.inference_mode():
                raw = chunk.unsqueeze(0).to(args.device)
                student_input = waveform_to_teacher_features(model, raw)
                teacher_target = teacher_rope_target(model, student_input)

            item = {
                "song": song_dir.name,
                "chunk_index": chunk_idx,
                "sample_rate": sample_rate,
                "waveform": chunk.cpu().half(),
                "stems": stem_chunk.cpu().half(),
                "student_input": student_input.squeeze(0).cpu().half(),
                "teacher_target": teacher_target.squeeze(0).cpu().half(),
            }
            torch.save(item, out_dir / f"{saved:08d}.pt")
            saved += 1

    print(f"saved {saved} feature chunks to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
