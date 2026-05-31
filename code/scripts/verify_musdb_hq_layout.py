import argparse
from pathlib import Path

import soundfile as sf


STEMS = ("mixture", "bass", "drums", "other")


def has_vocal(song_dir: Path) -> bool:
    return any((song_dir / f"{name}{ext}").exists() for name in ("vocals", "vocal") for ext in (".wav", ".flac"))


def has_file(song_dir: Path, name: str) -> bool:
    return any((song_dir / f"{name}{ext}").exists() for ext in (".wav", ".flac"))


def song_dirs(split_dir: Path) -> list[Path]:
    return sorted([p for p in split_dir.iterdir() if p.is_dir()]) if split_dir.exists() else []


def inspect_audio(path: Path) -> tuple[int, int]:
    info = sf.info(path)
    return int(info.samplerate), int(info.channels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    for split, expected in (("train", 100), ("test", 50)):
        split_songs = song_dirs(root / split)
        print(f"{split}: {len(split_songs)} songs; official expected {expected}")
        missing = []
        bad_audio = []
        for song in split_songs:
            required_ok = all(has_file(song, stem) for stem in STEMS) and has_vocal(song)
            if not required_ok:
                missing.append(song.name)
                continue
            mix = next((song / f"mixture{ext}" for ext in (".wav", ".flac") if (song / f"mixture{ext}").exists()))
            sr, channels = inspect_audio(mix)
            if sr != 44100 or channels != 2:
                bad_audio.append((song.name, sr, channels))

        if missing:
            print(f"  missing stems: {len(missing)}")
            for name in missing[:10]:
                print(f"    {name}")
        else:
            print("  all song folders have mixture/bass/drums/other/vocal(s)")

        if bad_audio:
            print(f"  bad sample-rate/channel examples: {bad_audio[:10]}")
        else:
            print("  checked mixtures are stereo 44.1 kHz")


if __name__ == "__main__":
    main()
