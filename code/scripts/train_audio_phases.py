import argparse
import json
import random
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
from student.teacher_features import student_features_to_stems, teacher_rope_target, waveform_to_teacher_features


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
    songs = [
        p for p in root.rglob("*")
        if p.is_dir() and ((p / "mixture.wav").exists() or (p / "mixture.flac").exists())
    ]
    return sorted(songs)


def read_segment(path: Path, start: int, frames: int) -> torch.Tensor:
    info = sf.info(path)
    start = max(0, min(start, max(0, info.frames - 1)))
    audio, _ = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    x = torch.from_numpy(audio).transpose(0, 1).contiguous()
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    elif x.shape[0] > 2:
        x = x[:2]
    if x.shape[-1] < frames:
        x = F.pad(x, (0, frames - x.shape[-1]))
    return x


def stem_paths(song_dir: Path) -> dict[str, Path]:
    return {
        "bass": find_audio(song_dir, ("bass", "target_bass")),
        "drums": find_audio(song_dir, ("drums", "target_drums")),
        "other": find_audio(song_dir, ("other", "target_other")),
        "vocal": find_audio(song_dir, ("vocals", "vocal", "target_vocals", "target_vocal")),
    }


class AudioSampler:
    def __init__(self, songs: list[Path], sample_rate: int, chunk_size: int, remix_prob: float = 0.0):
        self.songs = songs
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.remix_prob = remix_prob
        self.meta = []
        for song in songs:
            mix = find_audio(song, ("mixture",))
            info = sf.info(mix)
            self.meta.append({"song": song, "mix": mix, "stems": stem_paths(song), "frames": info.frames})

    def sample_one(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.remix_prob > 0 and random.random() < self.remix_prob:
            stem_audio = []
            for stem in STEMS:
                item = random.choice(self.meta)
                start = random.randint(0, max(0, item["frames"] - self.chunk_size))
                stem_audio.append(read_segment(item["stems"][stem], start, self.chunk_size))
            stems = torch.stack(stem_audio, dim=0)
            gains = torch.empty(4, 1, 1).uniform_(0.75, 1.25)
            stems = stems * gains
            mix = stems.sum(dim=0).clamp(-1.0, 1.0)
            return mix, stems

        item = random.choice(self.meta)
        start = random.randint(0, max(0, item["frames"] - self.chunk_size))
        mix = read_segment(item["mix"], start, self.chunk_size)
        stems = torch.stack([read_segment(item["stems"][stem], start, self.chunk_size) for stem in STEMS], dim=0)
        gain = random.uniform(0.8, 1.2)
        mix = mix * gain
        stems = stems * gain
        if random.random() < 0.5:
            mix = mix.flip(0)
            stems = stems.flip(1)
        return mix, stems

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        mixes, stems = zip(*(self.sample_one() for _ in range(batch_size)))
        return torch.stack(mixes, dim=0), torch.stack(stems, dim=0)


def make_student(name: str, hidden_size: int):
    if name == "v1":
        return RopeReplacementUNet(hidden_size=hidden_size)
    if name == "v2":
        return AxialRopeUNetV2(hidden_size=hidden_size)
    raise ValueError(f"Unknown model variant: {name}")


def load_student_checkpoint(student, path: str, device: str):
    if not path:
        return
    ckpt = torch_load(Path(path), device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    current = student.state_dict()
    matched = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    student.load_state_dict(matched, strict=False)
    print(f"loaded {len(matched)}/{len(current)} matching tensors from {path}")


def mrstft_loss(pred: torch.Tensor, target: torch.Tensor, sample_rate: int) -> torch.Tensor:
    pred = pred.reshape(-1, pred.shape[-1]).float()
    target = target.reshape(-1, target.shape[-1]).float()
    total = pred.new_tensor(0.0)
    for n_fft, hop in ((1024, 256), (2048, 512), (4096, 1024)):
        window = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)
        p = torch.stft(pred, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        t = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        total = total + F.l1_loss(torch.log1p(p.abs()), torch.log1p(t.abs()))
    return total / 3.0


def forward_student(teacher, student, mix: torch.Tensor):
    x = waveform_to_teacher_features(teacher, mix)
    h = student(x)
    stems = student_features_to_stems(teacher, h, mix.shape[-1], mix)
    return x, h, stems


def train_step(args, teacher, student, optimizer, mix, gt):
    optimizer.zero_grad(set_to_none=True)
    student_input, pred_hidden, pred = forward_student(teacher, student, mix)

    loss = pred.new_tensor(0.0)
    parts = {}
    if args.wave_weight:
        parts["wave"] = F.l1_loss(pred, gt)
        loss = loss + args.wave_weight * parts["wave"]
    if args.stft_weight:
        parts["stft"] = mrstft_loss(pred, gt, int(teacher.config.wave_sample_rate))
        loss = loss + args.stft_weight * parts["stft"]
    if args.mixture_weight:
        mix_pred = pred.sum(dim=1)
        parts["mixture"] = F.l1_loss(mix_pred, mix)
        loss = loss + args.mixture_weight * parts["mixture"]

    need_teacher = args.hidden_weight > 0 or args.teacher_stem_weight > 0
    if need_teacher:
        with torch.inference_mode():
            teacher_hidden = teacher_rope_target(teacher, student_input)
            teacher_stems = student_features_to_stems(teacher, teacher_hidden, mix.shape[-1], mix)
        if args.hidden_weight:
            parts["hidden"] = F.l1_loss(pred_hidden, teacher_hidden)
            loss = loss + args.hidden_weight * parts["hidden"]
        if args.teacher_stem_weight:
            parts["teacher_stem"] = F.l1_loss(pred, teacher_stems)
            loss = loss + args.teacher_stem_weight * parts["teacher_stem"]

    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
    optimizer.step()
    return float(loss.detach().cpu()), {k: float(v.detach().cpu()) for k, v in parts.items()}


@torch.no_grad()
def validate(args, teacher, student, sampler):
    student.eval()
    losses = []
    for _ in tqdm(range(args.val_steps), desc="val"):
        mix, gt = sampler.sample_batch(args.batch_size)
        mix = mix.to(args.device)
        gt = gt.to(args.device)
        _, _, pred = forward_student(teacher, student, mix)
        loss = F.l1_loss(pred, gt)
        losses.append(float(loss.cpu()))
    student.train()
    return sum(losses) / max(1, len(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--dataset-root", required=True, help="MUSDB-HQ train folder.")
    parser.add_argument("--run-dir", default="/kaggle/working/runs/audio_phases")
    parser.add_argument("--model-variant", choices=("v1", "v2"), default="v1")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--train-songs", type=int, default=80)
    parser.add_argument("--val-songs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--val-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-seconds", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--remix-prob", type=float, default=0.0)
    parser.add_argument("--wave-weight", type=float, default=1.0)
    parser.add_argument("--stft-weight", type=float, default=0.2)
    parser.add_argument("--hidden-weight", type=float, default=0.05)
    parser.add_argument("--teacher-stem-weight", type=float, default=0.0)
    parser.add_argument("--mixture-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(1234)
    torch.manual_seed(1234)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    teacher_dir = Path(args.project) / "teacher_model"
    teacher = AutoModel.from_pretrained(str(teacher_dir), trust_remote_code=True, torch_dtype=torch.float32).to(args.device)
    teacher = teacher.float().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    sr = int(teacher.config.wave_sample_rate)
    chunk_size = int(args.chunk_seconds * sr)
    songs = find_songs(Path(args.dataset_root))
    if len(songs) < args.train_songs + args.val_songs:
        print(f"warning: only found {len(songs)} songs")
    train_songs = songs[:args.train_songs]
    val_songs = songs[args.train_songs:args.train_songs + args.val_songs]
    if not val_songs:
        val_songs = train_songs[-min(5, len(train_songs)):]

    student = make_student(args.model_variant, int(teacher.config.hidden_size)).to(args.device)
    load_student_checkpoint(student, args.init_checkpoint, args.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    train_sampler = AudioSampler(train_songs, sr, chunk_size, remix_prob=args.remix_prob)
    val_sampler = AudioSampler(val_songs, sr, chunk_size, remix_prob=0.0)

    print(f"variant: {args.model_variant}")
    print(f"student params: {sum(p.numel() for p in student.parameters()):,}")
    print(f"train songs: {len(train_songs)} | val songs: {len(val_songs)}")
    print(f"run dir: {run_dir}")

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        student.train()
        running = 0.0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"epoch {epoch}")
        for _ in pbar:
            mix, gt = train_sampler.sample_batch(args.batch_size)
            mix = mix.to(args.device)
            gt = gt.to(args.device)
            loss, parts = train_step(args, teacher, student, optimizer, mix, gt)
            running += loss
            pbar.set_postfix(loss=f"{loss:.4f}")

        train_loss = running / max(1, args.steps_per_epoch)
        val_loss = validate(args, teacher, student, val_sampler)
        print(f"epoch {epoch}: train {train_loss:.6f} | val {val_loss:.6f}")

        ckpt = {
            "model": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_variant": args.model_variant,
            "hidden_size": int(teacher.config.hidden_size),
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / "last_audio.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, run_dir / "best_audio.pt")
            print(f"new best: {best:.6f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
