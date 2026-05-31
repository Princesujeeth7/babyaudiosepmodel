import argparse, json, random, sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from transformers import AutoModel

# Allow the script to run both on Kaggle (/kaggle/working/project) and from
# this repository layout (office_submission_minimal/code/scripts).
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
for path in (CODE_DIR, PROJECT_ROOT, Path("/kaggle/working/project")):
    if path.exists():
        sys.path.insert(0, str(path))

from student.rope_unet import RopeReplacementUNet
from student.teacher_features import waveform_to_teacher_features, teacher_rope_target, student_features_to_stems

STEMS = ("bass", "drums", "other", "vocal")
# Extra weight is given to bass and other because they were the weakest stems
# during evaluation. The order is bass, drums, other, vocal.
STEM_WEIGHTS = torch.tensor([1.5, 1.0, 1.7, 1.1]).view(1, 4, 1, 1)


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_audio(song_dir, names):
    for name in names:
        for ext in (".wav", ".flac"):
            p = song_dir / f"{name}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Missing {names} in {song_dir}")


def find_songs(root):
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_dir() and ((p / "mixture.wav").exists() or (p / "mixture.flac").exists())
    )


def read_segment(path, start, frames):
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


def stem_paths(song_dir):
    return {
        "bass": find_audio(song_dir, ("bass", "target_bass")),
        "drums": find_audio(song_dir, ("drums", "target_drums")),
        "other": find_audio(song_dir, ("other", "target_other")),
        "vocal": find_audio(song_dir, ("vocals", "vocal", "target_vocals", "target_vocal")),
    }


class Sampler:
    def __init__(self, songs, chunk_size, remix_prob=0.2):
        self.chunk_size = chunk_size
        self.remix_prob = remix_prob
        self.items = []
        for song in songs:
            mix = find_audio(song, ("mixture",))
            info = sf.info(mix)
            self.items.append({"mix": mix, "stems": stem_paths(song), "frames": info.frames})

    def sample_one(self):
        # Remix augmentation creates extra mixtures by combining stems from
        # different songs. This increases diversity without extra datasets.
        if random.random() < self.remix_prob:
            stems = []
            for stem in STEMS:
                item = random.choice(self.items)
                start = random.randint(0, max(0, item["frames"] - self.chunk_size))
                stems.append(read_segment(item["stems"][stem], start, self.chunk_size))
            stems = torch.stack(stems, 0)
            gains = torch.empty(4, 1, 1).uniform_(0.85, 1.15)
            stems = stems * gains
            mix = stems.sum(0).clamp(-1, 1)
            return mix, stems

        item = random.choice(self.items)
        start = random.randint(0, max(0, item["frames"] - self.chunk_size))
        mix = read_segment(item["mix"], start, self.chunk_size)
        stems = torch.stack([read_segment(item["stems"][s], start, self.chunk_size) for s in STEMS], 0)
        gain = random.uniform(0.9, 1.1)
        mix = mix * gain
        stems = stems * gain
        if random.random() < 0.5:
            mix = mix.flip(0)
            stems = stems.flip(1)
        return mix, stems

    def batch(self, batch_size):
        mixes, stems = zip(*(self.sample_one() for _ in range(batch_size)))
        return torch.stack(mixes, 0), torch.stack(stems, 0)


def weighted_l1(pred, target, weights):
    return ((pred - target).abs() * weights.to(pred.device)).mean()


def si_sdr_loss(pred, target, weights, eps=1e-8):
    """Negative stem-weighted SI-SDR; minimizing it maximizes SI-SDR."""
    pred = pred.float() - pred.float().mean(dim=-1, keepdim=True)
    target = target.float() - target.float().mean(dim=-1, keepdim=True)
    proj = (pred * target).sum(dim=-1, keepdim=True) * target / (target.pow(2).sum(dim=-1, keepdim=True) + eps)
    noise = pred - proj
    ratio = (proj.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps)
    sisdr = 10 * torch.log10(ratio + eps)
    sisdr = sisdr.mean(dim=-1)
    w = weights.view(1, 4).to(pred.device)
    return -(sisdr * w).mean()


def mrstft_loss(pred, target):
    """Magnitude multi-resolution STFT loss over all stems/channels."""
    pred = pred.reshape(-1, pred.shape[-1]).float()
    target = target.reshape(-1, target.shape[-1]).float()
    total = pred.new_tensor(0.0)
    for n_fft, hop in ((1024, 256), (2048, 512), (4096, 1024)):
        window = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)
        p = torch.stft(pred, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        t = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
        total = total + F.l1_loss(torch.log1p(p.abs()), torch.log1p(t.abs()))
    return total / 3.0


def decode_mask_and_stems(model, hidden, raw_audio):
    """
    Returns:
      mask: [B, 4, 2, F, T, 2]
      stems: [B, 4, 2, samples]
    """
    freq_model = model.freq_domain_model
    b, c, length = raw_audio.shape

    # This mirrors the teacher decoder path after the RoPE stack: final norm,
    # optional time deconvolution, mask estimation, complex mask application,
    # and iSTFT back to waveform stems.
    h = freq_model.final_norm(hidden)
    if freq_model.time_conv_length is not None:
        h = freq_model.time_deconv(h)
        h = rearrange(h, "b t n (d tc) -> b (t tc) n d", tc=freq_model.time_conv_length)

    with torch.autocast(device_type=raw_audio.device.type, enabled=False):
        packed = rearrange(raw_audio.float(), "b c t -> (b c) t")
        stft = torch.stft(
            packed,
            **model.stft_out_kwargs,
            window=torch.hann_window(
                model.stft_out_kwargs.get("win_length") or model.stft_out_kwargs["n_fft"],
                periodic=True,
                device=raw_audio.device,
                dtype=torch.float32,
            ),
            return_complex=True,
        )
        stft_real = torch.view_as_real(stft)

    t_frames = stft_real.shape[-2]
    h = h[:, :t_frames, :, :]

    mask = torch.stack([fn(h) for fn in freq_model.mask_estimators], dim=1)
    mask = rearrange(mask, "b n t (f c z) -> b n c f t z", z=2, c=c).float()

    with torch.autocast(device_type=raw_audio.device.type, enabled=False):
        stft_expanded = rearrange(stft_real, "(b c) f t z -> b 1 c f t z", b=b, c=c)
        masked = torch.view_as_complex(stft_expanded) * torch.view_as_complex(mask)
        masked = rearrange(masked, "b n c f t -> (b n c) f t")
        audio = torch.istft(
            masked,
            **model.stft_out_kwargs,
            window=torch.hann_window(
                model.stft_out_kwargs.get("win_length") or model.stft_out_kwargs["n_fft"],
                periodic=True,
                device=raw_audio.device,
                dtype=torch.float32,
            ),
            return_complex=False,
            length=length,
        )
        audio = rearrange(audio, "(b n c) t -> b n c t", b=b, n=model.config.num_stems, c=c)

    return mask, audio


@torch.no_grad()
def validate(teacher, student, sampler, args):
    student.eval()
    vals = []
    weights = STEM_WEIGHTS.to(args.device)
    for _ in tqdm(range(args.val_steps), desc="val"):
        mix, gt = sampler.batch(args.batch_size)
        mix = mix.to(args.device)
        gt = gt.to(args.device)
        x = waveform_to_teacher_features(teacher, mix)
        h = student(x)
        _, pred = decode_mask_and_stems(teacher, h, mix)
        vals.append(float(weighted_l1(pred, gt, weights).cpu()))
    student.train()
    return sum(vals) / max(1, len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--init-checkpoint", required=True)
    ap.add_argument("--run-dir", default="/kaggle/working/runs/phase8_v1_mask_distill")
    ap.add_argument("--train-songs", type=int, default=90)
    ap.add_argument("--val-songs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--val-steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--chunk-seconds", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--remix-prob", type=float, default=0.2)
    ap.add_argument("--wave-weight", type=float, default=1.0)
    ap.add_argument("--stft-weight", type=float, default=0.2)
    ap.add_argument("--sisdr-weight", type=float, default=0.03)
    ap.add_argument("--hidden-weight", type=float, default=0.05)
    ap.add_argument("--teacher-stem-weight", type=float, default=0.2)
    ap.add_argument("--mask-weight", type=float, default=0.2)
    ap.add_argument("--mixture-weight", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    random.seed(4321)
    torch.manual_seed(4321)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    teacher = AutoModel.from_pretrained(
        str(Path(args.project) / "teacher_model"),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(args.device).float().eval()

    for p in teacher.parameters():
        p.requires_grad_(False)

    songs = find_songs(Path(args.dataset_root))
    train_songs = songs[:args.train_songs]
    val_songs = songs[args.train_songs:args.train_songs + args.val_songs]
    chunk_size = int(args.chunk_seconds * int(teacher.config.wave_sample_rate))

    train_sampler = Sampler(train_songs, chunk_size, remix_prob=args.remix_prob)
    val_sampler = Sampler(val_songs, chunk_size, remix_prob=0.0)

    student = RopeReplacementUNet(hidden_size=int(teacher.config.hidden_size)).to(args.device)
    ckpt = torch_load(args.init_checkpoint, args.device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    student.load_state_dict(state)
    student.train()

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = STEM_WEIGHTS.to(args.device)

    print("student params:", sum(p.numel() for p in student.parameters()))
    print("stem weights:", STEM_WEIGHTS.view(-1).tolist())

    best = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"epoch {epoch}")

        for _ in pbar:
            mix, gt = train_sampler.batch(args.batch_size)
            mix = mix.to(args.device)
            gt = gt.to(args.device)

            opt.zero_grad(set_to_none=True)

            x = waveform_to_teacher_features(teacher, mix)
            h_student = student(x)
            mask_student, pred = decode_mask_and_stems(teacher, h_student, mix)

            loss = pred.new_tensor(0.0)
            loss = loss + args.wave_weight * weighted_l1(pred, gt, weights)
            loss = loss + args.stft_weight * mrstft_loss(pred, gt)
            loss = loss + args.sisdr_weight * si_sdr_loss(pred, gt, weights)
            loss = loss + args.mixture_weight * F.l1_loss(pred.sum(dim=1), mix)

            with torch.inference_mode():
                h_teacher = teacher_rope_target(teacher, x)
                mask_teacher, teacher_stems = decode_mask_and_stems(teacher, h_teacher, mix)

            loss = loss + args.hidden_weight * F.l1_loss(h_student, h_teacher)
            loss = loss + args.teacher_stem_weight * weighted_l1(pred, teacher_stems, weights)
            loss = loss + args.mask_weight * F.l1_loss(mask_student, mask_teacher)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            lv = float(loss.detach().cpu())
            running += lv
            pbar.set_postfix(loss=f"{lv:.4f}")

        train_loss = running / max(1, args.steps_per_epoch)
        val_loss = validate(teacher, student, val_sampler, args)

        print(f"epoch {epoch}: train {train_loss:.6f} | val {val_loss:.6f}")

        ckpt = {
            "model": student.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_variant": "v1",
            "hidden_size": int(teacher.config.hidden_size),
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / "last_audio.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, run_dir / "best_audio.pt")
            print(f"new best: {best:.6f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

if __name__ == "__main__":
    main()
