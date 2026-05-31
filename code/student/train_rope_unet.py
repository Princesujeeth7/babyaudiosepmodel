import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from student.rope_unet import RopeReplacementUNet, count_parameters


class FeatureCacheDataset(Dataset):
    def __init__(self, cache_dir: str):
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"No .pt feature files found in {cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        item = torch.load(self.files[idx], map_location="cpu")
        return {
            "student_input": item["student_input"].float(),
            "teacher_target": item["teacher_target"].float(),
            "song": item.get("song", ""),
            "chunk_index": item.get("chunk_index", -1),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir", default="runs/rope_unet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = FeatureCacheDataset(args.cache_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    first = dataset[0]["student_input"]
    hidden_size = first.shape[-1]
    model = RopeReplacementUNet(hidden_size=hidden_size).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))

    print(f"feature files: {len(dataset)}")
    print(f"hidden size: {hidden_size}")
    print(f"student params: {count_parameters(model):,}")

    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            x = batch["student_input"].to(args.device, non_blocking=True)
            y = batch["teacher_target"].to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp and args.device.startswith("cuda")):
                pred = model(x)
                loss_l1 = F.l1_loss(pred, y)
                loss_mse = F.mse_loss(pred, y)
                loss = loss_l1 + 0.1 * loss_mse

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach().cpu())
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.5f}")

        epoch_loss = running / max(1, len(loader))
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"epoch {epoch} loss {epoch_loss:.6f}")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": epoch_loss,
            "hidden_size": hidden_size,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if epoch_loss < best:
            best = epoch_loss
            torch.save(checkpoint, run_dir / "best.pt")
        if args.save_every and epoch % args.save_every == 0:
            torch.save(checkpoint, run_dir / f"epoch_{epoch:04d}.pt")

        with open(run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
