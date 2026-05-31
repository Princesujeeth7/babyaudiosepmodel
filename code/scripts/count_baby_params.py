from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from student.rope_unet import RopeReplacementUNet, count_parameters


def main() -> None:
    bundle = Path(__file__).resolve().parents[2]
    ckpt_path = bundle / "student_model" / "final_student_phase8_weights_only.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    hidden_size = int(ckpt.get("hidden_size", 384)) if isinstance(ckpt, dict) else 384
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = RopeReplacementUNet(hidden_size=hidden_size)
    model.load_state_dict(state)

    print(f"student hidden size: {hidden_size}")
    print(f"student trainable parameters: {count_parameters(model):,}")
    print(f"student weights file: {ckpt_path}")
    print(f"student weights size MB: {ckpt_path.stat().st_size / (1024 * 1024):.2f}")


if __name__ == "__main__":
    main()
