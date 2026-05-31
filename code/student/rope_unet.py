import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseTFBlock(nn.Module):
    """Lightweight time-frequency convolution block.

    The time branch models frame-to-frame evolution inside each band, and the
    frequency branch models nearby band relationships. Depthwise convolutions
    keep the parameter count low.
    """

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(1, channels)
        self.time_conv = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.freq_conv = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        # Axial time/frequency branches replace the dependency modeling that
        # the original teacher performed with RoPE Transformer attention.
        x = self.time_conv(x) + self.freq_conv(x)
        x = self.pointwise(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        ]
        layers.extend(DepthwiseTFBlock(out_channels) for _ in range(blocks))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, blocks: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)
        self.blocks = nn.Sequential(*(DepthwiseTFBlock(out_channels) for _ in range(blocks)))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.blocks(x)


class RopeReplacementUNet(nn.Module):
    """
    Replaces the expensive BS-RoFormer axial Transformer stack.

    Input/output shape is [batch, time, bands, hidden], matching the teacher
    feature interface after band split and optional time compression.
    """

    def __init__(
        self,
        hidden_size: int = 384,
        widths: tuple[int, ...] = (96, 160, 256, 384),
        blocks_per_level: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input = nn.Sequential(
            nn.Conv2d(hidden_size, widths[0], kernel_size=1),
            nn.GELU(),
            *(DepthwiseTFBlock(widths[0]) for _ in range(blocks_per_level)),
        )
        self.downs = nn.ModuleList(
            DownBlock(widths[i], widths[i + 1], blocks_per_level)
            for i in range(len(widths) - 1)
        )
        self.mid = nn.Sequential(*(DepthwiseTFBlock(widths[-1]) for _ in range(blocks_per_level)))
        self.ups = nn.ModuleList(
            UpBlock(widths[i + 1], widths[i], widths[i], blocks_per_level)
            for i in reversed(range(len(widths) - 1))
        )
        self.output = nn.Conv2d(widths[0], hidden_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.permute(0, 3, 1, 2).contiguous()

        skips = []
        x = self.input(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.mid(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        x = self.output(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x + residual


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
