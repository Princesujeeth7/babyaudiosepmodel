import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialGatedTFBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, time_dilation: int = 1):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.GroupNorm(1, channels)
        self.time_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(7, 1),
            padding=(3 * time_dilation, 0),
            dilation=(time_dilation, 1),
            groups=channels,
        )
        self.freq_conv = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, hidden * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        t = self.time_conv(x)
        f = self.freq_conv(x)
        x = self.fuse(torch.cat([t, f], dim=1))
        return residual + self.scale * x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, blocks: int):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), nn.GELU()]
        for i in range(blocks):
            layers.append(AxialGatedTFBlock(out_channels, time_dilation=2 ** (i % 3)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, blocks: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *(AxialGatedTFBlock(out_channels, time_dilation=2 ** (i % 3)) for i in range(blocks))
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.blocks(self.proj(x))


class AxialRopeUNetV2(nn.Module):
    """
    Stronger CNN/U-Net replacement for the BS-RoFormer RoPE Transformer stack.

    Input/output shape: [batch, time, bands, hidden].
    Each block has separate time-axis and frequency-axis branches, then fuses
    them with a gated 1x1 projection.
    """

    def __init__(
        self,
        hidden_size: int = 384,
        widths: tuple[int, ...] = (112, 192, 288, 384),
        blocks_per_level: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input = nn.Sequential(
            nn.Conv2d(hidden_size, widths[0], kernel_size=1),
            nn.GELU(),
            *(AxialGatedTFBlock(widths[0], time_dilation=2 ** (i % 3)) for i in range(blocks_per_level)),
        )
        self.downs = nn.ModuleList(
            DownBlock(widths[i], widths[i + 1], blocks_per_level) for i in range(len(widths) - 1)
        )
        self.mid = nn.Sequential(
            *(AxialGatedTFBlock(widths[-1], time_dilation=2 ** (i % 4)) for i in range(blocks_per_level + 1))
        )
        self.ups = nn.ModuleList(
            UpBlock(widths[i + 1], widths[i], widths[i], blocks_per_level)
            for i in reversed(range(len(widths) - 1))
        )
        self.output = nn.Conv2d(widths[0], hidden_size, kernel_size=1)
        self.out_scale = nn.Parameter(torch.tensor(0.2))

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
        x = self.output(x).permute(0, 2, 3, 1).contiguous()
        return residual + self.out_scale * x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
