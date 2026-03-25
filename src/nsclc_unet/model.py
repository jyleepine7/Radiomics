from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsclc_unet.config import ModelConfig


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        base = config.base_channels
        dropout = config.dropout

        self.embedding_pool = config.embedding_pool
        self.inc = DoubleConv(config.in_channels, base, dropout=dropout)
        self.down1 = DownBlock(base, base * 2, dropout=dropout)
        self.down2 = DownBlock(base * 2, base * 4, dropout=dropout)
        self.down3 = DownBlock(base * 4, base * 8, dropout=dropout)
        self.bottleneck = DownBlock(base * 8, base * 16, dropout=dropout)

        self.up1 = UpBlock(base * 16, base * 8, base * 8, dropout=dropout)
        self.up2 = UpBlock(base * 8, base * 4, base * 4, dropout=dropout)
        self.up3 = UpBlock(base * 4, base * 2, base * 2, dropout=dropout)
        self.up4 = UpBlock(base * 2, base, base, dropout=dropout)
        self.out_head = nn.Conv2d(base, config.out_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bottleneck(x4)
        return xb, (x1, x2, x3, x4)

    def decode(self, bottleneck: torch.Tensor, skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4 = skips
        x = self.up1(bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_head(x)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encode(x)
        avg_pool = F.adaptive_avg_pool2d(bottleneck, output_size=1).flatten(start_dim=1)
        if self.embedding_pool == "avg_max":
            max_pool = F.adaptive_max_pool2d(bottleneck, output_size=1).flatten(start_dim=1)
            return torch.cat([avg_pool, max_pool], dim=1)
        return avg_pool

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bottleneck, skips = self.encode(x)
        logits = self.decode(bottleneck, skips)
        if return_embedding:
            embedding = self.extract_embedding(x)
            return logits, embedding
        return logits

