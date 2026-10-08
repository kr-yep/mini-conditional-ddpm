from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return x + attended


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float):
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, emb_dim, dropout)
        self.res2 = ResBlock(out_channels, out_channels, emb_dim, dropout)
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, emb)
        x = self.res2(x, emb)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, emb_dim: int, dropout: float):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.res1 = ResBlock(out_channels + skip_channels, out_channels, emb_dim, dropout)
        self.res2 = ResBlock(out_channels, out_channels, emb_dim, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, emb)
        return self.res2(x, emb)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] | list[int] = (1, 2, 4),
        time_emb_dim: int = 256,
        class_emb_dim: int = 256,
        dropout: float = 0.1,
        use_attention: bool = False,
        attention_heads: int = 4,
    ):
        super().__init__()
        emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)
        self.class_proj = nn.Sequential(nn.SiLU(), nn.Linear(class_emb_dim, time_emb_dim))

        channels = [base_channels * m for m in channel_mults]
        self.init_conv = nn.Conv2d(image_channels, channels[0], 3, padding=1)

        downs = []
        in_ch = channels[0]
        self.skip_channels: list[int] = []
        for out_ch in channels:
            downs.append(DownBlock(in_ch, out_ch, emb_dim, dropout))
            self.skip_channels.append(out_ch)
            in_ch = out_ch
        self.downs = nn.ModuleList(downs)

        self.mid1 = ResBlock(channels[-1], channels[-1], emb_dim, dropout)
        self.mid_attn = SelfAttention2d(channels[-1], attention_heads) if use_attention else nn.Identity()
        self.mid2 = ResBlock(channels[-1], channels[-1], emb_dim, dropout)

        ups = []
        in_ch = channels[-1]
        for skip_ch in reversed(self.skip_channels):
            out_ch = skip_ch
            ups.append(UpBlock(in_ch, skip_ch, out_ch, emb_dim, dropout))
            in_ch = out_ch
        self.ups = nn.ModuleList(ups)

        self.out_norm = nn.GroupNorm(min(8, channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], image_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timesteps) + self.class_proj(self.class_emb(class_labels))
        x = self.init_conv(x)
        skips = []
        for down in self.downs:
            x, skip = down(x, emb)
            skips.append(skip)
        x = self.mid1(x, emb)
        x = self.mid_attn(x)
        x = self.mid2(x, emb)
        for up in self.ups:
            x = up(x, skips.pop(), emb)
        return self.out_conv(F.silu(self.out_norm(x)))
