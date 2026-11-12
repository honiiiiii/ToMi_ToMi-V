"""Multi-head temporal token mixing."""
import math

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F


class RegionMixer(nn.Module):
    """Learned mixing across temporal regions, shared across feature heads."""

    def __init__(self, num_regions):
        super().__init__()
        index = torch.arange(num_regions, dtype=torch.float32)
        distance = torch.abs(index[:, None] - index[None, :])
        if num_regions == 1:
            weights = torch.ones(1, 1)
        else:
            max_distance = distance.max().clamp_min(1.0)
            weights = torch.cos(distance / max_distance * math.pi / 4)
            weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
        self.conv = nn.Conv2d(num_regions, num_regions, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weights.unsqueeze(-1).unsqueeze(-1))

    def forward(self, x):
        return self.conv(x)


class MultiHeadTokenMixer(nn.Module):
    """Input: [batch*variables, regions, window_size, d_model]."""

    def __init__(self, dim, num_regions, heads, dropout, local_kernel_size, eps=1e-6):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.local_conv = nn.Conv2d(
            dim, dim, kernel_size=local_kernel_size,
            stride=1, padding=local_kernel_size // 2, groups=dim)
        self.region_mixer = RegionMixer(num_regions)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        batch_size = x.shape[0]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k = F.relu(q) + self.eps, F.relu(k) + self.eps
        local = self.local_conv(rearrange(v, 'b n w d -> b d n w'))
        local = rearrange(local, 'b d n w -> b n w d')
        q, k, v = (
            rearrange(t, 'b n w (h d) -> (b h) n w d',
                      h=self.num_heads, d=self.head_dim)
            for t in (q, k, v))
        key_transpose = k.transpose(-2, -1)
        summaries = self.region_mixer(torch.matmul(key_transpose, v))
        key_sum = key_transpose.sum(dim=-1, keepdim=True)
        normalizer = self.region_mixer(torch.matmul(q, key_sum)) + self.eps
        mixed = torch.matmul(q, summaries) / normalizer
        mixed = rearrange(mixed, '(b h) n w d -> b n w (h d)',
                          b=batch_size, h=self.num_heads)
        return self.to_out(mixed + local)


class TemporalBlock(nn.Module):
    """Mix temporal features independently for each variable."""

    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.window_size = config.window_size
        self.num_regions = math.ceil(self.seq_len / self.window_size)
        self.padded_len = self.num_regions * self.window_size
        self.input_projection = nn.Linear(config.embed_size, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.activation = nn.GELU()
        self.mixer = MultiHeadTokenMixer(
            config.d_model, self.num_regions, config.n_heads, config.dropout,
            config.local_kernel_size)
        self.output_projection = nn.Linear(config.d_model, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, channels, length, _ = x.shape
        hidden = self.activation(self.input_norm(self.input_projection(x)))
        if self.padded_len > length:
            hidden = F.pad(hidden, (0, 0, 0, self.padded_len - length))
        hidden = rearrange(hidden, 'b c (n w) d -> (b c) n w d',
                           n=self.num_regions, w=self.window_size)
        hidden = self.mixer(hidden)
        hidden = rearrange(hidden, '(b c) n w d -> b c (n w) d',
                           b=batch, c=channels, n=self.num_regions, w=self.window_size)
        hidden = self.output_projection(hidden[:, :, :length, :])
        return self.dropout(hidden + x)
