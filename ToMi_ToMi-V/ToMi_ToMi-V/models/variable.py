"""Gated low-rank cross-variable mixing."""
import torch
from torch import nn
from torch.nn import functional as F


class LowRankVariableMixer(nn.Module):
    """Input and output: [batch, variables, time, embedding]."""

    def __init__(self, channels, rank, dropout, gate_init):
        super().__init__()
        self.rank = min(rank, channels)
        self.down = nn.Parameter(torch.empty(channels, self.rank))
        self.up = nn.Parameter(torch.empty(self.rank, channels))
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.xavier_uniform_(self.down)
        nn.init.xavier_uniform_(self.up)

    def forward(self, x):
        mixed = torch.einsum('bcle,cr->brle', x, self.down)
        mixed = F.gelu(mixed)
        mixed = torch.einsum('brle,rc->bcle', mixed, self.up)
        return x + torch.tanh(self.gate) * self.dropout(mixed)
