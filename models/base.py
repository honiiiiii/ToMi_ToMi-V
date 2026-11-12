"""Shared embedding, temporal encoding, and forecasting path."""
import torch
from torch import nn

from layers.revin import RevIN
from .temporal import TemporalBlock
from .variable import LowRankVariableMixer


class ForecastModel(nn.Module):
    def __init__(self, config, cross_variable):
        super().__init__()
        self.config = config
        self.revin_layer = RevIN(config.enc_in, affine=True, subtract_last=False)
        self.embeddings = nn.Parameter(torch.randn(1, config.embed_size))
        self.encoder = nn.ModuleList(
            [TemporalBlock(config) for _ in range(config.num_hidden_layers)])
        if cross_variable:
            self.channel_mixer = LowRankVariableMixer(
                config.enc_in, config.channel_mixer_rank,
                config.channel_mixer_dropout, config.channel_mixer_gate_init)
        else:
            self.channel_mixer = nn.Identity()
        self.feature_projection = nn.Linear(config.embed_size * config.seq_len, config.d_model)
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff), nn.LeakyReLU(),
            nn.Linear(config.d_ff, config.pred_len))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.config.seq_len, self.config.enc_in):
            raise ValueError('Expected input [batch, seq_len, enc_in] matching ModelConfig')
        x = self.revin_layer(x, 'norm')
        x = x.transpose(-1, -2).unsqueeze(-1)
        if self.config.embed_size > 1:
            x = x * self.embeddings
        for block in self.encoder:
            x = block(x)
        x = self.channel_mixer(x)
        batch, channels, length, embedding = x.shape
        x = self.feature_projection(x.reshape(batch, channels, length * embedding))
        x = self.prediction_head(x).permute(0, 2, 1)
        return self.revin_layer(x, 'denorm')
