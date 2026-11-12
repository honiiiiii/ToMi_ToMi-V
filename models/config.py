"""Model configuration shared by ToMi and ToMi-V."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    window_size: int
    seq_len: int = 96
    pred_len: int = 96
    enc_in: int = 7
    embed_size: int = 16
    d_model: int = 128
    n_heads: int = 8
    num_hidden_layers: int = 1
    d_ff: int = 1024
    dropout: float = 0.1
    local_kernel_size: int = 5
    channel_mixer_rank: int = 4
    channel_mixer_dropout: float | None = None
    channel_mixer_gate_init: float = 0.0

    def __post_init__(self):
        for name in ('window_size', 'seq_len', 'pred_len', 'enc_in', 'embed_size',
                     'd_model', 'n_heads', 'num_hidden_layers', 'd_ff',
                     'local_kernel_size', 'channel_mixer_rank'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.window_size > self.seq_len:
            raise ValueError('window_size must not exceed seq_len')
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if self.local_kernel_size % 2 == 0:
            raise ValueError('local_kernel_size must be odd')
        if self.channel_mixer_dropout is None:
            self.channel_mixer_dropout = self.dropout
        for name in ('dropout', 'channel_mixer_dropout'):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f'{name} must be in [0, 1)')
