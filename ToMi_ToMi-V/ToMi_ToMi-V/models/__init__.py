from .config import ModelConfig
from .tomi import ToMi
from .tomi_v import ToMiV

MODELS = {'tomi': ToMi, 'tomi-v': ToMiV}

__all__ = ['ModelConfig', 'ToMi', 'ToMiV', 'MODELS']
