"""ToMi-V: temporal token mixing with cross-variable interaction."""
from .base import ForecastModel


class ToMiV(ForecastModel):
    def __init__(self, config):
        super().__init__(config, cross_variable=True)
