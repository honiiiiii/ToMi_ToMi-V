"""ToMi: variable-independent temporal token mixing."""
from .base import ForecastModel


class ToMi(ForecastModel):
    def __init__(self, config):
        super().__init__(config, cross_variable=False)
