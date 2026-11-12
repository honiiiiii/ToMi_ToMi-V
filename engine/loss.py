"""Joint pointwise and spectral training objective."""
import torch


def forecasting_loss(prediction, target, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    error = prediction - target
    time_loss = error.abs().mean() + error.square().mean()
    spectral_loss = (torch.fft.rfft(prediction, dim=1)
                     - torch.fft.rfft(target, dim=1)).abs().mean()
    return (1 - alpha) * time_loss + alpha * spectral_loss
