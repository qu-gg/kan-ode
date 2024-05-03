"""
@file metrics.py

Holds a variety of metric computing functions for time-series forecasting models
"""
import torch


def recon_mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for the given length of timesteps used for training """
    full_pixel_mses = (output[:, :kwargs['args'].timesteps] - target[:, :kwargs['args'].timesteps]) ** 2
    sequence_pixel_mse = torch.mean(full_pixel_mses, dim=(1, 2))
    return sequence_pixel_mse, torch.mean(sequence_pixel_mse), torch.std(sequence_pixel_mse)


def recon_mape(output, target, **kwargs):
    mape_top = torch.abs(output - target)
    mape_bot = torch.abs(target).nanmean()
    mape = mape_top / mape_bot
    return mape, torch.mean(mape), torch.std(mape)
