"""
@file utils.py

Utility functions across files
"""
import os
import numpy as np
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, inputs, targets):
        return (inputs - targets).abs()


def flatten_cfg(cfg: DictConfig):
    """ Utility function to flatten the primary submodules of a Hydra config """
    # Disable struct flag on the config
    OmegaConf.set_struct(cfg, False)

    # Loop through each item, merging with the main cfg if its another DictConfig
    for key, value in cfg.copy().items():
        if isinstance(value, DictConfig):
            cfg.merge_with(cfg.pop(key))

    return cfg


def get_model(name):
    """ Import and return the specific latent dynamics function by the given name"""
    # Lowercase name in case of misspellings
    name = name.lower()

    if name == "additive":
        from models.dynamics_functions.AdditiveModel import AdditiveModel
        return AdditiveModel

    if name == "kan_additive":
        from models.dynamics_functions.KANAdditiveModel import AdditiveModel
        return AdditiveModel

    if name == "kan_encoder_additive":
        from models.dynamics_functions.KANEncoderAdditiveModel import AdditiveModel
        return AdditiveModel

    # Given no correct model type, raise error
    raise NotImplementedError("Model type {} not implemented.".format(name))


def get_act(act="relu"):
    """
    Return torch function of a given activation function
    :param act: activation function
    :return: torch object
    """
    if act == "relu":
        return nn.ReLU()
    elif act == "leaky_relu":
        return nn.LeakyReLU(0.1)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "linear":
        return nn.Identity()
    elif act == 'softplus':
        return nn.modules.activation.Softplus()
    elif act == 'softmax':
        return nn.Softmax()
    elif act == "swish":
        return nn.SiLU()
    else:
        return None


def find_best_step(ckpt_folder):
    """
    Find the highest epoch in the Test Tube file structure.
    :param ckpt_folder: dir where the checpoints are being saved.
    :return: float of the highest epoch reached by the checkpoints.
    """
    best_ckpt = None
    best_step = None
    best = np.inf
    filenames = os.listdir(f"{ckpt_folder}/")
    for filename in filenames:
        if "last" in filename:
            continue

        test_value = float(filename[:-5].split("dst")[-1])
        test_step = int(filename.split('-')[0].replace('step', ''))
        if test_value < best:
            best = test_value
            best_ckpt = filename
            best_step = test_step

    return best_ckpt, best_step
