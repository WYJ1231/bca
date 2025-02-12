import os.path
from typing import Union

import torch
from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from jhammer.log import log

MODEL = "model"
OPTIMIZER = "optimizer"


def get_unwrapped_model(model: nn.Module):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    return model


def save_checkpoint(file, model: nn.Module, optimizer: Union[None, Optimizer] = None, **kwargs):
    checkpoint = {MODEL: get_unwrapped_model(model).state_dict()}
    if optimizer:
        checkpoint[OPTIMIZER] = optimizer.state_dict()
    for k, v in kwargs.items():
        if k in checkpoint:
            raise KeyError(f"Get duplicated key {k}.")
        checkpoint[k] = v
    file_path = os.path.split(file)[0]
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    torch.save(checkpoint, file)


def load_checkpoint(file, model: Union[nn.Module, None] = None, optimizer: Union[Optimizer, None] = None):
    log.info(f"Loading checkpoint {file}.")
    checkpoint = torch.load(file)
    if model:
        if MODEL not in checkpoint:
            log.warning(f"Checkpoint {file} does not include model state dict.")
        else:
            log.info("Loading model state.")
            model = get_unwrapped_model(model)
            model.load_state_dict(checkpoint[MODEL])

    if optimizer:
        if OPTIMIZER not in checkpoint:
            log.warning(f"Checkpoint {file} does not include optimizer state dict.")
        else:
            log.info(f"Loading optimizer state.")
            optimizer.load_state_dict(checkpoint[OPTIMIZER])
    return checkpoint
