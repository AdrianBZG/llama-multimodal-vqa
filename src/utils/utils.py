import os
from datetime import datetime
import torch
import random
import numpy as np

GLOBAL_SEED = 725612253


def get_available_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"  # Many issues with MPS so will force CPU here for now
    else:
        device = "cpu"

    return device


def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_save_folder(save_dir):
    """ Make save folder. If no name is given, create a folder with the current date and time."""
    save_name = datetime.now().strftime("%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path
