# convert all numpy arrays in dataSpec to torch tensors
import sys
import torch
import numpy as np


def to_tensor(var, device):
    if isinstance(var, np.ndarray):
        return torch.from_numpy(var).float().to(device)
    elif isinstance(var, (float, int)):
        return torch.tensor(var).float().to(device)
    else:
        raise ValueError(f"Unsupported variable type: {type(var)}")


def array2tensor(current_module, dirs):
    """
    Convert all numpy arrays in dataSpec or SAILdata to torch tensors
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for name in dirs:
        attr = getattr(current_module, name)
        if isinstance(attr, (np.ndarray, float, int)):
            # If the object is a numpy array, float, or int, convert it to a tensor
            setattr(current_module, name, to_tensor(attr, device))
