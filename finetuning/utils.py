import torch
from finetuning.model import PROiFHESi
import numpy as np 
from typing import Any

def print_attrs(name, obj):
    print(f"Object: {name}")
    for key, value in obj.attrs.items():
        print(f"    Attribute - {key}: {value}")


def print_dict(dict, name=None):
    if name:
        print(f"In {name}: ")
    else:
        print(f"In Dictionary: ")
    for key, value in dict.items():
        print(f"    {key}: {value}")


def is_in_args(args, name, default):
    """Checks if the parammeter is specified in the args Namespace
    If not, attributes him the default value
    """
    if name in args:
        para = getattr(args, name)
    else:
        para = default
    return para

def load_model_from_path(model_path, device):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    args.device = device
    model = PROiFHESi(
        args,
        target_names=checkpoint.get("target_names"),
    )
    model.network.load_state_dict(checkpoint["model_state_dict"])
    return model


def to_numpy(value: Any) -> np.ndarray:
    """Convert a tensor or array-like object to a NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)
