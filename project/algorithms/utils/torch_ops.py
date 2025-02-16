from typing import Dict

import torch
from torch import Tensor


def detach_tensor(*args):
    res = []
    for t in args:
        t.detach()
        res.append(t)
    return res


def state_dict_to(
    state_dict: Dict[str, Tensor], device: torch.device
) -> Dict[str, Tensor]:
    res = {}
    for k, v in state_dict.items():
        if isinstance(v, Tensor):
            v = v.to(device)
        elif isinstance(v, dict):
            v = state_dict_to(v, device)
        res[k] = v
    return res


def state_dict_to_cpu(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return state_dict_to(state_dict, torch.device("cpu"))


def state_dict_to_cuda(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return state_dict_to(state_dict, torch.device("cuda:0"))
