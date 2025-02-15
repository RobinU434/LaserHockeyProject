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
    return {k: v.to(device) for k, v in state_dict.items()}


def state_dict_to_cpu(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: v.cpu() for k, v in state_dict.items()}


def state_dict_to_cuda(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: v.cuda() for k, v in state_dict.items()}
