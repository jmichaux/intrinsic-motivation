import random
import numpy as np
import torch

def set_random_seed(seed, debug=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if debug:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
