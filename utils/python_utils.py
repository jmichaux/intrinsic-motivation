import random
import numpy as np
impoort torch

def set_random_seed(seed=None, debug=False):
    if seed is None:
        seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if debug:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
