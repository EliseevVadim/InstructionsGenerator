import gc
import torch

import numpy as np
import random


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def init_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
