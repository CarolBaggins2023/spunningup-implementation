import numpy as np
from typing import Tuple
import time
import os


def combined_shape(length: int, dim: [int, Tuple]) -> Tuple:
    # This function aims to return the right shape of observation and action buffer.
    # Because dim(of observation or action) may be a tuple, so directly executing np.zeros(length, dim)
    # to construct buffer can lead to a wrong shape of (int, tuple(int)), while what in need is (int, int, ...).
    if dim is None:
        return tuple([length, ])
    elif np.isscalar(dim):
        return tuple([length, dim])
    else:
        return tuple([length, *dim])


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    cumsum = 0
    cumsum_list = list()
    for elem in x[::-1]:
        cumsum = cumsum * discount + elem
        cumsum_list.append(cumsum)
    cumsum_list.reverse()
    return np.array(cumsum_list)


def count_variables(module) -> int:
    return sum([np.prod(p.shape) for p in module.parameters()])


def setup_logger_kwargs(exp_name: str, seed: int, data_dir: str):
    # Make a seed-specific subfolder in the experiment directory.
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
    
    logger_kwargs = dict(output_dir=os.path.join(data_dir, subfolder), exp_name=exp_name)
    return logger_kwargs
