import torch

from .callbacks import NGCallback
from .evalmeasure import accuracy, err10, r_squared
from .visualization import VisCP2D

rank_fun = lambda x: torch.argsort(torch.argsort(x, 1), 1)
neighbourhood_fun = lambda x, l: torch.exp(- rank_fun(x) / l)