import torch
from scipy.stats import linregress


def r_squared(p, y):
    _, _, r, _, _ = linregress(p.numpy(), y.numpy())
    return r**2

def err10(p, y):
    err = torch.abs(p - y)
    err10 = torch.sum(torch.where(err <= 0.1 * y, 1, 0)) / len(y)
    return err10

def accuracy(p, y):
    pref = 1 / (len(y))
    return pref * torch.sum(p.int() == y.int())
