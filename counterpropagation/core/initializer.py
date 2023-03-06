from typing import Callable

import torch
from prototorch.core.distances import squared_euclidean_distance
from prototorch.core.initializers import AbstractComponentsInitializer

from counterpropagation.utils import neighbourhood_fun


class ResponseLikeInitializer(AbstractComponentsInitializer):

    """

    Base Class to create components in the response-space.
    A corresponding generate funvtion needs to be created 
    for the respective response-space.

    """

    def __init__(self, x_train = None, noise: float = 0.0, transform: Callable = torch.nn.Identity()):
        super(ResponseLikeInitializer, self).__init__()
        self.x = x_train
        self.noise = noise
        self.transform = transform

    def generate_end_hook(self, x):
        drift = torch.rand_like(x) * self.noise
        components = self.transform(x + drift)
        return components
    
    def generate(self, num_components):
        raise NotImplementedError

class NGRInitializer(ResponseLikeInitializer):

    """

    Creates components in a neural gas like response-space.

    """

    def __init__(self, res_dim, lmbda: float = 1., **kwargs):
        super(NGRInitializer, self).__init__(**kwargs)

        self.res_dim = res_dim
        self.lmbda = lmbda

    def generate(self, num_components):
        inds = torch.LongTensor(self.res_dim).random_(0, len(self.x))
        res_proto_samples = self.x[inds]
        distances = squared_euclidean_distance(self.x, res_proto_samples)
        response = neighbourhood_fun(distances, self.lmbda) * distances
        sup_inds = torch.LongTensor(num_components).random_(0, len(self.x))
        sup_proto_samples = response[sup_inds]
        return self.generate_end_hook(sup_proto_samples)

NGRI = NGRInitializer
