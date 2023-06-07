from typing import Callable

import torch
from prototorch.core.distances import squared_euclidean_distance
from prototorch.core.initializers import AbstractComponentsInitializer
from torch_kmeans import KMeans

from counterpropagation.utils import neighbourhood_fun


class KMeans_Initializer(AbstractComponentsInitializer):
    ## uses https://pypi.org/project/torch-kmeans/
    def __init__(
            self,
            hparams,
            data: torch.Tensor,
            noise: float = 0.0,
            transform: Callable = torch.nn.Identity(),
    ):
        self.data = data
        self.noise = noise
        self.transform = transform
        self.model = KMeans(**hparams)
    
    def generate_end_hook(self, center):
        drift = torch.rand_like(center) * self.noise
        components = self.transform(center + drift)
        return components

    def generate(self, num_components=None):
        model_result = self.model(self.data.unsqueeze(0))
        centers = model_result.centers.squeeze()
        components = self.generate_end_hook(centers)
        return components


class ResponseLikeInitializer(AbstractComponentsInitializer):

    """

    Base Class to create components in the response-space.
    A corresponding generate funvtion needs to be created 
    for the respective response-space.

    For good initializations x should be equal or close 
    to the response space.
    """

    def __init__(self, x, response_protos, noise: float = 0.0, transform: Callable = torch.nn.Identity()):
        super(ResponseLikeInitializer, self).__init__()
        self.x = x
        self.res_protos = response_protos
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

    def __init__(self, lmbda: float = 1., **kwargs):
        super(NGRInitializer, self).__init__(**kwargs)

        self.lmbda = lmbda
    
    def generate(self, num_components):
        dists = squared_euclidean_distance(self.x, self.res_protos)
        response = neighbourhood_fun(dists, self.lmbda) * dists
        protos = torch.empty((num_components, response.shape[1]))
        for i in range(num_components):
            inds = torch.LongTensor(num_components).random_(0, len(self.x))
            protos[i] = torch.mean(response[inds], 0)
        return self.generate_end_hook(protos)
        

RELI = ResponseLikeInitializer
NGRI = NGRInitializer
KMEI = KMeans_Initializer
