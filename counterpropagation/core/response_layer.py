import torch
from prototorch.models.abstract import UnsupervisedPrototypeModel
from torch.nn import Parameter

from counterpropagation.utils import neighbourhood_fun


class ResponseModel(UnsupervisedPrototypeModel):

    """

    Base Class to create new models used for a Response in CP framework

    Inherits:
        UnsupervisedPrototypeModel: Base Class from Prototorch, since
                                    responses are assumed to have no
                                    other supervision then by the 
                                    subsequent supervised layer.


    """

    def __init__(self, hparams, **kwargs):
        super(ResponseModel, self).__init__(hparams, **kwargs)
    
    def forward(self, x):
        return self.response(x)

class NeuralGasResponse(ResponseModel):

    """

    Neural Gas like Response

    Inherits:
        ResponseModel : Base Class

    """

    def __init__(self, hparams, lmbda: float = 1., **kwargs):
        super(NeuralGasResponse, self).__init__(hparams, **kwargs)
        self.lmbda = Parameter(torch.tensor(lmbda), requires_grad=False)
    
    def response(self, x):
        distances = self.compute_distances(x)
        responses = neighbourhood_fun(distances, self.lmbda) * distances
        return responses

NGR = NeuralGasResponse

