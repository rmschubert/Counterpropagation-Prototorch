import torch
from prototorch.core.distances import euclidean_distance
from prototorch.models.abstract import ProtoTorchBolt
from prototorch.nn.wrappers import LambdaLayer
from torch.nn import Parameter

from counterpropagation.utils import neighbourhood_fun


class ResponseModel(ProtoTorchBolt):

    """

    Base Class to create new models used for a Response in CP framework

    Inherits:
        ProtoTorchBolt: Base Class from Prototorch, since
                        responses are assumed to have no
                        other supervision then by the 
                        subsequent supervised layer.
                        
                        Changed to ProtoTorchBolt,
                        since prototypes in reslayer
                        were not detected as leaf in 
                        grad-tree.

    """

    def __init__(self, hparams, **kwargs):
        super(ResponseModel, self).__init__(hparams, **kwargs)
        distance_fn = kwargs.get("response_distance", euclidean_distance)
        self.distance_layer = LambdaLayer(distance_fn)

        proto_init = kwargs.get("prototypes_initializer", None)
        if proto_init is not None:
            protos = proto_init.generate(self.hparams.n_resprotos)
            self.register_parameter("response_components", Parameter(protos))
    
    @property
    def prototypes(self):
        return self.response_components
        
    def compute_distances(self, x):
        return self.distance_layer(x, self.prototypes)
    
    def forward(self, x):
        return self.response(x)

    def response(self, x):
        raise NotImplementedError

class NeuralGasResponse(ResponseModel):

    """

    Neural Gas like Response

    Inherits:
        ResponseModel: Base Class

    """

    def __init__(self, hparams, lmbda: float = 1., **kwargs):
        super(NeuralGasResponse, self).__init__(hparams, **kwargs)
        self.lmbda = Parameter(torch.tensor(lmbda), requires_grad=False)
    
    def response(self, x):
        distances = self.compute_distances(x)
        responses = neighbourhood_fun(distances, self.lmbda) * distances
        return responses

REM = ResponseModel
NGR = NeuralGasResponse

