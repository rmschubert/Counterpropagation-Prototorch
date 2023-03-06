import torch
from prototorch.models.vis import Vis2DAbstract


class VisCP2D(Vis2DAbstract):

    """

    Visualization during Training based on Vis2DAbstract by Prototorch.
    
    Adapted for Counterpropagation to visualize the sup_layer in 
    the response-space (Only for Prototype-based sup_layer CP).
    Takes the first 2 dimensions for visualization.
    
    """

    def __init__(self, *args, **kwargs):
        super(VisCP2D, self).__init__(*args, **kwargs)

    def visualize(self, pl_module):
        with torch.no_grad():
            protos, plabel = pl_module.sup_layer.proto_layer()
            response = pl_module.res_layer(self.x_train)
            predictions = pl_module.predict((self.x_train, self.y_train))
            ax = self.setup_ax()
            self.plot_data(ax, response, predictions)
            self.plot_protos(ax, protos, plabel)
        
