from typing import Any, Callable, Optional

import torch
from prototorch.models.abstract import ProtoTorchBolt
from prototorch.nn.wrappers import LambdaLayer
from torch.nn import Linear, MSELoss
from torch.nn.parameter import Parameter

from counterpropagation.utils import accuracy, err10, r_squared


class CounterPropagation(ProtoTorchBolt):

    """

    Counterpropagation Model - Base Class

    Inherits:
        ProtoTorchBolt: ProtoTorchBolt to into the Prototorch Framework
    
    Forward Flow:
          Input -> ResponseModel(Input) -> SupervisedModel(Response) -> Loss(SupervisedModel)

    Backward Flow:
                            |----> ResponseModel(Input)          
                            |    
                            |
        Loss(SupervisedModel)
                            |                            
                            |
                            |----> SupervisedModel(Response)

    Thus, ResponseModel must be differentiable.

    """

    def __init__( 
        self, 
        hparams,
        resmodel: Any,
        supmodel: Optional[Any] = None,
        **kwargs 
        ):

        """

        Args:
            hparams (dict): hparams for the concatenated models
            resmodel (Any): Response Model
            supmodel (Optional[Any], optional): Supervised Model, can be None for regression,
                                                in which case an perceptron like layer is created.
                                                Defaults to None.

        Raises:
            ValueError: When resmodel is None.

        """

        super(CounterPropagation, self).__init__(hparams, **kwargs)


        if resmodel is None:
            raise ValueError("Response Model cannot be None")
        elif supmodel is None:
            self.res_layer = resmodel(hparams, prototypes_initializer=self.hparams.resprotosinit, **kwargs)
            self.sup_layer = supmodel
        else:
            self.res_layer = resmodel(hparams, prototypes_initializer=self.hparams.resprotosinit, **kwargs)
            self.sup_layer = supmodel(hparams, prototypes_initializer=self.hparams.supprotosinit, **kwargs)

    def forward(self, batch: Any):
        """

        To modify the forward a custom end_step function can be created,
        which depends on the batch (necessary for class-wise regression models)

        Args:
            batch (Any): Used for end_step

        Returns:
            torch.Tensor: the predictions.

        """

        x, y = batch
        response = self.res_layer(x)
        predictions = self.end_step((response, y))
        return predictions
    
    def _log_metric(self, fun: Callable, batch, tag: str):

        """

        Allows to log any custom metric which depends
        on predictions and targets.
        
        Args:
            fun (Callable): A custom metric m, s.t. m(predictions, targets)
            batch (Any): Current batch
            tag (str): Name of the metric

        """

        _, y = batch
        with torch.no_grad():
            predictions = self.forward(batch)
            result = fun(predictions, y)
        
        self.log( 
            tag,
            result,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
    
    def predict(self, batch):
        return self.forward(batch)

class CounterPropagationProtoModel(CounterPropagation):

    """

    Prototype-based Classification CP-Model, i.e. supmodel is a
    prototype-based Model

    """

    def __init__(self, hparams, **kwargs):
        """

        Args:
            hparams (dict): as in base class CounterPropagation

        Raises:
            ValueError: When supmodel is None.

        """

        super(CounterPropagationProtoModel, self).__init__(hparams, **kwargs)
        if self.sup_layer is None:
            raise ValueError("Supervised Layer in Prototype Modle cannot be None")

    def shared_step(self, batch, batch_idx, opt_idx=None):
        x, y = batch
        response = self.res_layer(x)
        distances, loss = self.sup_layer.shared_step((response, y), batch_idx)
        return distances, loss
    
    def training_step(self, batch, batch_idx, opt_idx=None):
        _, train_loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", train_loss)
        self._log_metric(accuracy, batch, tag="train_acc")
        return train_loss
    
    def test_step(self, batch, batch_idx):
        _, test_loss = self.shared_step(batch, batch_idx)
        self._log_metric(accuracy, batch, tag="test_acc")
        return test_loss
    
    def end_step(self, batch_reponse):
        response, _ = batch_reponse
        return self.sup_layer.predict(response)

class CounterPropagationRegModel(CounterPropagation):

    """

    Regression based CP-Model (for classification or regression).
    supmodel is allowed to be None in which case a torch.nn.Linear
    instance is created.

    """

    def __init__(self, hparams, **kwargs):

        """

        Regression-based CP-Model, allows supmodel to be None.

        Args:
            hparams (dict): as in base class CounterPropagation
        
        loss is by default the MSELoss.
        TODO: make loss customizable

        """

        super(CounterPropagationRegModel, self).__init__(hparams, **kwargs)

        if self.sup_layer is None:
            idim, odim = self.hparams.res_dim, self.hparams.out_dim
            self.sup_layer = Linear(idim, odim)
    
        self.loss = MSELoss(reduction='sum')

    def shared_step(self, batch, batch_idx, opt_idx=None):
        x, y = batch
        response = self.res_layer(x)
        preds = self.sup_layer(response).flatten()
        loss = self.loss(preds, 1.*y)
        return loss
    
    def training_step(self, batch, batch_idx, opt_idx=None):
        train_loss = self.shared_step(batch, batch_idx)
        self.log("MSE", train_loss)
        self._log_metric(r_squared, batch, tag="RSq")
        self._log_metric(err10, batch, tag="Err10")
        return train_loss
    
    def test_step(self, batch, batch_idx):
        test_loss = self.shared_step(batch, batch_idx)
        self._log_metric(r_squared, batch, tag="test_RSq")
        self._log_metric(err10, batch, tag="test_Err10")
        return test_loss

    def end_step(self, batch_reponse):
        response, _ = batch_reponse
        return self.sup_layer(response).flatten()

class CounterPropagationClassWiseRegModel(CounterPropagation):

    """

    Class-wise Regression based CP-Model, i.e. for each class there is
    a responsible perceptron.

    """

    def __init__(self, hparams, **kwargs):

        """

        When supmodel is None a linear-like layer is created by using
        res_dim and out_dim of the hparams. Therefore, out_dim is assumed
        to be at least the number of classes in the data to create that many
        perceptrons.
        
        TODO: Allow for multiple perceptrons per class by instead giving a distribution

        """

        super(CounterPropagationClassWiseRegModel, self).__init__(hparams, **kwargs)
        
        self.perceptron_label = torch.LongTensor(range(self.hparams.out_dim))

        idim, odim = self.hparams.res_dim, self.hparams.out_dim
        if self.sup_layer is None:
            _weights = torch.rand(idim, odim)
            self.register_parameter("weights", Parameter(_weights))
            self.sup_layer = LambdaLayer(lambda x: x @ self.weights)
    
    def _class_filter(self, y):
        plabs = self.perceptron_label
        if y.ndim == 1:
            return y.unsqueeze(-1) == plabs
        elif y.ndim == 2:
            return torch.max(y, -1).indices.unsqueeze(-1) == plabs
        else:
            raise TypeError(f"Only supported up to 2D labels, but got {y.ndim}D")

    def shared_step(self, batch, batch_idx, opt_idx=None):
        x, y = batch
        response = self.res_layer(x)
        raw_preds = self.sup_layer(response)
        filter_ = self._class_filter(y)
        preds = raw_preds[filter_]
        loss = self.loss(preds, y)
        return loss
    
    def end_step(self, batch_reponse):
        response, y = batch_reponse
        raw_preds = self.sup_layer(response)
        preds = raw_preds[self._class_filter(y)]
        return preds
        
    


CPPM = CounterPropagationProtoModel
CPRM = CounterPropagationRegModel
CCRM = CounterPropagationClassWiseRegModel
