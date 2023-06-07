from typing import Any, Callable, Optional

import torch
from prototorch.models.abstract import ProtoTorchBolt
from torch.nn import Linear, MSELoss
from torch.nn.parameter import Parameter

from counterpropagation.utils import accuracy


class CounterPropagation(ProtoTorchBolt):

    """

    Counterpropagation Model - Base Class

    Inherits:
        ProtoTorchBolt: ProtoTorchBolt to migrate into the Prototorch Framework
    
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

    Thus, ResponseModel must be differentiable by Version 1.0.1, however future updates 
    may include non-trainable Response Models.

    """

    def __init__( 
        self, 
        hparams,
        resmodel: Any,
        opsmodel: Optional[Any] = None,
        train_metrics: dict = dict(acc=accuracy),
        test_metrics: Optional[dict] = None,
        same_metrics: bool = True,
        **kwargs 
        ):

        """

        Args:
            hparams (dict): 
            hparams for the concatenated models
            
            resmodel (Any): 
            Response Model
            
            opsmodel (Optional[Any], None): 
            Supervised (Operation) Model (Layer), can be None for regression,
            in which case an perceptron like layer is created.
             Defaults to None.
            
            train_metrics (dict): 
            Metrics being used to monitor training performance in shape like:
            train_metrics = {"name_of_metric": callable_metric} 
             Defaults to accuracy.
            
            test_metrics (Optional[dict], None): 
            Metrics being used to monitor test performance. 
             Defaults to None.
            
            same_metrics (bool): 
            If True, test_metrics will equal train_metrics. If False a dict needs to be
            given to test_metrics with the corresponding metrics. 
             Defaults to True.

        """

        super(CounterPropagation, self).__init__(hparams, **kwargs)

        if opsmodel is None:
            self.res_layer = resmodel(hparams, prototypes_initializer=self.hparams.response_initializer, **kwargs)
            self.ops_layer = opsmodel
        else:
            self.res_layer = resmodel(hparams, prototypes_initializer=self.hparams.response_initializer, **kwargs)
            self.ops_layer = opsmodel(hparams, prototypes_initializer=self.hparams.operation_initializer, **kwargs)
        
        ## allow for customizable metrics
        ## and add test to every key in
        ## test metrics if test and train
        ## metrics are equal
        if same_metrics:
            met_copy = train_metrics.copy()
            for k in train_metrics.keys():
                new_k = "test_" + k
                met_copy[new_k] = train_metrics[k]
                del met_copy[k]
        else:
            assert test_metrics is not None, "Provide test_metrics if same_metrics is False"
            met_copy = test_metrics.copy()
            del test_metrics
            
        self.train_metrics = train_metrics
        self.test_metrics = met_copy

    def forward(self, batch: Any):
        """

        To modify the forward a custom end_step function can be created,
        which depends on the batch (necessary for class-wise regression models)

        Args:
            batch (Any): 
            Used for end_step

        Returns:
            torch.Tensor: the predictions.

        """

        x, y = batch
        response = self.res_layer(x)
        predictions = self.end_step(response, y)
        return predictions
    
    def _log_metric(self, fun: Callable, batch, tag: str):

        """

        Allows to log any custom metric which depends
        on predictions and targets.
        
        Args:
            fun (Callable): 
            A custom metric m, s.t. m(predictions, targets)

            batch (Any): 
            Current batch

            tag (str): 
            Name of the metric

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
    
    def training_step(self, batch, batch_idx):
        _, train_loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", train_loss)
        for k, v in self.train_metrics.items():
            self._log_metric(v, batch, tag=k)
        return train_loss
    
    def test_step(self, batch, batch_idx):
        _, test_loss = self.shared_step(batch, batch_idx)
        for k, v in self.test_metrics.items():
            self._log_metric(v, batch, tag=k)
        return test_loss

class CounterPropagationProtoModel(CounterPropagation):

    """

    Prototype-based Classification CP-Model, i.e. opsmodel is a
    prototype-based / distance-based Model

    """

    def __init__(self, hparams, **kwargs):
        """

        Args:
            hparams (dict): 
            as in base class CounterPropagation

        Raises:
            ValueError: When opsmodel is None.

        """

        super(CounterPropagationProtoModel, self).__init__(hparams, **kwargs)
        if self.ops_layer is None:
            raise ValueError("Supervised Layer in Prototype Model cannot be None")
        
        self.omega_available = False
        if hasattr(self.ops_layer, "_omega"):
            self.ops_omega = Parameter(self.ops_layer._omega)
            self.omega_available = True
        
        self.ops_dl = self.ops_layer.distance_layer
        self.ops_protos = Parameter(self.ops_layer.prototypes)

    def shared_step(self, batch, batch_idx, opt_idx=None):
        x, y = batch
        response = self.res_layer(x)
        
        if self.omega_available:
            distances = self.ops_dl(response, self.ops_protos, self.ops_omega)
        else:
            distances = self.ops_dl(response, self.ops_protos)
        
        _, ops_label = self.ops_layer.proto_layer()
        loss = self.ops_layer.loss(distances, y, ops_label)
        return distances, loss
    
    def end_step(self, batch_reponse, targets):
        response = batch_reponse
        return self.ops_layer.predict(response)

class CounterPropagationRegModel(CounterPropagation):

    """

    Regression based CP-Model (for classification or regression).
    opsmodel is allowed to be None in which case a torch.nn.Linear
    instance is created.

    """

    def __init__(self, hparams, loss: Callable = MSELoss(reduction="sum"), **kwargs):

        """

        Regression-based CP-Model, allows opsmodel to be None.

        Args:
            hparams (dict): 
            as in base class CounterPropagation
        
            loss (Callable, torch.nn.MSELoss):
            torch.Module like or any Callable (which is differentiable).
            Default loss is torch.nn.MSELoss.

        """

        super(CounterPropagationRegModel, self).__init__(hparams, **kwargs)

        self.idim = self.hparams.res_dim
        if self.ops_layer is None:
            self.ops_layer = Linear(self.idim, 1)
        
        self.loss = loss

    def shared_step(self, batch, batch_idx, opt_idx=None):
        x, y = batch
        response = self.res_layer(x)
        preds = self.ops_layer(response).squeeze()
        loss = self.loss(preds, 1. * y)
        return preds, loss

    def end_step(self, batch_reponse, targets):
        response = batch_reponse
        return self.ops_layer(response).squeeze()

class CounterPropagationClassWiseRegModel(CounterPropagationRegModel):

    """

    Class-wise Regression based CP-Model, i.e. for each class there is
    a responsible perceptron.

    """

    def __init__(self, hparams, **kwargs):

        """

        When opsmodel is None a linear-like layer is created by using
        res_dim and distribution provided in hparams.
        distribution is assumed to be of the shape dict(num_classes=n, per_class=p)


        """

        super(CounterPropagationClassWiseRegModel, self).__init__(hparams, **kwargs)

        nc, npc = self.hparams.distribution["num_classes"], self.hparams.distribution["per_class"]
        out_dim = nc * npc
        
        self.ops_layer = torch.nn.Linear(self.idim, out_dim)
        self.perceptron_label = torch.LongTensor([i for i in range(nc) for j in range(npc)])
    
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
        raw_preds = self.ops_layer(response).squeeze()
        filter_ = self._class_filter(y)
        preds = raw_preds[filter_]
        loss = self.loss(preds, 1.*y)
        return preds, loss
    
    def end_step(self, batch_reponse, targets):
        response, y = batch_reponse, targets
        raw_preds = self.ops_layer(response)
        preds = raw_preds[self._class_filter(y)]
        return preds
        
    


CPPM = CounterPropagationProtoModel
CPRM = CounterPropagationRegModel
CCRM = CounterPropagationClassWiseRegModel
