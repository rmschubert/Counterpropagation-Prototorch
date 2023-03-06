from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback


class NGCallback(Callback):
    """

    Callback for the schedule of the neighbourhood-range lambda in Neural Gas

    """
    def __init__(self, end_lmbda: Optional[float] = None):
        super(NGCallback, self).__init__()

        ## end_lmbda determines on which value lmbda is set at the end of the training
        ## if None the value given to the NG Layer is used instead
        if end_lmbda is None:
            self.end_lmbda = end_lmbda
        else:
            self.end_lmbda = torch.Tensor([end_lmbda])
    
    def new_lmbda(self, l, x):
        """

        returns new lambda based on epochs, exponential decrease

        Args:
            l (FloatTensor): old lambda
            x (float): power of lambda depending on epochs

        Returns:
            FloatTensor: new lambda

        """
        return l**x
    
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        sd = pl_module.res_layer.state_dict()
        if self.end_lmbda is None:
            l = sd["lmbda"]
        else:
            l = self.end_lmbda
        
        x = trainer.current_epoch / trainer.max_epochs
        new_l = self.new_lmbda(l, x)
        sd["lmbda"] = new_l
        pl_module.res_layer.load_state_dict(sd)


