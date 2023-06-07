from typing import Optional

import torch
from pytorch_lightning.callbacks import Callback

NG_DECAY = dict( 
    exp="Exponential decay of the form lambda ^ decay_constant",
    mul="Multiplicative decay of the form lambda * decay_constant"
)
DESCRIPTION = "Decay Options are:\n" + "\n".join([f"{k} - {v}" for (k, v) in NG_DECAY.items()])

class NGCallback(Callback):
    """

    Callback for the schedule of the neighborhood-range lambda in Neural Gas

    """
    def __init__(self, lmbda_start: Optional[float] = 1., gamma: float = 0.99, decay_type: str = "exp"):
        super(NGCallback, self).__init__()

        if lmbda_start is None:
            self.lmbda = lmbda_start
        else:
            self.lmbda_start = torch.Tensor([lmbda_start])
    
        assert decay_type in NG_DECAY, print(DESCRIPTION)
        self.decay_type = decay_type
        decay_funs = dict( 
            exp=self.exp_decay,
            mul=self.mul_decay,
        )
        self.decay_fun = decay_funs[self.decay_type]

        assert gamma <= 1. and gamma > 0., "gamma must be less or equal 1. and positive."
        self.gamma = gamma
    
    def exp_decay(self, l):
        return l ** self.gamma

    def mul_decay(self, l):
        return l * self.gamma
    
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        sd = pl_module.res_layer.state_dict()
        if self.lmbda_start is None:
            l = sd["lmbda"]
        else:
            l = self.lmbda_start
        
        sd["lmbda"] = self.decay_fun(l)
        pl_module.res_layer.load_state_dict(sd)


