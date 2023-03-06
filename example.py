import torch
from prototorch.core.initializers import MCI
from prototorch.datasets import Tecator
from prototorch.models import GLVQ
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import ExponentialLR

from counterpropagation.core import NGR
from counterpropagation.core.initializer import NGRI
from counterpropagation.cp import CPPM
from counterpropagation.utils import VisCP2D
from counterpropagation.utils.callbacks import NGCallback

## Simple Example to run a CP-Model
## without train/test splitting


resprotos = 5
supprotos = 1
batchsize = 64

if __name__ == "__main__":

    train_ds = Tecator()

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batchsize)

    n_c = len(torch.unique(train_ds.targets))

    hparams = dict( 
        distribution={ 
            "num_classes": n_c,
            "per_class": supprotos,
        },
        lr=1e-3,
        num_prototypes=resprotos,
        resprotosinit=MCI(train_ds.data),
        supprotosinit=NGRI(x_train=train_ds.data, res_dim=resprotos),
        bsize=batchsize,
    )
    
    cp_model = CPPM(
        hparams, 
        resmodel=NGR,
        supmodel=GLVQ,
        optimizer=torch.optim.Adam,
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.9, verbose=False),
    )


    trainer = Trainer(
        callbacks=[ 
            VisCP2D(train_ds),
            NGCallback(end_lmbda=0.6)
        ],
        detect_anomaly=True,
        max_epochs=100,
    )

    trainer.fit(cp_model, train_loader)
