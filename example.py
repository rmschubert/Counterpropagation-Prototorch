import torch
from prototorch.datasets import Tecator
from prototorch.models import GMLVQ
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import ExponentialLR

from counterpropagation import CPPM, KMEI, NGR, NGRI
from counterpropagation.utils import NGCallback, VisCP2D, accuracy, err10

## Simple Example to run a CP-Model
## without train/test splitting


resprotos = 5
opsprotos_per_class = 1
batchsize = 64
metrics = dict(acc=accuracy, Err10=err10)

if __name__ == "__main__":

    ## Training Data
    train_ds = Tecator()

    ## Create a loader for Data
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batchsize)

    ## used for distribution in hparams
    n_c = len(torch.unique(train_ds.targets))

    ## Initializer of (response) prototypes in the data space
    res_init = KMEI(data=train_ds.data, hparams=dict(n_clusters=resprotos, max_iter=100, tol=1e-7))

    ## hparams:
    ##  operation_initializer is used to init prototypes in response space
    ##  input_dim, latent_dim are used for Omega in GMLVQ
    hparams = dict(
        distribution={ 
            "num_classes": n_c,
            "per_class": opsprotos_per_class,
        },
        lr=5e-3,
        n_resprotos=resprotos,
        response_initializer=res_init,
        operation_initializer=NGRI(x=train_ds.data, response_protos=res_init.generate(), lmbda=0.9),
        bsize=batchsize,
        res_dim=resprotos,
        input_dim=resprotos,
        latent_dim=2,
    )
    
    ## The actual Counterprop-Model
    cp_model = CPPM(
        hparams,
        train_metrics=metrics, 
        resmodel=NGR,
        opsmodel=GMLVQ,
        optimizer=torch.optim.Adam,
        lr_scheduler=ExponentialLR,
        lr_scheduler_kwargs=dict(gamma=0.99, verbose=False),
    )

    ## Trainer instance
    trainer = Trainer(
        callbacks=[
            NGCallback(gamma=0.9),
            VisCP2D(train_ds),
        ],
        detect_anomaly=True,
        max_epochs=100,
    )

    ## Fit and Test
    trainer.fit(cp_model, train_loader)
    trainer.test(cp_model, train_loader)