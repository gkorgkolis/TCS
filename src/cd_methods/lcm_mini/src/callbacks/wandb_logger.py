from lightning.pytorch.loggers import WandbLogger

import wandb


class CustomWandbLogger(WandbLogger):
    def __init__(self, project, name, *args, **kwargs):
        super().__init__(project=project, name=name, *args, **kwargs)

    def log_hyperparams(self, params):
        # Custom logic for logging hyperparameters (optional)
        print("Logging hyperparameters to WandB...")
        super().log_hyperparams(params)