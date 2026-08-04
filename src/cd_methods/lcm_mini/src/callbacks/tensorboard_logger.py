from lightning.pytorch.loggers import TensorBoardLogger


class CustomTensorBoardLogger(TensorBoardLogger):
    def __init__(self, log_dir, name, *args, **kwargs):
        super().__init__(save_dir=log_dir, name=name, *args, **kwargs)

    def log_hyperparams(self, params):
        # Custom logic for logging hyperparameters (optional)
        print("Logging hyperparameters to TensorBoard...")
        super().log_hyperparams(params)