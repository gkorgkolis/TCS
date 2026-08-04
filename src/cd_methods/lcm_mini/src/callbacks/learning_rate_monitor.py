from lightning.pytorch.callbacks import LearningRateMonitor


class CustomLearningRateMonitor(LearningRateMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, trainer, pl_module):
        # Custom logic for learning rate monitoring (if any)
        print(f"Learning rate: {trainer.optimizers[0].param_groups[0]['lr']}")
        super().on_epoch_end(trainer, pl_module)