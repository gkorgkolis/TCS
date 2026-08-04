from lightning.pytorch.callbacks import ModelCheckpoint


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)