from lightning.pytorch.callbacks import EarlyStopping


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)