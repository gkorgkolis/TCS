from src.data_modules.data_module import MainDataModule


def load_data(path: str,
              batch_size: int=16,
              training_aids: bool=False,
              only_test: bool=False):

    if path is None:
        raise ValueError("Path must be specified.")
    
    if only_test:
        mod = MainDataModule(
            data_directory_path=path,
            training_dataset_name="train/train.pt",
            test_dataset_name="val/val.pt",
            validation_dataset_name="val/val.pt",
            batch_size=batch_size,
            training_aids=training_aids
        )
    else:
        mod = MainDataModule(
            data_directory_path=path,
            training_dataset_name="train/train.pt",
            test_dataset_name="test/test.pt",
            validation_dataset_name="val/val.pt",
            batch_size=batch_size,
            training_aids=training_aids
        )
    mod.setup(None)

    return mod.train_dataloader(), mod.val_dataloader(), mod.test_dataloader()