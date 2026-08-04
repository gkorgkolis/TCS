import os
import random
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append("..")
from src.utils.utils import lagged_batch_crosscorrelation


class MainDataLoader(Dataset):
    def __init__(self,
                 data_directory=None,
                 data=None,
                 use_shards=False,
                 num_shards_to_load=None,
                 training_aids=False,
                 binarize_labels=True):
        self.data_directory = data_directory
        self.training_aids = training_aids
        self.binarize_labels = binarize_labels
        self.use_shards = use_shards
        self.num_shards_to_load = num_shards_to_load

        # used by LazyShardDataset
        if data is not None:
            self.dataset = data
        else:
            self.dataset = self.load()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: list):
        """
        Args:
          idx (list)
        Returns:
          X_tuple: either just X, or a tuple with training aids
          Y_clas: binary adjacency if binarize_labels else DCE tensor
        """
        # Unwrap sample
        X = self.dataset[idx][0]  # time-series data
        Y = self.dataset[idx][-1]

        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y)

        X_components = [X]  # always start with X

        # --- Training Aids ---
        if self.training_aids:
            corr = lagged_batch_crosscorrelation(X.unsqueeze(0), Y.shape[-1])
            X_components.extend([corr])

        # Final input tuple
        X_tuple = tuple(X_components) if len(X_components) > 1 else X

        # --- Output label: Binary or Real-Valued Causal Effects ---
        Y_class = (torch.abs(Y) > 0).float() if self.binarize_labels else Y.float()

        #if self.regression_head:
        #    Y_reg = torch.unsqueeze(torch.sum(Y_class > 0), dim=-1).float()

        #    return X_tuple, (Y_class, Y_reg)
        
        return X_tuple, Y_class


    def on_epoch_start(self):
        random.shuffle(self.shard_files)

    def load(self):
        path = self.data_directory

        # Case 1: loading a single .pt file
        if os.path.isfile(path):
            try:
                data = torch.load(path, weights_only=False)
                if not data or not isinstance(data, list):
                    raise ValueError(f"Loaded data at {path} is malformed: {data}. It should be a list of two-dimensional tuples (data,label) \
                          where data is of shape (num_samples,num_vars) and label of shape (num_vars, num_vars, max_lag).")
                return data
            except Exception as e:
                print(f"Error loading dataset from {path}: {e}")
                raise e

        # Case 2: load from sharded directory
        elif os.path.isdir(path):
            shard_files = [f for f in os.listdir(path) if f.endswith(".pt")]
            if not shard_files:
                raise FileNotFoundError(f"No shard files found in directory {path}")

            # Randomly sampling shards
            num_to_load = min(self.num_shards_to_load, len(shard_files))
            selected = random.sample(shard_files, num_to_load)

            print(f"Loading {num_to_load} shards from {path}")
            all_data = []
            for shard in selected:
                shard_path = os.path.join(path, shard)
                try:
                    shard_data = torch.load(shard_path, weights_only=False)
                    if isinstance(shard_data, list):
                        all_data.extend(shard_data)
                    else:
                        print(f"Skipping malformed shard: {shard_path}")
                except Exception as e:
                    print(f"Error loading shard {shard_path}: {e}")

            if not all_data:
                raise ValueError(f"No valid data loaded from shards in {path}")
            return all_data

        else:
            raise FileNotFoundError(f"Dataset file or shard directory not found at {path}")
