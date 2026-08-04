from typing import Optional
import torch
import lightning.pytorch as L
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
from pathlib import Path
from src.data_modules.main_data_loader import MainDataLoader

class LazyShardDataset(Dataset):
    """
    Treats multiple .pt shard files as one large dataset.
    Each shard is lazily loaded and wrapped in MainDataLoader.
    """

    def __init__(self, base_dir, main_loader_kwargs):
        self.base_dir = Path(base_dir)
        self.main_loader_kwargs = main_loader_kwargs
        self.shard_files = sorted(self.base_dir.glob("*.pt"))
        if not self.shard_files:
            raise FileNotFoundError(f"No shard files found in {self.base_dir}")

        self.shard_lengths = []
        for f in self.shard_files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            self.shard_lengths.append(len(d))
            del d
        self.cumulative_lengths = [sum(self.shard_lengths[:i+1]) for i in range(len(self.shard_lengths))]
        print(f"[LazyShardDataset] Found {len(self.shard_files)} shards totaling {self.cumulative_lengths[-1]} samples.")

        self._cached_loader = None
        self._cached_index = None

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _load_shard(self, shard_idx):
        shard_path = self.shard_files[shard_idx]
        shard_data = torch.load(shard_path, map_location="cpu", weights_only=False)
        from src.data_modules.main_data_loader import MainDataLoader
        return MainDataLoader(data=shard_data, **self.main_loader_kwargs)

    def __getitem__(self, idx):
        import bisect
        shard_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        local_idx = idx if shard_idx == 0 else idx - self.cumulative_lengths[shard_idx - 1]

        # Lazy shard loading
        if self._cached_index != shard_idx:
            self._cached_loader = self._load_shard(shard_idx)
            self._cached_index = shard_idx

        sample = self._cached_loader[local_idx]

        # --- Deep clone to detach from shared storage ---
        def safe_clone(x):
            if isinstance(x, torch.Tensor):
                return x.clone()
            elif isinstance(x, (list, tuple)):
                return type(x)(safe_clone(xx) for xx in x)
            elif isinstance(x, dict):
                return {k: safe_clone(v) for k, v in x.items()}
            return x

        sample = safe_clone(sample)

        # --- Enforce fixed max sequence length ---
        # Prefer pulling from dataset args; fallback to 500
        MAX_SEQ_LEN = getattr(self, "max_seq_len", None)
        if MAX_SEQ_LEN is None and hasattr(self, "main_loader_kwargs"):
            MAX_SEQ_LEN = self.main_loader_kwargs.get("max_seq_len", 500)
        if MAX_SEQ_LEN is None:
            MAX_SEQ_LEN = 500  # fallback

        def pad_or_truncate(t):
            if not isinstance(t, torch.Tensor):
                return t
            if t.dim() == 2:  # [T, D] format
                T, D = t.shape
                if T > MAX_SEQ_LEN:
                    return t[:MAX_SEQ_LEN]
                elif T < MAX_SEQ_LEN:
                    pad = torch.zeros(MAX_SEQ_LEN - T, D, dtype=t.dtype, device=t.device)
                    return torch.cat([t, pad], dim=0)
            return t

        def fix_shapes(x):
            if isinstance(x, torch.Tensor):
                return pad_or_truncate(x)
            elif isinstance(x, (list, tuple)):
                return type(x)(fix_shapes(xx) for xx in x)
            elif isinstance(x, dict):
                return {k: fix_shapes(v) for k, v in x.items()}
            return x

        sample = fix_shapes(sample)

        # --- Optional: create attention mask if missing ---
        if isinstance(sample, dict) and "x" in sample and "mask" not in sample:
            seq_len = min(sample["x"].shape[0], MAX_SEQ_LEN)
            mask = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
            mask[:seq_len] = True
            sample["mask"] = mask

        return sample


    def on_epoch_start(self):
        random.shuffle(self.shard_files)
    

class MainDataModule(L.LightningDataModule):
    def __init__(self,
                 training_dataset_name: str,
                 test_dataset_name: str,
                 validation_dataset_name: str,
                 data_directory_path: str = None,
                 batch_size: int = 32,
                 n_vars: int = 1,
                 max_lag: int = 1,
                 training_aids: bool = False,
                 binarize_labels: bool = False,
                 use_shards: bool = False):
        super().__init__()
        self.data_directory_path = data_directory_path
        self.batch_size = batch_size
        self.n_vars = n_vars
        self.max_lag = max_lag
        self.training_dataset_name = training_dataset_name
        self.test_dataset_name = test_dataset_name
        self.validation_dataset_name = validation_dataset_name
        self.training_aids = training_aids
        #self.regression_head = regression_head
        self.binarize_labels = binarize_labels
        self.use_shards = use_shards

    def setup(self, stage: Optional[str] = None):
        from src.data_modules.main_data_loader import MainDataLoader
        from src.data_modules.data_module import LazyShardDataset

        main_loader_kwargs = dict(
            training_aids=self.training_aids,
            binarize_labels=self.binarize_labels
        )

        if stage in ("fit", None):
            if self.use_shards:
                self.train_dataset = LazyShardDataset(
                    base_dir=Path(self.data_directory_path) / "train",
                    main_loader_kwargs=main_loader_kwargs
                )
                self.validation_dataset = LazyShardDataset(
                    base_dir=Path(self.data_directory_path) / "val",
                    main_loader_kwargs=main_loader_kwargs
                )
            else:
                self.train_dataset = MainDataLoader(
                    data_directory=Path(self.data_directory_path) / self.training_dataset_name,
                    **main_loader_kwargs
                )
                self.validation_dataset = MainDataLoader(
                    data_directory=Path(self.data_directory_path) / self.validation_dataset_name,
                    **main_loader_kwargs
                )

        if stage in ("test", None):
            if self.use_shards:
                self.test_dataset = LazyShardDataset(
                    base_dir=Path(self.data_directory_path) / "test",
                    main_loader_kwargs=main_loader_kwargs
                )
            else:
                self.test_dataset = MainDataLoader(
                    data_directory=Path(self.data_directory_path) / self.test_dataset_name,
                    **main_loader_kwargs
                )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=False,
            shuffle=True, 
            persistent_workers=True
        )
        if hasattr(self.train_dataset, "on_epoch_start"):
            loader.on_epoch_start = self.train_dataset.on_epoch_start
        return loader

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=False, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=False, shuffle=False, persistent_workers=True)
