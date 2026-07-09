from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, data_dir, split=None):
        data_dir = Path(data_dir)
        self.X = np.load(data_dir / "X.npy", mmap_mode="r")
        self.y = np.load(data_dir / "y.npy", mmap_mode="r")
        meta = pd.read_csv(data_dir / "window_metadata.csv")
        self.meta = meta
        if split is not None:
            self.indices = meta.index[meta["split"] == split].to_numpy()
        else:
            self.indices = meta.index.to_numpy()
        if len(self.indices) == 0:
            raise ValueError(f"No samples found for split={split!r}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = torch.tensor(np.asarray(self.X[real_idx]), dtype=torch.float32)
        y = torch.tensor(np.asarray(self.y[real_idx]), dtype=torch.float32)
        return x, y

    def metadata(self):
        return self.meta.iloc[self.indices].reset_index(drop=True)
