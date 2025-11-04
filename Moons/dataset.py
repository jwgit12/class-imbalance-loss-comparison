import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
import numpy as np


class ImbalancedMoonsDataset(Dataset):
    def __init__(self, n_samples: int = 20000, noise: float = 0.1, imbalance_ratio: float = 0.01,
                 random_state: int = 42) -> None:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

        rng = np.random.RandomState(random_state)
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        keep1 = rng.choice(idx1, size=int(len(idx1) * imbalance_ratio), replace=False)
        keep_idx = np.concatenate([idx0, keep1])

        self.X: torch.Tensor = torch.tensor(X[keep_idx], dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(y[keep_idx], dtype=torch.float32)

        print("class counts (imbalanced):", np.bincount(y[keep_idx]))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
