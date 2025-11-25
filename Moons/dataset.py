import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
import numpy as np


class ImbalancedMoonsDataset(Dataset):
    def __init__(self, n_samples: int = 2000, noise: float = 0.1, imbalance_ratio: float = 0.01,
                 random_state: int = 42) -> None:
        """
        n_samples: The total number of samples to return (sum of both classes).
        imbalance_ratio: The fraction of the dataset that should be the minority class. (Out of 5000 samples how many are class 1) for 0.25 it means 1250 class 1 and 3750 class 0
        """
        rng = np.random.RandomState(random_state)

        # Calculate number of samples per class
        n_minority = int(n_samples * imbalance_ratio)
        n_majority = n_samples - n_minority

        # Generate enough data to satisfy the majority requirement.
        n_generation = n_majority * 2

        X_raw, y_raw = make_moons(n_samples=n_generation, noise=noise, random_state=random_state)

        # Retrieve indices for both classes
        idx0 = np.where(y_raw == 0)[0]
        idx1 = np.where(y_raw == 1)[0]

        # Select the exact number of samples required
        selected_idx0 = idx0[:n_majority]

        # We use choice here to ensure we pick randomly if we generated way more class 1s than needed
        selected_idx1 = rng.choice(idx1, size=n_minority, replace=False)

        keep_idx = np.concatenate([selected_idx0, selected_idx1])
        rng.shuffle(keep_idx)

        self.X: torch.Tensor = torch.tensor(X_raw[keep_idx], dtype=torch.float32)
        self.y: torch.Tensor = torch.tensor(y_raw[keep_idx], dtype=torch.float32)

        print(f"Requested n_samples: {n_samples}")
        print(f"Actual class counts: {np.bincount(self.y.numpy().astype(int))}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    #check for consistency after regenerating using random_state
    dataset = ImbalancedMoonsDataset(n_samples=5000, noise=0.25, imbalance_ratio=0.5, random_state=42)
    print(len(dataset))
    for x,y in dataset:
        print(x)
        print(y)
        break
        #tensor([-0.1197,  1.0368])
        #tensor(0.)