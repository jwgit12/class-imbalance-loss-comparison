import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
import numpy as np


class ImbalancedMoonsDataset(Dataset):
    def __init__(self, n_samples: int = 2000, noise: float = 0.1, imbalance_ratio: float = 0.01,
                 random_state: int = 42) -> None:
        """
        n_samples: The total number of samples to return (sum of both classes).
        imbalance_ratio: The fraction of the dataset that should be the minority class.
        """
        rng = np.random.RandomState(random_state)

        # 1. Calculate exact counts for each class to sum to n_samples
        n_minority = int(n_samples * imbalance_ratio)
        n_majority = n_samples - n_minority

        # 2. Generate enough data to satisfy the majority requirement.
        # make_moons generates 50/50, so we need 2 * n_majority to get enough class 0.
        # We generate a bit more just to be safe, though 2*n_majority is mathematically sufficient.
        n_generation = n_majority * 2

        X_raw, y_raw = make_moons(n_samples=n_generation, noise=noise, random_state=random_state)

        # 3. Retrieve indices for both classes
        idx0 = np.where(y_raw == 0)[0]
        idx1 = np.where(y_raw == 1)[0]

        # 4. Select the exact number of samples required
        # We take the first n_majority/n_minority indices (randomness is handled by make_moons + rng)
        # Note: We assume class 0 is majority and class 1 is minority
        selected_idx0 = idx0[:n_majority]

        # We use choice here to ensure we pick randomly if we generated way more class 1s than needed
        selected_idx1 = rng.choice(idx1, size=n_minority, replace=False)

        # 5. Concatenate and Create Tensors
        keep_idx = np.concatenate([selected_idx0, selected_idx1])

        # Optional: Shuffle the final indices so the dataset isn't ordered (0,0,0...1,1,1)
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
    dataset = ImbalancedMoonsDataset(n_samples=5000, noise=0.1, imbalance_ratio=0.01, random_state=42)
    print(len(dataset))
    for x,y in dataset:
        print(x)
        print(y)
        break
        #tensor([-0.1197,  1.0368])
        #tensor(0.)