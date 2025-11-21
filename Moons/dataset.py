import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
import numpy as np

class MoonsDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 20000,
        noise: float = 0.1,
        imbalance_ratio: float = 0.01,
        random_state: int = 42
    ) -> None:

        # Create the moon dataset with samples and noise
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

        rng = np.random.RandomState(random_state)
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]

        keep1 = rng.choice(idx1, size=int(len(idx1) * imbalance_ratio), replace=False)
        keep_idx = np.concatenate([idx0, keep1])

        self.X = torch.tensor(X[keep_idx], dtype=torch.float32)
        self.y = torch.tensor(y[keep_idx], dtype=torch.float32)

        print("class counts:", np.bincount(self.y.numpy().astype(int)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    #check for consistency after regenerating using random_state
    dataset = ImbalancedMoonsDataset(n_samples=5000, noise=0.1, imbalance_ratio=0.01, random_state=42)
    for x,y in dataset:
        print(x)
        print(y)
        break
        #tensor([-0.1197,  1.0368])
        #tensor(0.)