import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(MLP, self).__init__()
        self.fc1: nn.Linear = nn.Linear(input_size, hidden_size)
        self.relu: nn.ReLU = nn.ReLU()
        self.fc2: nn.Linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

