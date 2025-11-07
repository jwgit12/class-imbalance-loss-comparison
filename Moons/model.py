import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2) -> None:
        super(MLP, self).__init__()
        self.num_layers: int = num_layers
        self.layers: nn.ModuleList = nn.ModuleList()
        self.relu: nn.ReLU = nn.ReLU()

        # First layer
        self.layers.append(nn.Linear(input_size, hidden_size))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.relu(x)
        x = self.layers[-1](x)
        return x


