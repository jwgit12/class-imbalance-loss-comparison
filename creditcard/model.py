import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_size=64, num_layers=2):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_size),
                  nn.ReLU()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
