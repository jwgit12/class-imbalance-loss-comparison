import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        pt = torch.exp(-bce_loss)

        loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
