import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal loss:
        FL = (1-pt)^gamma * CE
    No alpha class balancing trick.
    """
    def __init__(self, alpha=0.5, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets.float())
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce

        return loss.mean() if self.reduction == "mean" else loss.sum()
