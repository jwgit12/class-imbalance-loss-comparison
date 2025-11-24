import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Alpha weighted focal loss for binary classification
    Small and reduced implementation adapted from: https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # We not that the implementation of the loss layer combines
        # the sigmoid operation for computing p with the loss computation.
        # (Retina Net Paper)
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # this serves as -log(pt)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # if y = 1 pt = p else its 1 -p (Retina Net Paper)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        focal_weight = (1 - p_t) ** self.gamma

        # weighting factor alpha for class 1 and 1-a for class 0 (Retina Net Paper)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        bce_loss = alpha_t * bce_loss
        loss = focal_weight * bce_loss

        return loss.mean()

if __name__ == "__main__":
    criterion = FocalLoss()
    inputs = torch.tensor([[0.2], [0.8], [0.4], [0.6]])
    targets = torch.tensor([[0], [1], [0], [1]])
    loss = criterion(inputs, targets)
    print(f"Focal Loss: {loss.item()}")