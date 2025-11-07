#TODO: Implement loss functions
#TODO: Implement loss functions
import torch
import torch.nn as nn

# Dice Loss for binary segmentation
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        # preds: logits (B,1,H,W) -> convert to sigmoid inside
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2 * intersection + self.eps) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.eps
        )

        return 1 - dice.mean()


# BCE + Dice combined (best for medical segmentation)
class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# default function so Trainer can import LOSS_FUNC
def build_loss():
    return DiceBCELoss()
