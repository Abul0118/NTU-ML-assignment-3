import torch

class DiceScore:
    def __init__(self, smooth: float = 1e-5):
        self.smooth = smooth

    def __call__(self, pred: torch.Tensor, label: torch.Tensor, mean: bool = True) -> torch.Tensor:
        """
        pred  : raw logits from model (B,1,H,W)
        label : ground truth mask (B,1,H,W)
        """

        # Apply sigmoid to logits
        pred = torch.sigmoid(pred)

        # Flatten tensors
        pred = pred.view(pred.size(0), -1)
        label = label.view(label.size(0), -1)

        # Intersection & Dice formula
        intersection = (pred * label).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            pred.sum(dim=1) + label.sum(dim=1) + self.smooth
        )

        return dice.mean() if mean else dice
