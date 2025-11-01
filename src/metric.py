import torch


class DiceScore:
    def __init__(self, smooth: float = 1e-5):
        self.smooth = smooth

    def __call__(self, pred: torch.Tensor, label: torch.Tensor, mean: bool = True) -> torch.Tensor:
        #TODO: Implement Dice Score calculation
        return dice.mean() if mean else dice
