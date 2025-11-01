import torch
from torch import nn
from torchvision.transforms.functional import resize
from tqdm import tqdm

from src.utils import dict_to_device


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        eval_func: callable,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        save_path: str = "best_model.pth",
        logger = None,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_func = eval_func
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.save_path = save_path
        self.logger = logger

    def train_one_epoch(self):
        self.model.train()
        for batch_data in tqdm(self.train_loader):
            batch_data = dict_to_device(batch_data, self.device)
            #TODO: Implement training step
            self.logger.log({"train/loss": loss.item()})
            self.logger.log({"train/lr": self.lr_scheduler.get_last_lr()[0]})

    def valid_one_epoch(self):
        scores = []
        self.model.eval()
        with torch.inference_mode():
            for batch_data in tqdm(self.valid_loader):
                #TODO: Implement validation step
        print(f"Validation Dice Score: {avg_score:.4f}")
        self.logger.log({"valid/dice_score": avg_score})
        return avg_score

    def fit(self, n_epoch):
        best_score = 0
        self.model.to(self.device)
        for _ in range(n_epoch):
            self.train_one_epoch()
            valid_score = self.valid_one_epoch()

            if valid_score > best_score:
                best_score = valid_score
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Successfully saved best model with score: {best_score:.4f}")
