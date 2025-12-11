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
        logger=None,
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
        total_loss = 0

        for batch_data in tqdm(self.train_loader):
            batch_data = dict_to_device(batch_data, self.device)

            images = batch_data["image"]
            masks = batch_data["label"]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Logging
            if self.logger:
                self.logger.log({"train/loss": loss.item()})
                self.logger.log({"train/lr": self.lr_scheduler.get_last_lr()[0]})

        self.lr_scheduler.step()
        return total_loss / len(self.train_loader)

    def valid_one_epoch(self):
        self.model.eval()
        scores = []

        with torch.inference_mode():
            for batch_data in tqdm(self.valid_loader):
                batch_data = dict_to_device(batch_data, self.device)
                images = batch_data["image"]
                masks = batch_data["label"]

                outputs = self.model(images)
                preds = torch.sigmoid(outputs)

                score = self.eval_func(preds, masks)
                scores.append(score)

        avg_score = sum(scores) / len(scores)

        print(f"Validation Dice Score: {avg_score:.4f}")

        if self.logger:
            self.logger.log({"valid/dice_score": avg_score})

        return avg_score

    def fit(self, n_epoch):
        best_score = 0
        self.model.to(self.device)

        for epoch in range(n_epoch):
            print(f"Epoch [{epoch+1}/{n_epoch}]")

            train_loss = self.train_one_epoch()
            valid_score = self.valid_one_epoch()

            print(f"Train Loss: {train_loss:.4f} | Val Dice: {valid_score:.4f}")

            if valid_score > best_score:
                best_score = valid_score
                torch.save(self.model.state_dict(), self.save_path)
                print(f"✅ Saved best model (Dice: {best_score:.4f})")

        print(f"Best Overall Dice Score: {best_score:.4f}")
