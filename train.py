from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.dataset import MedicalImageDataset, LoadImaged, ToTensord, ResizeImaged
from src.metric import DiceScore
from src.trainer import Trainer
from src.utils import set_random_seed
from src.loss import build_loss  # Dice+BCE combined


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="cnn_unet")
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.config}.yaml"))

    # ------------------- Transforms -------------------
    H = config.transform.resized_imaged.height
    W = config.transform.resized_imaged.width

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
            ResizeImaged(keys=["image", "label"], size=(H, W)),
        ]
    )
    valid_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ToTensord(keys=["image", "label"]),
            ResizeImaged(keys=["image", "label"], size=(H, W)),
        ]
    )

    # ------------------- Datasets & Loaders -------------------
    train_dataset = MedicalImageDataset(
        **config.train_dataset, transform=train_transform
    )
    train_loader = DataLoader(train_dataset, **config.train_loader)

    valid_dataset = MedicalImageDataset(
        **config.valid_dataset, transform=valid_transform
    )
    valid_loader = DataLoader(valid_dataset, **config.valid_loader)

    # ------------------- Device -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------- Model selection -------------------
    if config.name == "cnn_unet":
        from src.model import UNetSmall

        model = UNetSmall(in_ch=1, out_ch=1, base=16)

    elif config.name == "transformer_unet":
        from src.model_transformer_unet import build_model

        model = build_model()

    elif config.name == "transformer_unet_ssl":
        from src.model_transformer_unet import build_model

        model = build_model()
        # load SSL encoder weights into first conv block
        print("🔁 Loading SSL encoder weights into TransformerUNet.enc1 ...")
        ssl_state = torch.load(
            "checkpoints/ssl_enc1.pth", map_location=device
        )
        model.enc1.load_state_dict(ssl_state, strict=False)

    else:
        raise ValueError(f"Unknown config.name: {config.name}")

    # ------------------- Loss / Optim / Scheduler -------------------
    loss_func = build_loss()  # Dice + BCE
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.trainer.n_epoch
    )

    # ------------------- W&B -------------------
    wandb.init(project="ntucsie_ml_hw2", config=dict(model=config.name))
    wandb.watch(model, log="all")

    # ------------------- Trainer -------------------
    trainer = Trainer(
        model=model.to(device),
        device=device,
        criterion=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        eval_func=DiceScore(),
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_path=config.checkpoint.path,
        logger=wandb,
    )

    trainer.fit(**config.trainer)
