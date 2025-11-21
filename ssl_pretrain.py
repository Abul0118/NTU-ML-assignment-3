import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.dataset import MedicalImageDataset, LoadImaged, ToTensord, ResizeImaged
from src.model_ssl_encoder import build_ssl_encoder
from src.utils import set_random_seed


def add_noise(x, sigma=0.1):
    return x + sigma * torch.randn_like(x)


if __name__ == "__main__":
    set_random_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data for SSL: only images, no labels needed ---
    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            ToTensord(keys=["image"]),
            ResizeImaged(keys=["image"], size=(224, 224)),
        ]
    )

    dataset = MedicalImageDataset(
        data_dir="dataset/train",
        has_label=False,
        transform=transform,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # --- SSL model ---
    model = build_ssl_encoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epoch = 20

    for epoch in range(n_epoch):
        model.train()
        running_loss = 0.0

        for batch in loader:
            img = batch["image"].to(device)  # (B,1,H,W)
            noisy = add_noise(img)

            recon = model(noisy)

            loss = criterion(recon, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"[SSL] Epoch {epoch+1}/{n_epoch}  Loss: {epoch_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    # save only enc1, which matches TransformerUNet.enc1
    torch.save(model.enc1.state_dict(), "checkpoints/ssl_enc1.pth")
    print("✅ Saved SSL encoder weights to checkpoints/ssl_enc1.pth")
