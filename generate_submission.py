from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import Compose
from tqdm import tqdm

from src.dataset import MedicalImageDataset, LoadImaged, ToTensord, ResizeImaged
from src.utils import dict_to_device, set_random_seed
from src.model import UNetSmall


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--public_dir', type=str)
    parser.add_argument('--private_dir', type=str)
    return parser.parse_args()


def rle(image: np.ndarray) -> str:
    indices = np.where(image.flatten(order='F'))[0] + 1
    if len(indices) == 0:
        return "-1"
    prev, code = -1, []
    for ind in indices:
        if ind > prev + 1:
            code.append(ind)
            code.append(0)
        code[-1] += 1
        prev = ind

    return ' '.join(map(str, code))


def infer(dataset, model, device, prefix):
    pred_dict = {}
    model.eval()
    with torch.inference_mode():
        for data in tqdm(dataset):
            data = dict_to_device(data, device)
            image = data["image"].unsqueeze(0)   # (1,1,H,W)
            preds = model(image)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            preds = preds.squeeze().cpu().numpy()

            code = rle(preds)
            filename = data["filename"]
            pred_dict[f"{prefix}_{filename}"] = code
    return pred_dict


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.config}.yaml"))

    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            ToTensord(keys=["image"]),
            ResizeImaged(keys=["image"], size=(
                config.transform.resized_imaged.height,
                config.transform.resized_imaged.width
            )),
        ]
    )

    public_dataset = MedicalImageDataset(
        data_dir=args.public_dir, has_label=False, transform=transform
    )
    private_dataset = MedicalImageDataset(
        data_dir=args.private_dir, has_label=False, transform=transform
    )

    # -----------------------------------------------------------
    # 🔥 MODEL SWITCHING BASED ON CONFIG.NAME (ONLY THIS ADDED)
    # -----------------------------------------------------------
    if config.name == "cnn_unet":
        from src.model import UNetSmall
        model = UNetSmall(in_ch=1, out_ch=1, base=16)

    elif config.name == "transformer_unet":
        from src.model_transformer_unet import build_model
        model = build_model()

    elif config.name == "transformer_unet_ssl":
        from src.model_transformer_unet import build_model
        model = build_model()

        print("🔁 Loading SSL encoder weights into TransformerUNet.enc1 ...")
        ssl_state = torch.load("checkpoints/ssl_enc1.pth", map_location="cpu")
        model.enc1.load_state_dict(ssl_state, strict=False)

    elif config.name == "transformer_unet_ssl_aug":
        from src.model_transformer_unet import build_model
        model = build_model()

        print("🔁 Loading SSL encoder weights into TransformerUNet.enc1 (AUG MODE) ...")
        ssl_state = torch.load("checkpoints/ssl_enc1.pth", map_location="cpu")
        model.enc1.load_state_dict(ssl_state, strict=False)

    else:
        raise ValueError(f"❌ Unknown config.name: {config.name}")

    
    model.load_state_dict(torch.load(config.checkpoint.path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    public_pred = infer(public_dataset, model, device, "public")
    private_pred = infer(private_dataset, model, device, "private")
    all_pred = {**public_pred, **private_pred}

    df = pd.DataFrame.from_dict(all_pred, orient="index", columns=["label"])
    df.index = df.index.str.replace("public_", "").str.replace("private_", "")
    df.index.name = "row ID"
    df.reset_index().to_csv("submission.csv", index=False)

    print("✅ submission.csv created!")
