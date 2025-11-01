from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import Compose
from tqdm import tqdm

from src.dataset import MedicalImageDataset, LoadImaged, ToTensord, ResizeImaged
# from src.model import YOUR_MODEL
from src.utils import dict_to_device, set_random_seed


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


def infer(dataset, model, device):
    pred_dict = {}
    model.eval()
    with torch.inference_mode():
        for data in tqdm(dataset):
            data = dict_to_device(data, device)
            # TODO: Implement inference step
            code = rle(pred)
            pred_dict[data["filename"]] = code
    return pred_dict


if __name__ == "__main__":
    set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.config}.yaml"))
    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            ToTensord(keys=["image"]),
            ResizeImaged(keys=["image"], **config.transform.resized_imaged),
        ]
    )
    public_dataset = MedicalImageDataset(data_dir=args.public_dir, has_label=False, transform=transform)
    private_dataset = MedicalImageDataset(data_dir=args.private_dir, has_label=False, transform=transform)

    #TODO: Prepare the model for inference
    model = YOUR_MODEL
    model.load_state_dict(torch.load(config.checkpoint.path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    public_pred = infer(public_dataset, model, device)
    private_pred = infer(private_dataset, model, device)
    all_pred = public_pred | private_pred
    df = pd.DataFrame.from_dict(all_pred, orient="index", columns=["label"])
    df.index.name = "row ID"
    df.reset_index().to_csv("submission.csv", index=False)
