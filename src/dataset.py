import glob
import os
from typing import Optional, Union, List, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, resize


class MedicalImageDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        has_label: bool = True,
        transform: Optional[callable] = None
    ):
        self.data_dir = data_dir
        self.data_list = glob.glob(f"{data_dir}/image/*.tif")
        self.has_label = has_label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        if self.has_label:
            data =  {
                "filename": os.path.basename(self.data_list[idx]),
                "image": self.data_list[idx],
                "label": self.data_list[idx].replace("image", "label"),
            }
        else:
            data = {
                "filename": os.path.basename(self.data_list[idx]),
                "image": self.data_list[idx],
            }

        if self.transform:
            return self.transform(data)

        return data


class BaseTransform:
    def __init__(self, keys: List[str], *args, **kwargs):
        self.keys = keys
        self._parse_var(*args, **kwargs)

    def _parse_var(self, *args, **kwargs):
        pass

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        raise NotImplementedError

    def __call__(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        for key in self.keys:
            if key in data:
                data[key] = self._process(data[key], *args, **kwargs)
            else:
                raise KeyError(f"{key} is not a key in data.")
        return data


class LoadImaged(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(LoadImaged, self).__init__(keys, **kwargs)

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        return Image.open(data)


class ToTensord(BaseTransform):
    def __init__(self, keys, **kwargs):
        super(ToTensord, self).__init__(keys, **kwargs)

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        return to_tensor(data)


class ResizeImaged(BaseTransform):
    def __init__(self, keys, size, **kwargs):
        super(ResizeImaged, self).__init__(keys, **kwargs)
        self.size = size

    def _process(self, data: Dict[str, any], *args, **kwargs) -> Dict[str, any]:
        return resize(data, self.size)
