import os
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .segdataset import Segdataset
from ..preprocessing.color import printH
from ..utils import hex_to_rgb

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def _list_images_recursive(root: str) -> List[str]:
    files = []
    for d, _, fns in os.walk(root):
        for fn in sorted(fns):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                files.append(os.path.join(d, fn))
    return files

class RawInferenceDataset(Segdataset):
    def __init__(
        self,
        config: Dict,
        transform=None,
        return_data_path: bool = True
    ):
        super(RawInferenceDataset, self).__init__()
        self.name = "raw_inference"
        self.config = config
        self.transform = transform
        self.return_data_path = return_data_path

        self.data_source = self.config.get("dirs").get("data")
        if not self.data_source or not os.path.isdir(self.data_source):
            raise FileNotFoundError(f"data_folder not found! ({self.data_source})")

        printH(f"[Raw-Inference Dataset][{self.name}]", "creating dataloader...", "i")

        mapping_path = self.config.get("dirs").get("mapping")
        if not mapping_path or not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Metadata path not found! ({mapping_path})")
        with open(mapping_path, "r") as f:
            self.mapping = json.load(f)
        printH(f"[Raw-Inference Dataset][{self.name}]", "loaded the metadata!", "i")

        self.samples = _list_images_recursive(self.data_source)
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No images found in: {self.data_source}")
        self.samples = self.check_samples(self.samples)
        self.samples = np.asarray(self.samples)
        printH(f"[Raw-Inference Dataset][{self.name}]", f"found {len(self.samples)} samples!", "i")

        w, h = self.config.get("image").get("image_size")
        self.target_size: Optional[Tuple[int, int]] = (w, h)

        mean = self.config.get("image").get("normalize_mean", None)
        std = self.config.get("image").get("normalize_std", None)
        if isinstance(mean, str):
            mean = [float(x) for x in mean.split(",")]
        if isinstance(std, str):
            std = [float(x) for x in std.split(",")]
        self.norm_mean = mean
        self.norm_std = std

        self._build_id_to_color_lut()
        printH("[Raw-Inference Dataset][init]", f"id->color LUT built with shape {self.id_to_color_lut.shape}.", "i")

    def check_samples(self, samples: List[str]) -> List[str]:
        keep = []
        for s in samples:
            if os.path.exists(s):
                keep.append(s)
            else:
                printH("[Raw-Inference Dataset][check_samples]", f"file not found! {s}", "w")
        return keep

    def _build_id_to_color_lut(self) -> None:
        cls_map = self.mapping.get("target_classes", {})  # name -> id
        color_map = self.mapping.get("target_color", {})  # name -> hex
        id_to_color = {}
        for name, hex_code in color_map.items():
            cls_id = cls_map.get(name)
            if cls_id is not None:
                r, g, b = hex_to_rgb(hex_code.replace("#", ""))
                id_to_color[int(cls_id)] = (int(r), int(g), int(b))
        max_id = max(id_to_color.keys()) if id_to_color else 0
        lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
        for k, v in id_to_color.items():
            lut[k] = v
        self.id_to_color_lut = lut  # (num_classes, 3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.target_size is not None:
            w, h = self.target_size
            img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        if self.transform:
            augmented = self.transform(image=img_resized)
            x = augmented["image"]
        else:
            x = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            if self.norm_mean is not None and self.norm_std is not None:
                mean = torch.tensor(self.norm_mean, dtype=torch.float32)[:, None, None]
                std = torch.tensor(self.norm_std, dtype=torch.float32)[:, None, None]
                x = (x - mean) / std

        rgb_uint8 = img_resized.astype(np.uint8, copy=False)  
        return (x, rgb_uint8, img_path)

def get_raw_inference_dataloader(config: Dict) -> DataLoader:
    batch_size = config.get("data").get("batch_size", 1)
    num_workers = config.get("data").get("num_workers", 4)
    pin_memory = torch.cuda.is_available()


    dataset = RawInferenceDataset(config=config, transform=None, return_data_path=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return loader
