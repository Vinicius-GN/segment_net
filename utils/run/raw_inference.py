import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader

from utils.preprocessing.color import printH
from utils.head.segmentnet import SegmentNet

def _make_mosaic(rgb: np.ndarray, pred_rgb: np.ndarray, pad: int = 0) -> np.ndarray:
    if pad > 0:
        h = rgb.shape[0]
        spacer = np.zeros((h, pad, 3), dtype=np.uint8)
        return np.concatenate([rgb, spacer, pred_rgb], axis=1)
    return np.concatenate([rgb, pred_rgb], axis=1)

def run_raw_inference(
    model: SegmentNet,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    config: Dict
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    lut = getattr(dataloader.dataset, "id_to_color_lut", None)
    if lut is None:
        raise RuntimeError("id_to_color_lut not found in dataset. Make sure the loader builds it.")
    max_idx = lut.shape[0] - 1

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            x_img, rgbs, paths = batch  
            x_img = x_img.to(device)

            logits = model(x_img)
            preds = torch.argmax(logits, dim=1).cpu().numpy() 

            if isinstance(rgbs, torch.Tensor):
                rgbs_np = rgbs.numpy()
            else:
                rgbs_np = rgbs  
            if rgbs_np.ndim == 4 and rgbs_np.shape[1] == 3:
                rgbs_np = np.transpose(rgbs_np, (0, 2, 3, 1))
            rgbs_np = rgbs_np.astype(np.uint8, copy=False)

            for pred, rgb, path in zip(preds, rgbs_np, paths):
                pred = np.clip(pred.astype(np.int64, copy=False), 0, max_idx)
                pred_rgb = lut[pred]  

                mosaic = _make_mosaic(rgb, pred_rgb, pad=8)

                base = os.path.splitext(os.path.basename(path))[0]
                out_mask = os.path.join(save_dir, f"{base}_mask.png")
                out_mosaic = os.path.join(save_dir, f"{base}_mosaic.png")

                Image.fromarray(pred_rgb).save(out_mask)
                Image.fromarray(mosaic).save(out_mosaic)

    printH("[Image Segmentation][raw_inference]", "done!", "o")
