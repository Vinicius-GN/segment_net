import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

            out = model(x_img, return_att=True)

            if isinstance(out, tuple):
                logits, attn_maps = out
            else:
                logits = out
                attn_maps = None

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            attn_maps_np = None
            if attn_maps is not None:
                if attn_maps.dim() == 3:
                    attn_maps = attn_maps.unsqueeze(1)
                elif attn_maps.dim() != 4:
                    attn_maps = None
                if attn_maps is not None:
                    H, W = logits.shape[2], logits.shape[3]
                    attn_maps = F.interpolate(
                        attn_maps,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False
                    )
                    attn_maps_np = attn_maps.detach().cpu().numpy()

            if isinstance(rgbs, torch.Tensor):
                rgbs_np = rgbs.numpy()
            else:
                rgbs_np = rgbs
            if rgbs_np.ndim == 4 and rgbs_np.shape[1] == 3:
                rgbs_np = np.transpose(rgbs_np, (0, 2, 3, 1))
            rgbs_np = rgbs_np.astype(np.uint8, copy=False)

            for i, (pred, rgb, path) in enumerate(zip(preds, rgbs_np, paths)):
                pred = np.clip(pred.astype(np.int64, copy=False), 0, max_idx)
                pred_rgb = lut[pred]

                mosaic = _make_mosaic(rgb, pred_rgb, pad=8)

                base = os.path.splitext(os.path.basename(path))[0]
                out_mask = os.path.join(save_dir, f"{base}_mask.png")
                out_mosaic = os.path.join(save_dir, f"{base}_mosaic.png")

                Image.fromarray(pred_rgb).save(out_mask)
                Image.fromarray(mosaic).save(out_mosaic)

                if attn_maps_np is not None:
                    attn_i = attn_maps_np[i, 0]
                    out_attn = os.path.join(save_dir, f"{base}_attn.npy")
                    np.save(out_attn, attn_i)

    printH("[Image Segmentation][raw_inference]", "done!", "o")
