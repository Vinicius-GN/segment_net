import numpy as np 
import open3d as o3d
import os

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from typing import List
from tqdm import tqdm

from ..preprocessing.color import printH
from ..utils import hex_to_rgb

class Segdataset(Dataset):
    
    def check_samples(self, samples:List[str])->List[str]:

        keep_idx = []
        
        for idx, s in tqdm(enumerate(samples), total=len(samples)):
            img_path,  label_path = self.get_paths(s)
            if not (os.path.exists(img_path) and\
                    os.path.exists(label_path)):       
                printH("[check_samples]", f"file not found!\n\t{img_path}\n\t{label_path}", "w")
                
            else:
                keep_idx.append(idx)  
                                     
        return list(np.asarray(samples)[keep_idx])

    def _rgb_to_code(self, rgb_arr: np.ndarray) -> np.ndarray:
        rgb = rgb_arr.astype(np.uint32, copy=False)
        return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]

    def _build_color_lut(self) -> None:

        color_map = self.mapping["color_map"]  
        class_id = self.mapping["class_id"]   

        keys, vals = [], []
        for hx, name in color_map.items():
            r, g, b = hex_to_rgb(hx.replace("#", ""))  # returns (R,G,B)
            code = (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b)
            keys.append(code)
            vals.append(class_id[name])

        keys = np.asarray(keys, dtype=np.uint32)
        vals = np.asarray(vals, dtype=np.int16)

        sorter = np.argsort(keys)
        self.lut_sorted_keys = keys[sorter]
        self.lut_sorted_vals = vals[sorter]

    def map_class_from_color(self, source: np.ndarray, path: str) -> np.ndarray:
        # class_id conversion using a precomputed mapping.

        if source.ndim == 3 and source.shape[2] == 3:
            # (H, W, 3) -> pack to codes
            codes = self._rgb_to_code(source)  # (H, W)
            flat_codes = codes.ravel()  # (H*W,)
            out_shape = source.shape[:2]  # (H, W)
            add_channel = True
        elif source.ndim == 2 and source.shape[1] == 3:
            # (N, 3)
            flat_codes = self._rgb_to_code(source)             
            out_shape = (source.shape[0],)                    
            add_channel = True
        else:
            raise ValueError(f"Unsupported mask shape: {source.shape}")

        uniq_codes, inv = np.unique(flat_codes, return_inverse=True)

        idx = np.searchsorted(self.lut_sorted_keys, uniq_codes)
        in_range = (idx < self.lut_sorted_keys.size)

        matches = np.zeros_like(in_range, dtype=bool)
        matches[in_range] = (self.lut_sorted_keys[idx[in_range]] == uniq_codes[in_range])

        mapped = np.zeros_like(uniq_codes, dtype=np.int64)
        mapped[matches] = self.lut_sorted_vals[idx[matches]].astype(np.int64, copy=False)

        target_flat = mapped[inv]                               # (H*W,) or (N,)
        if len(out_shape) == 2:
            target = target_flat.reshape(out_shape)             # (H, W)
        else:
            target = target_flat                                # (N,)

        if add_channel:
            target = target[..., None]                          # (H, W, 1) or (N, 1)

        if not np.any(target):
            printH("[map_class_from_color][WARN]",f"All mapped values are 0 (unknown). Path: {path}", "w")

        return target.astype(np.int64, copy=False)
    
    def _build_labelids_lut(self) -> None:
        self.labelids_lut = None
        try:
            cid_labels = self.mapping.get("class_id_labels", None)   
            cid_13     = self.mapping.get("class_id", None)        

            if cid_labels and cid_13:
                idfine_to_name = {int(v): k for k, v in cid_labels.items()}
                max_id = max(idfine_to_name.keys()) if idfine_to_name else 0
                lut = np.zeros(max_id + 1, dtype=np.int64)  

                for idfine, name in idfine_to_name.items():
                    lut[idfine] = int(cid_13.get(name, 0))  

                self.labelids_lut = lut
                printH("[Segdataset][_build_labelids_lut]", f"LUT labelids built with {lut.size} entries.", "i")
            else:
                printH("[Segdataset][_build_labelids_lut][INFO]", "No 'class_id_labels' or 'class_id' in mapping.", "i")
        except Exception as e:
            printH("[Segdataset][_build_labelids_lut][WARN]", f"Failed to construct LUT: {e}", "w")
            self.labelids_lut = None

    def map_class_from_labelids(self, source: np.ndarray, path: str) -> np.ndarray:
        if source.ndim == 3:
            if source.shape[2] == 1:
                source = source[..., 0]
            elif source.shape[2] == 3:
                if np.all(source[...,0] == source[...,1]) and np.all(source[...,0] == source[...,2]):
                    source = source[..., 0]
                else:
                    source = source[..., 0]
            else:
                raise ValueError(f"Unsupported mask shape for labelids: {source.shape}")
        elif source.ndim != 2:
            raise ValueError(f"Unsupported mask shape for labelids: {source.shape}")

        ids = source.astype(np.int64, copy=False)

        if self.labelids_lut is not None:
            max_idx = self.labelids_lut.size - 1
            ids_clipped = np.clip(ids, 0, max_idx)
            ids_mapped = self.labelids_lut[ids_clipped]
        else:
            ids_mapped = ids

        out = ids_mapped[..., None]  # (H,W,1)

        if not np.any(out):
            printH("[map_class_from_labelids][WARN]", f"All values mapped to para 0 (unknown). Path: {path}", "w")

        return out

    def _build_target_palette(self) -> None:
        self.target_palette = {}
        try:
            tclasses = self.mapping.get("target_classes", {})
            tcolors  = self.mapping.get("target_color", {})
            for name, id13 in tclasses.items():
                hx = tcolors.get(name, "#000000")
                self.target_palette[int(id13)] = hex_to_rgb(hx.replace("#", ""))
            printH("[Segdataset][_build_target_palette]", f"Palette with {len(self.target_palette)} colors.", "i")
        except Exception as e:
            printH("[Segdataset][_build_target_palette][WARN]", f"Fail to build pallete: {e}", "w")
            self.target_palette = {}

    def colorize_ids_13(self, ids_hw1: np.ndarray) -> np.ndarray:
        assert ids_hw1.ndim == 3 and ids_hw1.shape[2] == 1, f"Expected (H,W,1), got {ids_hw1.shape}"
        ids = ids_hw1[..., 0]
        h, w = ids.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        if not hasattr(self, "target_palette") or not self.target_palette:
            self._build_target_palette()
        for k, (r, g, b) in self.target_palette.items():
            rgb[ids == k] = (r, g, b)
        return rgb

    
    def __len__(self):
       return len(self.samples)