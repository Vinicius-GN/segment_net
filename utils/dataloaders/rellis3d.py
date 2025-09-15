import numpy as np 
import open3d as o3d
import os
import yaml
import pandas as pd
import glob
import cv2
import json
import torch 

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm

from ..preprocessing.color import printH
from ..utils import hex_to_rgb
from .segdataset import Segdataset

class Rellis3DDataset(Segdataset):

    def __init__(self, 
                 set_name,
                 config,
                 num_samples:int=-1,
                 transform=None,
                 return_data_path:bool=False          
    ):
        super(Rellis3DDataset, self).__init__()
        
        self.set_name = set_name
        self.name = set_name        
        self.config = config    
        self.return_data_path = return_data_path
        self.data_source = self.config.get("dirs").get("data")   
        
        if self.set_name.find("_") > 0:
            self.set_name = self.set_name.split("_")[0]

        printH(f"[Rellis-3D Dataset][{self.name}]", "creating dataloader...", "i")

        if not os.path.exists(self.data_source):
            raise FileNotFoundError(f"data_dir not found! ({self.data_source})")
                
        self.transform = transform
        self.num_samples = num_samples
               
        if not os.path.exists(self.config.get("dirs").get("mapping")):
            raise FileNotFoundError(f"Metadata path not found! ({self.config.get('dirs').get('mapping')})")

        with open (self.config.get("dirs").get("mapping"), 'r') as f:
            self.mapping = json.load(f)
            
        printH(f"[Rellis-3D Dataset][{self.name}]", "loaded the metadata!", "i")       

        # source data path
        self.samples= []
        self.samples = np.loadtxt(self.config.get("splits").get(self.set_name), dtype=str)[:, 0]
        self.samples = np.char.add(self.data_source + "/", self.samples) 
        
        self.samples = self.check_samples(self.samples)
        self.samples = np.asarray(self.samples)
        
        printH(f"[Rellis-3D Dataset][{self.name}]", f"found {len(self.samples)} samples!", "i")

        if self.num_samples > 0:
            np.random.shuffle(self.samples)
            self.samples = self.samples[:self.num_samples]
            printH(f"[Rellis-3D Dataset][{self.name}]", f"using {len(self.samples)} samples!", "i")
        
        self._build_color_lut()
        printH("[Rellis-3D Dataset][init]", f"color LUT built with {self.lut_sorted_keys.size} entries.", "i")

        
    def get_paths(self, path:str)->Tuple[str, str, str]:

        label_path = path.replace(".jpg", ".png")
        label_path = label_path.replace("pylon_camera_node", "pylon_camera_node_label_color")
        
        return (path, label_path)
    
    def check_samples(self, samples:List[str])->List[str]:

        keep_idx = []
        
        for idx, s in tqdm(enumerate(samples), total=len(samples)):

            img_path,  label_path = self.get_paths(s)
        
            if not (os.path.exists(img_path) and\
                    os.path.exists(label_path)):
                
                printH("[Rellis-3D Dataset][check_samples]", f"file not found!\n\t{img_path}\n\t{label_path}", "w")
                
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
            printH("[Rellis-3D Dataset][map_class_from_color][WARN]",f"All mapped values are 0 (unknown). Path: {path}", "w")

        return target.astype(np.int64, copy=False)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, 
                    idx:int
    )->Tuple[torch.Tensor, torch.Tensor]:        
       
        image_path, image_label_path =\
            self.get_paths(self.samples[idx])  
               
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                      
            mask = cv2.imread(image_label_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            
            mask = self.map_class_from_color(mask, image_label_path)

            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask'].long()  
                
        except Exception as e:
            printH("[Rellis-3D Dataset][__getitem__][ERROR]", e, "e") 
                
        if self.return_data_path:
            return (img, mask, image_path)
        else:
            return (img, mask)