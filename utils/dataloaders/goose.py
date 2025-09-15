import numpy as np 
import open3d as o3d
import os
import yaml
import pandas as pd
import glob
import cv2
import json
import torch 
import time

import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm

from ..preprocessing.color import printH
from ..utils import hex_to_rgb
from .segdataset import Segdataset

class GooseDataset(Segdataset):

    def __init__(self, 
                 set_name,
                 config,
                 num_samples:int=-1,
                 transform=None,
                 return_data_path:bool=False           
    ):
        super(GooseDataset, self).__init__()

        self.set_name = set_name
        self.name = set_name
        self.return_data_path=return_data_path
        self.config = config   
        self.data_source = self.config.get("dirs").get("data") 
        self.prefer_labelids = True 
        self.label_source_stats = {"color": 0, "labelids": 0}

        
        if self.set_name.find("_") > 0:
            self.set_name = self.set_name.split("_")[0]

        printH(f"[Goose Dataset][{self.name}]", "creating dataloader...", "i")

        if not os.path.exists(self.data_source):
            raise FileNotFoundError(f"data_dir not found! ({self.data_source})")
        
        
        self.transform = transform
        self.num_samples = num_samples
               
        if not os.path.exists(self.config.get("dirs").get("mapping")):
            raise FileNotFoundError(f"Metadata path not found! ({self.config.get('dirs').get('mapping')})")

        with open (self.config.get("dirs").get("mapping"), 'r') as f:
            self.mapping = json.load(f)
            
        printH(f"[Goose Dataset][{self.name}]", "loaded the metadata!", "i")       

        if self.set_name == "test":
            self.set_name = "val"
            
        self.samples= []
                
        if not os.path.exists(os.path.join(self.data_source, "images", self.set_name)):
            raise FileNotFoundError(f"Data sequence not found! ({self.set_name})")
        else:
            self.samples.extend(
                        glob.glob(os.path.join(self.data_source, "images", self.set_name) + "/*/*vis.png")
                    )
                        
        self.samples = self.check_samples(self.samples)
        self.samples = np.asarray(self.samples)
        
    
        printH(f"[Goose Dataset][{self.name}]", f"found {len(self.samples)} samples!", "i")

        if self.num_samples > 0:
            np.random.shuffle(self.samples)
            self.samples = self.samples[:self.num_samples]
            printH(f"[Goose Dataset][{self.name}]", f"using {len(self.samples)} samples!", "i")
        
        self._build_color_lut()
        printH("[Goose Dataset][init]", f"color LUT built with {self.lut_sorted_keys.size} entries.", "i")

        self._build_labelids_lut()     
        self._build_target_palette()   

    def get_paths(self, path: str) -> Tuple[str, str]:
        base = path.replace("/images/", "/labels/")
        color_path    = base.replace("windshield_vis", "color")
        labelids_path = base.replace("windshield_vis", "labelids")

        if self.prefer_labelids and os.path.exists(labelids_path):
            chosen = labelids_path
        elif os.path.exists(color_path):
            chosen = color_path
        elif os.path.exists(labelids_path):
            chosen = labelids_path
        else:
            chosen = color_path 

        if "labelids" in chosen:
            self.label_source_stats["labelids"] += 1
        else:
            self.label_source_stats["color"] += 1

        return (path, chosen)
        
        
    def __getitem__(self, idx:int)->Tuple[torch.Tensor, torch.Tensor]:
        image_path, image_label_path = self.get_paths(self.samples[idx])
        try:
            img = cv2.imread(image_path)[:, :, ::-1]  # BGR->RGB

            if ("labelids" in image_label_path) or image_label_path.endswith("_labelids.png"):
                raw = cv2.imread(image_label_path, cv2.IMREAD_UNCHANGED)  
                mask = self.map_class_from_labelids(raw, image_label_path)   # -> (H,W,1) int64 (0..12)
            else:
                raw = cv2.imread(image_label_path)[:, :, ::-1]               # color RGB
                mask = self.map_class_from_color(raw, image_label_path)      # -> (H,W,1) int64 (0..12)

            if self.transform:
                augmented = self.transform(image=img, mask=mask)
                img  = augmented['image']
                mask = augmented['mask'].long()

        except Exception as e:
            printH("[Goose Dataset][__getitem__][ERROR]", e, "e")

        if self.return_data_path:
            return (img, mask, image_path)
        else:
            return (img, mask)
