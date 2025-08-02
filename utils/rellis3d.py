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
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from typing import List, Tuple
from PIL import Image
from tqdm import tqdm

from .color import printH
from .utils import hex_to_rgb

class Rellis3DDataset(Dataset):

    def __init__(self, 
                 set_name,
                 config,
                 num_samples:int=-1,
                 transform=None           
    ):
        super(Rellis3DDataset, self).__init__()
        
        self.set_name = set_name
        self.name = set_name        
        self.config = config    
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


    def map_class_from_color(self, source, path):
        
        unique_classes =  {hex_to_rgb(hex.replace("#","")) : self.mapping.get("class_id")[value]
                           for hex, value in self.mapping.get("color_map").items()}
                
        if source.ndim == 2:
            dim = (source.shape[0], 1)
        else:
            dim = (source.shape[0], source.shape[1], 1)

        target = np.zeros(dim)
        
        for uc, value in unique_classes.items():
            if source.ndim == 2:  # source shape is [n, 3]
                idx_uc = np.where((source == uc).all(axis=1))[0]
                target[idx_uc] = value
            elif source.ndim == 3:  # source shape is [n, m, 3]                
                idx_uc = np.where((source == uc).all(axis=2))
                target[idx_uc[0], idx_uc[1]] = value
            else:
                continue
            
        if np.all(target == 0):
            printH("[Rellis-3D Dataset][map_class_from_color][WARN]", 
                   f"all mapped values are 0 (unknown)! ({path})", "w")
            
        return target

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
                
        return (img, mask)