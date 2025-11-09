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

from ..preprocessing.color import printH
from ..utils import hex_to_rgb
from .segdataset import Segdataset

class BDD100KDataset(Segdataset):

    def __init__(self, 
                 set_name,
                 config,
                 num_samples:int=-1,
                 transform=None,
                 return_data_path:bool=False           
    ):
        super(BDD100KDataset, self).__init__()
        
        self.set_name = set_name
        self.name = set_name
        self.return_data_path=return_data_path        
        self.config = config    
        self.data_source = self.config.get("dirs").get("data")   
        
        if self.set_name.find("_") > 0:
            self.set_name = self.set_name.split("_")[0]

        printH(f"[BDD100k Dataset][{self.name}]", "creating dataloader...", "i")
        
        self.data_source = os.path.join(self.data_source, "10K")
        if not os.path.exists(self.data_source):
            raise FileNotFoundError(f"data_dir not found! ({self.data_source})")
                       
        self.transform = transform
        self.num_samples = num_samples
               
        if not os.path.exists(self.config.get("dirs").get("mapping")):
            raise FileNotFoundError(f"Metadata path not found! ({self.config.get('dirs').get('mapping')})")

        with open (self.config.get("dirs").get("mapping"), 'r') as f:
            self.mapping = json.load(f)
            
        printH(f"[BDD100k Dataset][{self.name}]", "loaded the metadata!", "i")       

        if self.set_name == "test":
            self.set_name = "val"
            
        self.config.get("splits").get(self.set_name)
        set_source = os.path.join(self.config.get("splits").get(self.set_name), "images", self.set_name)
        
        # source data path
        self.samples= []
        if not os.path.exists(set_source):
            raise FileNotFoundError(f"Data set source not found! ({self.set_name}:{set_source})")
        else:
            self.samples.extend(glob.glob(set_source + "/*.jpg"))
                        
        self.samples = self.check_samples(self.samples)
        self.samples = np.asarray(self.samples)
        
        printH(f"[BDD100k Dataset][{self.name}]", f"found {len(self.samples)} samples!", "i")

        if self.num_samples > 0:
            np.random.shuffle(self.samples)
            self.samples = self.samples[:self.num_samples]
            printH(f"[BDD100k Dataset][{self.name}]", f"using {len(self.samples)} samples!", "i")
        
        self._build_color_lut()
        printH("[BDD100k Dataset][init]", f"color LUT built with {self.lut_sorted_keys.size} entries.", "i")
        
    def get_paths(self, path:str)->Tuple[str, str, str]:

        label_path = path.replace("/images/", "/semantic_labels/color_labels/")
        label_path = label_path.replace(".jpg", "_train_color.png")
       
        return (path, label_path)
        
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
            print(img.shape, mask.shape, image_path)
            printH("[BDD100k Dataset][__getitem__][ERROR]", e, "e") 
                
        if self.return_data_path:
            return (img, mask, image_path)
        else:
            return (img, mask)