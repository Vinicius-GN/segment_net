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
from .a2d2_utils import (get_pc_labels, undistort_image,
                             map_lidar_points_onto_image,
                             undistort_semantic)
from .segdataset import Segdataset

class A2D2Dataset(Segdataset):

    def __init__(self, 
                 set_name,
                 config,
                 num_samples:int=-1,
                 transform=None,
                 return_data_path:bool=False           
    ):
        super(A2D2Dataset, self).__init__()

        self.set_name = set_name
        self.name = set_name
        self.return_data_path=return_data_path
        self.config = config    
        self.data_source = self.config.get("dirs").get("data")   
        
        if self.set_name.find("_") > 0:
            self.set_name = self.set_name.split("_")[0]

        printH(f"[A2D2 Dataset][{self.name}]", "creating dataloader...", "i")

        if not os.path.exists(self.data_source):
            raise FileNotFoundError(f"data_dir not found! ({self.data_source})")
        
        self.transform = transform

        self.num_samples = num_samples

        # metadata
        if not os.path.exists(self.config.get("dirs").get("metadata")):
            raise FileNotFoundError(f"Metadata path not found! ({self.config.get('dirs').get('metadata')})")
        
        with open (self.config.get("dirs").get("metadata"), 'r') as f:
            self.metadata = json.load(f)

        if not os.path.exists(self.config.get("dirs").get("mapping")):
            raise FileNotFoundError(f"Metadata path not found! ({self.config.get('dirs').get('mapping')})")

        with open (self.config.get("dirs").get("mapping"), 'r') as f:
            self.mapping = json.load(f)

        self.mapping_cam_name = {
            "frontleft":"front_left",
            "frontcenter":"front_center",
            "frontright":"front_right",
            "sideleft":"side_left",
            "sideright":"side_right",
            "rearcenter":"rear_center"
        }

        printH(f"[A2D2 Dataset][{self.name}]", "loaded the metadata!", "i")

        # source data path
        self.samples= []

        for seq_name in self.config.get("splits").get(self.set_name):
            if not os.path.exists(os.path.join(self.data_source, seq_name)):
                printH(f"[A2D2 Dataset][{self.name}][WARN]", f"Data sequence not found! ({seq_name})", "w")
            else:
                self.samples.extend(
                            glob.glob(os.path.join(self.data_source, f"{seq_name}/camera/*/*.png"))
                        )
        
        self.samples = self.check_samples(self.samples)
        self.samples = np.asarray(self.samples)

        printH(f"[A2D2 Dataset][{self.name}]", f"found {len(self.samples)} samples!", "i")

        if self.num_samples > 0:
            self.samples = self.samples[:self.num_samples]
            printH(f"[A2D2 Dataset][{self.name}]", f"using {len(self.samples)} samples!", "i")

        self._build_color_lut()
        printH("[A2d2 Dataset][init]", f"color LUT built with {self.lut_sorted_keys.size} entries.", "i")


    def get_paths(self, path:str)->Tuple[str, str, str]:

        label_path = path.replace("/camera/", "/label/")
        label_path = label_path.replace("_camera_", "_label_")
        
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
            printH("[A2D2 Dataset][__getitem__][ERROR]", e, "e") 
      
        if self.return_data_path:
            return (img, mask, image_path)
        else:
            return (img, mask)