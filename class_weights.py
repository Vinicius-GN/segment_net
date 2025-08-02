import os
import sys
import argparse 
import configparser
import copy
import json
import datetime
import shutil
import time
from datetime import timedelta
import warnings
import cv2

from collections import Counter
import matplotlib.pyplot as plt
from typing import Dict
import numpy as np
from tqdm import tqdm
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

from utils.utils import split_config
from utils.color import printH
from utils.tables import print_class_distribution_table
from utils.dataloaders import get_dataset

def parse_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--cfg",
		type=str,
		required=True,
		help="Directory where the config file is saved",
	)

	return parser.parse_args()

def compute_class_weights(class_count: dict, num_classes: int):
    total_pixels = sum(class_count.values())
    
    weights = {}
    for c in range(num_classes):
        count = class_count.get(c, 0)
        if count == 0:
            weights[c] = 0.0  
        else:
            weights[c] = total_pixels / (num_classes * count)
    return weights

def estimate_distribution(data:DataLoader):
    
    total = len(data) 
    
    class_count = {}
    total_pixels = 0
    
    for idx, (_, y_img) in tqdm(enumerate(data), total=total):
        
        y_img = y_img.flatten().numpy()        
        y_classes, y_counts = np.unique(y_img, return_counts=True)
        
        for c, n in zip(y_classes, y_counts):
            class_count[c] = class_count.get(c, 0) + n
            
        total_pixels+=len(y_img)

    class_distribution = {k:(v*100/total_pixels) for k, v in class_count.items()}
    
    return class_count, class_distribution


def get_dataloader(config, set_name):
    
    sample_transform = A.Compose([
            ToTensorV2()
        ])
    
    dataset = get_dataset(
                config     = config,
                set_name   = set_name,
                num_samples= -1,
                transform = sample_transform
            )

    return DataLoader(
                dataset,
                batch_size=1,
                shuffle=False
           )

def main(config:Dict, config_path:str):

    config = split_config(config)
    config.get("data").update(batch_size=1)
    num_classes = config.get("image").get("num_classes")
    
    with open (config.get("dirs").get("mapping"), 'r') as f:
            mapping = json.load(f)
        
    mapping = {v:k for k, v in mapping.get("target_classes").items()}
       
    train_dataloader = get_dataloader(config, "train")
    val_dataloader   = get_dataloader(config, "val")
    test_dataloader  = get_dataloader(config, "test")

    printH("[Class Weights][main][train]", "estimating class weights for images:", "i")
    train_count, train_distribution = estimate_distribution(train_dataloader)
    train_weights = compute_class_weights(train_count, num_classes)

    printH("[Class Weights][main][val]", "estimating class weights for images:", "i")
    val_count, val_distribution = estimate_distribution(val_dataloader)
    val_weights = compute_class_weights(val_count, num_classes)
    
    printH("[Class Weights][main][test]", "estimating class weights for images:", "i")
    test_count, test_distribution = estimate_distribution(test_dataloader)
    test_weights = compute_class_weights(test_count, num_classes)
    
    
    printH("[Class Weights][main]", "result!!!", "i")
    
    if config.get("data").get("dataset") == "bdd100k" or\
       config.get("data").get("dataset") == "goose":
        print_class_distribution_table(
            mapping,
            train_distribution, train_weights, train_count,
            val_distribution, val_weights, val_count,
            test_distribution, test_weights, test_count,
            include_test_for_final_distribution=False
        )
    else:
        print_class_distribution_table(
            mapping,
            train_distribution, train_weights, train_count,
            val_distribution, val_weights, val_count,
            test_distribution, test_weights, test_count,
            include_test_for_final_distribution=True
        )
    
    printH("[Class Weights][main]", "done!!!", "o")


if __name__ == "__main__":

    printH("[Class Weights][main]", "running....", "i")

    args = parse_arguments()

    if not os.path.exists(args.cfg):
        raise FileNotFoundError(("config file not found!"
                                    f"(cfg:{args.cfg})"))

    config = configparser.ConfigParser()
    config.read(args.cfg)    
    config = config._sections   

    for key, param in config.get("DIRS").items():
        if not os.path.exists(param):
            raise FileNotFoundError((f"{key} not found!"
                                        f"({key}:{param})"))
            
    main(config, args.cfg)
