
import warnings

from typing import Dict, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import cv2

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

from utils.preprocessing.preprocessing import PILNumpyToTensor
from utils.dataloaders.raw_inference_loader import get_raw_inference_dataloader
from utils.dataloaders.a2d2 import A2D2Dataset
from utils.dataloaders.rudg import RUDGDataset
from utils.dataloaders.rellis3d import Rellis3DDataset
from utils.dataloaders.goose import GooseDataset
from utils.dataloaders.bdd100k import BDD100KDataset


"""
    get dataset based on the configuration
    Args:        
        config (Dict): configuration dictionary
                       + config.get("data").get("dataset") specifies the name of dataset
        set_name (str): name of the set to load (train, val, or test)
        num_samples (int): number of samples to load, -1 for all samples
        transform (callable, optional): transformation to apply to the dataset
    Returns:
        Dataset: dataset object
"""
def get_dataset(
    config:Dict,
    set_name:str,
    num_samples:int=-1,
    transform=None,
    return_data_path:bool=False
)->Dataset:
    
    if config.get("data").get("dataset") == "a2d2":
        return A2D2Dataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform,
                    return_data_path=return_data_path
                )
    elif config.get("data").get("dataset") == "rugd":
        return RUDGDataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform,
                    return_data_path=return_data_path
                )
    elif config.get("data").get("dataset") == "rellis3d":
        return Rellis3DDataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform,
                    return_data_path=return_data_path
            )
    elif config.get("data").get("dataset") == "goose":
        return GooseDataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform,
                    return_data_path=return_data_path
            )
    elif config.get("data").get("dataset") == "bdd100k":
        return BDD100KDataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform,
                    return_data_path=return_data_path
            )
    else:
        raise ValueError(f"Unknown dataset: {config.get('data').get('dataset')}")
       

"""
    get dataloaders for training and validation datasets
    Args:
        config (Dict): configuration dictionary
        return_plot_sets (bool): whether return the plot sets (default: True)
    Returns:
        train_dataloader (DataLoader): dataloader for training data 
        val_dataloader (DataLoader): dataloader for validation data
        
        - if return_plot_sets is true: 
        plot_train_dataloader (DataLoader): dataloader for plotting training data
        plot_val_dataloader (DataLoader): dataloader for plotting validation data
"""
def get_train_val_dataloaders(
    config:Dict,
    return_plot_sets:bool=True,
    return_data_path:bool=False
)->Union[
    Tuple[DataLoader, DataLoader, DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader]
]:
    ENABLE_AUG = True
    w, h = config.get("image").get("image_size")

    train_transform = A.Compose([
        A.LongestMaxSize(max_size=max(w, h), p=1.0),

        A.PadIfNeeded(
            min_height=h, min_width=w,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            p=1.0
        ),

        A.HorizontalFlip(p=0.5 if ENABLE_AUG else 0.0),
        A.Affine(
            scale=(0.97, 1.03),
            translate_percent={"x": 0.03, "y": 0.03},
            rotate=(-2, 2),
            shear=None,
            p=0.20 if ENABLE_AUG else 0.0
        ),

        A.OneOf([
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
            A.RandomGamma(gamma_limit=(92,108), p=1.0),
            A.PlanckianJitter(mode="blackbody", temperature_limit=(3500, 7500), sampling_method="uniform", p=1.0),
        ], p=0.5 if ENABLE_AUG else 0.0),

        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 20.0), per_channel=False, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=(3,5), p=1.0),
        ], p=0.15 if ENABLE_AUG else 0.0),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.LongestMaxSize(max_size=max(w, h), p=1.0),
        A.PadIfNeeded(
            min_height=h, min_width=w,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
            p=1.0
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
      

    train_set = get_dataset(
                    config     = config,
                    set_name   = 'train',
                    num_samples= config.get("data").get("num_samples"),
                    transform = train_transform,
                    return_data_path = return_data_path
                )
    val_set = get_dataset(
                    config     = config,
                    set_name   = 'val',
                    num_samples= config.get("data").get("num_samples"),
                    transform = val_transform,
                    return_data_path = return_data_path
                )
    
    train_dataloader = DataLoader(
                            train_set, 
                            batch_size=config.get("data").get("batch_size"), 
                            shuffle=True
                        )
    val_dataloader   = DataLoader(
                            val_set, 
                            batch_size=config.get("data").get("batch_size")
                        )
    
    if return_plot_sets:
        
        sample_transform = A.Compose([
            A.Resize(height=h, width=w),
            ToTensorV2()
        ])
        
        plot_train_set = get_dataset(
                    config     = config,
                    set_name   = 'train_plot',
                    num_samples= config.get("data").get("num_samples_plot"),
                    transform = sample_transform,
                    return_data_path = return_data_path
                )
        plot_val_set = get_dataset(
                    config     = config,
                    set_name   = 'val_plot',      
                    num_samples= config.get("data").get("num_samples_plot"),
                    transform = sample_transform,
                    return_data_path = return_data_path
                )

        plot_train_dataloader = DataLoader(
                                plot_train_set,
                                batch_size=1,
                                shuffle=False
                              )
        plot_val_dataloader = DataLoader(
                                plot_val_set,
                                batch_size=1,
                                shuffle=False
                              )        
         
        return (train_dataloader, 
                val_dataloader, 
                plot_train_dataloader, 
                plot_val_dataloader)
    else: 
        return (train_dataloader, 
                val_dataloader)


"""
    get dataloader for testing datasets
    Args:
        config (Dict): configuration dictionary
    Returns:
        test_dataloader (DataLoader): dataloader for testing data 
"""
def get_test_dataloader(
    config:Dict,
    return_data_path:bool= False
)->Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    
    w, h = config.get("image").get("image_size")  
        
    test_transform = A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
        
    test_set = get_dataset(
                    config     = config,
                    set_name   = 'test',
                    num_samples= config.get("data").get("num_samples"),
                    transform = test_transform,
                    return_data_path = return_data_path
                )
    
    test_dataloader   = DataLoader(
                            test_set, 
                            batch_size=config.get("data").get("batch_size")
                        )
            
    
    return test_dataloader



