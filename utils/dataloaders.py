
import warnings

from typing import Dict, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

from utils.preprocessing import PILNumpyToTensor
from utils.a2d2 import A2D2Dataset
from utils.rudg import RUDGDataset
from utils.rellis3d import Rellis3DDataset
from utils.goose import GooseDataset
from utils.bdd100k import BDD100KDataset


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
    transform=None
)->Dataset:
    
    if config.get("data").get("dataset") == "a2d2":
        return A2D2Dataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform
                )
    elif config.get("data").get("dataset") == "rugd":
        return RUDGDataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform
                )
    elif config.get("data").get("dataset") == "rellis3d":
        return Rellis3DDataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform
            )
    elif config.get("data").get("dataset") == "goose":
        return GooseDataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform
            )
    elif config.get("data").get("dataset") == "bdd100k":
        return BDD100KDataset(
                    set_name   = set_name,
                    config     = config,
                    num_samples= num_samples,
                    transform = transform
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
    return_plot_sets:bool=True
)->Union[
    Tuple[DataLoader, DataLoader, DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader]
]:
    
    w, h = config.get("image").get("image_size")  
        
    train_transform = A.Compose([
        A.SmallestMaxSize(max_size=w * 2, p=1.0),
        A.RandomCrop(height=h, width=w, p=1.0),
        A.OneOf([
            A.NoOp(),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Compose([
                A.HorizontalFlip(p=1.0), 
                A.VerticalFlip(p=1.0)
            ]),
            A.Affine(
                translate_percent={"x": 0.02, "y": 0.02}, 
                scale=(1.0, 1.1), 
                rotate=0,
                p=0.2
            ),
        ], p=0.50),
        A.OneOf([
            A.NoOp(),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(brightness=0.2, 
                          contrast=0.2, 
                          saturation=0.2, 
                          hue=0.1,
                          p=1.0),
            A.RandomGamma(p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, 
                                 sat_shift_limit=30, 
                                 val_shift_limit=20, 
                                 p=1.0),
            A.PlanckianJitter(mode="blackbody", 
                              temperature_limit=(3000, 9000),
                              sampling_method="uniform",
                              p=1.0),
        ], p=0.80),   
        A.OneOf([
            A.NoOp(),   
            A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=0.50),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])    
    
    val_transform = A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
      

    train_set = get_dataset(
                    config     = config,
                    set_name   = 'train',
                    num_samples= config.get("data").get("num_samples"),
                    transform = train_transform
                )
    val_set = get_dataset(
                    config     = config,
                    set_name   = 'val',
                    num_samples= config.get("data").get("num_samples"),
                    transform = val_transform
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
                    transform = sample_transform
                )
        plot_val_set = get_dataset(
                    config     = config,
                    set_name   = 'val_plot',      
                    num_samples= config.get("data").get("num_samples_plot"),
                    transform = sample_transform
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
    config:Dict
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
                    transform = test_transform
                )
    
    test_dataloader   = DataLoader(
                            test_set, 
                            batch_size=config.get("data").get("batch_size")
                        )
            
    
    return test_dataloader



