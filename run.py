import os
import argparse 
import configparser
import json
import datetime
import shutil
import warnings

from typing import Dict, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary


from utils.utils import split_config
from utils.color import printH
from utils.loss import SegmentLoss

from utils.dataloaders import get_train_val_dataloaders, get_test_dataloader

from utils.training import train_model
from utils.testing import test_model

from utils.segmentnet import SegmentNet

"""
    parse command line arguments    
    Returns:
        args (argparse.Namespace): parsed arguments
"""
def parse_arguments()-> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--cfg",
		type=str,
		required=True,
		help="Directory where the config file is saved",
	)

	return parser.parse_args()


"""
    main function to train.py script
    Args:
        config (Dict): configuration dictionary
        config_path (str): path to the configuration file
    Returns:
        None
"""
def main(config:Dict, config_path:str):
    
    config = split_config(config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if (config.get("mode").get("mode") == "train") or\
       (config.get("mode").get("mode") == "resume"):
            
        train_dataloader, val_dataloader, plot_train_dataloader, plot_val_dataloader =\
            get_train_val_dataloaders(config)
            
    elif (config.get("mode").get("mode") == "test"):
        
        test_dataloader =\
            get_test_dataloader(config)
    else:
        raise ValueError(f"Invalid mode: {config.get('mode').get('mode')}. "
                         "Valid modes are 'train', 'resume', or 'test'.")
        
            
    model = SegmentNet(config=config)
    model.to(device)

    printH("[Image Segmentation][main]", f"model summary:", "i")
    w, h = config.get("image").get("image_size")  
    summary(model, input_size=(1, 3, h, w))
    
    criterion = SegmentLoss(config=config, device=device)

    optimizer = torch.optim.AdamW(
                    model.parameters(), 
                    lr=config.get("model").get("learning_rate"), 
                    weight_decay=config.get("model").get("weight_decay")
                )
    
    epochs = config.get("train").get("epochs")
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
   
    if config.get("mode").get("mode") == "train":

        data_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backbone_name = config.get("backbone").get("type")
        head_name = config.get("head").get("type")
        attn_name =  config.get("attention").get("type")
        dataset_name = config.get("data").get("dataset")
        
        log_dir = os.path.join(config.get("dirs").get("logs"), f"{dataset_name}_{backbone_name}_{attn_name}_{head_name}_{data_str}")

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        
        os.makedirs(log_dir)
        shutil.copy(config_path,
                    os.path.join(log_dir, "config.ini"))

        printH("[Image Segmentation][main]", f"log_dir: {log_dir}", "i")

        history = {
                    "train":{
                        "loss": [],
                        "inf_time":[],
                        "image":{
                                "dice":[],
                                "confmat":[]
                        }
                    },
                    "val":{
                        "loss": [],
                        "inf_time":[],
                        "image":{
                                "dice":[],
                                "confmat":[]
                        }
                    },
                    "test":{
                        "loss": [],
                        "inf_time":[],
                        "image":{
                                "dice":[],
                                "confmat":[]
                        }
                    },
                    "best_val_epoch":0,
                    "best_val_loss":float("inf"),
                    "last_epoch":0,
                    "time":[]                
                }

        train_model(model=model,
                    train_dataloader=train_dataloader, 
                    val_dataloader=val_dataloader,
                    plot_train_dataloader=plot_train_dataloader,
                    plot_val_dataloader=plot_val_dataloader, 
                    criterion=criterion, 
                    optimizer=optimizer,
                    scheduler=scheduler, 
                    num_epochs=epochs, 
                    device=device, 
                    history=history,
                    save_dir=log_dir,
                    config=config,
                    best_val_loss=float("inf"),
                    init_epoch=0)
    
    elif config.get("mode").get("mode") == "resume":

        printH("[Image Segmentation][main]", f"resume...", "i")

        log_dir = config.get("dirs").get("logs")

        if not os.path.exists(log_dir):
            raise FileNotFoundError(f"Checkpoint folder not found! (log_dir:{log_dir})")
        
        printH("[Image Segmentation][main]", f"log_dir: {log_dir}", "i")

        checkpoint_path = os.path.join(log_dir, "last_model.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found! (checkpoint:{checkpoint_path})")
        
        printH("[Image Segmentation][main]", f"checkpoint: {checkpoint_path}", "i")

        history_path = os.path.join(log_dir, "history.json")

        if not os.path.exists(history_path):
            raise FileNotFoundError(f"History file not found! (history file:{history_path})")
        
        with open (history_path, 'r') as f:
            history = json.load(f)
        
        best_val_loss = history.get("best_val_loss")
        init_epoch =  history.get("last_epoch")+1
       
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)

        train_model(model=model,
                    train_dataloader=train_dataloader, 
                    val_dataloader=val_dataloader,
                    plot_train_dataloader=plot_train_dataloader,
                    plot_val_dataloader=plot_val_dataloader,   
                    criterion=criterion, 
                    optimizer=optimizer, 
                    scheduler=scheduler, 
                    num_epochs=epochs, 
                    device=device, 
                    history=history,
                    save_dir=log_dir,
                    config=config,
                    best_val_loss=best_val_loss,
                    init_epoch=init_epoch)
        
    elif config.get("mode").get("mode") == "test":
        
        printH("[Image Segmentation][main]", f"testing...", "i")

        log_dir = config.get("dirs").get("logs")

        if not os.path.exists(log_dir):
            raise FileNotFoundError(f"Checkpoint folder not found! (log_dir:{log_dir})")
        
        printH("[Image Segmentation][main]", f"log_dir: {log_dir}", "i")

        checkpoint_path = os.path.join(log_dir, "best_model.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found! (checkpoint:{checkpoint_path})")
        
        printH("[Image Segmentation][main]", f"checkpoint: {checkpoint_path}", "i")
                
        history_path = os.path.join(log_dir, "history.json")

        if not os.path.exists(history_path):
            raise FileNotFoundError(f"History file not found! (history file:{history_path})")
        
        with open (history_path, 'r') as f:
            history = json.load(f)
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
                
        test_model(model=model,
                   test_dataloader= test_dataloader,
                   criterion=criterion,    
                   device=device, 
                   save_dir=log_dir,
                   history=history,
                   config=config)
    


if __name__ == "__main__":

    printH("[Image Segmentation][main]", "creating model...", "i")

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