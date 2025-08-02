import os
import json
import time
from datetime import timedelta
import numpy as np

from typing import Dict, Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.utils import hex_to_rgb
from utils.color import printH
from utils.metrics import image_dice, metrics_from_confmat
from utils.tables import print_metric_table
from utils.image_grid import save_grid_samples
from utils.loss import SegmentLoss
from utils.segmentnet import SegmentNet


"""
    train model
    
    Args:
        model (SegmentNet): model to train
        train_dataloader (DataLoader): dataloader for training data
        val_dataloader (DataLoader): dataloader for validation data
        criterion (SegmentLoss): loss function
        optimizer (torch.optim.Optimizer): optimizer for training
        scheduler (torch.optim.lr_scheduler.LRScheduler): learning rate scheduler
        num_epochs (int): number of epochs to train
        device (torch.device): device to train on (cpu or cuda)
        save_dir (str): directory to save the model and history
        config (Dict): configuration dictionary
        history (Dict): dictionary to store training history
        best_val_loss (float): best validation loss so far
        init_epoch (int): initial epoch to start training from
        plot_train_dataloader (Optional[DataLoader]): dataloader for plotting training data
        plot_val_dataloader (Optional[DataLoader]): dataloader for plotting validation data
    Returns:
        None     
"""
def train_model(
    model:SegmentNet,
    train_dataloader:DataLoader, 
    val_dataloader:DataLoader, 
    criterion:SegmentLoss, 
    optimizer:torch.optim.Optimizer,
    scheduler:torch.optim.lr_scheduler.LRScheduler,
    num_epochs:int, 
    device:torch.device, 
    save_dir:str,
    config:Dict,
    history:Dict,
    best_val_loss:float=float('inf'),
    init_epoch:int=0,
    plot_train_dataloader:Optional[DataLoader]=None,
    plot_val_dataloader:Optional[DataLoader]=None
):
    best_model_path = os.path.join(save_dir, "best_model.pth")
    last_model_path = os.path.join(save_dir, "last_model.pth")
    history_path    = os.path.join(save_dir, "history.json")

    save_image_path = os.path.join(save_dir, "images")
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)
        
    save_image_path_steps = os.path.join(save_image_path, "steps")
    if not os.path.exists(save_image_path_steps):
        os.makedirs(save_image_path_steps)
             
    steps_train = config.get("checkpoint").get("steps_train")
    num_steps_train = len(train_dataloader)

    patience = config.get("train").get("patience")
    last_best_epoch = init_epoch

    with open (config.get("dirs").get("mapping"), 'r') as f:
        mapping = json.load(f)

        #{name, num} -> {num, name}
        class_id_mapping_image      = {v:k for k, v in mapping.get("target_classes").items()}

        color_mapping_image = {}
        for k, v in mapping.get("target_color").items():
            if mapping.get("target_classes").get(k) not in color_mapping_image:
                color_mapping_image[mapping.get("target_classes").get(k)] = hex_to_rgb(v.replace("#",""))

    
    
    print("")

    column_width = 5    
    init_train_time = time.time()
    total_train_steps = len(train_dataloader)
    mean_step_train_time = []

    classes_weights = torch.tensor(config.get("loss").get("class_weights"))
    num_classes = config.get("image").get("num_classes")

    grad_norm = config.get("model").get("grad_norm")
    steps_optimizer = config.get("train").get("steps_minibatch_optimizer")
    steps_plot_train = config.get("checkpoint").get("steps_plot_train")

    for epoch in range(init_epoch, num_epochs):
        init_epoch_time = time.time()

        ##################################### TRAINING ##########################################
        model.train()
        train_loss = 0.0
        train_img_dice_mean = 0.0
        train_inf_time_mean = 0.0
        train_confmat = np.zeros((num_classes, num_classes), dtype=np.uint64)
        
        optimizer.zero_grad()
        for step, (x_img, y_img) in enumerate(train_dataloader):
            init_step_time = time.time()
            
            x_img, y_img = x_img.to(device), y_img.to(device)
            
            batch_size = x_img.shape[0]
            
            train_step_inf_time = time.time()
            logits = model(x_img)
            train_step_inf_time = (time.time() - train_step_inf_time)/batch_size
            
                                                    
            loss = criterion(logits=logits, targets=y_img)         
            loss.backward()
            
            if (step+1)%steps_optimizer == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            
            # dice
            img_dice = image_dice(
                y_pred=logits,
                y_true=y_img,
                weights=classes_weights
            )          

            train_img_dice_mean += img_dice
            train_inf_time_mean += train_step_inf_time
            
            # confusion matrix
            preds = torch.argmax(logits, dim=1).view(-1).cpu().numpy()
            target = y_img.view(-1).cpu().numpy()
            batch_cm = np.bincount(
                num_classes * target + preds, minlength=num_classes ** 2
            ).reshape(num_classes, num_classes).astype(np.uint64)
            train_confmat += batch_cm

            history["train"]["loss"].append([epoch, step, loss.item()]) 
            history["train"]["inf_time"].append([epoch, step, train_step_inf_time])
            history["train"]["image"]["dice"].append([epoch, step, img_dice])
                        
            mean_step_train_time.append(time.time()-init_step_time)

            if (step%steps_train == 0):
                (_, train_micro_acc, _),\
                (_, train_micro_miou, _) =\
                    metrics_from_confmat(confmat=train_confmat)
                
                t_loss = f"{train_loss/(step+1):.3f}".rjust(column_width+1, " ")
                
                t_dice_img = f"{train_img_dice_mean/(step+1):.3f}".rjust(column_width, " ")
                t_acc_img = f"{train_micro_acc:.2f}".rjust(column_width, " ")
                t_miou_img = f"{train_micro_miou:.2f}".rjust(column_width, " ")
                t_inf_time = f"{train_inf_time_mean/(step+1):.4f} s".rjust(column_width, " ")

                remaining_steps = total_train_steps - step + 1
                time_per_step = np.mean(mean_step_train_time)
                remaining_time = remaining_steps * time_per_step
                
                print(f"Epoch: [{epoch+1:04d}|{num_epochs}]", end="")
                print(f"[{step+1:04d}|{num_steps_train}] | ", end="")
                print(f"Time: [{timedelta(seconds=time_per_step)}] | ", end="")
                print(f"loss: {t_loss} | ", end="")                    
                print(f"dice: {t_dice_img} | ", end="")
                print(f"acc: {t_acc_img} | ", end="")
                print(f"miou: {t_miou_img} | ", end="")
                print(f"inf. time: {t_inf_time} | ", end="")
                print(f"R. Time [{timedelta(seconds=remaining_time)}]")
        
            if (step%steps_plot_train == 0):
                if plot_train_dataloader is not None:
                    with torch.no_grad():
                        save_grid_samples(model, 
                                          plot_train_dataloader, 
                                          f"train_{epoch}_{step}", 
                                          save_image_path_steps, 
                                          device=device,
                                          mapping=color_mapping_image)
                        
                    model.train()

        scheduler.step()

        train_loss /= len(train_dataloader)
        train_img_dice_mean /= len(train_dataloader)
        train_inf_time_mean /= len(train_dataloader)         
        history["train"]["image"]["confmat"].append({
                                                "epoch": epoch,
                                                "shape": train_confmat.shape,
                                                "data": train_confmat.flatten().tolist()
                                            })
               
        (train_macro_acc, train_micro_acc, train_acc),\
        (train_macro_miou, train_micro_miou, train_iou) =\
            metrics_from_confmat(confmat=train_confmat)
                
        ###################################### VALIDATION ########################################
        model.eval()
        val_loss = 0.0
        val_img_dice_mean = 0.0
        val_inf_time_mean = 0.0
        val_confmat = np.zeros((num_classes, num_classes), dtype=np.uint64)

        with torch.no_grad():
            for step, (x_img, y_img) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):

                x_img, y_img= x_img.to(device),  y_img.to(device)

                batch_size = x_img.shape[0]
                
                val_step_inf_time = time.time()
                logits = model(x_img)
                val_step_inf_time = (time.time() - val_step_inf_time)/batch_size

                loss = criterion(logits=logits, targets=y_img)

                val_loss += loss.item()
        
                # dice
                img_dice = image_dice(
                    y_pred=logits,
                    y_true=y_img,
                    weights=classes_weights
                )     

                val_img_dice_mean += img_dice
                val_inf_time_mean += val_step_inf_time
                
                # confusion matrix
                preds = torch.argmax(logits, dim=1).view(-1).cpu().numpy()
                target = y_img.view(-1).cpu().numpy()
                batch_cm = np.bincount(
                    num_classes * target + preds, minlength=num_classes ** 2
                ).reshape(num_classes, num_classes).astype(np.uint64)
                val_confmat += batch_cm

                history["val"]["loss"].append([epoch, step, loss.item()])
                history["val"]["inf_time"].append([epoch, step, val_step_inf_time])
                history["val"]["image"]["dice"].append([epoch, step, img_dice])

                
        val_loss /= len(val_dataloader)
        val_img_dice_mean /= len(val_dataloader)
        val_inf_time_mean /= len(val_dataloader) 
        history["val"]["image"]["confmat"].append({
                                                "epoch": epoch,
                                                "shape": val_confmat.shape,
                                                "data": val_confmat.flatten().tolist()
                                            })
       
        (val_macro_acc, val_micro_acc, val_acc),\
        (val_macro_miou, val_micro_miou, val_iou) =\
            metrics_from_confmat(confmat=val_confmat)
            
        print("-"*30)
        print("")
        print(f"Epoch: [{epoch+1:04d}|{num_epochs}] | ", end="")
        print(f"Time: {timedelta(seconds=time.time()-init_epoch_time)}")
        print("")

        t_loss = f"{train_loss:.3f}".rjust(column_width+1, " ")
        t_dice_img = f"{train_img_dice_mean:.3f}".rjust(column_width, " ")
        t_micro_acc_img = f"{train_micro_acc:.2f}".rjust(column_width, " ")
        t_micro_miou_img = f"{train_micro_miou:.2f}".rjust(column_width, " ")
        t_macro_acc_img = f"{train_macro_acc:.2f}".rjust(column_width, " ")
        t_macro_miou_img = f"{train_macro_miou:.2f}".rjust(column_width, " ")
        t_inf_time = f"{train_inf_time_mean:.4f} s".rjust(column_width, " ")

        v_loss = f"{val_loss:.3f}".rjust(column_width+1, " ")
        v_dice_img = f"{val_img_dice_mean:.3f}".rjust(column_width, " ")
        v_micro_acc_img = f"{val_micro_acc:.2f}".rjust(column_width, " ")
        v_micro_miou_img = f"{val_micro_miou:.2f}".rjust(column_width, " ")
        v_macro_acc_img = f"{val_macro_acc:.2f}".rjust(column_width, " ")
        v_macro_miou_img = f"{val_macro_miou:.2f}".rjust(column_width, " ")
        v_inf_time = f"{val_inf_time_mean:.4f} s".rjust(column_width, " ")
        
        print(f"Train | loss: {t_loss} | ", end="")
        print(f"dice: {t_dice_img} | ", end="")
        print(f"micro acc: {t_micro_acc_img} | ", end="")
        print(f"micro miou: {t_micro_miou_img} | ", end="")
        print(f"macro acc: {t_macro_acc_img} | ", end="")
        print(f"macro miou: {t_macro_miou_img} | ", end="")
        print(f"inf. time: {t_inf_time} |")

        print(f"Val   | loss: {v_loss} | ", end="")
        print(f"dice: {v_dice_img} | ", end="")
        print(f"micro acc: {v_micro_acc_img} | ", end="")
        print(f"micro miou: {v_micro_miou_img} | ", end="")
        print(f"macro acc: {v_macro_acc_img} | ", end="")
        print(f"macro miou: {v_macro_miou_img} | ", end="")
        print(f"inf. time: {v_inf_time} |")
        print("")

        print(f"{'='*40} IMAGE {'='*40}") 
        print_metric_table(train_acc, 
                           val_acc,
                           train_iou, 
                           val_iou,
                           class_id_mapping_image, 
                           epoch=epoch)
        
        print("")  
        
        if plot_train_dataloader is not None:
            save_grid_samples(model, 
                            plot_train_dataloader, 
                            f"train_{epoch}", 
                            save_image_path, 
                            device=device,
                            mapping=color_mapping_image)
        if plot_val_dataloader is not None:
            save_grid_samples(model, 
                            plot_val_dataloader,
                            f"val_{epoch}", 
                            save_image_path, 
                            device=device,
                            mapping=color_mapping_image)
        
        # Save best model
        if val_loss < best_val_loss:
            printH("[Image Segmentation][train]", "saving best model..", "c")
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            history["best_val_epoch"] = epoch 
            last_best_epoch = epoch

        # Save last model
        torch.save(model.state_dict(), last_model_path)

        # Save history
        history["best_val_loss"] = best_val_loss
        history["last_epoch"] = epoch
        history["time"].append((epoch, time.time()-init_epoch_time))
        with open(history_path, "w") as f:
            json.dump(history, f)

        if epoch > (last_best_epoch + patience):
            printH("[Image Segmentation][train]", "early stop!", "m")
            break

        print("-"*30)

    printH("[Image Segmentation][train]", "done!", "o")
    printH("[Image Segmentation][train]", f"elapsed time: {timedelta(seconds=time.time()-init_train_time)}", "o")

