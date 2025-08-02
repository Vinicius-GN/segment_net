import os
import json
import time
from datetime import timedelta
import numpy as np

from typing import Dict, Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.utils import hex_to_rgb, to_serializable
from utils.color import printH
from utils.metrics import image_dice, metrics_from_confmat, classification_report_from_confmat
from utils.tables import print_test_metric_table, print_report_table, print_test_macro_table
from utils.image_grid import save_grid_samples
from utils.loss import SegmentLoss
from utils.segmentnet import SegmentNet



def test_model(
    model:SegmentNet,
    test_dataloader:DataLoader, 
    criterion:SegmentLoss, 
    device:torch.device, 
    save_dir:str,
    config:Dict,
    history:Dict
):
    history_path = os.path.join(save_dir, "history.json")
    
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
    init_test_time = time.time()
    total_test_steps = len(test_dataloader)
    mean_step_test_time = []

    classes_weights = torch.tensor(config.get("loss").get("class_weights"))
    num_classes = config.get("image").get("num_classes")

    model.eval()
    test_loss = 0.0
    test_img_dice_mean = 0.0    
    confmat_global = np.zeros((num_classes, num_classes), dtype=np.uint64)

    with torch.no_grad():
        for step, (x_img, y_img) in tqdm(enumerate(test_dataloader), total=total_test_steps):

            x_img, y_img= x_img.to(device),  y_img.to(device)

            logits = model(x_img)

            batch_size =  x_img.shape[0]
            
            train_step_inf_time = time.time()
            loss = criterion(logits=logits, targets=y_img)
            train_step_inf_time = (time.time() - train_step_inf_time)/batch_size
            mean_step_test_time.append(train_step_inf_time)

            test_loss += loss.item()
    
            img_dice = image_dice(
                y_pred=logits,
                y_true=y_img,
                weights=classes_weights
            )
            test_img_dice_mean += img_dice
                        
            # confusion matrix
            preds = torch.argmax(logits, dim=1).view(-1).cpu().numpy()
            target = y_img.view(-1).cpu().numpy()
            batch_cm = np.bincount(
                num_classes * target + preds, minlength=num_classes ** 2
            ).reshape(num_classes, num_classes).astype(np.uint64)
            confmat_global += batch_cm

            # history
            history["test"]["loss"].append([0, step, loss.item()])
            history["test"]["image"]["dice"].append([0, step, img_dice])
            
            
        history["test"]["image"]["confmat"].append({
                                                "shape": confmat_global.shape,
                                                "data": confmat_global.flatten().tolist()
                                            })
        history["test"]["inf_time"] = mean_step_test_time
                    
        test_loss /= total_test_steps
        test_img_dice_mean /= total_test_steps
        test_inf_time_mean = np.mean(mean_step_test_time)
               
        print("-"*30)
        print("")
        print(f"Time: {timedelta(seconds=time.time()-init_test_time)}")
        print("")

        (test_macro_acc,  test_micro_acc,  test_acc),\
        (test_macro_miou, test_micro_miou, test_iou) =\
            metrics_from_confmat(confmat=confmat_global)
        
        v_loss           = f"{test_loss:2.3f}".rjust(column_width+1, " ")
        v_dice_img       = f"{test_img_dice_mean:2.3f}".rjust(column_width, " ")
        v_macro_acc_img  = f"{test_macro_acc:2.2f}".rjust(column_width, " ")
        v_macro_miou_img = f"{test_macro_miou:2.2f}".rjust(column_width, " ")
        v_micro_acc_img  = f"{test_micro_acc:2.2f}".rjust(column_width, " ")
        v_micro_miou_img = f"{test_micro_miou:2.2f}".rjust(column_width, " ")
        v_inf_time_img   = f"{test_inf_time_mean:2.4f}".rjust(column_width, " ")
        
        print(f"Test   | loss: {v_loss} | ", end="")
        print(f"dice: {v_dice_img} | ", end="")
        print(f"macro acc: {v_macro_acc_img} | ", end="")
        print(f"macro miou: {v_macro_miou_img} | ", end="")
        print(f"micro acc: {v_micro_acc_img} | ", end="")
        print(f"micro miou: {v_micro_miou_img} | ", end="")
        print(f"inf. time: {v_inf_time_img} |")
        print("")

       
        
        class_names = [v for _, v in class_id_mapping_image.items()]
        report = classification_report_from_confmat(confmat_global, class_names=class_names)
                
        
        print(f"{'-'*32} IMAGE {'-'*32}") 
        test_recall    = [report[k]["recall"] for k in class_names]
        test_precision = [report[k]["precision"] for k in class_names]
        test_f1score   = [report[k]["f1-score"] for k in class_names]
        test_support   = [report[k]["support"] for k in class_names]
        
        print_test_metric_table(test_acc, 
                                test_iou,
                                test_recall,
                                test_precision,
                                test_f1score,
                                test_support,
                                class_names)
        
        print("")           
        
        print_test_macro_table(report)  
        
        print("")   
        
        # Save history
        with open(history_path, "w") as f:
            json.dump(history, f)
            
        report_path = history_path.replace("history", "report")
        with open(report_path, "w") as f:
            json.dump(report, f, default=to_serializable, indent=4)

        print("-"*71)

    printH("[Image Segmentation][test]", "done!", "o")
    printH("[Image Segmentation][test]", f"elapsed time: {timedelta(seconds=time.time()-init_test_time)}", "o")