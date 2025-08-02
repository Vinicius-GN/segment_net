from typing import Dict
import torch
import torch.nn.functional as F
import json
from torchmetrics.functional.segmentation import mean_iou
from .dice_loss import _expand_onehot_labels_dice, dice_loss
import numpy as np

def compute_mean_iou(preds, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum().item()
        union = (pred_cls + target_cls).clamp(0, 1).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0

def image_metrics(y_pred:torch.Tensor,
                  y_true:torch.Tensor,
                  weights:torch.Tensor=None):
    
    #dice
    one_hot_target = y_true.squeeze(-1)
    if (y_pred.shape != y_true.shape):
        one_hot_target = _expand_onehot_labels_dice(y_pred, y_true.squeeze(-1))
    
    batch_size = y_true.shape[0]
    weights = weights.unsqueeze(0).expand(batch_size, -1).to(y_pred.device)
    softmax_pred = y_pred.softmax(dim=1)
    dice = dice_loss(softmax_pred,
                     one_hot_target,
                     weight=weights).item()
    #acc
    batch_size, num_classes, height, width = y_pred.shape
    preds = torch.argmax(y_pred, dim=1).view(-1).cpu()
    target = y_true.view(-1).cpu()

    
    one_hot_pred = torch.nn.functional.one_hot(preds, num_classes=num_classes).view(batch_size, num_classes, height, width).long().to(y_pred.device)
    # one_hot_target = one_hot_target.long().to(y_pred.device)
    one_hot_target = torch.nn.functional.one_hot(target, num_classes=num_classes).view(batch_size, num_classes, height, width).long().to(y_pred.device)
    
    miou_per_classes = mean_iou(preds=one_hot_pred, 
                                target=one_hot_target, 
                                num_classes=num_classes, 
                                per_class=True,
                                input_format="one-hot").mean(dim=0).squeeze()
    miou_mean = mean_iou(preds=one_hot_pred, 
                         target=one_hot_target, 
                         num_classes=num_classes, 
                         per_class=False,
                         input_format="one-hot").mean().item()
   
    per_class_miou = {int(idx):m.item() for idx, m in enumerate(miou_per_classes)}
    
    acc = (preds == target).float().mean().item()
    num_classes = y_pred.size(1)
    per_class_accuracy = {}
    
    for cls in range(num_classes):
        cls_mask = (target == cls)

        if cls_mask.sum() > 0:
            cls_accuracy = (preds[cls_mask] == target[cls_mask]).float().mean().item()
        else:
            cls_accuracy = float('nan')

        per_class_accuracy[cls] = cls_accuracy
    
    return dice, acc, miou_mean, per_class_accuracy, per_class_miou


def image_dice(y_pred:torch.Tensor,
                y_true:torch.Tensor,
                weights:torch.Tensor=None):
    
    #dice
    one_hot_target = y_true.squeeze(-1)
    if (y_pred.shape != y_true.shape):
        one_hot_target = _expand_onehot_labels_dice(y_pred, y_true.squeeze(-1))
    
    batch_size = y_true.shape[0]
    weights = weights.unsqueeze(0).expand(batch_size, -1).to(y_pred.device)
    softmax_pred = y_pred.softmax(dim=1)
    dice = dice_loss(softmax_pred,
                     one_hot_target,
                     weight=weights).item()
   
    
    return dice

def metrics_from_confmat(
    confmat
):
    
    TP = np.diag(confmat)
    FP = confmat.sum(axis=0) - TP
    FN = confmat.sum(axis=1) - TP
    TN = confmat.sum() - (TP + FP + FN)
           
    iou = TP / (TP + FP + FN + 1e-6) 
    acc = TP / (TP + FN + 1e-6)   
            
    macro_miou = np.nanmean(iou)
    macro_acc = np.nanmean(acc)
        
    micro_miou = TP.sum() / (TP + FP + FN).sum()
    micro_acc = TP.sum() / confmat.sum()
        
    return (macro_acc, micro_acc, acc), (macro_miou, micro_miou, iou)

    
def classification_report_from_confmat(confmat, class_names=None):
    TP = np.diag(confmat)
    FP = confmat.sum(axis=0) - TP
    FN = confmat.sum(axis=1) - TP
    support = confmat.sum(axis=1)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    acc = TP / (support + 1e-6)

    report = {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": acc,
        "support": support
    }

    macro_avg = {
        "precision": np.mean(precision),
        "recall": np.mean(recall),
        "f1-score": np.mean(f1),
        "support": np.sum(support)
    }

    weighted_avg = {
        "precision": np.average(precision, weights=support),
        "recall": np.average(recall, weights=support),
        "f1-score": np.average(f1, weights=support),
        "support": np.sum(support)
    }

    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(TP))]

    full_report = {
        name: {k: float(report[k][i]) for k in report}
        for i, name in enumerate(class_names)
    }
    full_report["macro avg"] = macro_avg
    full_report["weighted avg"] = weighted_avg
    full_report["accuracy"] = float(TP.sum()) / float(confmat.sum())

    return full_report