import torch

from typing import List, Tuple, Dict
import torch.nn.functional as F

from monai.losses import HausdorffDTLoss
from .dice_loss import DiceLoss
from .lovasz_losses import lovasz_softmax
from .boundary_loss import MulticlassBoundaryLoss

class SegmentLoss(torch.nn.Module):

    def __init__(self, config:Dict, device:torch.device):

        super(SegmentLoss, self).__init__()
        
        self.config = config
        self.device = device

        alpha=self.config.get("loss").get("alpha") 
        gamma=self.config.get("loss").get("gamma")
        reduction="mean"
        weights=self.config.get("loss").get("class_weights")
        weights=torch.tensor(weights, device=self.device)
        
        self.scale = self.config.get("loss").get("loss_scale")
                            
        if self.config.get("loss").get("type") == "focal_dice":
        
            self.loss= FocalDice(alpha=alpha, 
                                 gamma=gamma, 
                                 reduction=reduction, 
                                 weights=weights)
            
        elif self.config.get("loss").get("type") == "dice":
            
            self.loss = DiceLoss(use_sigmoid=False, 
                                 reduction=reduction)
                       
        elif self.config.get("loss").get("type") == "cross_entropy":
            
            self.loss = torch.nn.CrossEntropyLoss(
                reduction=reduction,
                label_smoothing=0.1     
            )
            
        elif self.config.get("loss").get("type") == "focal_cross_entropy":
            
            self.loss = FocalLoss(alpha=alpha, 
                                  gamma=gamma, 
                                  reduction=reduction, 
                                  weights=weights)
            
        elif self.config.get("loss").get("type") == "lovasz_softmax":
            
            self.loss = LovaszSoftmax()
        
        elif self.config.get("loss").get("type") == "hausdorff_dt_dice":
            
            self.loss = HausdorffDTDice(
                reduction=reduction
            )
            
        elif self.config.get("loss").get("type") == "boundary_dice":
            self.loss = MulticlassBoundaryDiceLoss(
                weights=weights,
                reduction=reduction,
                max_distance=20
            )
        else:
            raise ValueError(f"Loss type {self.config.get('loss').get('type')} not supported!")

    def forward(self, logits, targets):
        
        logits = logits.to(self.device)
        targets = targets.to(self.device).squeeze(-1)
        
        return self.scale*self.loss(logits, targets)
    
    @property
    def loss_name(self):
        return self.config.get("loss").get("type")


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights


    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=0.1, weight=self.weights)
        pt = torch.exp(-ce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class FocalDice(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean", weights=None):
        super(FocalDice, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights

        self.dice = DiceLoss(use_sigmoid=False, reduction="none")

    def forward(self, logits, targets):

        batch_size = targets.shape[0]
        
        class_weights = self.weights.unsqueeze(0).expand(batch_size, -1)
        
        dice_loss = self.dice(logits, targets, weight=class_weights)

        pt = torch.exp(-dice_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * dice_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
        
class LovaszSoftmax(torch.nn.Module):
    def __init__(self):
        super(LovaszSoftmax, self).__init__()
        
    def forward(self, logits, targets):
        probs = logits.softmax(dim=1)
        loss = lovasz_softmax(probs, targets, 
                              classes="all",
                              per_image=False)        
        return loss
    
    
    
class HausdorffDTDice(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super(HausdorffDTDice, self).__init__()
        
        self.loss_fn = HausdorffDTLoss(
            include_background=False,
            reduction=reduction
        )
        
        self.dice = DiceLoss(use_sigmoid=False, reduction=reduction)
        
    def forward(self, logits, targets):
        C = logits.shape[1]
        probs = logits.softmax(dim=1)
        
        targets_onehot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        
        hausdorffdT_loss = self.loss_fn(probs, targets_onehot) 
        dice_loss =  self.dice(logits, targets)

        return 0.5*hausdorffdT_loss + dice_loss
    
class MulticlassBoundaryDiceLoss(torch.nn.Module):
    def __init__(self, weights=None, reduction='mean', max_distance:int=20):
        super(MulticlassBoundaryDiceLoss, self).__init__()
        
        self.boundary_fn = MulticlassBoundaryLoss(
                class_weights=weights,
                reduction=reduction,
                max_distance = max_distance
        )
        
        self.dice = DiceLoss(use_sigmoid=False, reduction=reduction)
                
    def forward(self, logits, targets):
                
        boundary_loss = self.boundary_fn(logits, targets) 
        dice_loss =  self.dice(logits, targets)

        return 0.5*boundary_loss + dice_loss