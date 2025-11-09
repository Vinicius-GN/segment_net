import torch

from typing import List, Tuple, Dict
from typing import Optional
import torch.nn.functional as F

from monai.losses import HausdorffDTLoss
from .dice_loss import DiceLoss
from .lovasz_losses import lovasz_softmax
from .boundary_loss import MulticlassBoundaryLoss
from .log_cosh_dice import LogCoshDiceLoss
from .sensitivity_specificity import SensitivitySpecificityLoss
from .focal_tversky import FocalTverskyLoss
from .jaccard import JaccardLoss
from .top_k import TopKLoss
from .dmce import DistanceMapCELoss
from .conditional_boundary_loss import ConditionalBoundaryLoss
from .active_boundary_loss import ActiveBoundaryLoss
from .inverseform_loss import InverseFormLoss

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
        elif self.config.get("loss").get("type")== "jaccard_iou":
            
            self.loss = JaccardLoss(
                smooth=1.0
            )

        elif self.config.get("loss").get("type") == "focal_tversky":
            self.loss = FocalTverskyLoss(
                alpha = 0.3,
                gamma = 4./3., 
                beta  = 0.7,
                smooth= 1.0
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
        elif self.config.get("loss").get("type") == "log_cosh_dice":
            
            self.loss = LogCoshDiceLoss(
                use_sigmoid=False,
                reduction=reduction
            )
        elif self.config.get("loss").get("type") == "sensitivity_specificity":
            
            self.loss = SensitivitySpecificityLoss(
                use_sigmoid=False,
                reduction=reduction
            )
        elif self.config.get("loss").get("type") == "top_k":
            self.loss = TopKLoss(
                 k_frac=0.25, gamma=2.0
            )
        elif self.config.get("loss").get("type") == "dmce_dice":
            self.loss = DistanceMapCEDiceLoss(
                 reduction=reduction
            )
        elif self.config.get("loss").get("type") == "conditional_boundary_dice":
            self.loss = ConditionalBoundaryDiceLoss(
                kernel_size=self.config.get("loss").get("ccas_kernel", 5),
                alpha=self.config.get("loss").get("a2c_alpha", 0.1),
                beta=self.config.get("loss").get("a2pn_beta", 0.5),
                reduction=reduction,
                class_weights=weights,
            )

        elif self.config.get("loss").get("type") == "active_boundary_dice":
            self.loss = ActiveBoundaryDiceLoss(
                boundary_ratio=self.config.get("loss").get("boundary_ratio", 0.01),
                theta=self.config.get("loss").get("theta", 20.0),
                label_smoothing_max=self.config.get("loss").get("abl_label_smoothing_max", 0.0),
                use_2n_for_pb=self.config.get("loss").get("use_2n_for_pb", True),
                dilate_pb=self.config.get("loss").get("dilate_pb", 0),
                reduction=reduction
            )

        elif self.config.get("loss").get("type") == "inverseform_dice":
            self.loss = InverseFormDiceLoss(
                mode=self.config.get("loss").get("if_mode", "euclidean_affine"),
                tile=self.config.get("loss").get("if_tile", 64),
                stride=self.config.get("loss").get("if_stride", None),
                reduce=self.config.get("loss").get("if_reduce", 4),
                lam_geo=self.config.get("loss").get("if_lam_geo", 0.1),
                itn_weights=self.config.get("loss").get("itn_weights", None),
                freeze_itn=self.config.get("loss").get("freeze_itn", True),
                reduction=reduction,
                dice_cls=DiceLoss(use_sigmoid=False, reduction=reduction),
                lambda_if=self.config.get("loss").get("lambda_if", 0.5)
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
    
class DistanceMapCEDiceLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(DistanceMapCEDiceLoss, self).__init__()
        self.dmce = DistanceMapCELoss(reduction=reduction)
        self.dice = DiceLoss(use_sigmoid=False, reduction=reduction)

    def forward(self, logits, targets):
        dmce_loss = self.dmce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return 0.5*dmce_loss + dice_loss

class ConditionalBoundaryDiceLoss(torch.nn.Module):
    def __init__(self, kernel_size:int=5, alpha:float=0.1, beta:float=0.5, reduction:str='mean', class_weights=None):
        super().__init__()
        self.cbl = ConditionalBoundaryLoss(kernel_size=kernel_size, alpha=alpha, beta=beta, reduction=reduction, class_weights=class_weights)
        self.dice = DiceLoss(use_sigmoid=False, reduction=reduction)
    def forward(self, logits, targets):
        cbl = self.cbl(logits, targets)
        dsc = self.dice(logits, targets)
        return 0.5*cbl + dsc


class ActiveBoundaryDiceLoss(torch.nn.Module):
    def __init__(self, boundary_ratio=0.01, theta=20.0, label_smoothing_max=0.0, use_2n_for_pb=True, dilate_pb=0, reduction="mean"):
        super(ActiveBoundaryDiceLoss, self).__init__()
        self.abl = ActiveBoundaryLoss(
            boundary_ratio=boundary_ratio,
            theta=theta,
            label_smoothing_max=label_smoothing_max,
            use_2n_for_pb=use_2n_for_pb,
            dilate_pb=dilate_pb,
            reduction=reduction
        )
        self.dice = DiceLoss(use_sigmoid=False, reduction=reduction)

    def forward(self, logits, targets):
        abl_loss = self.abl(logits, targets)
        dice_loss = self.dice(logits, targets)
        return 0.5*abl_loss + dice_loss

class InverseFormDiceLoss(torch.nn.Module):
    def __init__(self, mode:str="euclidean_affine", tile:int=64, stride:Optional[int]=None, reduce:int=4, lam_geo:float=0.1, itn:Optional[torch.nn.Module]=None, itn_weights:Optional[str]=None, freeze_itn:bool=True, reduction:str="mean", dice_cls:Optional[torch.nn.Module]=None, lambda_if:float=0.5):
        super().__init__()
        self.if_loss = InverseFormLoss(mode=mode, tile=tile, stride=stride, reduce=reduce, lam_geo=lam_geo, itn=itn, itn_weights=itn_weights, freeze_itn=freeze_itn, reduction=reduction)
        self.dice = dice_cls
        self.lambda_if = lambda_if
    def forward(self, logits, targets):
        lif = self.if_loss(logits, targets)
        if self.dice is None: return lif
        return self.lambda_if*lif + self.dice(logits, targets)
