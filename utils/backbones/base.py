import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Dict, List


class BaseBackbone(nn.Module):
    
    def __init__(self, config:Dict, num_feature_layers):
        
        super(BaseBackbone, self).__init__()
        
        self.config = config
        self.aggregate_fn = self.config.get("backbone").get("aggregate")
        self.layer_weights = self.config.get("backbone").get("layer_weights")
        self.fpn_channels =  self.config.get("backbone").get("fpn_out_channels")              
        self.num_feature_layers = num_feature_layers
        
        
        batch_channels = self.fpn_channels
        if self.aggregate_fn == "concat":
            batch_channels = batch_channels*num_feature_layers

        norm_type = self.config.get("model").get("norm_fn")
        if norm_type == "batch_norm":
            self.norm   = nn.BatchNorm2d(batch_channels)
        elif norm_type == "group_norm":
            num_groups = max(4, batch_channels//32)
            self.norm   = nn.GroupNorm(num_groups, batch_channels)

        self.dropout = nn.Dropout2d(p=self.config.get("backbone").get("dropout"), inplace=True) 
        self.activation = nn.GELU()

    """
        aggregate_image_features: aggregate the features from the FPN 

        Args: 
        - img_feat (Dict[str,torch.Tensor]) : a dictionary with the features from the FPN

        Options:
        - aggregation options:
            + sum : use bilinear interpolation to upsample all feature vectors
                        to the same dimension and return the sum of all vectors
            + concat: use bilinear interpolation to upsample all feature vectors
                        to the same dimension and return the concatenation of all vectors
            + weighted_sum: use bilinear interpolation to upsample all feature vectors
                        to the same dimension and return the weighted sum of all vectors
            + max_pool: use bilinear interpolation to upsample all feature vectors
                        to the same dimension and return the max pooling of all vectors
        Returns:
        - aggregate_features (torch.Tensor)
    """
    def aggregate_image_features(self, img_feat: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = sorted(img_feat.keys())  
        base_shape = img_feat[keys[0]].shape[2:]

        upsampled_feats = [
            F.interpolate(img_feat[k], size=base_shape, mode='bilinear', align_corners=True)
            if img_feat[k].shape[2:] != base_shape else img_feat[k]
            for k in keys
        ]

        if self.aggregate_fn == "sum":
            return torch.stack(upsampled_feats, dim=0).sum(dim=0)
        elif self.aggregate_fn == "concat":
            return torch.cat(upsampled_feats, dim=1)
        elif self.aggregate_fn == "weighted_sum":
            if not self.layer_weights or len(self.layer_weights) != len(keys):
                raise ValueError("`layer_weights` must match number of feature maps.")
            weighted = [
                w * feat for w, feat in zip(self.layer_weights, upsampled_feats)
            ]
            return sum(weighted)
        elif self.aggregate_fn == "max_pool":
            stacked = torch.stack(upsampled_feats, dim=0)
            return torch.max(stacked, dim=0).values
        else:
            raise ValueError(f"Unsupported aggregation function: {self.aggregate_fn}")
        