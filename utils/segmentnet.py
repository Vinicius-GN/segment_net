import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchvision.transforms.functional import normalize as normalize_image

from typing import Dict, Tuple, List, Union
from itertools import product

from utils.color import printH
from utils.backbone import (Resnet18_FPN, MobilenetV3_FPN,
                            EfficientNetB0_FPN, DeeplabV3MobilenetV3_FPN, 
                            MobileViT_FPN, DeiT3Small_FPN,
                            EfficientFormer_FPN, PiTXS_FPN, 
                            SegFormerB0_FPN,LeViT_FPN,
                            SAM2_Hiera_FPN, TinyViT_FPN,
                            FastViT_FPN, ConvNeXtV2_FPN,
                            MaxxViTV2_FPN, EdgeNeXt_FPN)

from utils.head import (SEConvInterpHead, DepthwiseSeparableHead,
                        LightweightTransformerHead)


from utils.attention import (SEAttention, ClassChannelAttention,
                             ClassSpatialAttention, ClassQueryAttention)

class SegmentNet(nn.Module):

    def __init__(self, config:Dict):

        super(SegmentNet, self).__init__()

        self.config = config

        printH("[SegmentNet]", "init...", "i")

        ## Encoder - Backbone
        backbone_name = self.config.get("backbone").get("type") 
        if backbone_name == "resnet18":
            self.backbone = Resnet18_FPN(config=self.config)
        elif backbone_name == "mobilenetv3":
            self.backbone = MobilenetV3_FPN(config=self.config)
        elif backbone_name == "efficientnetb0":
            self.backbone = EfficientNetB0_FPN(config=self.config)
        elif backbone_name == "deeplabv3_mobilenetv3":
            self.backbone = DeeplabV3MobilenetV3_FPN(config=self.config)
        elif backbone_name == "mobilevit":
            self.backbone = MobileViT_FPN(config=self.config)
        elif backbone_name == "deit3_small":
            self.backbone = DeiT3Small_FPN(config=self.config)
        elif backbone_name == "efficientformer":
            self.backbone = EfficientFormer_FPN(config=self.config)
        elif backbone_name == "levit":
            self.backbone = LeViT_FPN(config=self.config)
        elif backbone_name == "segformerb0":
            self.backbone = SegFormerB0_FPN(config=self.config)
        elif backbone_name == "pitxs":
            self.backbone = PiTXS_FPN(config=self.config)
        elif backbone_name == "sam2_hiera":
            self.backbone = SAM2_Hiera_FPN(config=self.config)
        elif backbone_name == "tinyvit":
            self.backbone = TinyViT_FPN(config=self.config)
        elif backbone_name == "fastvit":
            self.backbone = FastViT_FPN(config=self.config)
        elif backbone_name == "convnextv2":
            self.backbone = ConvNeXtV2_FPN(config=self.config)
        elif backbone_name == "maxxvitv2":
            self.backbone = MaxxViTV2_FPN(config=self.config)
        elif backbone_name == "edgenext":
            self.backbone = EdgeNeXt_FPN(config=self.config)
        else:
            raise ValueError(f"Image backbone invalid! {backbone_name}")
     
     
        ## Decoder
        head_name = self.config.get("head").get("type") 
        if head_name == "se_conv_interp":
            self.head = SEConvInterpHead(
                            config=self.config, 
                            num_classes=self.config.get("image").get("num_classes"),
                            num_feature_layers=self.backbone.num_feature_layers
                        )
        elif head_name == "depthwise_nn":
            self.head = DepthwiseSeparableHead(
                            config=self.config, 
                            num_classes=self.config.get("image").get("num_classes"),
                            num_blocks=self.config.get("head").get("num_blocks"),
                            num_feature_layers=self.backbone.num_feature_layers
                        )
        elif head_name == "transformer":
            self.head = LightweightTransformerHead(
                            config=self.config, 
                            num_classes=self.config.get("image").get("num_classes"),
                            num_layers=self.config.get("head").get("num_blocks"),
                            num_feature_layers=self.backbone.num_feature_layers,
                            num_heads=self.config.get("head").get("num_heads_transformer")
                        )        
        else: 
            raise ValueError(f"Model head invalid! {head_name}")
        
        
        ## Attention
        attn_type = self.config.get("attention").get("type")
        self.use_attention = self.config.get("attention").get("use_attention")
        
        if self.use_attention:
            channel_size = self.config.get("backbone").get("fpn_out_channels")
            if self.config.get("backbone").get("aggregate") == "concat":            
                channel_size = channel_size*self.backbone.num_feature_layers
            else:
                channel_size = channel_size
                
                            
            if attn_type == "se_channel":
                self.attention = SEAttention(
                                    channel_size, 
                                    reduction=self.config.get("attention").get("reduction_rate")
                                )
            elif attn_type == "spatial":
                self.attention = ClassSpatialAttention(
                                    channel_size, 
                                    num_classes=self.config.get("image").get("num_classes"),
                                    dropout=self.config.get("attention").get("dropout")
                                 )
            elif attn_type == "query":
                embed_dim = max(16, channel_size//self.config.get("attention").get("reduction_rate")) 
                self.attention = ClassQueryAttention(
                                    channel_size, 
                                    embed_dim=embed_dim,
                                    num_classes=self.config.get("image").get("num_classes"),
                                    dropout=self.config.get("attention").get("dropout"),
                                    global_per_class = False
                                )
        
            elif attn_type == "class_channel":
                self.attention = ClassChannelAttention(
                                    channel_size, 
                                    num_classes=self.config.get("image").get("num_classes"),
                                    dropout=self.config.get("attention").get("dropout")
                                )
            else:
                raise ValueError(f"Attention type invalid! {attn_type}")
            
        
    def forward(self, x):
        
        x = self.backbone(x)     
        
        # attention
        if self.use_attention:
            x = x + self.attention(x)
            
        x = self.head(x)       

        return x
    
    def predict(self, 
                x:torch.Tensor, 
                normalize:bool=False):
        
        self.eval()

        if normalize:
            x = x/255.
            x = normalize_image(x,
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])

        y = self.forward(x)
        y = y.softmax(dim=1)
        
        return torch.argmax(y, dim=1)