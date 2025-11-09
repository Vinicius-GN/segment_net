import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchvision.transforms.functional import normalize as normalize_image

from typing import Dict, Tuple, List, Union
from itertools import product

from utils.preprocessing.color import printH
import utils.backbones as backbones

from utils.head.head import (SEConvInterpHead, DepthwiseSeparableHead,
                        LightweightTransformerHead, SegFormerAllMLPHead, DeepLabV3PlusHead, PSPNetHead, CCNetHead, LightHamHead, DenseASPPHead, Mask2FormerHead)


from utils.attention.attention import (SEAttention, ClassChannelAttention,
                             ClassSpatialAttention, ClassQueryAttention)

from utils.attention.ocr_attention import OCRAttention
from utils.attention.dual_attention import DANetModule
from utils.attention.polarized_selfattention import ParallelPolarizedSelfAttention, SequentialPolarizedSelfAttention
from utils.attention.efficientvit_attention import EfficientViTMSA
from utils.attention.pvtspatialreduction_attention import PVTSpatialReductionAttention

class SegmentNet(nn.Module):

    def __init__(self, config:Dict):

        super(SegmentNet, self).__init__()

        self.config = config
        #print(config)

        printH("[SegmentNet]", "init...", "i")

        ## Encoder - Backbone
        backbone_name = self.config.get("backbone").get("type") 
        if backbone_name == "resnet18":
            self.backbone = backbones.Resnet18_FPN(config=self.config)
        elif backbone_name == "mobilenetv3":
            self.backbone = backbones.MobilenetV3_FPN(config=self.config)
        elif backbone_name == "efficientnetb0":
            self.backbone = backbones.backbones.EfficientNetB0_FPN(config=self.config)
        elif backbone_name == "deeplabv3_mobilenetv3":
            self.backbone = backbones.DeeplabV3MobilenetV3_FPN(config=self.config)
        elif backbone_name == "mobilevit":
            self.backbone = backbones.MobileViT_FPN(config=self.config)
        elif backbone_name == "deit3_small":
            self.backbone = backbones.DeiT3Small_FPN(config=self.config)
        elif backbone_name == "efficientformer":
            self.backbone = backbones.EfficientFormer_FPN(config=self.config)
        elif backbone_name == "levit":
            self.backbone = backbones.LeViT_FPN(config=self.config)
        elif backbone_name == "segformerb0":
            self.backbone = backbones.SegFormerB0_FPN(config=self.config)            
        elif backbone_name == "segformerb2":
            self.backbone = backbones.SegFormerB2_FPN(config=self.config)
        elif backbone_name == "segformerb3":
            self.backbone = backbones.SegFormerB3_FPN(config=self.config)
        elif backbone_name == "segformerb4":
            self.backbone = backbones.SegFormerB4_FPN(config=self.config)
        elif backbone_name == "segformerb5":
            self.backbone = backbones.SegFormerB5_FPN(config=self.config)
        elif backbone_name == "swinv2":
            self.backbone = backbones.SwinV2_FPN(config=self.config)
        elif backbone_name == "pitxs":
            self.backbone = backbones.PiTXS_FPN(config=self.config)
        elif backbone_name == "sam2_hiera":
            self.backbone = backbones.SAM2_Hiera_FPN(config=self.config)
        elif backbone_name == "tinyvit":
            self.backbone = backbones.TinyViT_FPN(config=self.config)
        elif backbone_name == "fastvit":
            self.backbone = backbones.FastViT_FPN(config=self.config)
        elif backbone_name == "convnextv2":
            self.backbone = backbones.ConvNeXtV2_FPN(config=self.config)
        elif backbone_name == "maxxvitv2":
            self.backbone = backbones.MaxxViTV2_FPN(config=self.config)
        elif backbone_name == "edgenext":
            self.backbone = backbones.EdgeNeXt_FPN(config=self.config)
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
        elif head_name == "segformer_h":
            self.head = SegFormerAllMLPHead(
                config=self.config,
                num_classes=self.config.get("image").get("num_classes"),
                num_feature_layers=self.backbone.num_feature_layers,
            )
        elif head_name == "deeplabv3_h":
            self.head = DeepLabV3PlusHead(
                            config=self.config, 
                            num_classes=self.config.get("image").get("num_classes"),
                            num_feature_layers=self.config.get("head").get("num_blocks"),
                        )  
            
        elif head_name == "pspnet_h":
            self.head = PSPNetHead(
                config=self.config,
                num_classes=self.config.get("image").get("num_classes"),
                num_feature_layers=self.backbone.num_feature_layers,
            )

        elif head_name == "ccnet_h":
            self.head = CCNetHead(
                config=self.config,
                num_classes=self.config.get("image").get("num_classes"),
                num_feature_layers=self.backbone.num_feature_layers,
            )
        
        elif head_name == "lightham_h":
            self.head = LightHamHead(
                config=self.config,
                num_classes=self.config.get("image").get("num_classes"),
                num_feature_layers=self.backbone.num_feature_layers,
            )

        elif head_name == "denseaspp_h":
            self.head = DenseASPPHead(
                config=self.config,
                num_classes=self.config.get("image").get("num_classes"),
                num_feature_layers=self.backbone.num_feature_layers,
            )

        elif head_name == "mask2former_h":
            self.head = Mask2FormerHead(
                config=self.config,
                num_classes=self.config.get("image").get("num_classes"),
                num_feature_layers=self.backbone.num_feature_layers,
            )

        else: 
            raise ValueError(f"Model head invalid! {head_name}")
        
        
        ## Attention
                ## Attention
        attn_cfg = self.config.get("attention")
        attn_type = attn_cfg.get("type")
        self.use_attention = attn_cfg.get("use_attention")

        if self.use_attention:
            channel_size = self.config.get("backbone").get("fpn_out_channels")
            if self.config.get("backbone").get("aggregate") == "concat":
                channel_size = channel_size * self.backbone.num_feature_layers

            rr = attn_cfg.get("reduction_rate", 16)
            dr = attn_cfg.get("dropout", 0.0)
            num_heads = self.config.get("head").get("num_heads_transformer", 8)

            if attn_type == "se_channel":
                self.attention = SEAttention(
                    channel_size,
                    reduction=rr
                )

            elif attn_type == "spatial":
                self.attention = ClassSpatialAttention(
                    channel_size,
                    num_classes=self.config.get("image").get("num_classes"),
                    dropout=dr
                )

            elif attn_type == "query":
                embed_dim = max(16, channel_size // max(1, rr))
                self.attention = ClassQueryAttention(
                    channel_size,
                    embed_dim=embed_dim,
                    num_classes=self.config.get("image").get("num_classes"),
                    dropout=dr,
                    global_per_class=False
                )

            elif attn_type == "class_channel":
                self.attention = ClassChannelAttention(
                    channel_size,
                    num_classes=self.config.get("image").get("num_classes"),
                    dropout=dr
                )

            elif attn_type == "ocr":
                key_dim = max(32, channel_size // max(1, rr))
                self.attention = OCRAttention(
                    in_channels=channel_size,
                    num_classes=self.config.get("image").get("num_classes"),
                    key_channels=key_dim,
                    value_channels=key_dim,
                    dropout=dr,
                    scale=1
                )

            elif attn_type == "danet":
                self.attention = DANetModule(
                    in_channels=channel_size,
                    reduction=self.config["attention"].get("reduction_rate", 4),
                    dropout=self.config["attention"].get("dropout", 0.0),
                )

            elif attn_type == "psa_parallel":
                self.attention = ParallelPolarizedSelfAttention(
                    channel=channel_size,
                    reduction=max(1, rr)
                )

            elif attn_type == "psa_sequential":
                order = attn_cfg.get("psa_order", "cs")
                self.attention = SequentialPolarizedSelfAttention(
                    channel=channel_size,
                    reduction=max(1, rr),
                    order=order
                )

            elif attn_type == "efficientvit":
                head_dim = attn_cfg.get("head_dim", max(16, channel_size // max(1, rr)))
                ksizes = tuple(attn_cfg.get("head_kernel_sizes", (3, 5)))
                self.attention = EfficientViTMSA(
                    channels=channel_size,
                    head_dim=head_dim,
                    head_kernel_sizes=ksizes
                )

            elif attn_type == "pvt_sra":
                sr_ratio = attn_cfg.get("sr_ratio", 2)
                self.attention = PVTSpatialReductionAttention(
                    channels=channel_size,
                    embed_dim=channel_size,
                    num_heads=num_heads,
                    mode="sra",
                    sr_ratio=sr_ratio,
                    attn_drop=dr,
                    proj_drop=dr
                )

            elif attn_type == "pvt_linear":
                pool_size = attn_cfg.get("pool_size", 7)
                self.attention = PVTSpatialReductionAttention(
                    channels=channel_size,
                    embed_dim=channel_size,
                    num_heads=num_heads,
                    mode="linear",
                    pool_size=pool_size,
                    attn_drop=dr,
                    proj_drop=dr
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