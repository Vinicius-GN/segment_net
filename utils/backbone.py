import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
import timm
from typing import Dict, List
from torchvision.models.feature_extraction import get_graph_node_names
from transformers import SegformerModel, SegformerConfig



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
        
        
class Resnet18_FPN(BaseBackbone):

    def __init__(self, config):        
               
        super(Resnet18_FPN, self).__init__(config, 4)

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = create_feature_extractor(
            resnet,
            return_nodes={
                'layer1': 'feat1',  # high resolution, local features, low level abstraction
                'layer2': 'feat2', 
                'layer3': 'feat3',  
                'layer4': 'feat4',  # low resolution, global features, high level abstraction
            }
        )

        self.fpn = FeaturePyramidNetwork(
            [64, 128, 256, 512], 
            out_channels=self.fpn_channels
        )      
       
    def forward(self, x:torch.Tensor):
        
        x = self.backbone(x)
        x = self.fpn(x)
        
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
    
    
    
class MobilenetV3_FPN(BaseBackbone):

    def __init__(self, config):
                
        super(MobilenetV3_FPN, self).__init__(config, 5)

        mobilenetv3 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        self.backbone = create_feature_extractor(
            mobilenetv3,
            return_nodes={
                'features.2': 'feat1',   # P1 (High resolution, low-level)
                'features.4': 'feat2',   # P2
                'features.7': 'feat3',   # P3
                'features.12': 'feat4',  # P4
                'features.15': 'feat5',  # P5 (Low resolution, high-level)
            }
        )

        self.fpn = FeaturePyramidNetwork(
            [24, 40, 80, 112, 160], 
            out_channels=self.fpn_channels
        )
    
    def forward(self, x:torch.Tensor):
       
        x = self.backbone(x)        
        x = self.fpn(x)
        
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x)  
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
    
    
class EfficientNetB0_FPN(BaseBackbone):

    def __init__(self, config):
                
        super(EfficientNetB0_FPN, self).__init__(config, 4)

        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        self.backbone = create_feature_extractor(
            efficientnet,
            return_nodes={
                'features.2': 'feat1',   # P1 (High resolution, low-level)
                'features.3': 'feat2',   # P2
                'features.5': 'feat3',   # P3
                'features.7': 'feat4',   # P4 (Low resolution, high-level)
            }
        )

        self.fpn = FeaturePyramidNetwork(
            [24, 40, 112, 320], 
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):
        
        x = self.backbone(x)        
        x = self.fpn(x)
                
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x)  
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
    
class DeeplabV3MobilenetV3_FPN(BaseBackbone):

    def __init__(self, config):
        
        super(DeeplabV3MobilenetV3_FPN, self).__init__(config, 5)

        deeplab = models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        )
                
        self.backbone = create_feature_extractor(
            deeplab,
            return_nodes = {
                'backbone.1': 'feat1',  # P1 (High resolution, low-level)
                'backbone.3': 'feat2',  # P2
                'backbone.6': 'feat3',  # P3
                'backbone.9': 'feat4',  # P4
                'backbone.13': 'feat5', # P5 (Low resolution, high-level)
            }
        )

        self.fpn = FeaturePyramidNetwork(
            [16, 24, 40, 80, 160], 
            out_channels=self.fpn_channels
        )
    
    def forward(self, x:torch.Tensor):
        
        x = self.backbone(x)   
        x = self.fpn(x)

        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x

class MobileViT_FPN(BaseBackbone):

    def __init__(self, config):
              
        super(MobileViT_FPN, self).__init__(config, 6)

        mobilevit = timm.create_model("mobilevitv2_200.cvnets_in22k_ft_in1k_384", pretrained=True)

        self.layer_1 = mobilevit.stem
        self.layer_2 = mobilevit.stages[0]
        self.layer_3 = mobilevit.stages[1]
        self.layer_4 = mobilevit.stages[2]
        self.layer_5 = mobilevit.stages[3]
        self.layer_6 = mobilevit.stages[4]

        self.fpn = FeaturePyramidNetwork(
            [64, 128, 256, 512, 768, 1024], 
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):

        feat1 = self.layer_1(x) 
        feat2 = self.layer_2(feat1)
        feat3 = self.layer_3(feat2)
        feat4 = self.layer_4(feat3)
        feat5 = self.layer_5(feat4)
        feat6 = self.layer_6(feat5)

        x = self.fpn({
            "feat1":feat1,
            "feat2":feat2,
            "feat3":feat3,
            "feat4":feat4,
            "feat5":feat5,
            "feat6":feat6
        })

        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
    
class DeiT3Small_FPN(BaseBackbone):

    def __init__(self, config):
        super(DeiT3Small_FPN, self).__init__(config, 5)

        deit_small = timm.create_model("deit3_small_patch16_384.fb_in22k_ft_in1k", pretrained=True)
    
        self.backbone = create_feature_extractor(
            deit_small,
            return_nodes = {
                'blocks.2': 'feat1',   # P1 (High resolution, low-level)
                'blocks.4': 'feat2',   # P2
                'blocks.6': 'feat3',   # P3
                'blocks.8': 'feat4',   # P4
                'blocks.10': 'feat5',  # P5 (Low resolution, high-level)
            }
        )

        self.fpn = FeaturePyramidNetwork(
            [384, 384, 384, 384, 384],  
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        
        # apply backbone
        x = self.backbone(x)  
       
        # reshape transformer features to = [batch_size, channels, 2*height, 2*width]
        for key in x:
            batch_size, seq_length, embed_dim = x[key].shape
            height = width = int((seq_length-1)**0.5)

            patch_features = x[key][:, 1:, :]
            patch_features = patch_features.permute(0, 2, 1).reshape(batch_size, embed_dim, height, width)
            x[key] = patch_features
                    
        x = self.fpn(x)      

        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
    
class EfficientFormer_FPN(BaseBackbone):

    def __init__(self, config):
        super(EfficientFormer_FPN, self).__init__(config, 4)

        efficientformer = timm.create_model("efficientformerv2_s0.snap_dist_in1k", pretrained=True)
        
        self.backbone = timm.models.create_feature_extractor(
            efficientformer,
            return_nodes={
                'stages.0': 'feat1',
                'stages.1': 'feat2',
                'stages.2': 'feat3',
                'stages.3': 'feat4',
            }
        )

        self.fpn = FeaturePyramidNetwork(
            [32, 48, 96, 176],  
            out_channels=self.fpn_channels
        )
        
        self.fpn_input_res = nn.Sequential(
            nn.Conv2d(3, self.fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU( ),
        )
                
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(2 * self.fpn_channels, self.fpn_channels, 3, 1, 1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.fpn_channels, self.fpn_channels, 4, 2, 1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.fpn_channels, self.fpn_channels, 4, 2, 1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU()
        )


    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
                
        x_res = self.fpn_input_res(x)
        
        x = self.backbone(x)
        
        x = self.fpn(x)      

        # [batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        
        # residual connection to recover local features
        x_res = F.interpolate(x_res, size=x.shape[-2:], mode='bilinear')
        x = torch.cat([x, x_res], dim=1)
        x = self.fusion_proj(x)
        
        # regularization
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
    
class LeViT_FPN(BaseBackbone):

    def __init__(self, config):
        super(LeViT_FPN, self).__init__(config, 4)

        levit = timm.create_model("levit_conv_384.fb_dist_in1k", pretrained=True)
        
        self.layer_1 = levit.stem
        self.layer_2 = levit.stages[0]
        self.layer_3 = levit.stages[1]
        self.layer_4 = levit.stages[2]

        self.fpn = FeaturePyramidNetwork(
            [384, 384, 512, 768],  
            out_channels=self.fpn_channels
        )
        
        self.fpn_input_res = nn.Sequential(
            nn.Conv2d(3, self.fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU( ),
        )
                
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(2 * self.fpn_channels, self.fpn_channels, 3, 1, 1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.fpn_channels, self.fpn_channels, 4, 2, 1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.fpn_channels, self.fpn_channels, 4, 2, 1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU()
        )

    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x_res = self.fpn_input_res(x)
        
        feat_1 = self.layer_1(x)       
        feat_2 = self.layer_2(feat_1)
        feat_3 = self.layer_3(feat_2)
        feat_4 = self.layer_4(feat_3)
                
        x = self.fpn({
            'feat1':feat_1,
            'feat2':feat_2,
            'feat3':feat_3,
            'feat4':feat_4
        })      

        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        
        # residual connection to recover local features
        x_res = F.interpolate(x_res, size=x.shape[-2:], mode='bilinear')
        x = torch.cat([x, x_res], dim=1)
        x = self.fusion_proj(x)
        
        # regularization        
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
    
class SegFormerB0_FPN(BaseBackbone):

    def __init__(self, config):
       
        super(SegFormerB0_FPN, self).__init__(config, 4)


        segformer_config = SegformerConfig.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            output_hidden_states=True
        )
        self.backbone = SegformerModel.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", 
            config=segformer_config
        )
        
        self.fpn = FeaturePyramidNetwork(
            [32, 64, 160, 256],  
            out_channels=self.fpn_channels
        )


    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        
        x = self.backbone(x)
        
        x = self.fpn( {
            'feat1': x.hidden_states[0], 
            'feat2': x.hidden_states[1],  
            'feat3': x.hidden_states[2],  
            'feat4': x.hidden_states[3], 
        })      

        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
    
class PiTXS_FPN(BaseBackbone):

    def __init__(self, config):
        super(PiTXS_FPN, self).__init__(config, 4)

        pitxs = timm.create_model("pit_xs_224.in1k", pretrained=True)
        
        self.backbone = create_feature_extractor(
            pitxs,
            return_nodes={
                'patch_embed': 'feat1',
                'transformers.0': 'feat2',
                'transformers.1': 'feat3',
                'transformers.2': 'feat4',
            }
        )
        
        self.fpn = FeaturePyramidNetwork(
            [96, 96, 192, 384],  
            out_channels=self.fpn_channels
        )
        
        self.fpn_input_res = nn.Sequential(
            nn.Conv2d(3, self.fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU( ),
        )
                
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(2 * self.fpn_channels, self.fpn_channels, 3, 1, 1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.fpn_channels, self.fpn_channels, 4, 2, 1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.fpn_channels, self.fpn_channels, 4, 2, 1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.GELU()
        )

    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
             
        x_res = self.fpn_input_res(x)
        
        x = self.backbone(x)
        x = self.fpn(x) 
                
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        
        # residual connection to recover local features
        x_res = F.interpolate(x_res, size=x.shape[-2:], mode='bilinear')
        x = torch.cat([x, x_res], dim=1)
        x = self.fusion_proj(x)
        
        # regularization
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    

class SAM2_Hiera_FPN(BaseBackbone):

    def __init__(self, config):
        super(SAM2_Hiera_FPN, self).__init__(config, 5)

        
        sam2_hiera = timm.create_model("sam2_hiera_tiny.fb_r896", pretrained=True)
       
        self.backbone = create_feature_extractor(
            sam2_hiera,
            return_nodes={
                'patch_embed': 'feat1',
                'blocks.0'   : 'feat2',
                'blocks.4'   : 'feat3',
                'blocks.7'   : 'feat4',
                'blocks.11'  : 'feat5',
            }
        )
        
        self.fpn = FeaturePyramidNetwork(
            [96, 96, 384, 384, 768],  
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(896, 896), mode='bilinear', align_corners=False)
         
        x = self.backbone(x)
        
        # [batch_size, height, width, channels] -> [batch_size, channels, height, width]
        x = {k:v.permute(0, 3, 1, 2) for k, v in x.items()} 
        
        x = self.fpn(x)     
                 
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
    
    
class TinyViT_FPN(BaseBackbone):

    def __init__(self, config):
        super(TinyViT_FPN, self).__init__(config, 5)
        
        self.tinyvit = timm.create_model("tiny_vit_11m_224.in1k", pretrained=True)
               
        self.fpn = FeaturePyramidNetwork(
            [64, 64, 128, 256, 448],  
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
       
        feat1 = self.tinyvit.patch_embed(x)
        feat2 = self.tinyvit.stages[0](feat1)
        feat3 = self.tinyvit.stages[1](feat2)
        feat4 = self.tinyvit.stages[2](feat3)
        feat5 = self.tinyvit.stages[3](feat4)

        x = {
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4,
            'feat5': feat5,
        }
        
        x = self.fpn(x)     
                 
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
    

class FastViT_FPN(BaseBackbone):

    def __init__(self, config):
        super(FastViT_FPN, self).__init__(config, 5)

        
        fastvit = timm.create_model("fastvit_sa12.apple_in1k", pretrained=True)
       
        self.backbone = create_feature_extractor(
            fastvit,
            return_nodes={
                'stem'       : 'feat1',
                'stages.0'   : 'feat2',
                'stages.1'   : 'feat3',
                'stages.2'   : 'feat4',
                'stages.3'   : 'feat5',
            }
        )
        
        
        self.fpn = FeaturePyramidNetwork(
            [64, 64, 128, 256, 512],  
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
         
        x = self.backbone(x)        
        x = self.fpn(x)     
        
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
    

class ConvNeXtV2_FPN(BaseBackbone):

    def __init__(self, config):
        super(ConvNeXtV2_FPN, self).__init__(config, 5)

        
        convnextv2 = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k_384", pretrained=True)
       
        self.backbone = create_feature_extractor(
            convnextv2,
            return_nodes={
                'stem'       : 'feat1',
                'stages.0'   : 'feat2',
                'stages.1'   : 'feat3',
                'stages.2'   : 'feat4',
                'stages.3'   : 'feat5',
            }
        )
        
        
        self.fpn = FeaturePyramidNetwork(
            [96, 96, 192, 384, 768],  
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
         
        x = self.backbone(x)        
        x = self.fpn(x)     
        
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
    
class MaxxViTV2_FPN(BaseBackbone):

    def __init__(self, config):
        super(MaxxViTV2_FPN, self).__init__(config, 5)

        
        maxxvitv2 = timm.create_model("maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k", pretrained=True)
       
        self.backbone = create_feature_extractor(
            maxxvitv2,
            return_nodes={
                'stem'       : 'feat1',
                'stages.0'   : 'feat2',
                'stages.1'   : 'feat3',
                'stages.2'   : 'feat4',
                'stages.3'   : 'feat5',
            }
        )
        
        self.fpn = FeaturePyramidNetwork(
            [128, 128, 256, 512, 1024],  
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.backbone(x)      
        x = self.fpn(x)
        
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x
    

class EdgeNeXt_FPN(BaseBackbone):

    def __init__(self, config):
        super(EdgeNeXt_FPN, self).__init__(config, 5)
        
        self.edgenext = timm.create_model("edgenext_base.in21k_ft_in1k", pretrained=True)
        
        self.fpn = FeaturePyramidNetwork(
            [80, 80, 160, 288, 584],  
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):
        # resize to transform model image size
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        feat1 = self.edgenext.stem(x)
        feat2 = self.edgenext.stages[0](feat1)
        feat3 = self.edgenext.stages[1](feat2)
        feat4 = self.edgenext.stages[2](feat3)
        feat5 = self.edgenext.stages[3](feat4)
        
        x = {
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4,
            'feat5': feat5,
        } 
        x = self.fpn(x)
        
        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x