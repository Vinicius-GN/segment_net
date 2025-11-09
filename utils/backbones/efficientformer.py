import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
import timm

from ..aggregation.base import BaseBackbone

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