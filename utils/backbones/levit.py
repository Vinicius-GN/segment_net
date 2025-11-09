import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
import timm
from ..aggregation.base import BaseBackbone


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