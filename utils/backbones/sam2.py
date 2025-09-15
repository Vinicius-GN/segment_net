import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
import timm

from .base import BaseBackbone

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