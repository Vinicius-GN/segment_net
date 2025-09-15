import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor

from .base import BaseBackbone

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
    