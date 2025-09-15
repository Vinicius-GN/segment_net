import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor

from .base import BaseBackbone


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