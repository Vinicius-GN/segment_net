import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
import timm

from ..aggregation.base import BaseBackbone

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
    