import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
import timm

from .base import BaseBackbone


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
    