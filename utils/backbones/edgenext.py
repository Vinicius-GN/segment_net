import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
import timm

from ..aggregation.base import BaseBackbone

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