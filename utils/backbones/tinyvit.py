import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
import timm
from .base import BaseBackbone

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