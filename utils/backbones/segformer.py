

import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision.ops import FeaturePyramidNetwork
from transformers import SegformerModel, SegformerConfig

from .base import BaseBackbone


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
    
class SegFormerB2_FPN(BaseBackbone):

    def __init__(self, config):
       
        super(SegFormerB2_FPN, self).__init__(config, 4)


        segformer_config = SegformerConfig.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            output_hidden_states=True
        )
        self.backbone = SegformerModel.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512", 
            config=segformer_config
        )
        

        self.fpn = FeaturePyramidNetwork(
            [64, 128, 320, 512],  
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