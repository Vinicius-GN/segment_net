
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
from transformers import Swinv2Config, Swinv2Model

from ..aggregation.base import BaseBackbone


class SwinV2_FPN(BaseBackbone):

    def __init__(self, config):
       
        super(SwinV2_FPN, self).__init__(config, 4)


        swinv2_config = Swinv2Config.from_pretrained(
            "microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft",
            output_hidden_states=True
        )
        self.backbone = Swinv2Model.from_pretrained(
            "microsoft/swinv2-base-patch4-window12to16-192to256-22kto1k-ft",
            config=swinv2_config
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=self.fpn_channels
        )

    def forward(self, x: torch.Tensor):
        
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = self.backbone(x).hidden_states
        reshaped_states = []
        for i in range(len(x)):
            b, n, c = x[i].shape
            h = w = int(n**0.5)
            reshaped_states.append(x[i].permute(0, 2, 1).reshape(b, c, h, w))
           
        features = {
            "feat1": reshaped_states[0],  
            "feat2": reshaped_states[1],  
            "feat3": reshaped_states[2],  
            "feat4": reshaped_states[3],  
        }

        x = self.fpn(features)

        x = self.aggregate_image_features(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
    