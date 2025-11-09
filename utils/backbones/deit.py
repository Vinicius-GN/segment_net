import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
import timm

from ..aggregation.base import BaseBackbone


class DeiT3Small_FPN(BaseBackbone):

    def __init__(self, config):
        super(DeiT3Small_FPN, self).__init__(config, 5)

        deit_small = timm.create_model("deit3_small_patch16_384.fb_in22k_ft_in1k", pretrained=True)
    
        self.backbone = create_feature_extractor(
            deit_small,
            return_nodes = {
                'blocks.2': 'feat1',   # P1 (High resolution, low-level)
                'blocks.4': 'feat2',   # P2
                'blocks.6': 'feat3',   # P3
                'blocks.8': 'feat4',   # P4
                'blocks.10': 'feat5',  # P5 (Low resolution, high-level)
            }
        )

        self.fpn = FeaturePyramidNetwork(
            [384, 384, 384, 384, 384],  
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):
        
        # resize to transform model image size
        x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        
        # apply backbone
        x = self.backbone(x)  
       
        # reshape transformer features to = [batch_size, channels, 2*height, 2*width]
        for key in x:
            batch_size, seq_length, embed_dim = x[key].shape
            height = width = int((seq_length-1)**0.5)

            patch_features = x[key][:, 1:, :]
            patch_features = patch_features.permute(0, 2, 1).reshape(batch_size, embed_dim, height, width)
            x[key] = patch_features
                    
        x = self.fpn(x)      

        #[batch_size, channel, width, height]
        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
    