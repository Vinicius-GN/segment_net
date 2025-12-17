import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor

from ..aggregation.base import BaseBackbone

class Resnet18_FPN(BaseBackbone):

    def __init__(self, config):        
               
        super(Resnet18_FPN, self).__init__(config, 4)

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = create_feature_extractor(
            resnet,
            return_nodes={
                'layer1': 'feat1',  # high resolution, local features, low level abstraction
                'layer2': 'feat2', 
                'layer3': 'feat3',  
                'layer4': 'feat4',  # low resolution, global features, high level abstraction
            }
        )

        self.fpn = FeaturePyramidNetwork(
            [64, 128, 256, 512], 
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
    
class Resnet18(BaseBackbone):

    def __init__(self, config):        
               
        super(Resnet18, self).__init__(config, 4)

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = create_feature_extractor(
            resnet,
            return_nodes={
                'layer1': 'feat1',  # high resolution, local features, low level abstraction
                'layer2': 'feat2', 
                'layer3': 'feat3',  
                'layer4': 'feat4',  # low resolution, global features, high level abstraction
            }
        )

        self.up = nn.Conv2d(512, 1024, kernel_size=1)
        self.shuffle = nn.PixelShuffle(2)

       
    def forward(self, x:torch.Tensor):
        
        x = self.backbone(x)['feat4']
        x= self.up(x)
        x= self.shuffle(x)
        
        #[batch_size, channel, width, height]
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x