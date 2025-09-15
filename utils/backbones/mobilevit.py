import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FeaturePyramidNetwork
import timm

from .base import BaseBackbone


class MobileViT_FPN(BaseBackbone):

    def __init__(self, config):
              
        super(MobileViT_FPN, self).__init__(config, 6)

        mobilevit = timm.create_model("mobilevitv2_200.cvnets_in22k_ft_in1k_384", pretrained=True)

        self.layer_1 = mobilevit.stem
        self.layer_2 = mobilevit.stages[0]
        self.layer_3 = mobilevit.stages[1]
        self.layer_4 = mobilevit.stages[2]
        self.layer_5 = mobilevit.stages[3]
        self.layer_6 = mobilevit.stages[4]

        self.fpn = FeaturePyramidNetwork(
            [64, 128, 256, 512, 768, 1024], 
            out_channels=self.fpn_channels
        )

    def forward(self, x:torch.Tensor):

        feat1 = self.layer_1(x) 
        feat2 = self.layer_2(feat1)
        feat3 = self.layer_3(feat2)
        feat4 = self.layer_4(feat3)
        feat5 = self.layer_5(feat4)
        feat6 = self.layer_6(feat5)

        x = self.fpn({
            "feat1":feat1,
            "feat2":feat2,
            "feat3":feat3,
            "feat4":feat4,
            "feat5":feat5,
            "feat6":feat6
        })

        x = self.aggregate_image_features(x) 
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x