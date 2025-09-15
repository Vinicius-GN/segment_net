import torch
import torch.nn as nn
import torch.nn.functional as F

# based on: https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch
#         : https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 bias=False):
        
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            groups=in_channels,
            bias=bias
        )
        
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, 
            stride=1, 
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x   
    
class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self,  
                 in_channels, 
                 out_channels, 
                 scale_factor=2, 
                 norm_fn='batch_norm'):
        
        super(DepthwiseSeparableConvBlock, self).__init__()
        
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ds_conv = DepthwiseSeparableConv(in_channels, out_channels)

        if norm_fn == 'batch_norm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_fn == 'group_norm':
            groups = max(2, out_channels // 16)
            self.norm = nn.GroupNorm(groups, out_channels)
        else:
            raise ValueError(f"Unsupported normalization: {norm_fn}")

        self.activation = nn.GELU()

        if in_channels != out_channels:
            self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        
        identity = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        identity = self.res_proj(identity)

        x = self.ds_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

        x += identity
        x = self.activation(x)
        
        return x