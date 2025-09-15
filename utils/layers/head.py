
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple

from utils.layers.depthwise_separable_conv import DepthwiseSeparableConvBlock
from utils.layers.deeplabv3 import DeepLabV3PlusCore
class Interpolate2D(nn.Module):
    def __init__ (self, scale, mode):

        super(Interpolate2D, self).__init__()
        self.scale=scale
        self.mode=mode

    def forward(self, x):
        return F.interpolate(x, 
                             scale_factor=self.scale,
                             mode=self.mode, 
                             align_corners=True)


class SEConvInterpHead(nn.Module):

    def __init__(self, config, num_classes:int, num_feature_layers:int):
        super(SEConvInterpHead, self).__init__()

        self.config = config
        self.channel_size = self.config.get("backbone").get("fpn_out_channels")
        self.image_size = list(reversed(self.config.get("image").get("image_size")))
        self.dropout_prob = self.config.get("head").get("dropout")

        self.opt_latency = self.config.get("head").get("opt_latency")
        
        # projection
        if self.config.get("backbone").get("aggregate") == "concat":            
            in_channels = self.channel_size*num_feature_layers
            num_groups = self.channel_size//8
        else:
            in_channels = self.channel_size
            num_groups = self.channel_size//8

        # head 
        self.conv1 = nn.Conv2d(in_channels, 
                               in_channels//2,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               dilation=2)
        
        self.conv2 = nn.Conv2d(in_channels//2, 
                               in_channels//4,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               dilation=2)

        self.conv3 = nn.Conv2d(in_channels//4, 
                               in_channels//8,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               dilation=2)

        self.conv4 = nn.Conv2d(in_channels//8, 
                               in_channels//16,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               dilation=2)
            
        self.classifier = nn.Conv2d(in_channels//16, 
                                  num_classes,
                                  kernel_size=1)
        
        # norm
        if self.config.get("model").get("norm_fn") == "batch_norm":
            self.n1   = nn.BatchNorm2d(in_channels//2)
            self.n2   = nn.BatchNorm2d(in_channels//4)
            self.n3   = nn.BatchNorm2d(in_channels//8)
            self.n4   = nn.BatchNorm2d(in_channels//16)
        elif self.config.get("model").get("norm_fn") == "group_norm":
            num_groups = max(2, in_channels//16)
            self.n1   = nn.GroupNorm(num_groups, in_channels//2)
            self.n2   = nn.GroupNorm(num_groups, in_channels//4)
            self.n3   = nn.GroupNorm(num_groups, in_channels//8)
            self.n4   = nn.GroupNorm(num_groups, in_channels//16)

    def forward(self, x):
        # head
        x = F.gelu(self.n1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = F.gelu(self.n2(self.conv2(x)))
        x = F.dropout(x, self.dropout_prob)
 
        x = F.gelu(self.n3(self.conv3(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = F.gelu(self.n4(self.conv4(x)))
        x = F.dropout(x, self.dropout_prob)
  
        # classify
        if self.opt_latency:
            x = self.classifier(x)
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)  
        else:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
            x = self.classifier(x)          
        return x
        
class DepthwiseSeparableHead(nn.Module):
    
    def __init__(self, config, 
                 num_classes:int, 
                 num_feature_layers:int, 
                 num_blocks:int=4):
        super(DepthwiseSeparableHead, self).__init__()

        self.config = config
        
        self.image_size = list(reversed(self.config.get("image").get("image_size")))
        self.opt_latency = self.config.get("head").get("opt_latency")
                
        channel_size = self.config.get("backbone").get("fpn_out_channels")
        num_classes = num_classes
        dropout_prob = self.config.get("head").get("dropout")
        norm_fn = self.config.get("model").get("norm_fn") 
        

        if self.config.get("backbone").get("aggregate") == "concat":            
            in_channels = channel_size*num_feature_layers
        else:
            in_channels = channel_size
            
        intermediate_channels = [channel_size]
        for i in range(1, num_blocks):
            intermediate_channels.append(
                max(8, intermediate_channels[i-1]//2)
            )
            
        layers = []        
        layers.append(
            nn.Dropout(dropout_prob, inplace=True)
        )
        
        prev_channels = in_channels
        for out_channels in intermediate_channels:
            layers.append(
                DepthwiseSeparableConvBlock(prev_channels, 
                                            out_channels, 
                                            scale_factor=1.25, 
                                            norm_fn=norm_fn)
            )
            prev_channels = out_channels
            
        self.upsample_blocks = nn.Sequential(*layers)        
        self.classifier = nn.Conv2d(prev_channels, num_classes, kernel_size=1)
       

    def forward(self, x):    
        x = self.upsample_blocks(x)
        
        # classify
        if self.opt_latency:
            x = self.classifier(x)
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)  
        else:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
            x = self.classifier(x)    
        return x


class LightweightTransformerHead(nn.Module):
     
    def __init__(self, config, 
                 num_classes:int, 
                 num_feature_layers:int,
                 num_heads:int=4, 
                 num_layers:int=4):
        
        super(LightweightTransformerHead, self).__init__()
        
        self.config = config
        
        self.image_size = list(reversed(self.config.get("image").get("image_size")))
        self.opt_latency = self.config.get("head").get("opt_latency")
        
        channel_size = self.config.get("backbone").get("fpn_out_channels")
        num_classes = num_classes
        dropout_prob = self.config.get("head").get("dropout")
                
        if self.config.get("backbone").get("aggregate") == "concat":            
            in_channels = channel_size*num_feature_layers
        else:
            in_channels = channel_size
                        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        emb_dim = max(32, in_channels // 4)
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=1)

        self.pos_emb = nn.Parameter(
            torch.randn(1, emb_dim, self.image_size[0]//4, self.image_size[1]//4)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=channel_size,
            dropout=dropout_prob,
            activation='gelu',
            batch_first=True  # (B, HW, C)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Conv2d(emb_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.proj(x) 
        
        # pooling to reduce cost
        x = self.pool(x)
        _, _, HP, WP = x.shape
        
        # Positional Encoding
        pos = F.interpolate(self.pos_emb, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        x = x + pos
   
        #  [B, C, H, W] -> [B, H*W, C]
        x = x.flatten(2).permute(0, 2, 1)  

        # Transformer layers
        x = self.transformer(x)

        #  [B, H*W, C] -> [B, C, H, W]
        x = x.permute(0, 2, 1).view(B, -1, HP, WP)
        
        # Upsample (due pooling)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        # classify
        if self.opt_latency:
            x = self.classifier(x)
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)  
        else:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=False)
            x = self.classifier(x)    
        
        return x

class SegFormerAllMLPHead(nn.Module):
    #SegFormer-style all-MLP decoder head
  
    def __init__(self, config, num_classes: int, num_feature_layers: int, embed_dim: int = None):
        super().__init__()
        self.config = config
        self.image_size = list(reversed(self.config.get("image").get("image_size")))
        self.opt_latency = self.config.get("head").get("opt_latency")
        self.dropout_prob = self.config.get("head").get("dropout")
        self.norm_fn = self.config.get("model").get("norm_fn")

        self.fpn_out = self.config.get("backbone").get("fpn_out_channels")
        self.aggregate = self.config.get("backbone").get("aggregate")
        self.num_scales = int(num_feature_layers)

        if embed_dim is None:
            embed_dim = max(32, self.fpn_out)   # mild default
        self.embed_dim = embed_dim

        self.proj = nn.ModuleList([
            nn.Conv2d(self.fpn_out, self.embed_dim, kernel_size=1, bias=False)
            for _ in range(self.num_scales)
        ])
        if self.norm_fn == "batch_norm":
            self.proj_norm = nn.ModuleList([nn.BatchNorm2d(self.embed_dim) for _ in range(self.num_scales)])
        elif self.norm_fn == "group_norm":
            ng = max(2, self.embed_dim // 16)
            self.proj_norm = nn.ModuleList([nn.GroupNorm(ng, self.embed_dim) for _ in range(self.num_scales)])
        else:
            self.proj_norm = nn.ModuleList([nn.Identity() for _ in range(self.num_scales)])

        fused_in = self.embed_dim * self.num_scales
        self.fuse = nn.Conv2d(fused_in, self.embed_dim, kernel_size=3, padding=1, bias=False)
        if self.norm_fn == "batch_norm":
            self.fuse_norm = nn.BatchNorm2d(self.embed_dim)
        elif self.norm_fn == "group_norm":
            ng = max(2, self.embed_dim // 16)
            self.fuse_norm = nn.GroupNorm(ng, self.embed_dim)
        else:
            self.fuse_norm = nn.Identity()

        self.dropout = nn.Dropout(self.dropout_prob, inplace=False)
        self.classifier = nn.Conv2d(self.embed_dim, num_classes, kernel_size=1)

    def _split_concat_tensor(self, x: torch.Tensor):
        B, C, H, W = x.shape
        expected = self.fpn_out * self.num_scales
        if C != expected:
            return [x]
        return list(torch.split(x, self.fpn_out, dim=1))

    def _ensure_list_features(self, x):
        if isinstance(x, (list, tuple)):
            feats = list(x)
        else:
            feats = self._split_concat_tensor(x)
        if len(feats) < self.num_scales:
            feats = feats + [feats[-1]] * (self.num_scales - len(feats))
        elif len(feats) > self.num_scales:
            feats = feats[:self.num_scales]
        return feats

    def forward(self, x):
        feats = self._ensure_list_features(x)

        sizes = [f.shape[-2:] for f in feats]
        H_ref, W_ref = max(sizes, key=lambda s: s[0] * s[1])

        proj_up = []
        for i, f in enumerate(feats):
            y = self.proj[i](f)
            y = self.proj_norm[i](y)
            y = F.gelu(y)
            if y.shape[-2:] != (H_ref, W_ref):
                y = F.interpolate(y, size=(H_ref, W_ref), mode="bilinear", align_corners=False)
            proj_up.append(y)

        z = torch.cat(proj_up, dim=1) 
        z = self.fuse_norm(F.gelu(self.fuse(z)))
        z = self.dropout(z)

        if self.opt_latency:
            z = self.classifier(z)
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
        else:
            z = F.interpolate(z, size=self.image_size, mode="bilinear", align_corners=False)
            z = self.classifier(z)
        return z

class DeepLabV3PlusHead(nn.Module):

    def __init__(self, config, num_classes: int, num_feature_layers: int):
        super().__init__()
        self.config = config
        self.num_classes = int(num_classes)
        self.num_feature_layers = int(num_feature_layers)

        self.image_size = list(reversed(self.config.get("image").get("image_size")))
        self.opt_latency = bool(self.config.get("head").get("opt_latency"))
        norm_fn = self.config.get("model").get("norm_fn") or "batch_norm"
        dropout = float(self.config.get("head").get("dropout", 0.1))

        fpn_out = int(self.config.get("backbone").get("fpn_out_channels"))
        aggregate = (self.config.get("backbone").get("aggregate") or "").lower()

        aspp_channels    = int(self.config.get("head").get("aspp_channels", 256))
        decoder_channels = int(self.config.get("head").get("decoder_channels", 256))
        low_channels     = int(self.config.get("head").get("low_channels", 48))

        os_val = int(self.config.get("head").get("output_stride", 16))
        if os_val == 8:
            aspp_rates = self.config.get("head").get("aspp_rates", [12, 24, 36])
        else:
            aspp_rates = self.config.get("head").get("aspp_rates", [6, 12, 18])

        if aggregate == "concat":
            in_ch_high = fpn_out * self.num_feature_layers
            in_ch_low  = in_ch_high  
        else:
            in_ch_high = fpn_out
            in_ch_low  = fpn_out

        self.core = DeepLabV3PlusCore(
            in_ch_high=in_ch_high,
            in_ch_low=in_ch_low,
            aspp_channels=aspp_channels,
            decoder_channels=decoder_channels,
            low_channels=low_channels,
            aspp_rates=aspp_rates,
            norm_fn=norm_fn,
            dropout=dropout
        )
        self.classifier = nn.Conv2d(decoder_channels if aggregate != "concat" else aspp_channels,
                                    self.num_classes, kernel_size=1)

        self._aggregate = aggregate

    @staticmethod
    def _ensure_list(x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]):
        feats = self._ensure_list(x)

        if len(feats) > 1 and self._aggregate != "concat":
            f_high, f_low = self.core.pick_high_low(feats)
            y = self.core.forward_pair(f_high, f_low)
            if self.opt_latency:
                y = self.classifier(y)
                y = F.interpolate(y, size=self.image_size, mode="bilinear", align_corners=False)
            else:
                y = F.interpolate(y, size=self.image_size, mode="bilinear", align_corners=False)
                y = self.classifier(y)
            return y

        y = self.core.forward_single(feats[0])
        if self.opt_latency:
            y = self.classifier(y)
            y = F.interpolate(y, size=self.image_size, mode="bilinear", align_corners=False)
        else:
            y = F.interpolate(y, size=self.image_size, mode="bilinear", align_corners=False)
            y = self.classifier(y)
        return y