import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, groups=1, act=True, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.act(x); x = self.drop(x)
        return x

class PAMEfficient(nn.Module):
    def __init__(self, in_channels, inter_channels=None, tile_hw: int = 64, dropout=0.0):
        super().__init__()
        inter_channels = inter_channels or max(8, in_channels // 4)
        self.query = ConvBNReLU(in_channels, inter_channels, k=1, act=False)
        self.key   = ConvBNReLU(in_channels, inter_channels, k=1, act=False)
        self.value = ConvBNReLU(in_channels, inter_channels, k=1, act=False, dropout=dropout)
        self.proj  = ConvBNReLU(inter_channels, inter_channels, k=1, act=False)
        self.scale = inter_channels ** -0.5
        self.tile  = tile_hw

    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query(x).view(b, -1, h * w).transpose(1, 2)     
        k = self.key(x).view(b, -1, h * w)                       
        v = self.value(x).view(b, -1, h * w).transpose(1, 2)     

        tile = self.tile
        out = x.new_zeros(b, h * w, v.size(-1))
        for i in range(0, h * w, tile):
            qi = q[:, i:i + tile, :]                              
            attn = torch.bmm(qi, k) * self.scale                  
            attn = F.softmax(attn, dim=-1)
            out[:, i:i + tile, :] = torch.bmm(attn, v)            

        out = out.transpose(1, 2).contiguous().view(b, -1, h, w)   
        out = self.proj(out)                                      
        return out

class CAMEfficient(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dropout=0.0):
        super().__init__()
        inter_channels = inter_channels or max(8, in_channels // 4)
        self.theta = ConvBNReLU(in_channels, inter_channels, k=1, act=False)
        self.phi   = ConvBNReLU(in_channels, inter_channels, k=1, act=False)
        self.g     = ConvBNReLU(in_channels, inter_channels, k=1, act=False, dropout=dropout)
        self.proj  = ConvBNReLU(inter_channels, inter_channels, k=1, act=False)
        self.scale = inter_channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.size()
        hw = h * w
        theta = self.theta(x).view(b, -1, hw)                     
        phi   = self.phi(x).view(b, -1, hw)                       
        g     = self.g(x).view(b, -1, hw)                         

        attn = torch.bmm(theta, phi.transpose(1, 2)) * self.scale  
        attn = F.softmax(attn, dim=-1)
        out  = torch.bmm(attn, g).view(b, -1, h, w)                
        out  = self.proj(out)                                     
        return out

class DANetModule(nn.Module):
    def __init__(self, in_channels, reduction=4, dropout=0.0, tile_hw=64):
        super().__init__()
        inter = max(8, in_channels // reduction)

        self.pre_p = ConvBNReLU(in_channels, inter, k=1)
        self.pam   = PAMEfficient(inter, inter_channels=inter, tile_hw=tile_hw, dropout=dropout)
        self.post_p= ConvBNReLU(inter, inter, k=1, dropout=dropout)

        self.pre_c = ConvBNReLU(in_channels, inter, k=1)
        self.cam   = CAMEfficient(inter, inter_channels=inter, dropout=dropout)
        self.post_c= ConvBNReLU(inter, inter, k=1, dropout=dropout)

        self.conv_fuse = ConvBNReLU(inter, in_channels, k=1, act=True, dropout=dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        fp = self.pre_p(x)
        fp = self.pam(fp)
        fp = self.post_p(fp)            

        fc = self.pre_c(x)
        fc = self.cam(fc)
        fc = self.post_c(fc)            

        fused_inter = self.alpha * fp + self.beta * fc            
        y = self.conv_fuse(fused_inter)                           
        y = y + x                                            
        return y
