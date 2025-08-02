import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        reduced_channels = max(2, channels // reduction)

        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)

        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        # [B, C, H, W]
        batch, channels, _, _ = x.size()
        
        # [B, C, H, W] -> [B, C]
        squeeze = self.global_avg_pool(x).view(batch, channels)
        
        # [B, C] -> [B, R]
        excitation = F.relu(self.fc1(squeeze))
        
        # [B, R] -> [B, C, 1, 1]: weights
        excitation = self.sigmoid(self.fc2(excitation)).view(batch, channels, 1, 1)
                
        return x * excitation
         
class ClassChannelAttention(nn.Module):
    def __init__(self, channels, num_classes, dropout):
        super(ClassChannelAttention, self).__init__()
        self.num_classes = num_classes
        self.channel_attention = nn.Parameter(torch.randn(num_classes, channels))
        
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # [B, C, H, W]
        B, C, H, W = x.shape
        
        # [B, K, C]
        attn_weights = self.channel_attention.unsqueeze(0).expand(B, -1, -1)  
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # [B, C, H, W] -> [B, 1, C, H, W]
        x = x.unsqueeze(1)
        
        # [B, 1, C, H, W] -> [B, K, C, H, W]
        x = attn_weights.view(B, self.num_classes, C, 1, 1) * x  

        # [B, K, C, H, W] -> [B, C, H, W]
        x = x.sum(dim=1)
        
        return x  
    
    
class ClassSpatialAttention(nn.Module):
    def __init__(self, channels, num_classes, dropout):
        super(ClassSpatialAttention, self).__init__()
        
        self.attn_conv = nn.Conv2d(channels, num_classes, kernel_size=1)
        
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # [B, C, H, W] -> [B, K, H, W]
        attn_map = torch.sigmoid(self.attn_conv(x)) 
        attn_map = self.attn_dropout(attn_map)
        
        # [B, C, H, W] -> [B, 1, C, H, W]
        x = x.unsqueeze(1)  
        
        # [B, 1, C, H, W] * [B, K, H, W] -> [B, K, C, H, W] 
        x = attn_map.unsqueeze(2) * x  
        
        # [B, K, C, H, W] -> [B, C, H, W]
        x = x.sum(dim=1)
        return x
    
    
class ClassQueryAttention(nn.Module):
    def __init__(self, channels, embed_dim, num_classes, dropout, global_per_class=True):
        super(ClassQueryAttention, self).__init__()
        
        self.global_per_class = global_per_class
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        self.query_embed = nn.Parameter(torch.randn(num_classes, embed_dim))
    
        self.key_proj = nn.Conv2d(channels, embed_dim, kernel_size=1)
        self.value_proj = nn.Conv2d(channels, embed_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(embed_dim, channels, kernel_size=1)
        
        self.attn_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        K = self.num_classes

        # [B, C, H, W] -> [B, H*W, embed_dim]
        key = self.key_proj(x).flatten(2).permute(0, 2, 1)  
        # [B, C, H, W] -> [B, H*W, embed_dim]
        value = self.value_proj(x).flatten(2).permute(0, 2, 1) 
        # [B, K, embed_dim]
        query = self.query_embed.unsqueeze(0).expand(B, K, -1)

        # [B, K, embed_dim]*[B, H*W, embed_dim] -> [B, K, H*W]
        attn = torch.einsum('bkd,bnd->bkn', query, key) / (self.embed_dim ** 0.5)  
        attn = attn.softmax(-1)
        attn = self.attn_dropout(attn)

        # global weights
        if self.global_per_class:
            # [B, K, H*W]*[B, H*W, embed_dim] -> [B, K, embed_dim]
            weighted = torch.einsum('bkn,bnd->bkd', attn, value)  

            # [B, K, d] -> [B*K, embed_dim, 1, 1]
            weighted = weighted.view(B*K, self.embed_dim, 1, 1)
    
            #[B*K, embed_dim, 1, 1] -> [B, K, C, 1, 1]
            out = self.out_proj(weighted).view(B, K, C, 1, 1)  
            # [B, C, H, W] -> [B, 1, C, H, W]
            x = x.unsqueeze(1)   
            # [B, K, C, H, W] -> [B, C, H, W]      
            return (x * out).sum(dim=1)  
        
        else: #spatial
            # [B, K, H*W] -> [B, K, H, W]
            attn = attn.view(B, K, H, W)       
            # [B, K, 1, H, W]*[B, 1, C, H, W] -> [B, K, C, H, W] 
            x = attn.unsqueeze(2) * x.unsqueeze(1) 
            # [B, K, C, H, W] -> [B, C, H, W]
            x = x.sum(dim=1) 
            return x
            