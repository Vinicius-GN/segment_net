import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionWrapper(nn.Module):

    def __init__(self, attn_fn, channels):

        super(AttentionWrapper, self).__init__()
        print(channels)
        self.attn_fn = attn_fn

        self.norm_1 = nn.BatchNorm2d(channels)
        self.norm_2 = nn.BatchNorm2d(channels)

        self.alpha = nn.Parameter(torch.tensor(1e-6))
        self.beta  = nn.Parameter(torch.tensor(1e-6))

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels*4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels*4, channels, kernel_size=1),
        )

    
    def forward(self, x, return_scores=False):

        x = x + self.alpha*self.attn_fn(self.norm_1(x))

        x = x + self.beta*self.mlp(self.norm_2(x))

        return x
        