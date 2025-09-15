import torch
import torch.nn as nn   

def flatten(input, target, ignore_index):
    num_class = input.size(1)
    input = input.permute(0, 2, 3, 1).contiguous()
    
    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)
    
    mask = (target_flatten != ignore_index)
    input_flatten = input_flatten[mask]
    target_flatten = target_flatten[mask]
    
    return input_flatten, target_flatten

class JaccardLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
      
    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        input = torch.softmax(input, dim=1)
        num_classes = input.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]
            
            intersection = (input_c * target_c).sum()
            total = (input_c + target_c).sum()
            union = total - intersection
            IoU = (intersection + self.smooth)/(union + self.smooth)
            
            losses.append(1-IoU)
        
        losses = torch.stack(losses)
        loss = losses.mean()
        return loss