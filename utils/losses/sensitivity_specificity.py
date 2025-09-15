import torch
import torch.nn as nn


class SensitivitySpecificityLoss(nn.Module):
    def __init__(self, use_sigmoid: bool = False, reduction: str = 'mean'):
        super(SensitivitySpecificityLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.use_sigmoid:
            inputs = torch.sigmoid(inputs)
        else:
            inputs = torch.softmax(inputs, dim=1)

        if targets.ndim == 3 and inputs.ndim == 4 and inputs.shape[1] > 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])
            targets = targets.permute(0, 3, 1, 2).float()

        elif targets.ndim == 3 and inputs.ndim == 4 and inputs.shape[1] == 1:
            targets = targets.unsqueeze(1).float()

        true_positive = (inputs * targets).sum(dim=(2, 3))
        true_negative = ((1 - inputs) * (1 - targets)).sum(dim=(2,3))
        false_negative = ((1 - inputs) * targets).sum(dim=(2, 3))
        false_positive = (inputs * (1 - targets)).sum(dim=(2, 3))

        sensitivity = true_positive / (true_positive + false_negative + 1e-6)
        specificity = true_negative / (true_negative + false_positive + 1e-6)

        loss = 1 - (sensitivity + specificity) / 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
if __name__ == "__main__":
    # Example usage
    inputs = torch.randn(4, 1, 256, 256)  # Batch of 4 images
    targets = torch.randint(0, 2, (4, 1, 256, 256)).float()  # Binary targets

    loss_fn = SensitivitySpecificityLoss(use_sigmoid=True)
    loss = loss_fn(inputs, targets)
    print(f"Sensitivity-Specificity Loss: {loss.item()}")