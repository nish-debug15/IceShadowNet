import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class IceResNet(nn.Module):
    """
    ResNet18 implementation for Lunar Ice Classification.
    
    Domain Mismatch Tradeoff for Viva:
    - Pretrained (ImageNet): ImageNet models are trained on RGB optical photos (3 channels, visual features like edges, dog faces, etc.). 
      SAR data (2 channels, radar backscatter, speckle noise) has fundamentally different statistics and physical meaning.
      Transfer learning might still work if the early layers act as generic edge/texture extractors, but we must adapt the first layer 
      to accept 2 channels. We freeze the early layers and fine-tune the later ones to prevent catastrophic forgetting.
    - Scratch: Training from scratch avoids the domain mismatch, but requires significantly more SAR data to converge 
      and avoid overfitting, which might be a challenge with a small single-crater dataset.
    
    This module supports both variants via the `pretrained` flag.
    """
    def __init__(self, in_channels=2, num_classes=1, pretrained=True):
        super(IceResNet, self).__init__()
        
        if pretrained:
            # Variant (b): Adapted from pretrained ImageNet weights
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            
            # Modify first conv layer to accept `in_channels` instead of 3
            # We initialize the new conv layer by averaging the RGB weights across channels
            original_conv1 = self.model.conv1
            self.model.conv1 = nn.Conv2d(in_channels, original_conv1.out_channels, 
                                         kernel_size=original_conv1.kernel_size, 
                                         stride=original_conv1.stride, 
                                         padding=original_conv1.padding, 
                                         bias=original_conv1.bias is not None)
            
            with torch.no_grad():
                # Average the weights of the 3 ImageNet channels and replicate/slice for in_channels
                avg_weight = torch.mean(original_conv1.weight, dim=1, keepdim=True)
                self.model.conv1.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))
                
            # Freeze early layers strategy: Freeze layer1 and layer2, fine-tune layer3, layer4 and fc
            for name, param in self.model.named_parameters():
                if "layer1" in name or "layer2" in name or "conv1" in name or "bn1" in name:
                    param.requires_grad = False
                    
        else:
            # Variant (a): Trained from scratch
            self.model = resnet18(weights=None)
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # Modify the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Test Pretrained variant
    model_pretrained = IceResNet(pretrained=True)
    dummy_input = torch.randn(2, 2, 256, 256)
    out1 = model_pretrained(dummy_input)
    print("Pretrained output shape:", out1.shape)
    
    # Test Scratch variant
    model_scratch = IceResNet(pretrained=False)
    out2 = model_scratch(dummy_input)
    print("Scratch output shape:", out2.shape)
