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
            
            # Modify first conv layer only if in_channels != 3
            # If in_channels == 3, keep the native pretrained conv1 weights as-is
            if in_channels != 3:
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
                
            # Freeze early layers strategy: freeze ONLY model.bn1, model.layer1, model.layer2 params.
            # conv1, layer3, layer4, and fc all remain trainable. Exact references, no name-substring matching.
            for module in [self.model.bn1, self.model.layer1, self.model.layer2]:
                for param in module.parameters():
                    param.requires_grad = False
                    
        else:
            # Variant (a): Trained from scratch
            self.model = resnet18(weights=None)
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # Modify the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Print parameter summary at startup
        self.print_parameter_summary()

    def print_parameter_summary(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        print(f"Parameter Summary: Trainable={trainable:,} ({100*trainable/total:.1f}%), Frozen={frozen:,}, Total={total:,}")
        for name, mod in [('conv1', self.model.conv1), ('bn1', self.model.bn1),
                          ('layer1', self.model.layer1), ('layer2', self.model.layer2),
                          ('layer3', self.model.layer3), ('layer4', self.model.layer4),
                          ('fc', self.model.fc)]:
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            f = sum(p.numel() for p in mod.parameters() if not p.requires_grad)
            req = "trainable" if t > 0 else "frozen"
            print(f"  {name:7s}: {req:9s} (trainable={t:,}, frozen={f:,})")
        
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
