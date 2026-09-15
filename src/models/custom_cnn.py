import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    """
    Custom CNN Architecture for SAR Ice Classification.
    
    Design Justification for Viva:
    - Input: 2 channels (DFSAR provides dual-frequency, e.g., L-band and S-band or different polarizations like HH, HV).
    - Architecture: We use a sequence of Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d blocks. 
      BatchNorm is crucial here because SAR data often has high dynamic range and speckle noise, 
      which makes activations unstable.
    - Dropout: Applied before the final fully connected layers to prevent overfitting on our 
      (potentially small, single-crater) dataset.
    - Capacity: Kept relatively small (compared to deep ResNets) to avoid memorizing 
      threshold-like features and encourage learning spatial textures.
    """
    def __init__(self, in_channels=2, num_classes=1):
        super(CustomCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to handle varying input sizes, outputting 4x4 spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output logits
        
        return x

if __name__ == "__main__":
    # Quick shape test
    model = CustomCNN(in_channels=2, num_classes=1)
    dummy_input = torch.randn(8, 2, 256, 256) # Batch of 8, 2 channels, 256x256
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}") # Should be [8, 1]
    
    # Calculate parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {param_count:,}")
