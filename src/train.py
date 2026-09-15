import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Assuming config parsing is simple or manual for now
from src.datasets.spatial_split import spatial_block_split
from src.datasets.ice_dataset import IcePatchDataset
from src.models.custom_cnn import CustomCNN

def train():
    print("Starting training pipeline...")
    
    # 1. Configuration (Simulated from config.yaml)
    batch_size = 16
    epochs = 5
    learning_rate = 0.001
    patch_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Dataset splitting
    # Create dummy coordinates representing the large image area
    print("Generating synthetic coordinates for spatial split...")
    coords = [(x, y) for x in range(0, 1000, 256) for y in range(0, 1000, 256)]
    
    train_coords, val_coords, test_coords = spatial_block_split(
        coordinates=coords, 
        grid_size=(2, 2), 
        train_ratio=0.7, 
        val_ratio=0.15
    )
    print(f"Split sizes -> Train: {len(train_coords)}, Val: {len(val_coords)}, Test: {len(test_coords)}")
    
    # 3. DataLoaders
    train_dataset = IcePatchDataset(train_coords, patch_size=patch_size, augment=True)
    val_dataset = IcePatchDataset(val_coords, patch_size=patch_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Model setup
    model = CustomCNN(in_channels=2, num_classes=1).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    # 5. Loss and Optimizer
    # Class imbalance: Use weighted BCE Loss. Assuming ice is minority class.
    # pos_weight > 1 increases weight on positive (ice) class.
    pos_weight = torch.tensor([3.0]).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 6. Training loop
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    # 7. Save checkpoint
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/custom_cnn_checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # 8. Save loss curves
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Weighted BCE)')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/loss_curves.png")
    print("Loss curves saved to reports/loss_curves.png")

if __name__ == '__main__':
    train()
