import torch
from torch.utils.data import Dataset
import numpy as np

class IcePatchDataset(Dataset):
    """
    PyTorch Dataset for Lunar Ice Patches.
    """
    def __init__(self, coordinates, data_dir=None, patch_size=256, augment=False):
        """
        Args:
            coordinates (list): List of (x, y) coordinates defining the patches.
            data_dir (str): Directory containing real DFSAR data.
            patch_size (int): Size of the patches.
            augment (bool): Whether to apply on-the-fly augmentation.
        """
        self.coordinates = coordinates
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.augment = augment
        
    def __len__(self):
        return len(self.coordinates)
        
    def __getitem__(self, idx):
        # coord = self.coordinates[idx]
        
        # TODO: Load real patch from data_dir using the coordinate.
        # Since real DFSAR data isn't downloaded yet, we use synthetic tensors.
        # Note: This is a dummy generation for pipeline verification.
        
        # Simulated 2-channel SAR data (e.g., CPR and DOP mapped to features, or raw channels)
        image = np.random.randn(2, self.patch_size, self.patch_size).astype(np.float32)
        
        # Dummy label: 1 if potential ice, 0 if non-ice
        label = np.random.randint(0, 2)
        
        if self.augment:
            # Simple on-the-fly augmentation: random flips and rotations
            # Valid for SAR intensity data (assuming rotation doesn't break polarimetric physical meaning in this simplified context)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1) # Horizontal flip
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2) # Vertical flip
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(1, 2))
            
        # Ensure contiguous array after numpy ops
        image = np.ascontiguousarray(image)
        
        return torch.from_numpy(image), torch.tensor([label], dtype=torch.float32)
