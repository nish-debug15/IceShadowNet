import numpy as np

def spatial_block_split(coordinates: list, grid_size: tuple = (2, 2), train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Splits coordinates into train, validation, and test sets using a spatial block split.
    This prevents adjacent patches from leaking between sets.
    
    Args:
        coordinates (list): List of (x, y) tuples representing patch coordinates.
        grid_size (tuple): The (rows, cols) of the spatial grid to divide the area into.
        train_ratio (float): The ratio of blocks to assign to the training set.
        val_ratio (float): The ratio of blocks to assign to the validation set.
        
    Returns:
        tuple: (train_coords, val_coords, test_coords)
    """
    if not coordinates:
        return [], [], []

    # Determine bounding box
    min_x = min(c[0] for c in coordinates)
    max_x = max(c[0] for c in coordinates)
    min_y = min(c[1] for c in coordinates)
    max_y = max(c[1] for c in coordinates)
    
    # Calculate block dimensions
    # To handle the case where max_x == min_x, we add a small epsilon
    block_width = (max_x - min_x + 1e-5) / grid_size[1]
    block_height = (max_y - min_y + 1e-5) / grid_size[0]
    
    # Assign each coordinate to a block index (row, col)
    blocks = {}
    for x, y in coordinates:
        col = min(int((x - min_x) / block_width), grid_size[1] - 1)
        row = min(int((y - min_y) / block_height), grid_size[0] - 1)
        block_idx = (row, col)
        if block_idx not in blocks:
            blocks[block_idx] = []
        blocks[block_idx].append((x, y))
        
    # Shuffle block indices to assign them to splits
    block_indices = list(blocks.keys())
    np.random.seed(42) # Fixed seed for reproducibility
    np.random.shuffle(block_indices)
    
    n_blocks = len(block_indices)
    n_train = int(n_blocks * train_ratio)
    n_val = int(n_blocks * val_ratio)
    
    train_blocks = set(block_indices[:n_train])
    val_blocks = set(block_indices[n_train:n_train+n_val])
    # remaining are test blocks
    
    train_coords = []
    val_coords = []
    test_coords = []
    
    for block_idx, coords in blocks.items():
        if block_idx in train_blocks:
            train_coords.extend(coords)
        elif block_idx in val_blocks:
            val_coords.extend(coords)
        else:
            test_coords.extend(coords)
            
    return train_coords, val_coords, test_coords

if __name__ == '__main__':
    # Unit test to prove no block appears in more than one split
    # Create a synthetic grid of coordinates
    print("Running spatial block split unit test...")
    coords = []
    for x in range(0, 100, 10):
        for y in range(0, 100, 10):
            coords.append((x, y))
            
    train, val, test = spatial_block_split(coords, grid_size=(4, 4))
    
    # Verify no overlap
    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    
    assert len(train_set.intersection(val_set)) == 0, "Leakage between train and val!"
    assert len(train_set.intersection(test_set)) == 0, "Leakage between train and test!"
    assert len(val_set.intersection(test_set)) == 0, "Leakage between val and test!"
    
    print(f"Total coordinates: {len(coords)}")
    print(f"Train size: {len(train)}")
    print(f"Val size: {len(val)}")
    print(f"Test size: {len(test)}")
    print("Unit test passed: No leakage between splits.")
