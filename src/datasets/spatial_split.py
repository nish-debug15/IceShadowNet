import numpy as np

def assert_no_extent_leakage(
    train_coords: list,
    val_coords: list,
    test_coords: list,
    patch_size: int = 64,
) -> None:
    """
    Assert that no two patches across different splits have overlapping pixel extents.
    Each patch with origin (r, c) occupies [r, r + patch_size) x [c, c + patch_size).
    """
    splits = [("train", train_coords), ("val", val_coords), ("test", test_coords)]
    for i in range(len(splits)):
        name1, set1 = splits[i]
        for j in range(i + 1, len(splits)):
            name2, set2 = splits[j]
            for r1, c1 in set1:
                for r2, c2 in set2:
                    if abs(r1 - r2) < patch_size and abs(c1 - c2) < patch_size:
                        raise AssertionError(
                            f"PIXEL EXTENT LEAKAGE DETECTED between {name1} at ({r1}, {c1}) "
                            f"and {name2} at ({r2}, {c2}) with patch_size={patch_size}!"
                        )


def spatial_block_split(
    coordinates: list,
    grid_size: tuple = (2, 2),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    patch_size: int = 64,
    buffer_pixels: int = 0,
    seed: int = 42,
):
    """
    Splits coordinates into train, validation, and test sets using a spatial block split.
    This prevents adjacent patches from leaking between sets.
    
    Args:
        coordinates (list): List of (x, y) tuples representing patch coordinates.
        grid_size (tuple): The (rows, cols) of the spatial grid to divide the area into.
        train_ratio (float): The ratio of blocks to assign to the training set.
        val_ratio (float): The ratio of blocks to assign to the validation set.
        patch_size (int): Size of the square patch in pixels.
        buffer_pixels (int): Buffer width in pixels around block boundaries. Patches whose
            pixel extent comes within buffer_pixels of an adjacent block belonging to a
            different split are dropped to guarantee zero pixel-level leakage.
        seed (int): Random seed for block assignment shuffling.
        
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
    coord_to_block = {}
    for x, y in coordinates:
        col = min(int((x - min_x) / block_width), grid_size[1] - 1)
        row = min(int((y - min_y) / block_height), grid_size[0] - 1)
        block_idx = (row, col)
        coord_to_block[(x, y)] = block_idx
        if block_idx not in blocks:
            blocks[block_idx] = []
        blocks[block_idx].append((x, y))
        
    # Shuffle block indices to assign them to splits
    block_indices = list(blocks.keys())
    np.random.seed(seed) # Fixed seed for reproducibility
    np.random.shuffle(block_indices)
    
    n_blocks = len(block_indices)
    n_train = int(n_blocks * train_ratio)
    n_val = int(n_blocks * val_ratio)
    
    train_blocks = set(block_indices[:n_train])
    val_blocks = set(block_indices[n_train:n_train+n_val])
    # remaining are test blocks
    
    block_to_split = {}
    for b in train_blocks:
        block_to_split[b] = "train"
    for b in val_blocks:
        block_to_split[b] = "val"
    for b in block_indices[n_train+n_val:]:
        block_to_split[b] = "test"
    
    train_coords = []
    val_coords = []
    test_coords = []
    
    for (x, y), (row, col) in coord_to_block.items():
        my_split = block_to_split.get((row, col), "test")
        
        # Apply boundary buffer guard if requested
        if buffer_pixels > 0:
            drop = False
            # Check left neighbor along x (col - 1)
            if col > 0 and block_to_split.get((row, col - 1), my_split) != my_split:
                boundary_x = min_x + col * block_width
                if (x - boundary_x) < buffer_pixels:
                    drop = True
            # Check right neighbor along x (col + 1)
            if col < grid_size[1] - 1 and block_to_split.get((row, col + 1), my_split) != my_split:
                boundary_x = min_x + (col + 1) * block_width
                if (boundary_x - (x + patch_size)) < buffer_pixels:
                    drop = True
            # Check top neighbor along y (row - 1)
            if row > 0 and block_to_split.get((row - 1, col), my_split) != my_split:
                boundary_y = min_y + row * block_height
                if (y - boundary_y) < buffer_pixels:
                    drop = True
            # Check bottom neighbor along y (row + 1)
            if row < grid_size[0] - 1 and block_to_split.get((row + 1, col), my_split) != my_split:
                boundary_y = min_y + (row + 1) * block_height
                if (boundary_y - (y + patch_size)) < buffer_pixels:
                    drop = True
            
            if drop:
                continue
                
        if my_split == "train":
            train_coords.append((x, y))
        elif my_split == "val":
            val_coords.append((x, y))
        else:
            test_coords.append((x, y))
            
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
