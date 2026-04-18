import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio


class HLSBurnScarsDataset(Dataset):
    """
    Dataset for HLS burn scars tiles.
    Input shape:  (6, 1, H, W) -- channels, timestep, height, width (Prithvi format)
    Mask shape:   (H, W)       -- 0=unburned, 1=burned, -1=nodata (ignored in loss)
    """
    def __init__(self, tile_dir, augment=False):
        self.tile_dir = tile_dir
        self.augment  = augment

        # Find all input tiles and pair with masks
        all_inputs = sorted(glob.glob(os.path.join(tile_dir, '*_merged.tif')))

        # Only keep pairs where both input and mask exist
        self.pairs = []
        for inp in all_inputs:
            mask = inp.replace('_merged.tif', '.mask.tif')
            if os.path.exists(mask):
                self.pairs.append((inp, mask))

        print(f"Dataset: {len(self.pairs)} valid input/mask pairs found in {os.path.basename(tile_dir)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, mask_path = self.pairs[idx]

        # Load 6-band input
        with rasterio.open(input_path) as src:
            img = src.read().astype(np.float32)   # (6, H, W)

        # Data is already in [0, 1] range — just clip to be safe
        img = np.clip(img, 0, 1)
        img = np.nan_to_num(img, nan=0.0)

        # Add time dimension for Prithvi: (6, H, W) -> (6, 1, H, W)
        img = img[:, np.newaxis, :, :]

        # Load mask — values are -1 (nodata), 0 (unburned), 1 (burned)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)   # (H, W)

        # Simple augmentation — random horizontal/vertical flip
        if self.augment:
            if np.random.rand() > 0.5:
                img  = img[:, :, :, ::-1].copy()   # horizontal flip
                mask = mask[:, ::-1].copy()
            if np.random.rand() > 0.5:
                img  = img[:, :, ::-1, :].copy()   # vertical flip
                mask = mask[::-1, :].copy()

        return {
            'image': torch.tensor(img),
            'mask':  torch.tensor(mask),
        }