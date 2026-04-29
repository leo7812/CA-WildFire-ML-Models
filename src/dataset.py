import os
import glob
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio


def parse_date_from_filename(filename):
    """
    Extracts date from HLS filename.
    Example: subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif
    '2018190' = year 2018, day-of-year 190
    Returns a pandas Timestamp or None if parsing fails.
    """
    basename = os.path.basename(filename)
    parts    = basename.split('.')
    for part in parts:
        if len(part) == 7 and part.isdigit():
            year = int(part[:4])
            doy  = int(part[4:])
            try:
                date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
                return pd.Timestamp(date)
            except Exception:
                return None
    return None


def load_weather_df(csv_path):
    """
    Loads and preprocesses the weather dataframe.
    Returns a dataframe indexed by date for fast lookup.
    """
    wc_df = pd.read_csv(csv_path)
    wc_df['DATE'] = pd.to_datetime(wc_df['DATE'])
    wc_df = wc_df.set_index('DATE')
    return wc_df


WEATHER_COLS = [
    'TEMP_RANGE',
    'LAGGED_AVG_WIND_SPEED',
    'AVG_WIND_SPEED',
    'MAX_TEMP',
    'WIND_TEMP_RATIO'
]

# Approximate mean and std for normalization
# These will be computed from the dataset on first run
WEATHER_MEAN = np.array([15.0, 5.0, 5.0, 75.0, 0.07], dtype=np.float32)
WEATHER_STD  = np.array([8.0,  3.0, 3.0, 15.0, 0.05], dtype=np.float32)


class HLSBurnScarsDataset(Dataset):
    """
    Dataset for HLS burn scars tiles with optional weather conditioning.

    Input shape:   (6, 1, H, W) -- channels, timestep, height, width
    Mask shape:    (H, W)       -- 0=unburned, 1=burned, -1=nodata
    Weather shape: (5,)         -- normalized weather scalars (if weather_csv provided)
    """
    def __init__(self, tile_dir, weather_csv=None, augment=False):
        self.tile_dir    = tile_dir
        self.augment     = augment
        self.use_weather = weather_csv is not None
        self.wc_df       = None

        if self.use_weather:
            self.wc_df = load_weather_df(weather_csv)
            print(f'Weather data loaded: {len(self.wc_df)} days')

        # Find all input/mask pairs
        all_inputs = sorted(glob.glob(os.path.join(tile_dir, '*_merged.tif')))
        self.pairs = []
        for inp in all_inputs:
            mask = inp.replace('_merged.tif', '.mask.tif')
            if os.path.exists(mask):
                self.pairs.append((inp, mask))

        print(f'Dataset: {len(self.pairs)} valid pairs in {os.path.basename(tile_dir)}')

    def __len__(self):
        return len(self.pairs)

    def _get_weather(self, input_path):
        """Look up weather scalars for this tile's date."""
        date = parse_date_from_filename(input_path)
        if date is None or self.wc_df is None:
            return np.zeros(len(WEATHER_COLS), dtype=np.float32)

        # Find closest date within 3 days
        try:
            idx = self.wc_df.index.get_indexer([date], method='nearest')[0]
            row = self.wc_df.iloc[idx]
            weather = row[WEATHER_COLS].values.astype(np.float32)
            # Normalize
            weather = (weather - WEATHER_MEAN) / (WEATHER_STD + 1e-6)
            # Replace any NaNs with 0
            weather = np.nan_to_num(weather, nan=0.0)
        except Exception:
            weather = np.zeros(len(WEATHER_COLS), dtype=np.float32)

        return weather

    def __getitem__(self, idx):
        input_path, mask_path = self.pairs[idx]

        # Load 6-band image
        with rasterio.open(input_path) as src:
            img = src.read().astype(np.float32)   # (6, H, W)
        img = np.clip(img, 0, 1)
        img = np.nan_to_num(img, nan=0.0)
        img = img[:, np.newaxis, :, :]             # (6, 1, H, W)

        # Load mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)    # (H, W)
        mask = np.nan_to_num(mask, nan=0).clip(-1, 1)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img  = img[:, :, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() > 0.5:
                img  = img[:, :, ::-1, :].copy()
                mask = mask[::-1, :].copy()

        item = {
            'image': torch.tensor(img),
            'mask':  torch.tensor(mask),
        }

        if self.use_weather:
            item['weather'] = torch.tensor(self._get_weather(input_path))

        return item