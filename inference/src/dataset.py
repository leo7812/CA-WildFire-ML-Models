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


def parse_tile_id_from_filename(filename):
    """
    Extracts the MGRS tile ID from an HLS filename.
    Example: subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif -> 'T10SEH'
    Returns None if parsing fails.
    """
    basename = os.path.basename(filename)
    for part in basename.split('.'):
        if len(part) == 6 and part[0] == 'T' and part[1:3].isdigit() and part[3:].isalpha():
            return part
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


def find_pairs(tile_dir):
    """Scans a directory for (input, mask) tif pairs."""
    all_inputs = sorted(glob.glob(os.path.join(tile_dir, '*_merged.tif')))
    pairs = []
    for inp in all_inputs:
        mask = inp.replace('_merged.tif', '.mask.tif')
        if os.path.exists(mask):
            pairs.append((inp, mask))
    return pairs


def group_split_pairs(pairs, val_fraction=1/3, seed=42):
    """
    Splits (input, mask) pairs into train/val groups by MGRS tile ID -- see
    src/dataset.py (training-side counterpart) for the full rationale; kept
    in sync manually since this is a separate deployment copy.

    sklearn is imported locally rather than at module level: this package is
    bundled into the inference container, and inference/requirements.txt does
    not otherwise depend on scikit-learn -- only scripts/evaluate.py (run in
    the full dev environment) calls this function.

    Returns (train_pairs, val_pairs, train_tiles, val_tiles).
    """
    from sklearn.model_selection import GroupShuffleSplit

    tile_ids = [parse_tile_id_from_filename(inp) for inp, _ in pairs]
    unparsed = [inp for (inp, _), tile in zip(pairs, tile_ids) if tile is None]
    if unparsed:
        raise ValueError(f'Could not parse MGRS tile ID for {len(unparsed)} files, e.g. {unparsed[:3]}')

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    indices = np.arange(len(pairs))
    train_idx, val_idx = next(splitter.split(indices, groups=tile_ids))

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs   = [pairs[i] for i in val_idx]
    train_tiles = {tile_ids[i] for i in train_idx}
    val_tiles   = {tile_ids[i] for i in val_idx}

    overlap = train_tiles & val_tiles
    assert not overlap, f'Tile leakage after group split: {sorted(overlap)}'

    return train_pairs, val_pairs, train_tiles, val_tiles


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

    `tile_dir_or_pairs` accepts either a directory to scan (legacy, single-dir
    usage -- still used by predict.py) or a pre-built list of (input_path,
    mask_path) pairs, which is what group_split_pairs() produces for callers
    that pool multiple directories and re-split by tile group.

    Input shape:   (6, 1, H, W) -- channels, timestep, height, width
    Mask shape:    (H, W)       -- 0=unburned, 1=burned, -1=nodata
    Weather shape: (5,)         -- normalized weather scalars (if weather_csv provided)
    """
    def __init__(self, tile_dir_or_pairs, weather_csv=None, augment=False):
        self.augment     = augment
        self.use_weather = weather_csv is not None
        self.wc_df       = None

        if self.use_weather:
            self.wc_df = load_weather_df(weather_csv)
            print(f'Weather data loaded: {len(self.wc_df)} days')

        if isinstance(tile_dir_or_pairs, (str, os.PathLike)):
            self.pairs = find_pairs(tile_dir_or_pairs)
            print(f'Dataset: {len(self.pairs)} valid pairs in {os.path.basename(str(tile_dir_or_pairs))}')
        else:
            self.pairs = list(tile_dir_or_pairs)
            print(f'Dataset: {len(self.pairs)} valid pairs (pre-split pool)')

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