import os
import sys
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import glob

# Add src to path if running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import WeatherConditionedWildfire
from dataset import HLSBurnScarsDataset, WEATHER_COLS, WEATHER_MEAN, WEATHER_STD, parse_date_from_filename

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = '/Users/leonardofloresgonzalez/wildfire_project'
WEIGHTS_PATH = f'{BASE_DIR}/checkpoints/wildfire_weather_weights.pt'
WEATHER_CSV  = f'{BASE_DIR}/data/CA_Weather_Fire_Dataset_1984-2025-WeatherConditionsDaysOfFire.csv'
VAL_DIR      = f'{BASE_DIR}/data/hls_burn_scars/validation'


def load_model(weights_path=WEIGHTS_PATH, device=None):
    """Load the weather-conditioned Prithvi model from saved weights."""
    if device is None:
        device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

    model = WeatherConditionedWildfire(freeze_backbone=False)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    model.to(device)
    print(f'Model loaded on {device}')
    return model, device


def predict(model, tif_path, weather_scalars=None, device='mps'):
    """
    Run inference on a single GeoTIFF tile.

    Args:
        model:           loaded WeatherConditionedWildfire
        tif_path:        path to a 6-band HLS merged GeoTIFF
        weather_scalars: optional np.array of shape (5,) with raw weather values
                         [TEMP_RANGE, LAGGED_AVG_WIND_SPEED, AVG_WIND_SPEED,
                          MAX_TEMP, WIND_TEMP_RATIO]
                         If None, zeros are used (weather-agnostic prediction)
        device:          'mps', 'cuda', or 'cpu'

    Returns:
        mask:   (H, W) binary numpy array — 1=burned, 0=unburned
        probs:  (H, W) float numpy array — burn probability 0-1
        acres:  float — predicted burned area in acres
    """
    with rasterio.open(tif_path) as src:
        img = src.read().astype(np.float32)   # (6, H, W)

    img = np.clip(img, 0, 1)
    img = np.nan_to_num(img, nan=0.0)
    img = img[:, np.newaxis, :, :]            # (6, 1, H, W)

    image_tensor = torch.tensor(img).unsqueeze(0).to(device)  # (1, 6, 1, H, W)

    if weather_scalars is not None:
        # Normalize using dataset constants
        weather_norm = (weather_scalars - WEATHER_MEAN) / (WEATHER_STD + 1e-6)
        weather_norm = np.nan_to_num(weather_norm, nan=0.0)
    else:
        weather_norm = np.zeros(5, dtype=np.float32)

    weather_tensor = torch.tensor(weather_norm).unsqueeze(0).to(device)  # (1, 5)

    with torch.no_grad():
        logits = model(image_tensor, weather_tensor)
        probs  = logits.softmax(dim=1)[0, 1].cpu().numpy()   # burn probability
        mask   = logits.argmax(dim=1)[0].cpu().numpy()       # binary mask

    burned_pixels = (mask == 1).sum()
    acres = burned_pixels * 900 / 4047

    return mask, probs, acres


def visualize(tif_path, mask, probs, acres, actual_acres=None):
    """Visualize input RGB, predicted mask and burn probability."""
    with rasterio.open(tif_path) as src:
        img = src.read().astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # RGB
    rgb = img[[2,1,0]].transpose(1,2,0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    axes[0].imshow(rgb)
    axes[0].set_title('Input RGB (pre-fire)')
    axes[0].axis('off')

    # NIR
    axes[1].imshow(img[3], cmap='RdYlGn')
    axes[1].set_title('NIR (vegetation health)')
    axes[1].axis('off')

    # Predicted mask
    axes[2].imshow(mask, cmap='hot', vmin=0, vmax=1)
    title = f'Predicted Mask\n{acres:.0f} acres'
    if actual_acres is not None:
        error = abs(acres - actual_acres) / (actual_acres + 1e-6) * 100
        title += f'\n(actual: {actual_acres:.0f}, error: {error:.1f}%)'
    axes[2].set_title(title)
    axes[2].axis('off')

    # Probability heatmap
    im = axes[3].imshow(probs, cmap='inferno', vmin=0, vmax=1)
    axes[3].set_title('Burn Probability')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.suptitle(os.path.basename(tif_path)[:60], fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import pandas as pd
    from dataset import parse_date_from_filename

    WEATHER_COLS = ['TEMP_RANGE', 'LAGGED_AVG_WIND_SPEED', 'AVG_WIND_SPEED',
                    'MAX_TEMP', 'WIND_TEMP_RATIO']

    # Load model
    model, device = load_model()

    # Load raw weather dataframe
    wc_df = pd.read_csv(WEATHER_CSV)
    wc_df['DATE'] = pd.to_datetime(wc_df['DATE'])
    wc_df = wc_df.set_index('DATE')

    # Load validation dataset for ground truth
    val_ds = HLSBurnScarsDataset(VAL_DIR)

    for idx in range(3):
        input_path, mask_path = val_ds.pairs[idx]

        with rasterio.open(mask_path) as src:
            gt = src.read(1).astype(np.float32)
        actual_acres = (gt == 1).sum() * 900 / 4047

        # Get raw weather values for this tile's date
        date = parse_date_from_filename(input_path)
        try:
            i = wc_df.index.get_indexer([date], method='nearest')[0]
            weather_raw = wc_df.iloc[i][WEATHER_COLS].values.astype(np.float32)
        except Exception:
            weather_raw = None

        print(f'\n{os.path.basename(input_path)[:55]}')
        if weather_raw is not None:
            print(f'Weather: temp={weather_raw[3]:.1f}F  wind={weather_raw[2]:.1f}mph')

        mask, probs, acres = predict(model, input_path, weather_raw, device=device)
        visualize(input_path, mask, probs, acres, actual_acres=actual_acres)

        print(f'Predicted: {acres:.0f} acres | Actual: {actual_acres:.0f} acres | '
              f'Error: {abs(acres - actual_acres) / actual_acres * 100:.1f}%')