import os
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from model import build_prithvi_task

BASE_DIR     = '/Users/leonardofloresgonzalez/wildfire_project'
WEIGHTS_PATH = f'{BASE_DIR}/checkpoints/wildfire_weights.pt'


def load_model(weights_path=WEIGHTS_PATH):
    task = build_prithvi_task(freeze_backbone=False)
    task.model.load_state_dict(
        torch.load(weights_path, map_location='cpu')
    )
    task.eval()
    return task


def predict_tile(task, tif_path, device='mps'):
    task.to(device)

    with rasterio.open(tif_path) as src:
        img = src.read().astype(np.float32)   # (6, H, W)

    img = np.clip(img, 0, 1)
    img = np.nan_to_num(img, nan=0.0)
    img = img[:, np.newaxis, :, :]            # (6, 1, H, W)

    tensor = torch.tensor(img).unsqueeze(0).to(device)  # (1, 6, 1, H, W)

    with torch.no_grad():
        output = task.model(tensor)
        probs  = output.output.softmax(dim=1)[0, 1].cpu().numpy()  # burn probability
        mask   = output.output.argmax(dim=1)[0].cpu().numpy()      # binary mask

    # Estimate fire size (each pixel = 30m x 30m = 900 sq meters)
    burned_pixels  = (mask == 1).sum()
    predicted_acres = burned_pixels * 900 / 4047

    return mask, probs, predicted_acres


def visualize(tif_path, mask, probs, predicted_acres):
    with rasterio.open(tif_path) as src:
        img = src.read().astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rgb = img[[2,1,0]].transpose(1,2,0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    axes[0].imshow(rgb)
    axes[0].set_title('Input RGB')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Predicted Burn Mask')
    axes[1].axis('off')

    im = axes[2].imshow(probs, cmap='inferno', vmin=0, vmax=1)
    axes[2].set_title(f'Burn Probability\n~{predicted_acres:.0f} acres')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.suptitle(f'Predicted burned area: {predicted_acres:.0f} acres', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Test on one validation tile
    val_dir  = f'{BASE_DIR}/data/hls_burn_scars/validation'
    test_tif = os.path.join(val_dir, os.listdir(val_dir)[0].replace('.mask.tif', '_merged.tif'))

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print('Loading model...')
    task = load_model()

    print(f'Running inference on {os.path.basename(test_tif)}...')
    mask, probs, acres = predict_tile(task, test_tif, device=device)

    print(f'Predicted burned area: {acres:.0f} acres')
    visualize(test_tif, mask, probs, acres)