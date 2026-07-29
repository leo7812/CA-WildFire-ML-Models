import os
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dataset import HLSBurnScarsDataset, find_pairs, group_split_pairs
from model import WeatherConditionedWildfire

# ── Paths (local M2) ──────────────────────────────────────────────────────────
BASE_DIR    = '/Users/leonardofloresgonzalez/wildfire_project'
TRAIN_DIR   = f'{BASE_DIR}/data/hls_burn_scars/training'
VAL_DIR     = f'{BASE_DIR}/data/hls_burn_scars/validation'
WEATHER_CSV = f'{BASE_DIR}/data/CA_Weather_Fire_Dataset_1984-2025-WeatherConditionsDaysOfFire.csv'
CKPT_DIR    = f'{BASE_DIR}/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Hyperparameters (small for local testing) ─────────────────────────────────
BATCH_SIZE    = 2
NUM_WORKERS   = 0
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 2
VAL_FRACTION  = 1/3    # matches upstream's original 2/3 train : 1/3 val scale
SPLIT_SEED    = 42

# ── Pool both dirs, re-split by MGRS tile group (zero tile overlap) ───────────
all_pairs = find_pairs(TRAIN_DIR) + find_pairs(VAL_DIR)
train_pairs, val_pairs, train_tiles, val_tiles = group_split_pairs(
    all_pairs, val_fraction=VAL_FRACTION, seed=SPLIT_SEED
)

print(f'Pooled {len(all_pairs)} scenes across {len(train_tiles) + len(val_tiles)} tiles')
print(f'Train: {len(train_pairs)} scenes / {len(train_tiles)} tiles')
print(f'Val:   {len(val_pairs)} scenes / {len(val_tiles)} tiles')
print(f'Tile overlap between splits: {len(train_tiles & val_tiles)} (must be 0)')

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = HLSBurnScarsDataset(train_pairs, weather_csv=WEATHER_CSV, augment=True)
val_ds   = HLSBurnScarsDataset(val_pairs,   weather_csv=WEATHER_CSV, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

# ── Device ────────────────────────────────────────────────────────────────────
accelerator = (
    'gpu' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f'Using: {accelerator}')

# ── Phase 1 ───────────────────────────────────────────────────────────────────
print('\n=== Phase 1: Training weather encoder + decoder (backbone frozen) ===')
model = WeatherConditionedWildfire(freeze_backbone=True, lr=1e-4)

trainer_p1 = Trainer(
    max_epochs=PHASE1_EPOCHS,
    accelerator=accelerator,
    precision='32',
    logger=False,
    enable_checkpointing=False,
    log_every_n_steps=5,
)
trainer_p1.fit(model, train_loader, val_loader)
print('Phase 1 complete!')

# ── Phase 2 ───────────────────────────────────────────────────────────────────
print('\n=== Phase 2: End-to-end fine-tuning ===')
for param in model.parameters():
    param.requires_grad = True

trainer_p2 = Trainer(
    max_epochs=PHASE2_EPOCHS,
    accelerator=accelerator,
    precision='32',
    logger=False,
    enable_checkpointing=False,
    log_every_n_steps=5,
)
trainer_p2.fit(model, train_loader, val_loader)
print('Phase 2 complete!')

# ── Save ──────────────────────────────────────────────────────────────────────
weights_path = f'{CKPT_DIR}/wildfire_weather_weights.pt'
torch.save(model.state_dict(), weights_path)
print(f'Weights saved to {weights_path}')