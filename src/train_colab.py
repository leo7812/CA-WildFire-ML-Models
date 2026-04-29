import os
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dataset import HLSBurnScarsDataset
from model import WeatherConditionedWildfire

# ── Paths (Colab) ─────────────────────────────────────────────────────────────
BASE_DIR    = '/content/drive/MyDrive/wildfire_project'
TRAIN_DIR   = '/content/hls_burn_scars/training'
VAL_DIR     = '/content/hls_burn_scars/validation'
WEATHER_CSV = f'{BASE_DIR}/CA_Weather_Fire_Dataset_1984-2025-WeatherConditionsDaysOfFire.csv'
CKPT_DIR    = f'{BASE_DIR}/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE    = 16
NUM_WORKERS   = 2
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 20

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = HLSBurnScarsDataset(TRAIN_DIR, weather_csv=WEATHER_CSV, augment=True)
val_ds   = HLSBurnScarsDataset(VAL_DIR,   weather_csv=WEATHER_CSV, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

# ── Phase 1: Freeze backbone, train weather encoder + decoder ─────────────────
print('\n=== Phase 1: Training weather encoder + decoder (backbone frozen) ===')
model = WeatherConditionedWildfire(freeze_backbone=True, lr=1e-4)

trainer_p1 = Trainer(
    max_epochs=PHASE1_EPOCHS,
    accelerator='gpu',
    precision='bf16-mixed',
    logger=False,
    callbacks=[
        EarlyStopping(monitor='val/loss', patience=5, verbose=True),
        ModelCheckpoint(
            dirpath=CKPT_DIR,
            filename='weather-phase1-{epoch:02d}-{val_loss:.4f}',
            save_top_k=2,
            monitor='val/loss'
        ),
    ],
    log_every_n_steps=10,
)
trainer_p1.fit(model, train_loader, val_loader)
print('Phase 1 complete!')

# ── Phase 2: Unfreeze everything ──────────────────────────────────────────────
print('\n=== Phase 2: End-to-end fine-tuning (all layers unfrozen) ===')
for param in model.parameters():
    param.requires_grad = True

# Re-configure optimizer with lower backbone LR
for pg in model.optimizers().param_groups:
    pg['lr'] = 1e-5

trainer_p2 = Trainer(
    max_epochs=PHASE2_EPOCHS,
    accelerator='gpu',
    precision='bf16-mixed',
    logger=False,
    callbacks=[
        EarlyStopping(monitor='val/loss', patience=5, verbose=True),
        ModelCheckpoint(
            dirpath=CKPT_DIR,
            filename='weather-phase2-best-{val_loss:.4f}',
            save_top_k=1,
            monitor='val/loss'
        ),
    ],
    log_every_n_steps=10,
)
trainer_p2.fit(model, train_loader, val_loader)
print('Phase 2 complete!')

# ── Save final weights ────────────────────────────────────────────────────────
weights_path = f'{CKPT_DIR}/wildfire_weather_weights.pt'
torch.save(model.state_dict(), weights_path)
print(f'\nFinal weights saved to {weights_path}')