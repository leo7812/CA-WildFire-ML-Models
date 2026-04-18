import os
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from dataset import HLSBurnScarsDataset
from model import build_prithvi_task

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = '/Users/leonardofloresgonzalez/wildfire_project'
TRAIN_DIR = f'{BASE_DIR}/data/hls_burn_scars/training'
VAL_DIR   = f'{BASE_DIR}/data/hls_burn_scars/validation'
CKPT_DIR  = f'{BASE_DIR}/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
# These are intentionally small for local testing on M2
# On Colab A100 use: BATCH_SIZE=16, PHASE1_EPOCHS=20, PHASE2_EPOCHS=10
BATCH_SIZE     = 4
NUM_WORKERS    = 0
PHASE1_EPOCHS  = 5
PHASE2_EPOCHS  = 3

# ── Datasets & Dataloaders ────────────────────────────────────────────────────
train_ds = HLSBurnScarsDataset(TRAIN_DIR, augment=True)
val_ds   = HLSBurnScarsDataset(VAL_DIR,   augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ── Device ────────────────────────────────────────────────────────────────────
accelerator = (
    'gpu' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f'Using accelerator: {accelerator}')

# ── Phase 1: Train decoder only ───────────────────────────────────────────────
print('\n=== Phase 1: Training decoder (backbone frozen) ===')
task = build_prithvi_task(freeze_backbone=True)

trainer_p1 = Trainer(
    max_epochs=PHASE1_EPOCHS,
    accelerator=accelerator,
    precision='32',
    logger=False,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3, verbose=True),
        ModelCheckpoint(
            dirpath=CKPT_DIR,
            filename='phase1-{epoch:02d}-{val_loss:.4f}',
            save_top_k=2,
            monitor='val_loss'
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ],
    log_every_n_steps=5,
)
trainer_p1.fit(task, train_loader, val_loader)

# ── Phase 2: End-to-end fine-tuning ───────────────────────────────────────────
print('\n=== Phase 2: End-to-end fine-tuning (backbone unfrozen) ===')
for param in task.model.parameters():
    param.requires_grad = True

for pg in task.optimizers().param_groups:
    pg['lr'] = 1e-5

trainer_p2 = Trainer(
    max_epochs=PHASE2_EPOCHS,
    accelerator=accelerator,
    precision='32',
    logger=False,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=3, verbose=True),
        ModelCheckpoint(
            dirpath=CKPT_DIR,
            filename='phase2-best-{val_loss:.4f}',
            save_top_k=1,
            monitor='val_loss'
        ),
    ],
    log_every_n_steps=5,
)
trainer_p2.fit(task, train_loader, val_loader)

# ── Save final weights ────────────────────────────────────────────────────────
weights_path = f'{CKPT_DIR}/wildfire_weights.pt'
torch.save(task.model.state_dict(), weights_path)
print(f'\nFinal weights saved to {weights_path}')