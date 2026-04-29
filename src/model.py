import torch
import torch.nn as nn
import terratorch
import terratorch.tasks
from lightning import LightningModule


def build_prithvi_task(freeze_backbone=True):
    task = terratorch.tasks.SemanticSegmentationTask(
        model_factory='EncoderDecoderFactory',
        model_args={
            'backbone': 'prithvi_eo_v2_300',
            'backbone_pretrained': True,
            'backbone_num_frames': 1,
            'backbone_bands': [
                'BLUE', 'GREEN', 'RED',
                'NIR_NARROW', 'SWIR_1', 'SWIR_2'
            ],
            'backbone_coords_encoding': [],
            'necks': [
                {'name': 'SelectIndices', 'indices': [5, 11, 17, 23]},
                {'name': 'ReshapeTokensToImage', 'effective_time_dim': 1},
                {'name': 'LearnedInterpolateToPyramidal'},
            ],
            'decoder': 'UNetDecoder',
            'decoder_channels': [256, 128, 64, 32],
            'head_dropout': 0.1,
            'num_classes': 2,
        },
        loss='ce',
        class_weights=torch.tensor([1.0, 2.0]),
        ignore_index=-1,
        lr=1e-4,
        optimizer='AdamW',
        freeze_backbone=freeze_backbone,
        freeze_decoder=False,
    )
    return task


class WeatherConditionedWildfire(LightningModule):
    """
    Prithvi segmentation + weather conditioning.

    Weather scalars are encoded into a (B, 2) bias term that gets
    added directly to the final (B, 2, H, W) logits. This is the
    simplest and most stable way to condition on tabular inputs
    without needing to hook into intermediate feature maps.

    Intuition: weather shifts the model's overall confidence about
    burned vs unburned — high wind/temp increases burn probability
    uniformly, then Prithvi's spatial features determine the shape.
    """
    def __init__(self, weather_dim=5, freeze_backbone=True, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        # Base Prithvi task
        self.prithvi_task = build_prithvi_task(freeze_backbone=freeze_backbone)

        # Weather encoder: 5 scalars -> 2 class logit biases
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2),   # 2 outputs = bias for [unburned, burned] logits
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0]),
            ignore_index=-1
        )

    def forward(self, image, weather):
        # Prithvi spatial prediction: (B, 2, H, W)
        output = self.prithvi_task.model(image)
        logits = output.output   # (B, 2, H, W)

        # Weather bias: (B, 5) -> (B, 2) -> (B, 2, 1, 1)
        weather_bias = self.weather_encoder(weather)
        weather_bias = weather_bias.unsqueeze(-1).unsqueeze(-1)  # broadcast over H, W

        # Add weather bias to spatial logits
        out = logits + weather_bias   # (B, 2, H, W)
        return out

    def training_step(self, batch, batch_idx):
        image   = batch['image']
        mask    = batch['mask']
        weather = batch.get('weather', torch.zeros(image.shape[0], 5, device=self.device))

        logits = self(image, weather)
        loss   = self.criterion(logits, mask)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image   = batch['image']
        mask    = batch['mask']
        weather = batch.get('weather', torch.zeros(image.shape[0], 5, device=self.device))

        logits = self(image, weather)
        loss   = self.criterion(logits, mask)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        backbone_params = list(self.prithvi_task.model.parameters())
        weather_params  = list(self.weather_encoder.parameters())
        return torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.lr * 0.1},
            {'params': weather_params,  'lr': self.lr},
        ])