import torch
import torch.nn as nn
import torch.nn.functional as F
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
        ignore_index=-1,
        lr=1e-4,
        optimizer='AdamW',
        freeze_backbone=freeze_backbone,
        freeze_decoder=False,
    )
    return task


class FocalLoss(nn.Module):
    """
    Multi-class focal loss. Down-weights easy, already-confident pixels (both
    classes) so gradient concentrates on hard/ambiguous ones, instead of
    relying on a single blunt class-weight multiplier for the ~8:1 imbalance.
    """
    def __init__(self, alpha=(1.0, 1.0), gamma=2.0, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        valid = target != self.ignore_index
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction='none')
        p_t = torch.exp(-ce)
        alpha_t = self.alpha[target.clamp(min=0)]
        focal = alpha_t * (1 - p_t) ** self.gamma * ce
        denom = valid.sum().clamp(min=1)
        return focal[valid].sum() / denom


class TverskyLoss(nn.Module):
    """
    Soft Tversky loss on the burned-class probability. alpha weights false
    positives, beta weights false negatives -- alpha > beta trades recall for
    precision by penalizing false positives harder.
    """
    def __init__(self, alpha=0.7, beta=0.3, ignore_index=-1, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        valid = (target != self.ignore_index).float()
        target_burned = (target == 1).float()
        probs = torch.softmax(logits, dim=1)[:, 1] * valid
        tp = (probs * target_burned).sum()
        fp = (probs * (1 - target_burned) * valid).sum()
        fn = ((1 - probs) * target_burned).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal + Tversky hybrid: focal handles the ~8:1 pixel imbalance without an
    aggressive class-weight multiplier, Tversky's alpha > beta then explicitly
    penalizes false positives harder for precision. This is the single source
    of truth for the segmentation criterion -- build_prithvi_task()'s task is
    only used for its backbone/decoder; its own loss config is not on the
    training path.
    """
    def __init__(self, focal_alpha=(1.0, 3.0), focal_gamma=2.0,
                 tversky_alpha=0.7, tversky_beta=0.3, tversky_weight=1.0,
                 ignore_index=-1):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, ignore_index=ignore_index)
        self.tversky_weight = tversky_weight

    def forward(self, logits, target):
        return self.focal(logits, target) + self.tversky_weight * self.tversky(logits, target)


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

        # Loss: Focal + Tversky hybrid (see FocalTverskyLoss docstring)
        self.criterion = FocalTverskyLoss(ignore_index=-1)

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