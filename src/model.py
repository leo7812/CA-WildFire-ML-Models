import torch
import terratorch
import terratorch.tasks


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
        class_weights=torch.tensor([1.0, 2.0]),  # upweight burned class for small fires
        ignore_index=-1,
        lr=1e-4,
        optimizer='AdamW',
        freeze_backbone=freeze_backbone,
        freeze_decoder=False,
    )
    return task