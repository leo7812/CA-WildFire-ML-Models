import os
import io
import sys
import base64
import numpy as np
import torch
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from model import WeatherConditionedWildfire
from dataset import WEATHER_MEAN, WEATHER_STD

app = FastAPI()

WEIGHTS_PATH = os.path.join(BASE_DIR, '..', 'checkpoints', 'wildfire_weather_weights.pt')
TILES_DIR    = os.path.join(BASE_DIR, 'tiles')
STATIC_DIR   = os.path.join(BASE_DIR, 'static')

model = None

@app.on_event('startup')
async def load_model():
    global model
    print(f'Loading weights from {WEIGHTS_PATH}...')
    model = WeatherConditionedWildfire(freeze_backbone=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    print('Model ready.')

class PredictRequest(BaseModel):
    tile_index:      int   = 0
    max_temp:        float = 85.0
    wind_speed:      float = 10.0
    temp_range:      float = 20.0
    lagged_wind:     float = 6.0
    wind_temp_ratio: float = 0.12

@app.post('/predict')
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    import glob
    tiles = sorted(glob.glob(os.path.join(TILES_DIR, '*_merged.tif')))
    if req.tile_index >= len(tiles):
        raise HTTPException(status_code=400, detail=f'Tile index out of range (max {len(tiles)-1})')

    tile_path = tiles[req.tile_index]
    mask_path = tile_path.replace('_merged.tif', '.mask.tif')

    with rasterio.open(tile_path) as src:
        img = src.read().astype(np.float32)
    img = np.clip(img, 0, 1)
    img = np.nan_to_num(img, nan=0.0)
    img = img[:, np.newaxis, :, :]

    with rasterio.open(mask_path) as src:
        gt = src.read(1).astype(np.float32)
    actual_acres = float((gt == 1).sum() * 900 / 4047)

    weather_raw = np.array([
        req.temp_range,
        req.lagged_wind,
        req.wind_speed,
        req.max_temp,
        req.wind_temp_ratio,
    ], dtype=np.float32)
    weather_norm = (weather_raw - WEATHER_MEAN) / (WEATHER_STD + 1e-6)
    weather_norm = np.nan_to_num(weather_norm, nan=0.0)

    image_tensor   = torch.tensor(img).unsqueeze(0)
    weather_tensor = torch.tensor(weather_norm).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor, weather_tensor)
        probs  = logits.softmax(dim=1)[0, 1].numpy()
        mask   = logits.argmax(dim=1)[0].numpy()

    predicted_acres = float((mask == 1).sum() * 900 / 4047)

    def arr_to_b64(arr):
        arr_norm = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype(np.uint8)
        img_pil  = Image.fromarray(arr_norm)
        buf      = io.BytesIO()
        img_pil.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    rgb = img[[2,1,0], 0].transpose(1,2,0)
    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6) * 255).astype(np.uint8)
    rgb_pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    rgb_pil.save(buf, format='PNG')
    rgb_b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({
        'predicted_acres': round(predicted_acres),
        'actual_acres':    round(actual_acres),
        'error_pct':       round(abs(predicted_acres - actual_acres) / (actual_acres + 1e-6) * 100, 1),
        'tile_name':       os.path.basename(tile_path).replace('subsetted_512x512_','').replace('_merged.tif',''),
        'rgb_image':       rgb_b64,
        'mask_image':      arr_to_b64(mask.astype(np.float32)),
        'prob_image':      arr_to_b64(probs),
        'gt_image':        arr_to_b64((gt == 1).astype(np.float32)),
    })

@app.get('/health')
async def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.get('/')
async def root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
