import os
import io
import sys
import base64
import datetime
import numpy as np
import torch
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds
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
STATIC_DIR   = os.path.join(BASE_DIR, 'static')
GDAL_AUTH_FILE = '/tmp/gdal_auth_local.hdr'

STAC_URL        = 'https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search'
HLS_COLLECTION  = 'HLSS30.v2.0'
HLS_BANDS       = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
TILE_PX         = 512
PIXEL_M         = 30
CLOUD_THRESHOLD = 10

model = None


@app.on_event('startup')
async def load_model():
    global model

    print(f'Loading weights from {WEIGHTS_PATH}...')
    model = WeatherConditionedWildfire(freeze_backbone=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    print('Model ready.')

    token = os.environ.get('NASA_EARTHDATA_TOKEN', '').strip()
    if token:
        with open(GDAL_AUTH_FILE, 'w') as f:
            f.write(f'Authorization: Bearer {token}\n')
        os.environ['GDAL_HTTP_HEADER_FILE']              = GDAL_AUTH_FILE
        os.environ['GDAL_HTTP_MERGE_CONSECUTIVE_RANGES'] = 'YES'
        os.environ['GDAL_DISABLE_READDIR_ON_OPEN']       = 'EMPTY_DIR'
        os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS']   = '.tif'
        os.environ['GDAL_HTTP_MAX_RETRY']                = '3'
        os.environ['GDAL_HTTP_RETRY_DELAY']              = '5'
        os.environ['GDAL_HTTP_TIMEOUT']                  = '60'
        print('NASA Earthdata token loaded — live HLS streaming enabled.')
    else:
        print('WARNING: NASA_EARTHDATA_TOKEN not set. Live tile fetching will fail.')
        print('  Export it before starting: export NASA_EARTHDATA_TOKEN=<your_token>')


# ── Request schema ────────────────────────────────────────────────────────────
class LivePredictRequest(BaseModel):
    lat:             float
    lon:             float
    max_temp:        float = 85.0
    wind_speed:      float = 10.0
    temp_range:      float = 20.0
    lagged_wind:     float = 6.0
    wind_temp_ratio: float = 0.12


# ── NASA STAC helpers (same as main.py) ───────────────────────────────────────
def _search_stac(lat: float, lon: float, days: int) -> list:
    end   = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=days)
    dt    = f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    buf  = 0.02
    bbox = [lon - buf, lat - buf, lon + buf, lat + buf]

    resp = requests.post(
        STAC_URL,
        json={'collections': [HLS_COLLECTION], 'bbox': bbox, 'datetime': dt, 'limit': 20},
        timeout=30,
    )
    resp.raise_for_status()

    items = resp.json().get('features', [])
    clear = [
        it for it in items
        if float(it.get('properties', {}).get('eo:cloud_cover', 100)) < CLOUD_THRESHOLD
    ]
    clear.sort(key=lambda x: x['properties']['datetime'], reverse=True)
    return clear


def _fetch_window(item: dict, lat: float, lon: float) -> np.ndarray:
    tile_crs = None
    window   = None
    bands    = []

    for band_name in HLS_BANDS:
        if band_name not in item['assets']:
            raise ValueError(f'Band {band_name} missing from STAC item {item["id"]}')

        vsi_url = '/vsicurl/' + item['assets'][band_name]['href']

        with rasterio.open(vsi_url) as src:
            if tile_crs is None:
                tile_crs = src.crs
                (cx,), (cy,) = warp_transform('EPSG:4326', tile_crs, [lon], [lat])
                half   = TILE_PX * PIXEL_M / 2
                window = from_bounds(
                    cx - half, cy - half, cx + half, cy + half,
                    src.transform,
                )

            raw = src.read(
                1,
                window=window,
                out_shape=(TILE_PX, TILE_PX),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=0,
            ).astype(np.float32)

        raw[raw < -1000] = 0.0
        raw /= 10000.0
        bands.append(raw)

    img = np.stack(bands, axis=0)
    np.clip(img, 0.0, 1.0, out=img)
    np.nan_to_num(img, copy=False, nan=0.0)
    return img


# ── Image helpers ─────────────────────────────────────────────────────────────
def _arr_to_b64(arr: np.ndarray) -> str:
    lo, hi = arr.min(), arr.max()
    norm = ((arr - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    buf  = io.BytesIO()
    Image.fromarray(norm).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _mask_to_b64(mask: np.ndarray) -> str:
    """Render binary mask with fire colormap: burned=orange, unburned=dark."""
    h, w = mask.shape
    rgb = np.full((h, w, 3), 17, dtype=np.uint8)  # dark background #111111
    rgb[mask > 0.5] = [255, 77, 26]               # fire orange #ff4d1a
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ── /predict_live ─────────────────────────────────────────────────────────────
@app.post('/predict_live')
async def predict_live(req: LivePredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

    if not os.environ.get('GDAL_HTTP_HEADER_FILE'):
        raise HTTPException(
            status_code=503,
            detail='NASA_EARTHDATA_TOKEN not configured — set the env var and restart the server',
        )

    # Try date ranges: 30 → 90 → 180 days
    stac_item = None
    for days in [30, 90, 180]:
        results = _search_stac(req.lat, req.lon, days=days)
        if results:
            stac_item = results[0]
            break

    if stac_item is None:
        raise HTTPException(
            status_code=404,
            detail='no_clear_tile: No cloud-free HLS tile found within 180 days',
        )

    try:
        img = _fetch_window(stac_item, req.lat, req.lon)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'tile_fetch_error: {exc}')

    img_t          = img[:, np.newaxis, :, :]
    image_tensor   = torch.tensor(img_t).unsqueeze(0)

    weather_raw    = np.array(
        [req.temp_range, req.lagged_wind, req.wind_speed, req.max_temp, req.wind_temp_ratio],
        dtype=np.float32,
    )
    weather_norm   = np.nan_to_num((weather_raw - WEATHER_MEAN) / (WEATHER_STD + 1e-6), nan=0.0)
    weather_tensor = torch.tensor(weather_norm).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor, weather_tensor)
        probs  = logits.softmax(dim=1)[0, 1].numpy()
        mask   = logits.argmax(dim=1)[0].numpy()

    predicted_acres = float((mask == 1).sum() * 900 / 4047)
    peak_prob       = round(float(probs.max()) * 100, 1)

    rgb = img[[2, 1, 0]].transpose(1, 2, 0)
    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    rgb_b64 = base64.b64encode(buf.getvalue()).decode()

    props     = stac_item['properties']
    tile_date = props['datetime'][:10]
    item_id   = stac_item.get('id', '')
    parts     = item_id.split('.')
    mgrs      = parts[2] if len(parts) > 2 else item_id

    return JSONResponse({
        'predicted_acres': round(predicted_acres),
        'peak_prob':       peak_prob,
        'rgb_image':       rgb_b64,
        'mask_image':      _mask_to_b64(mask.astype(np.float32)),
        'prob_image':      _arr_to_b64(probs),
        'tile_date':       tile_date,
        'tile_id':         mgrs,
        'cloud_cover':     props.get('eo:cloud_cover'),
    })


# ── Health + static ───────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    return {
        'status':       'ok',
        'model_loaded': model is not None,
        'nasa_token':   bool(os.environ.get('GDAL_HTTP_HEADER_FILE')),
    }

@app.get('/')
async def root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
