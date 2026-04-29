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
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from google.cloud import storage, secretmanager
from PIL import Image

sys.path.insert(0, '/app/src')

from model import WeatherConditionedWildfire
from dataset import WEATHER_MEAN, WEATHER_STD

# ── Config ────────────────────────────────────────────────────────────────────
BUCKET_NAME  = 'sjsu-wildfire-models'
WEIGHTS_FILE = 'wildfire_weather_weights.pt'
WEIGHTS_PATH = '/tmp/wildfire_weather_weights.pt'

STAC_URL        = 'https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search'
HLS_COLLECTION  = 'HLSS30.v2.0'
HLS_BANDS       = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
TILE_PX         = 512
PIXEL_M         = 30
CLOUD_THRESHOLD = 10
GDAL_AUTH_FILE  = '/tmp/gdal_auth.hdr'

model = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model

    # Weights from GCS
    print('Downloading weights from GCS...')
    gcs = storage.Client()
    gcs.bucket(BUCKET_NAME).blob(WEIGHTS_FILE).download_to_filename(WEIGHTS_PATH)
    print('Weights downloaded. Loading model...')
    model = WeatherConditionedWildfire(freeze_backbone=False, backbone_pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    print('Model ready.')

    # NASA Earthdata token from Secret Manager
    print('Loading NASA Earthdata token...')
    sm     = secretmanager.SecretManagerServiceClient()
    name   = 'projects/sjsu-ds-projects/secrets/NASA_EARTHDATA_TOKEN/versions/latest'
    token  = sm.access_secret_version(name=name).payload.data.decode('UTF-8').strip()

    # Write GDAL auth header file for vsicurl COG streaming
    with open(GDAL_AUTH_FILE, 'w') as f:
        f.write(f'Authorization: Bearer {token}\n')

    os.environ['GDAL_HTTP_HEADER_FILE']             = GDAL_AUTH_FILE
    os.environ['GDAL_HTTP_MERGE_CONSECUTIVE_RANGES'] = 'YES'
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN']      = 'EMPTY_DIR'
    os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS']  = '.tif'
    os.environ['GDAL_HTTP_MAX_RETRY']               = '3'
    os.environ['GDAL_HTTP_RETRY_DELAY']             = '5'
    os.environ['GDAL_HTTP_TIMEOUT']                 = '60'
    print('NASA token loaded and GDAL vsicurl configured.')

    yield

app = FastAPI(lifespan=lifespan)

# ── Request schemas ───────────────────────────────────────────────────────────
class LivePredictRequest(BaseModel):
    lat:             float
    lon:             float
    max_temp:        float = 85.0
    wind_speed:      float = 10.0
    temp_range:      float = 20.0
    lagged_wind:     float = 6.0
    wind_temp_ratio: float = 0.12

# ── STAC helpers ──────────────────────────────────────────────────────────────
def _search_stac(lat: float, lon: float, days: int) -> list:
    """Return cloud-free HLSS30 items intersecting (lat, lon), newest first."""
    end   = datetime.datetime.utcnow()
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
    """
    Stream-read a TILE_PX×TILE_PX window at PIXEL_M resolution from all 6 HLS bands
    via GDAL vsicurl. Returns float32 (6, 512, 512) surface reflectance in [0, 1].
    """
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
                # Reproject click point to tile CRS (uses rasterio's bundled PROJ)
                (cx,), (cy,) = warp_transform('EPSG:4326', tile_crs, [lon], [lat])
                half   = TILE_PX * PIXEL_M / 2   # 7 680 m
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

        raw[raw < -1000] = 0.0   # HLS fill value is -9999 (int16)
        raw /= 10000.0           # int16 DN → surface reflectance
        bands.append(raw)

    img = np.stack(bands, axis=0)   # (6, 512, 512)
    np.clip(img, 0.0, 1.0, out=img)
    np.nan_to_num(img, copy=False, nan=0.0)
    return img


def _arr_to_b64(arr: np.ndarray) -> str:
    lo, hi = arr.min(), arr.max()
    norm = ((arr - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    buf  = io.BytesIO()
    Image.fromarray(norm).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ── /predict_live ─────────────────────────────────────────────────────────────
@app.post('/predict_live')
async def predict_live(req: LivePredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet')

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

    # (6, 512, 512) → (6, 1, 512, 512) → batch tensor
    img_t          = img[:, np.newaxis, :, :]
    image_tensor   = torch.tensor(img_t).unsqueeze(0)   # (1, 6, 1, 512, 512)

    weather_raw    = np.array(
        [req.temp_range, req.lagged_wind, req.wind_speed, req.max_temp, req.wind_temp_ratio],
        dtype=np.float32,
    )
    weather_norm   = np.nan_to_num((weather_raw - WEATHER_MEAN) / (WEATHER_STD + 1e-6), nan=0.0)
    weather_tensor = torch.tensor(weather_norm).unsqueeze(0)  # (1, 5)

    with torch.no_grad():
        logits = model(image_tensor, weather_tensor)
        probs  = logits.softmax(dim=1)[0, 1].numpy()   # (512, 512) burn probability
        mask   = logits.argmax(dim=1)[0].numpy()        # (512, 512) binary mask

    predicted_acres = float((mask == 1).sum() * 900 / 4047)
    peak_prob       = round(float(probs.max()) * 100, 1)   # highest burn probability in window (%)

    # RGB: B04=red (idx 2), B03=green (idx 1), B02=blue (idx 0)
    rgb = img[[2, 1, 0]].transpose(1, 2, 0)
    rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    rgb_b64 = base64.b64encode(buf.getvalue()).decode()

    # Tile metadata
    props     = stac_item['properties']
    tile_date = props['datetime'][:10]
    item_id   = stac_item.get('id', '')
    parts     = item_id.split('.')
    mgrs      = parts[2] if len(parts) > 2 else item_id   # e.g. T10SGF

    return JSONResponse({
        'predicted_acres': round(predicted_acres),
        'peak_prob':       peak_prob,
        'rgb_image':       rgb_b64,
        'mask_image':      _arr_to_b64(mask.astype(np.float32)),
        'prob_image':      _arr_to_b64(probs),
        'tile_date':       tile_date,
        'tile_id':         mgrs,
        'cloud_cover':     props.get('eo:cloud_cover'),
    })


# ── Health + static ───────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.get('/')
async def root():
    return FileResponse('/app/static/index.html')

app.mount('/static', StaticFiles(directory='/app/static'), name='static')
