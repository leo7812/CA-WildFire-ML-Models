import os
import io
import sys
import time
import shutil
import base64
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.cloud import secretmanager
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, '/app/src')

from model import WeatherConditionedWildfire
from dataset import WEATHER_MEAN, WEATHER_STD

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT  = 'sjsu-ds-projects'
BUCKET_NAME  = 'sjsu-wildfire-models'
HF_REPO_ID   = 'leo7812/firecast-wildfire'
WEIGHTS_FILE = 'wildfire_weather_weights.pt'
WEIGHTS_PATH = '/tmp/wildfire_weather_weights.pt'

STAC_URL        = 'https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search'
HLS_COLLECTION  = 'HLSS30.v2.0'
HLS_BANDS       = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
TILE_PX         = 512
PIXEL_M         = 30
CLOUD_THRESHOLD = 10
GDAL_AUTH_FILE  = '/tmp/gdal_auth.hdr'
CACHE_TTL       = 1800  # seconds before a prefetched tile expires

model          = None
_startup_error = None
_tile_cache: dict   = {}  # key → {img, stac_item, ts}
_tile_pending: set  = set()  # keys currently being fetched


def _cache_key(lat: float, lon: float) -> str:
    return f"{round(lat, 2)},{round(lon, 2)}"


def _startup():
    global model, _startup_error
    try:
        if not os.path.exists(WEIGHTS_PATH):
            try:
                from google.cloud import storage
                print('Downloading weights from GCS...')
                gcs  = storage.Client(project=GCP_PROJECT)
                blob = gcs.bucket(BUCKET_NAME).blob(WEIGHTS_FILE)
                blob.download_to_filename(WEIGHTS_PATH, timeout=300)
                print(f'GCS download complete ({os.path.getsize(WEIGHTS_PATH) / 1024 / 1024:.0f} MB).')
            except Exception as gcs_err:
                print(f'GCS unavailable ({gcs_err}), falling back to Hugging Face Hub...')
                cached = hf_hub_download(repo_id=HF_REPO_ID, filename=WEIGHTS_FILE)
                shutil.copy(cached, WEIGHTS_PATH)
                print(f'HF download complete ({os.path.getsize(WEIGHTS_PATH) / 1024 / 1024:.0f} MB).')
        else:
            print(f'Using cached weights at {WEIGHTS_PATH}.')

        print('Building model architecture...')
        model = WeatherConditionedWildfire(freeze_backbone=False, backbone_pretrained=False)
        print('Architecture built. Loading state dict...')
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
        model.eval()
        print('Model ready.')

        print('Loading NASA Earthdata token...')
        token = os.environ.get('NASA_EARTHDATA_TOKEN', '').strip()
        if not token:
            try:
                sm    = secretmanager.SecretManagerServiceClient()
                name  = 'projects/sjsu-ds-projects/secrets/NASA_EARTHDATA_TOKEN/versions/latest'
                token = sm.access_secret_version(name=name).payload.data.decode('UTF-8').strip()
                print('NASA token loaded from Secret Manager.')
            except Exception as sm_err:
                raise RuntimeError(
                    f'NASA_EARTHDATA_TOKEN env var not set and Secret Manager unavailable: {sm_err}'
                )
        else:
            print('NASA token loaded from environment variable.')

        with open(GDAL_AUTH_FILE, 'w') as f:
            f.write(f'Authorization: Bearer {token}\n')

        os.environ['GDAL_HTTP_HEADER_FILE']              = GDAL_AUTH_FILE
        os.environ['GDAL_HTTP_MERGE_CONSECUTIVE_RANGES'] = 'YES'
        os.environ['GDAL_DISABLE_READDIR_ON_OPEN']       = 'EMPTY_DIR'
        os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS']   = '.tif'
        os.environ['GDAL_HTTP_MAX_RETRY']                = '3'
        os.environ['GDAL_HTTP_RETRY_DELAY']              = '5'
        os.environ['GDAL_HTTP_TIMEOUT']                  = '60'
        print('Startup complete.')
    except Exception as exc:
        _startup_error = str(exc)
        print(f'STARTUP FAILED: {exc}', flush=True)
        import traceback; traceback.print_exc()


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_startup, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


# ── Request schemas ───────────────────────────────────────────────────────────
class PrefetchRequest(BaseModel):
    lat: float
    lon: float


class LivePredictRequest(BaseModel):
    lat:             float
    lon:             float
    max_temp:        float = 85.0
    wind_speed:      float = 10.0
    temp_range:      float = 20.0
    lagged_wind:     float = 6.0
    wind_temp_ratio: float = 0.12


# ── STAC + tile helpers ───────────────────────────────────────────────────────
def _search_stac(lat: float, lon: float, days: int) -> list:
    end   = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=days)
    dt    = f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    buf   = 0.02
    bbox  = [lon - buf, lat - buf, lon + buf, lat + buf]

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
    def _fetch_band(band_name: str) -> tuple[str, np.ndarray]:
        if band_name not in item['assets']:
            raise ValueError(f'Band {band_name} missing from STAC item {item["id"]}')
        vsi_url = '/vsicurl/' + item['assets'][band_name]['href']
        with rasterio.open(vsi_url) as src:
            (cx,), (cy,) = warp_transform('EPSG:4326', src.crs, [lon], [lat])
            half   = TILE_PX * PIXEL_M / 2
            window = from_bounds(cx - half, cy - half, cx + half, cy + half, src.transform)
            raw    = src.read(
                1, window=window, out_shape=(TILE_PX, TILE_PX),
                resampling=Resampling.bilinear, boundless=True, fill_value=0,
            ).astype(np.float32)
        raw[raw < -1000] = 0.0
        raw /= 10000.0
        return band_name, raw

    with ThreadPoolExecutor(max_workers=len(HLS_BANDS)) as ex:
        futures  = {ex.submit(_fetch_band, b): b for b in HLS_BANDS}
        results  = {}
        for fut in as_completed(futures):
            name, data = fut.result()
            results[name] = data

    img = np.stack([results[b] for b in HLS_BANDS], axis=0)
    np.clip(img, 0.0, 1.0, out=img)
    np.nan_to_num(img, copy=False, nan=0.0)
    return img


def _do_prefetch(key: str, lat: float, lon: float):
    try:
        for days in [30, 90, 180]:
            results = _search_stac(lat, lon, days=days)
            for stac_item in results:
                try:
                    img = _fetch_window(stac_item, lat, lon)
                    if not _tile_usable(img):
                        print(f'Prefetch: skipping boundary tile {stac_item.get("id","?")} for {key}')
                        continue
                    _tile_cache[key] = {'img': img, 'stac_item': stac_item, 'ts': time.time()}
                    print(f'Prefetch complete for {key}')
                    return
                except Exception as exc:
                    print(f'Prefetch: error fetching {stac_item.get("id","?")}: {exc}')
        print(f'Prefetch: no usable tile for {key}')
    except Exception as exc:
        print(f'Prefetch error for {key}: {exc}')
    finally:
        _tile_pending.discard(key)


def _tile_usable(img: np.ndarray, max_nodata_frac: float = 0.20) -> bool:
    """Return False if more than max_nodata_frac of pixels are all-zero (boundary fill)."""
    all_zero = (img == 0).all(axis=0)
    return all_zero.mean() < max_nodata_frac


def _arr_to_b64(arr: np.ndarray) -> str:
    lo, hi = arr.min(), arr.max()
    norm = ((arr - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    buf  = io.BytesIO()
    Image.fromarray(norm).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def _mask_to_b64(mask: np.ndarray) -> str:
    h, w = mask.shape
    rgb = np.full((h, w, 3), 17, dtype=np.uint8)
    rgb[mask > 0.5] = [255, 77, 26]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


# ── /prefetch ─────────────────────────────────────────────────────────────────
@app.post('/prefetch')
async def prefetch(req: PrefetchRequest):
    if model is None:
        return {'status': 'warming_up', 'error': _startup_error}
    key = _cache_key(req.lat, req.lon)
    if key in _tile_cache:
        return {'status': 'cached'}
    if key not in _tile_pending:
        _tile_pending.add(key)
        threading.Thread(target=_do_prefetch, args=(key, req.lat, req.lon), daemon=True).start()
    return {'status': 'fetching'}


# ── /predict_live ─────────────────────────────────────────────────────────────
@app.post('/predict_live')
async def predict_live(req: LivePredictRequest):
    if model is None:
        detail = f'startup_error: {_startup_error}' if _startup_error else 'Model not loaded yet'
        raise HTTPException(status_code=503, detail=detail)

    # Use prefetched tile if available, otherwise fetch now
    key    = _cache_key(req.lat, req.lon)
    cached = _tile_cache.pop(key, None)

    if cached and (time.time() - cached['ts']) < CACHE_TTL:
        img       = cached['img']
        stac_item = cached['stac_item']
        if not _tile_usable(img):
            img, stac_item = None, None  # cached tile was bad, re-fetch
    else:
        img, stac_item = None, None

    t_fetch_start = time.time()
    if img is None:
        t_stac_start = time.time()
        for days in [30, 90, 180]:
            results = _search_stac(req.lat, req.lon, days=days)
            for candidate in results:
                try:
                    fetched = _fetch_window(candidate, req.lat, req.lon)
                    if not _tile_usable(fetched):
                        print(f'Skipping boundary tile {candidate.get("id","?")}')
                        continue
                    img, stac_item = fetched, candidate
                    break
                except Exception as exc:
                    print(f'Tile fetch error {candidate.get("id","?")}: {exc}')
            if img is not None:
                break
        t_stac_end = time.time()
        print(f'[TIMING] STAC search + tile fetch: {t_stac_end - t_stac_start:.2f}s')
    else:
        print(f'[TIMING] Tile served from cache')

    if img is None:
        raise HTTPException(
            status_code=404,
            detail='no_clear_tile: No usable cloud-free HLS tile found within 180 days',
        )

    img_t          = img[:, np.newaxis, :, :]
    image_tensor   = torch.tensor(img_t).unsqueeze(0)

    weather_raw    = np.array(
        [req.temp_range, req.lagged_wind, req.wind_speed, req.max_temp, req.wind_temp_ratio],
        dtype=np.float32,
    )
    weather_norm   = np.nan_to_num((weather_raw - WEATHER_MEAN) / (WEATHER_STD + 1e-6), nan=0.0)
    weather_tensor = torch.tensor(weather_norm).unsqueeze(0)

    t_infer_start = time.time()
    with torch.no_grad():
        logits = model(image_tensor, weather_tensor)

        # Compute weather risk (0–1) from raw physical values
        weather_risk = float(np.clip(
            (weather_raw[3] - 50) / 65 * 0.4 +  # max_temp
            (weather_raw[2] / 40) * 0.4 +         # wind_speed
            (weather_raw[4] / 0.5) * 0.2,         # wind_temp_ratio
            0.0, 1.0,
        ))

        # Boost burn-class logit by weather risk so that extreme conditions
        # produce visibly larger predicted burn areas
        logits = logits.clone()
        logits[:, 1, :, :] += weather_risk * 8.0

        probs = logits.softmax(dim=1)[0, 1].numpy()

        # Dynamic threshold: 0.5 at no risk, drops to 0.05 at max risk
        threshold = max(0.05, 0.5 - weather_risk * 0.45)
        mask = (probs >= threshold).astype(np.int32)

    t_infer_end = time.time()
    print(f'[TIMING] Model inference: {t_infer_end - t_infer_start:.2f}s')
    print(f'[TIMING] Total (fetch + inference): {t_infer_end - t_fetch_start:.2f}s')

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


# ── Health ────────────────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    return {'status': 'ok', 'model_loaded': model is not None, 'startup_error': _startup_error}
