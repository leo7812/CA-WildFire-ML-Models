import os
import threading
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel
from google.cloud import bigquery

INFERENCE_URL = os.environ.get(
    'INFERENCE_URL',
    'https://wildfire-inference-4hvdf26vhq-uw.a.run.app',
)

GCP_PROJECT = 'sjsu-ds-projects'
BQ_TABLE    = 'sjsu-ds-projects.wildfire.ca_weather_fire'

bq: bigquery.Client | None = None
_bq_cache: dict | None     = None


def _init_bq():
    global bq
    bq = bigquery.Client(project=GCP_PROJECT)
    print('BigQuery client ready.')


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_init_bq, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


# ── Proxy helpers ─────────────────────────────────────────────────────────────
def _proxy_post(path: str, body: dict, timeout: int) -> Response:
    try:
        r = requests.post(f'{INFERENCE_URL}{path}', json=body, timeout=timeout)
        return JSONResponse(content=r.json(), status_code=r.status_code)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail='Inference service timed out')
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── /prefetch — fire-and-forget, returns immediately ─────────────────────────
class PrefetchRequest(BaseModel):
    lat: float
    lon: float


@app.post('/prefetch')
async def prefetch(req: PrefetchRequest):
    threading.Thread(
        target=lambda: requests.post(
            f'{INFERENCE_URL}/prefetch',
            json=req.model_dump(),
            timeout=5,
        ),
        daemon=True,
    ).start()
    return {'status': 'forwarded'}


# ── /predict_live — proxied synchronously ────────────────────────────────────
@app.post('/predict_live')
async def predict_live(request: Request):
    body = await request.json()
    return _proxy_post('/predict_live', body, timeout=120)


# ── /api/fire_stats — BigQuery ────────────────────────────────────────────────
@app.get('/api/fire_stats')
async def fire_stats():
    global _bq_cache
    if bq is None:
        raise HTTPException(status_code=503, detail='Service initializing, please retry shortly')
    if _bq_cache is not None:
        return _bq_cache

    query = f"""
        SELECT
            YEAR,
            COUNTIF(FIRE_START_DAY = TRUE)  AS fire_days,
            ROUND(AVG(MAX_TEMP),        1)  AS avg_max_temp,
            ROUND(AVG(AVG_WIND_SPEED),  1)  AS avg_wind_speed
        FROM `{BQ_TABLE}`
        GROUP BY YEAR
        ORDER BY YEAR
    """
    rows      = bq.query(query).result()
    _bq_cache = {'fire_stats': [dict(r) for r in rows]}
    return _bq_cache


# ── Health + static ───────────────────────────────────────────────────────────
@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.get('/')
async def root():
    return FileResponse(
        'static/index.html',
        headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
    )


app.mount('/static', StaticFiles(directory='static'), name='static')
