# FIRECAST — California Wildfire Spread Predictor

A machine learning web app that predicts wildfire burn probability for any location in California using live NASA satellite imagery and user-specified weather conditions.

**Live site:** https://sjsu-ds-projects.appspot.com/

https://github.com/user-attachments/assets/943e56c9-1c10-40b0-9841-89e43d8c65fc

---

## Summary

This repo combines a fine-tuned geospatial foundation model (Prithvi-EO-2.0), live NASA Sentinel-2 satellite imagery, and historical California weather data to estimate wildfire spread risk. Users pick a location on a map, adjust weather sliders, and receive a burn probability heatmap and predicted acreage in real time.

---

## Metrics

Validation performance (264-tile validation split, HLS Burn Scars dataset):
- IoU: 0.844
- F1 / Dice: 0.915
- Precision: 0.931 | Recall: 0.900
- Pixel accuracy: 0.985

---

## Local Setup

**Requirements:** Python 3.11

```bash
pip install -r requirements.txt
```

### Web App

```bash
cd deploy
uvicorn main:app --reload
```

No credentials needed. The historical fire chart falls back to a bundled CSV snapshot when BigQuery is unavailable.

### Inference Service

```bash
cd inference
NASA_EARTHDATA_TOKEN=<your_token> uvicorn main:app --reload
```

**Model weights** (~1.2 GB) are hosted publicly on Hugging Face at [leo7812/firecast-wildfire](https://huggingface.co/leo7812/firecast-wildfire) and download automatically on first startup — no manual download required.

**NASA Earthdata token** is required to fetch live Sentinel-2 satellite tiles. Register for a free account at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov), then pass your token via the `NASA_EARTHDATA_TOKEN` environment variable.

### Dependency Summary

| Dependency | Web App | Inference Service | How to get |
|---|---|---|---|
| Model weights | — | auto-downloaded from HF Hub | automatic |
| NASA Earthdata token | — | required | free at urs.earthdata.nasa.gov |
| GCP / BigQuery | optional | optional | falls back automatically |

---

## Pipeline

```
1. Data Collection
   ├── HLS Burn Scars (Hugging Face) — 804 labeled 512×512 satellite scenes
   └── CA Weather & Fire CSV (CAL FIRE / NOAA) — daily records 1984–2025

2. Data Storage
   └── Weather/fire CSV → uploaded to BigQuery

3. Model Training
   ├── src/train_colab.py — fine-tunes Prithvi-EO-2.0 on HLS burn scars
   └── Weather encoder MLP trained jointly; checkpoint saved to GCS

4. Inference Service
   └── inference/ — Dockerized FastAPI on Cloud Run
       ├── Fetches live Sentinel-2 tiles from NASA STAC on demand
       └── Runs forward pass → burn mask + probability heatmap

5. Web Application
   └── deploy/ — FastAPI on App Engine Standard
       ├── Proxies prediction requests to the inference service
       ├── Queries BigQuery for historical fire stats
       └── Serves the frontend (deploy/static/index.html)
```

---

## Repository Structure

```
wildfire_project/
├── src/                        # Model architecture, dataset loader, and training scripts
├── inference/                  # Dockerized ML inference service deployed on Cloud Run
├── deploy/                     # Web application deployed on App Engine Standard
│   ├── static/                 # Single-page frontend (index.html)
│   └── fire_stats_fallback.csv # Bundled CA fire stats snapshot (1984–2025)
├── notebooks/                  # Exploratory data analysis
├── data/                       # Local training data (not tracked in git)
└── checkpoints/                # Model weight files (not tracked in git)
```

---

## System Design

```
User Browser
     │
     ▼
App Engine Standard              BigQuery (primary)
(deploy/)            ─────────►  sjsu-ds-projects.wildfire.ca_weather_fire
     │               ─────────►  fire_stats_fallback.csv (if BigQuery unavailable)
     │
     ▼
Cloud Run (inference/)
     │
     ├── NASA STAC CMR  ────────► live Sentinel-2 tiles (30m resolution)
     ├── GCS Bucket     ────────► model weights (primary, same-region)
     └── Hugging Face   ────────► model weights (fallback, leo7812/firecast-wildfire)
```

**Scaling:**

- App Engine Standard scales between 0–3 instances based on CPU utilization (65% threshold).
- Cloud Run scales to zero when idle and spins up on demand; one worker per instance due to model memory.
- BigQuery results are cached in memory after the first request to avoid repeated query costs.

---

## Inference Service

**Location:** `inference/`
**Docker:** `inference/Dockerfile`
**Model training:** `src/train_colab.py`, `src/model.py`

The service loads a fine-tuned **Prithvi-EO-2.0** (300M parameter geospatial foundation model) with a 5-scalar weather conditioning MLP. On each request it fetches the most recent cloud-free Sentinel-2 chip for the given coordinates from NASA's STAC API, normalizes the 6 spectral bands, and runs a forward pass.

| Endpoint | Input | Output |
| --- | --- | --- |
| `POST /predict_live` | `lat, lon, max_temp, wind_speed, temp_range, lagged_wind, wind_temp_ratio` | burn mask image, probability heatmap, predicted acres, peak probability |
| `POST /prefetch` | `lat, lon` | warms tile cache in background |
| `GET /health` | — | service status, model load state |

---

## Cloud Data

**What:** Daily California weather and fire occurrence records, 1984–2025. Key fields: `MAX_TEMP`, `AVG_WIND_SPEED`, `LAGGED_AVG_WIND_SPEED`, `TEMP_RANGE`, `WIND_TEMP_RATIO`, `FIRE_START_DAY`.

**How stored:** Uploaded from a local CSV into BigQuery table `sjsu-ds-projects.wildfire.ca_weather_fire`.

**How consumed:** Queried by the web app at startup and cached in memory. Rendered on the Home page as an interactive bar chart showing wildfire days per year (1984–2025). A bundled snapshot (`deploy/fire_stats_fallback.csv`) is used automatically when BigQuery is unavailable.

---

## Technical Challenges

A few problems that came up during development and how they were solved:

- **Dependency conflict on Cloud Run:** `terratorch` silently upgraded `segmentation-models-pytorch` past the version the model was trained against, causing shape mismatches at inference time. Fixed by pinning both packages explicitly in `requirements.txt`.
- **Cold-start timeouts:** Cloud Run health checks were failing because model weight loading (~1.2 GB from GCS/Hugging Face) took longer than the default startup probe window. Solved by loading the model on a background thread from FastAPI's `lifespan` handler (`inference/main.py`) so the server binds to the port and starts responding to `/health` immediately; `/health` and `/predict_live` report a `warming_up` state until the load finishes.
- **OOM kills under load:** The 300M-parameter model plus a full forward pass exceeded default Cloud Run memory limits. Resolved by capping the inference container to a single Uvicorn worker (`--workers 1` in `inference/Dockerfile`), so only one copy of the model is resident in memory per instance instead of one per worker.
- **Dynamic location support:** Initial version only supported a fixed set of pre-cached tiles. Rebuilt the inference path to fetch live Sentinel-2 tiles from NASA's STAC API on demand for any California coordinate, with cloud-cover fallback logic when the freshest tile is obscured.

---

## Tech Stack

- **ML:** PyTorch, Prithvi-EO-2.0 (300M), `segmentation-models-pytorch`, `terratorch`
- **Backend:** FastAPI, Uvicorn, Docker
- **Infra:** Google Cloud Run, App Engine Standard, BigQuery, Google Cloud Storage
- **Data:** NASA HLS / Sentinel-2 (STAC API), CAL FIRE / NOAA historical records, Hugging Face Hub
