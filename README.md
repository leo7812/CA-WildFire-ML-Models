# FIRECAST — California Wildfire Spread Predictor

A machine learning web app that predicts wildfire burn probability for any location in California using live NASA satellite imagery and user-specified weather conditions.

**Live site:** https://wildfire-final-512104156662.us-west1.run.app/

---

## Summary

This repo combines a fine-tuned geospatial foundation model (Prithvi-EO-2.0), live NASA Sentinel-2 satellite imagery, and historical California weather data to estimate wildfire spread risk. Users pick a location on a map, adjust weather sliders, and receive a burn probability heatmap and predicted acreage in real time.

---

## Setup

**Requirements:** Python 3.11, GCP credentials with access to `sjsu-ds-projects`

```bash
pip install -r requirements.txt
```

Run the web app locally:
```bash
cd deploy
uvicorn main:app --reload
```

Run the inference service locally (requires model weights in GCS):
```bash
cd inference
uvicorn main:app --reload
```

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
├── src/            # Model architecture, dataset loader, and training scripts
├── inference/      # Dockerized ML inference service deployed on Cloud Run
├── deploy/         # Web application deployed on App Engine Standard
│   └── static/     # Single-page frontend (index.html)
├── notebooks/      # Exploratory data analysis
├── data/           # Local training data (not tracked in git)
└── checkpoints/    # Model weight files (not tracked in git)
```

---

## System Design

```
User Browser
     │
     ▼
App Engine Standard              BigQuery
(deploy/)            ─────────►  sjsu-ds-projects.wildfire.ca_weather_fire
     │                           (historical CA weather & fire stats)
     │
     ▼
Cloud Run (inference/)
     │
     ├── NASA STAC CMR  ────────► live Sentinel-2 tiles (30m resolution)
     └── GCS Bucket     ────────► model weights (wildfire_weather_weights.pt)
```

**Scaling:**
- App Engine Standard scales between 1–3 instances based on CPU utilization (65% threshold).
- Cloud Run scales to zero when idle and spins up on demand; one worker per instance due to model memory.
- BigQuery results are cached in memory after the first request to avoid repeated query costs.

---

## Inference Service

**Location:** `inference/`  
**Docker:** `inference/Dockerfile`  
**Model training:** `src/train_colab.py`, `src/model.py`

The service loads a fine-tuned **Prithvi-EO-2.0** (300M parameter geospatial foundation model) with a 5-scalar weather conditioning MLP. On each request it fetches the most recent cloud-free Sentinel-2 chip for the given coordinates from NASA's STAC API, normalizes the 6 spectral bands, and runs a forward pass.

| Endpoint | Input | Output |
|---|---|---|
| `POST /predict_live` | `lat, lon, max_temp, wind_speed, temp_range, lagged_wind, wind_temp_ratio` | burn mask image, probability heatmap, predicted acres, peak probability |
| `POST /prefetch` | `lat, lon` | warms tile cache in background |
| `GET /health` | — | service status, model load state |

---

## Cloud Data

**What:** Daily California weather and fire occurrence records, 1984–2025. Key fields: `MAX_TEMP`, `AVG_WIND_SPEED`, `LAGGED_AVG_WIND_SPEED`, `TEMP_RANGE`, `WIND_TEMP_RATIO`, `FIRE_START_DAY`.

**How stored:** Uploaded from a local CSV into BigQuery table `sjsu-ds-projects.wildfire.ca_weather_fire`.

**How consumed:** Queried by the web app at startup and cached in memory. Rendered on the Home page as an interactive bar chart showing wildfire days per year (1984–2025).

---

## Website

https://wildfire-final-512104156662.us-west1.run.app/
