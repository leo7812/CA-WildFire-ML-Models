FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgdal-dev gdal-bin libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tiles ./tiles
COPY static ./static
COPY main.py .

ENV PORT=8080
CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1