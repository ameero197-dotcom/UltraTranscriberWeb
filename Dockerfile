# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system deps (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install gunicorn

# Copy app
COPY app/ /app/

# Env
ENV BILLING_MODE=ads \
    APP_SECRET=change-me-please \
    BASE_URL=http://localhost:7860

EXPOSE 7860

# Use gunicorn (production-style)
CMD ["gunicorn", "UltraTranscriberWeb:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:7860", "--workers", "2"]
