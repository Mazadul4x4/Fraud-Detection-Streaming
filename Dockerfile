# Container image for the real-time fraud scoring API.
# Build:  docker compose build api      Run:  docker compose up -d api

FROM python:3.11-slim

# No .pyc files, unbuffered logs (shown immediately in `docker compose logs`)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# XGBoost needs the OpenMP runtime library
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilt when requirements change
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code (only what the API needs)
COPY src/ src/

# Never run as root inside the container
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
