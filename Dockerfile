# Use a slim base with Python 3.12
FROM python:3.12-slim

# Workdir
WORKDIR /app

# Helpful defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ="America/Denver"

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Entrypoint
CMD ["python", "alpaca_backtester.py"]
