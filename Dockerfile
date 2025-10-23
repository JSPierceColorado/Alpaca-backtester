# Use a slim base with Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Environment defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ="America/Denver"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code (make sure main.py is at repo root)
COPY . .

# Entry point (matches your repo file)
CMD ["python", "main.py"]
