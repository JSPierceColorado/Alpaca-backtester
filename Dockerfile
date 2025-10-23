# Use a slim base with Python 3.12
FROM python:3.12-slim

# Create and set work directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment defaults (override in Railway)
ENV TZ="America/Denver" \
    PYTHONUNBUFFERED=1

# Entry point
CMD ["python", "alpaca_backtester.py"]
