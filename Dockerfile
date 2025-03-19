FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to utilize Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port (will be overridden by environment variable if set)
ARG PORT=5000
EXPOSE ${PORT}

# Run with optimized settings configured through environment variables
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-5000} --workers ${WORKERS:-4} --log-level ${LOG_LEVEL:-info}