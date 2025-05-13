FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run as non-root user for better security
RUN useradd -m appuser
USER appuser

# Default command - can be overridden in docker-compose or when running container
CMD ["python", "-m", "src.gal_friday.main"]
