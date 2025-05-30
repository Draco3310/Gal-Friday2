# Multi-stage build for production Gal-Friday deployment

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt requirements-test.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run tests and type checking
RUN pip install --no-cache-dir -r requirements-test.txt && \
    python -m pytest tests/ --no-cov && \
    python -m mypy gal_friday/

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r galfriday && useradd -r -g galfriday galfriday

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=galfriday:galfriday gal_friday ./gal_friday
COPY --chown=galfriday:galfriday scripts ./scripts
COPY --chown=galfriday:galfriday config ./config

# Create directories for runtime
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R galfriday:galfriday /app

# Switch to non-root user
USER galfriday

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GAL_FRIDAY_ENV=production

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# Expose ports
EXPOSE 8000 8080

# Entry point script
COPY --chown=galfriday:galfriday scripts/deploy/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "gal_friday.main"] 