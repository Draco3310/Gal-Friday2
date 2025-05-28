# Gal-Friday Production Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying Gal-Friday to production, including infrastructure setup, deployment procedures, monitoring configuration, and operational best practices.

## Table of Contents
1. [Infrastructure Requirements](#infrastructure-requirements)
2. [Pre-deployment Checklist](#pre-deployment-checklist)
3. [Database Setup](#database-setup)
4. [Application Deployment](#application-deployment)
5. [Configuration Management](#configuration-management)
6. [Security Hardening](#security-hardening)
7. [Monitoring & Alerting](#monitoring--alerting)
8. [Performance Tuning](#performance-tuning)
9. [Backup & Recovery](#backup--recovery)
10. [Operational Procedures](#operational-procedures)

## Infrastructure Requirements

### Minimum Hardware Requirements
- **Application Servers**: 2x (for redundancy)
  - CPU: 8 cores
  - RAM: 16GB
  - Storage: 100GB SSD
  - Network: 1Gbps

- **Database Servers**:
  - PostgreSQL: 16 cores, 32GB RAM, 500GB SSD
  - InfluxDB: 8 cores, 16GB RAM, 1TB SSD
  - Redis: 4 cores, 8GB RAM, 50GB SSD

### Network Architecture
```
Internet
    |
Load Balancer (HTTPS)
    |
    +-- App Server 1
    |       |
    +-- App Server 2
            |
    +-- PostgreSQL (Primary)
    |       |
    |       +-- PostgreSQL (Replica)
    |
    +-- InfluxDB
    |
    +-- Redis
```

### Software Requirements
- Operating System: Ubuntu 22.04 LTS
- Python: 3.11+
- Docker: 24.0+
- PostgreSQL: 14+
- InfluxDB: 2.0+
- Redis: 7.0+
- Nginx: 1.18+

## Pre-deployment Checklist

### Code Preparation
- [ ] All tests passing
- [ ] Code review completed
- [ ] Security scan passed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Version tagged in Git

### Infrastructure
- [ ] Servers provisioned
- [ ] Network configured
- [ ] SSL certificates obtained
- [ ] DNS configured
- [ ] Firewall rules set
- [ ] Load balancer configured

### Dependencies
- [ ] Python packages locked (`requirements.txt`)
- [ ] System packages documented
- [ ] External API credentials ready
- [ ] Database schemas prepared

## Database Setup

### PostgreSQL Setup

1. **Install PostgreSQL**:
```bash
sudo apt update
sudo apt install postgresql-14 postgresql-contrib
```

2. **Configure PostgreSQL**:
```bash
# Edit postgresql.conf
sudo nano /etc/postgresql/14/main/postgresql.conf

# Key settings:
max_connections = 200
shared_buffers = 8GB
effective_cache_size = 24GB
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 20MB
min_wal_size = 2GB
max_wal_size = 8GB
```

3. **Create Database and User**:
```sql
CREATE USER gal_friday WITH ENCRYPTED PASSWORD 'secure_password';
CREATE DATABASE gal_friday_prod OWNER gal_friday;
GRANT ALL PRIVILEGES ON DATABASE gal_friday_prod TO gal_friday;

-- Enable required extensions
\c gal_friday_prod
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

4. **Run Migrations**:
```bash
cd /app/gal_friday
python -m gal_friday.dal.migrations.migrate --production
```

### InfluxDB Setup

1. **Install InfluxDB**:
```bash
wget https://dl.influxdata.com/influxdb/releases/influxdb2-2.7.1-amd64.deb
sudo dpkg -i influxdb2-2.7.1-amd64.deb
sudo systemctl start influxdb
```

2. **Initial Configuration**:
```bash
influx setup \
  --org gal-friday \
  --bucket market-data \
  --username admin \
  --password secure_password \
  --retention 90d
```

3. **Create Buckets**:
```bash
influx bucket create --name predictions --retention 30d
influx bucket create --name metrics --retention 7d
```

### Redis Setup

1. **Install Redis**:
```bash
sudo apt install redis-server
```

2. **Configure Redis**:
```bash
# Edit redis.conf
sudo nano /etc/redis/redis.conf

# Key settings:
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Application Deployment

### Docker Deployment

1. **Build Docker Image**:
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 gal_friday && chown -R gal_friday:gal_friday /app
USER gal_friday

# Run application
CMD ["python", "-m", "gal_friday.main"]
```

2. **Docker Compose Configuration**:
```yaml
version: '3.8'

services:
  app:
    build: .
    image: gal-friday:latest
    deploy:
      replicas: 2
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - ENV=production
      - DB_HOST=postgres
      - REDIS_HOST=redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    networks:
      - gal-friday-net

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    networks:
      - gal-friday-net

networks:
  gal-friday-net:
    driver: bridge
```

### Systemd Service

```ini
# /etc/systemd/system/gal-friday.service
[Unit]
Description=Gal-Friday Trading System
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=gal_friday
Group=gal_friday
WorkingDirectory=/opt/gal-friday
Environment="PATH=/opt/gal-friday/venv/bin"
ExecStart=/opt/gal-friday/venv/bin/python -m gal_friday.main
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

## Configuration Management

### Production Configuration

```yaml
# config/production.yaml
environment: production

database:
  host: ${DB_HOST}
  port: 5432
  name: gal_friday_prod
  user: ${DB_USER}
  password: ${DB_PASSWORD}
  pool_size: 20
  max_overflow: 10

influxdb:
  url: ${INFLUX_URL}
  token: ${INFLUX_TOKEN}
  org: gal-friday
  timeout: 30000

redis:
  host: ${REDIS_HOST}
  port: 6379
  password: ${REDIS_PASSWORD}
  db: 0

kraken:
  api_key: ${KRAKEN_API_KEY}
  api_secret: ${KRAKEN_API_SECRET}
  websocket_url: wss://ws.kraken.com
  rest_url: https://api.kraken.com

trading:
  enabled: true
  max_position_size: 10000
  max_daily_trades: 100
  risk_per_trade: 0.02

models:
  registry_path: /data/models
  cache_enabled: true
  cache_ttl: 3600

monitoring:
  metrics_enabled: true
  metrics_port: 9090
  health_check_interval: 60

logging:
  level: INFO
  format: json
  file: /var/log/gal-friday/app.log
  max_size: 100MB
  backup_count: 10

performance:
  memory_limit_mb: 14336  # 14GB
  gc_threshold_mb: 2048
  cache_sizes:
    model: 100
    prediction: 5000
    feature: 2000
```

### Environment Variables

```bash
# .env.production
# Database
DB_HOST=10.0.1.10
DB_USER=gal_friday
DB_PASSWORD=<secure_password>

# InfluxDB
INFLUX_URL=http://10.0.1.11:8086
INFLUX_TOKEN=<secure_token>

# Redis
REDIS_HOST=10.0.1.12
REDIS_PASSWORD=<secure_password>

# Kraken API
KRAKEN_API_KEY=<api_key>
KRAKEN_API_SECRET=<api_secret>

# Application
SECRET_KEY=<secret_key>
API_KEY=<dashboard_api_key>
```

## Security Hardening

### Application Security

1. **API Authentication**:
```python
# Implement API key validation
API_KEY_HEADER = "X-API-Key"
VALID_API_KEYS = set(os.environ.get("API_KEYS", "").split(","))

async def verify_api_key(request: Request):
    api_key = request.headers.get(API_KEY_HEADER)
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
```

2. **Rate Limiting**:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.get("/api/predict")
@limiter.limit("100/minute")
async def predict(request: Request):
    # Handle prediction
```

3. **Input Validation**:
```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    trading_pair: str
    features: Dict[str, float]
    
    @validator('trading_pair')
    def validate_trading_pair(cls, v):
        if v not in ["XRP/USD", "DOGE/USD"]:
            raise ValueError("Invalid trading pair")
        return v
```

### Network Security

1. **Firewall Rules**:
```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 443/tcp # HTTPS
sudo ufw allow from 10.0.1.0/24 to any port 5432  # PostgreSQL
sudo ufw allow from 10.0.1.0/24 to any port 6379  # Redis
sudo ufw enable
```

2. **SSL/TLS Configuration**:
```nginx
server {
    listen 443 ssl http2;
    server_name gal-friday.example.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    location / {
        proxy_pass http://app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring & Alerting

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'gal-friday'
    static_configs:
      - targets: ['app1:9090', 'app2:9090']
    
  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Grafana Dashboards

1. **System Overview Dashboard**:
   - Request rate and latency
   - Error rate
   - CPU and memory usage
   - Database connections
   - Cache hit rates

2. **Trading Performance Dashboard**:
   - P&L over time
   - Win rate
   - Position sizes
   - Model accuracy
   - Prediction latency

3. **Model Lifecycle Dashboard**:
   - Active experiments
   - Drift metrics
   - Retraining jobs
   - Model performance comparison

### Alert Rules

```yaml
# alerts.yml
groups:
  - name: gal-friday
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 > 14000
        for: 10m
        annotations:
          summary: "Memory usage exceeds 14GB"
          
      - alert: ModelDriftDetected
        expr: model_drift_score > 0.15
        for: 30m
        annotations:
          summary: "Significant model drift detected"
          
      - alert: DatabaseConnectionPoolExhausted
        expr: postgresql_connections_active / postgresql_connections_max > 0.9
        for: 5m
        annotations:
          summary: "Database connection pool nearly exhausted"
```

## Performance Tuning

### Application Optimization

1. **Connection Pooling**:
```python
# Database connection pool
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

2. **Caching Strategy**:
```python
# Redis caching for predictions
@cached("prediction", ttl=300)
async def get_prediction(model_id: str, features: Dict[str, float]):
    # Expensive prediction logic
    pass
```

3. **Async Processing**:
```python
# Use asyncio for concurrent operations
async def process_batch(items: List[Any]):
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results
```

### Database Optimization

1. **Indexes**:
```sql
-- Critical indexes for performance
CREATE INDEX idx_orders_timestamp_pair ON orders(created_at, trading_pair);
CREATE INDEX idx_predictions_model_time ON predictions(model_id, created_at);
CREATE INDEX idx_positions_pair_status ON positions(trading_pair, status);
```

2. **Partitioning**:
```sql
-- Partition large tables by date
CREATE TABLE market_data_2024_01 PARTITION OF market_data
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

## Backup & Recovery

### Backup Strategy

1. **Database Backups**:
```bash
# Daily full backup
0 2 * * * pg_dump -U gal_friday gal_friday_prod | gzip > /backup/postgres/daily_$(date +\%Y\%m\%d).sql.gz

# Hourly incremental backup using pg_basebackup
0 * * * * pg_basebackup -D /backup/postgres/incremental -Ft -z -P
```

2. **Application State Backup**:
```bash
# Backup models and configurations
0 3 * * * tar -czf /backup/models_$(date +\%Y\%m\%d).tar.gz /data/models
0 3 * * * tar -czf /backup/config_$(date +\%Y\%m\%d).tar.gz /app/config
```

### Recovery Procedures

1. **Database Recovery**:
```bash
# Restore from backup
gunzip < /backup/postgres/daily_20240115.sql.gz | psql -U gal_friday gal_friday_prod

# Point-in-time recovery
pg_restore -U gal_friday -d gal_friday_prod /backup/postgres/incremental/base.tar
```

2. **Application Recovery**:
```bash
# Restore models
tar -xzf /backup/models_20240115.tar.gz -C /

# Restore configuration
tar -xzf /backup/config_20240115.tar.gz -C /

# Restart services
sudo systemctl restart gal-friday
```

## Operational Procedures

### Deployment Process

1. **Blue-Green Deployment**:
```bash
# 1. Deploy to green environment
docker-compose -f docker-compose.green.yml up -d

# 2. Run health checks
./scripts/health_check.sh green

# 3. Switch traffic
./scripts/switch_traffic.sh green

# 4. Monitor for issues
./scripts/monitor_deployment.sh

# 5. If issues, rollback
./scripts/switch_traffic.sh blue
```

2. **Database Migration**:
```bash
# 1. Backup current database
pg_dump -U gal_friday gal_friday_prod > backup_pre_migration.sql

# 2. Run migrations
python -m gal_friday.dal.migrations.migrate --production

# 3. Verify migration
psql -U gal_friday -d gal_friday_prod -c "SELECT version FROM schema_migrations;"
```

### Monitoring Checklist

**Daily**:
- [ ] Check system health dashboard
- [ ] Review overnight trading performance
- [ ] Check for any critical alerts
- [ ] Verify backup completion
- [ ] Review error logs

**Weekly**:
- [ ] Analyze model performance metrics
- [ ] Review A/B test results
- [ ] Check drift detection reports
- [ ] Update documentation
- [ ] Security scan

**Monthly**:
- [ ] Performance analysis
- [ ] Capacity planning review
- [ ] Security audit
- [ ] Disaster recovery test
- [ ] Update dependencies

### Incident Response

1. **Severity Levels**:
   - **P1**: System down, no trading possible
   - **P2**: Degraded performance, partial functionality
   - **P3**: Minor issues, no immediate impact
   - **P4**: Improvements, optimizations

2. **Response Procedures**:
```
P1 Incident:
1. Page on-call engineer
2. Create incident channel
3. Stop trading if necessary
4. Investigate root cause
5. Implement fix
6. Post-mortem within 24h

P2 Incident:
1. Alert team via Slack
2. Investigate within 1h
3. Implement fix within 4h
4. Document in runbook
```

### Maintenance Windows

**Scheduled Maintenance**:
- Time: Sundays 02:00-04:00 UTC
- Frequency: Monthly
- Activities:
  - System updates
  - Database maintenance
  - Model updates
  - Performance optimization

**Emergency Maintenance**:
- Notify users 15 minutes in advance
- Minimize downtime
- Document all changes
- Post-maintenance verification

## Conclusion

This deployment guide provides a comprehensive framework for deploying and operating Gal-Friday in production. Key success factors:

1. **Automation**: Automate deployment, monitoring, and recovery
2. **Redundancy**: Multiple app servers, database replicas
3. **Monitoring**: Comprehensive metrics and alerting
4. **Security**: Defense in depth approach
5. **Documentation**: Keep runbooks and procedures updated

Regular reviews and updates of these procedures ensure smooth operation and continuous improvement of the system. 