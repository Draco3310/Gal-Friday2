Database Configuration

The production configuration specifies PostgreSQL and InfluxDB details loaded from environment variables, showing typical keys like DB_HOST, DB_USER, DB_PASSWORD, INFLUX_URL, and INFLUX_TOKEN{line_range_start=276 line_range_end=312 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L276-L312"}{line_range_start=340 line_range_end=367 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L340-L367"}.

gal_friday/database.py uses ConfigManager to build the SQLAlchemy connection string at runtime, validating required fields (host, port, name, username, password) before creating the async engine with pooling options.

Schema files (e.g., db/schema/001_create_logs_table.sql) define tables such as logs with indexes, indicating database migrations are expected via Alembic.

Non‑functional requirements specify access to running PostgreSQL v13+ and InfluxDB v2.x services as mandatory infrastructure{line_range_start=154 line_range_end=155 path=docs/Gal-Friday/Phase 1 - Requirements Analysis & Planning/srs_gal_friday_v0.1.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 1 - Requirements Analysis & Planning/srs_gal_friday_v0.1.md#L154-L155"}{line_range_start=405 line_range_end=406 path=docs/Gal-Friday/Phase 1 - Requirements Analysis & Planning/srs_gal_friday_v0.1.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 1 - Requirements Analysis & Planning/srs_gal_friday_v0.1.md#L405-L406"}.

The documentation does not state strict response‑time SLAs, but performance tuning recommendations include connection pooling and index usage for efficient queries{line_range_start=518 line_range_end=524 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L518-L524"}{line_range_start=528 line_range_end=548 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L528-L548"}.

API Integration

Production configuration expects Kraken credentials via KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables{line_range_start=305 line_range_end=309 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L305-L309"}{line_range_start=360 line_range_end=363 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L360-L363"}.

The configuration uses websocket_url: wss://ws.kraken.com and rest_url: https://api.kraken.com (supporting the main endpoints) with options for sandbox or other endpoints configurable via YAML, so a sandbox/testnet setup is feasible.

Rate limits are addressed through a configurable rate_limit_delay and max_requests_per_minute in the backtesting data-loading section of config.yaml, implying the application is prepared for Kraken’s standard rate limits.

Technical Indicators

The enabled features in config.yaml include RSI, MACD, Bollinger Bands, and ATR with configurable periods and parameters.

docs/technical_analysis_migration.md describes a pluggable technical analysis module supporting production calculations (via pandas-ta), a testing stub, and an optional TA-Lib wrapper with automatic fallback if TA-Lib is not available.

Indicators can be calculated on demand; there is also mention of potential caching and parallelization in future enhancements.

Testing Environment

The setup script can provision local PostgreSQL, InfluxDB, and Redis instances using Docker Compose for development and testing.

Unit tests mock configuration and database connections (e.g., tests/dal/test_connection_pool.py) without requiring real credentials, demonstrating the preferred approach of mocking external dependencies during tests.

Documentation references using test databases or mocks for repositories, supporting maintainability and testability of the DAL{line_range_start=126 line_range_end=126 path=docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/folders/dal.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/folders/dal.md#L126-L126"}.

Error Handling & Monitoring

The monitoring subsystem defines an AlertingSystem supporting channels such as Email (SendGrid), SMS (Twilio), Discord, and Slack for notifying stakeholders about critical events{line_range_start=52 line_range_end=67 path=docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/folders/monitoring.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/folders/monitoring.md#L52-L67"}.

The production deployment guide includes Prometheus configuration and Grafana dashboards, along with alert rules for high error rates, memory usage, model drift, and database connection exhaustion{line_range_start=452 line_range_end=485 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L452-L485"}{line_range_start=496 line_range_end=525 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L496-L525"}.

Logging can route to console, rotating JSON files, PostgreSQL, and InfluxDB, ensuring detailed audit trails with sensitive data filtering{line_range_start=1 line_range_end=29 path=docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/modules/logger_service.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/modules/logger_service.md#L1-L29"}{line_range_start=30 line_range_end=44 path=docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/modules/logger_service.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/modules/logger_service.md#L30-L44"}.

Deployment Considerations

The project supports Docker deployment with multi-stage builds, Docker Compose for orchestration, and Kubernetes manifests for production deployment, enabling horizontal scaling of stateless components (replicas){line_range_start=174 line_range_end=208 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L174-L208"}{line_range_start=210 line_range_end=236 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L210-L236"}{line_range_start=426 line_range_end=426 path=docs/Gal-Friday/Phase 1 - Requirements Analysis & Planning/srs_gal_friday_v0.1.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 1 - Requirements Analysis & Planning/srs_gal_friday_v0.1.md#L426-L426"}.

Network security guidance includes firewall rules and TLS configuration to secure database connections and application traffic{line_range_start=414 line_range_end=446 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L414-L446"}{line_range_start=432 line_range_end=447 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L432-L447"}.

Secrets are retrieved from environment variables or a secrets manager, with optional encryption at rest to protect sensitive data{line_range_start=59 line_range_end=59 path=docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/folders/utils.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/folders/utils.md#L59-L59"}{line_range_start=343 line_range_end=367 path=docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md git_url="https://github.com/Draco3310/Gal-Friday2/blob/main/docs/Gal-Friday/Phase 4 - Full Implementation/production_deployment_guide.md#L343-L367"}.

These references provide the necessary context for implementing the outstanding work items. The codebase expects configuration via YAML files supplemented with environment variables for secrets, uses Docker/Docker Compose (and optionally Kubernetes) for deployment, relies on PostgreSQL and InfluxDB for storage, integrates with Kraken APIs using standard credentials, and features a pluggable technical analysis system with fallback options. Monitoring and alerting leverage Prometheus/Grafana and an extensible AlertingSystem to deliver notifications through multiple channels