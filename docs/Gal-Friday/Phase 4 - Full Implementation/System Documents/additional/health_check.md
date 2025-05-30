# Health Check Module (`gal_friday/health_check.py`) Documentation

## Module Overview

The `gal_friday/health_check.py` module provides a robust and extensible system for performing comprehensive health checks within the Gal-Friday trading application. Its primary purpose is to monitor the status and responsiveness of the application itself, its critical dependencies (like databases, exchange APIs, caching layers), and underlying system resources. This enables proactive issue detection, facilitates integration with orchestration platforms (e.g., Kubernetes liveness and readiness probes), and contributes to the overall reliability and operational stability of the system.

## Key Data Structures

The module defines several data structures to represent health status and check results:

-   **`HealthStatus(Enum)`**:
    -   An enumeration representing the possible outcomes of a health check for a component or the overall system.
    -   Members:
        -   `HEALTHY`: The component or system is functioning correctly.
        -   `DEGRADED`: The component or system is functioning but with issues or reduced performance; it might be approaching an unhealthy state.
        -   `UNHEALTHY`: The component or system is not functioning correctly and may require intervention.

-   **`HealthCheckResult(Dataclass)`**:
    -   A dataclass to store the outcome of a single, specific health check.
    -   Fields:
        -   `check_name (str)`: The unique name of the health check (e.g., "DatabaseConnection", "CPULoad").
        -   `status (HealthStatus)`: The result of this specific check.
        -   `message (str, Optional)`: A human-readable message providing more details, especially in case of DEGRADED or UNHEALTHY status.
        -   `details (Dict[str, Any], Optional)`: A dictionary for any additional structured information or metrics related to the check (e.g., current CPU load, memory usage).
        -   `timestamp (datetime)`: When the check was performed.
        -   `is_critical (bool)`: Indicates if a failure of this check should render the overall system readiness as UNHEALTHY.

-   **`SystemHealth(Dataclass)`**:
    -   A dataclass that aggregates the results from all individual health checks.
    -   Fields:
        -   `overall_status (HealthStatus)`: The aggregated health status of the entire system. Typically, if any critical check is UNHEALTHY, the overall status is UNHEALTHY. If any check is DEGRADED (and no critical checks are UNHEALTHY), the overall status is DEGRADED. Otherwise, it's HEALTHY.
        -   `service_name (str)`: Name of the service providing the health report (e.g., "GalFridayApp").
        -   `timestamp (datetime)`: Timestamp of when the aggregated health report was generated.
        -   `checks (List[HealthCheckResult])`: A list of `HealthCheckResult` objects from all performed checks.
        -   `system_info (Dict[str, Any], Optional)`: General system information like uptime, version.

## Core Components

### `HealthChecker(ABC)` (Abstract Base Class)

-   **Purpose:** Defines the abstract contract for all specific health check implementations.
-   **Key Attributes & Methods:**
    -   `name (str)`: A unique and descriptive name for the health check.
    -   `critical (bool)`: A flag indicating whether a failure (UNHEALTHY status) of this specific check should cause the overall system readiness to be reported as UNHEALTHY. Non-critical checks might result in a DEGRADED overall status.
    -   `async check() -> HealthCheckResult`: An abstract asynchronous method that concrete checker implementations must override. This method performs the actual health check logic and returns a `HealthCheckResult` object.

### Concrete Checker Implementations

These classes inherit from `HealthChecker` and implement specific checks:

-   **`LivenessChecker(HealthChecker)`**:
    -   **Purpose:** Performs a basic responsiveness check of the application's asyncio event loop.
    -   **Logic:** Its `check()` method might simply return `HealthStatus.HEALTHY` if it's able to execute, indicating the event loop is not blocked indefinitely. More advanced checks could involve scheduling a quick no-op task and ensuring it completes.
    -   **Critical:** Usually `True`.

-   **`MemoryChecker(HealthChecker)`**:
    -   **Purpose:** Monitors system RAM usage against configurable warning and critical thresholds.
    -   **Logic:** Uses `psutil.virtual_memory().percent` to get current memory usage. Compares this against `memory_warning_threshold` (for DEGRADED status) and `memory_critical_threshold` (for UNHEALTHY status) loaded from configuration.
    -   **Critical:** Configurable, often `True` for critical threshold.

-   **`CPUChecker(HealthChecker)`**:
    -   **Purpose:** Monitors system CPU load (average over a short interval) against configurable thresholds.
    -   **Logic:** Uses `psutil.cpu_percent(interval=1)` to get current CPU load. Compares against `cpu_warning_threshold` and `cpu_critical_threshold` from configuration.
    -   **Critical:** Configurable, often `True` for critical threshold.

-   **`DatabaseChecker(HealthChecker)`**:
    -   **Purpose:** Tests connectivity to the configured PostgreSQL database.
    -   **Logic:** Attempts to establish a connection and perform a simple query (e.g., `SELECT 1`) using `asyncpg` directly or via a session from an SQLAlchemy `async_sessionmaker` (if the checker is provided with one).
    -   **Critical:** Usually `True`.

-   **`ExchangeAPIChecker(HealthChecker)`**:
    -   **Purpose:** Tests connectivity to the configured exchange's public API (e.g., Kraken's server time endpoint).
    -   **Logic:** Makes a simple, non-authenticated GET request (e.g., to `/0/public/Time` for Kraken) using `aiohttp`. Checks for a successful HTTP status code (e.g., 200 OK) and valid response structure.
    -   **Critical:** Usually `True`.

-   **`RedisChecker(HealthChecker)`**: (Optional dependency)
    -   **Purpose:** Tests connectivity to the configured Redis instance if Redis is used for caching or other purposes.
    -   **Logic:** Attempts to connect to Redis using `aioredis` and execute a simple command like `PING`.
    -   **Critical:** Configurable, depends on how critical Redis is to the application's core functionality.

-   **`ComponentChecker(HealthChecker)`**:
    -   **Purpose:** A generic checker that allows custom, application-specific health check functions to be easily integrated into the health check system.
    -   **Logic:** Takes an asynchronous check function (`Callable[[], Awaitable[HealthCheckResult]]`) as a parameter during its initialization. Its `check()` method simply `await`s this provided function.
    -   **Critical:** Determined by the `critical` flag passed during its registration.

### `HealthCheckService` Class

-   **Purpose:** Orchestrates all registered health checkers, executes them (periodically or on demand), and provides an aggregated system health status.
-   **Initialization (`__init__`):**
    -   **Parameters:** `config_manager (ConfigManager)`, `logger_service (LoggerService)`, and potentially direct dependencies like an SQLAlchemy `async_sessionmaker` if `DatabaseChecker` is initialized here.
    -   **Actions:**
        -   Stores `config_manager` and `logger_service`.
        -   Initializes an empty list `self._checkers: List[HealthChecker]`.
        -   Loads health check configurations (intervals, thresholds).
        -   Instantiates and registers a set of default checkers based on configuration (e.g., `LivenessChecker`, `MemoryChecker` if `health.check_memory` is true, `CPUChecker`, `DatabaseChecker`, `ExchangeAPIChecker`, `RedisChecker` if enabled).
        -   Initializes `_periodic_check_task = None` and `_last_system_health: Optional[SystemHealth] = None`.

-   **`add_component_check(name: str, check_func: Callable[[], Awaitable[HealthCheckResult]], critical: bool) -> None`**:
    -   Allows other parts of the application to dynamically register custom health checks by providing a name, an awaitable check function, and a criticality flag.
    -   Creates a `ComponentChecker` instance with these parameters and adds it to `self._checkers`.

-   **`async check_health() -> SystemHealth`**:
    -   Asynchronously executes the `check()` method of all registered `HealthChecker` instances concurrently using `asyncio.gather()`.
    -   Aggregates the `HealthCheckResult`s from all checkers.
    -   Determines the `overall_status`:
        -   `UNHEALTHY` if any critical check is `UNHEALTHY`.
        -   `DEGRADED` if any check is `DEGRADED` (and no criticals are `UNHEALTHY`).
        -   `HEALTHY` otherwise.
    -   Constructs and returns a `SystemHealth` object.
    -   Updates `self._last_system_health` with the new result.

-   **`async get_liveness() -> Tuple[HealthStatus, HealthCheckResult]`**:
    -   Specifically runs (or gets the latest result from) the `LivenessChecker`.
    -   Returns a simple status (`HealthStatus`) and the `HealthCheckResult`, suitable for Kubernetes liveness probes. The HTTP endpoint would typically return 200 OK if HEALTHY, 503 Service Unavailable otherwise.

-   **`async get_readiness() -> SystemHealth`**:
    -   Returns the system readiness status, typically by returning the `self._last_system_health` (if periodic checks are enabled and a recent result is available) or by triggering a new `await self.check_health()`.
    -   Suitable for Kubernetes readiness probes. The HTTP endpoint would return 200 OK if `overall_status` is HEALTHY, 503 otherwise.

-   **`async start_periodic_checks() -> None`**:
    -   If `health.check_interval_seconds` is configured to be positive, creates and starts an asyncio background task that calls `self.check_health()` repeatedly at this interval.
    -   Stores the task in `self._periodic_check_task`.

-   **`async stop_periodic_checks() -> None`**:
    -   If `_periodic_check_task` is running, cancels it and awaits its completion.

## Usage and Integration

-   **HTTP Endpoints for Orchestrators:** The `HealthCheckService` is primarily designed to be consumed by HTTP endpoints exposed by the main Gal-Friday application (e.g., if it's a FastAPI or AIOHTTP web application).
    -   An endpoint like `/health/live` would call `health_service.get_liveness()` and return an appropriate HTTP status code (200 for HEALTHY, 503 for UNHEALTHY). This is used by Kubernetes (or similar orchestrators) to determine if the application instance needs to be restarted.
    -   An endpoint like `/health/ready` would call `health_service.get_readiness()` and return an HTTP status based on `SystemHealth.overall_status`. This tells the orchestrator whether the application instance is ready to receive traffic or perform work.
-   **Internal Monitoring:** The `MonitoringService` might also consume the `SystemHealth` status from `HealthCheckService` as one of its inputs for making decisions about the overall system state or for triggering specific alerts via the `AlertingSystem`.
-   **Startup Integration:** The `HealthCheckService` is typically instantiated and started by the main application orchestrator (`GalFridayApp` in `main.py`) during the application startup sequence. Custom component checks can be added by other services during their own initialization.

## Dependencies

-   **Standard Libraries:** `asyncio`, `datetime`, `enum`, `http` (for status codes, if directly used).
-   **Third-Party Libraries:**
    -   `psutil`: For system resource monitoring (CPU, memory).
    -   `aiohttp`: Used by `ExchangeAPIChecker` for making asynchronous HTTP requests.
    -   `asyncpg`: Used by `DatabaseChecker` for direct PostgreSQL connection testing.
    -   `aioredis` (Optional): Used by `RedisChecker` if Redis integration is enabled.
-   **Core Application Modules:**
    -   `gal_friday.config_manager.ConfigManager` (or a protocol providing configuration access).
    -   `gal_friday.logger_service.LoggerService`.

## Configuration

The `HealthCheckService` and its checkers rely on configurations typically found under a `health` section in `config.yaml`, as well as configurations for dependencies like database and exchange APIs:

-   **`health.check_interval_seconds (int)`**: Interval for periodic background health checks.
-   **`health.check_memory (bool)`**: Enable/disable memory checks.
-   **`health.memory_warning_threshold (float)`**: Memory usage percentage for DEGRADED status.
-   **`health.memory_critical_threshold (float)`**: Memory usage percentage for UNHEALTHY status.
-   **`health.check_cpu (bool)`**: Enable/disable CPU checks.
-   **`health.cpu_warning_threshold (float)`**: CPU load percentage for DEGRADED status.
-   **`health.cpu_critical_threshold (float)`**: CPU load percentage for UNHEALTHY status.
-   **`database.connection_string (str)`**: Used by `DatabaseChecker` to connect to PostgreSQL.
-   **`exchange.api_url (str)`** (or a specific public endpoint like `kraken.public_api_url`): Used by `ExchangeAPIChecker`.
-   **`redis.url (str)`** (and other Redis connection params): Used by `RedisChecker` if enabled.

## Adherence to Standards

The `health_check.py` module implements standard operational practices crucial for modern distributed systems and microservices:
-   **Liveness Probes:** Allow orchestration platforms to detect and restart unresponsive application instances.
-   **Readiness Probes:** Enable orchestrators to route traffic only to application instances that are fully initialized and capable of serving requests or performing work.
-   **Dependency Checking:** Proactively monitors the health of critical external and internal dependencies.
This structured approach to health monitoring is essential for building a resilient, self-aware, and operationally manageable trading system.
