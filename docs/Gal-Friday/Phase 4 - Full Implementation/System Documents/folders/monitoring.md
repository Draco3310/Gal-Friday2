# Monitoring Folder (`gal_friday/monitoring`) Documentation

## Folder Overview

The `gal_friday/monitoring` folder is dedicated to providing a comprehensive suite of tools and services for system observability, real-time monitoring, performance tracking, and alerting within the Gal-Friday trading system. These components are essential for maintaining operational awareness, ensuring system stability, diagnosing issues, and making informed decisions about the bot's performance and health. The folder typically includes a backend for a web-based dashboard, services for data aggregation, an alerting system, and performance monitoring utilities.

## Key Modules and Their Roles

The `monitoring` folder comprises several key modules that work together to deliver a full observability solution:

### `dashboard_backend.py` (FastAPI application)

-   **Purpose:** Implements the backend server for the Gal-Friday monitoring dashboard, typically using the FastAPI framework.
-   **Key Features & Components:**
    -   **REST API Endpoints:** Provides a set of HTTP API endpoints that the dashboard's frontend can query to fetch data. This includes endpoints for:
        -   Current system status (e.g., running, halted, from `MonitoringService`).
        -   Portfolio state summaries (equity, P&L, positions, from `PortfolioManager`).
        -   Key performance metrics.
        -   Recent orders and trade history.
        -   Model registry information and experiment status (from `model_lifecycle` components).
    -   **WebSocket Endpoint (`/ws`):** Offers a WebSocket connection endpoint for broadcasting real-time updates from the Gal-Friday system to connected dashboard clients. This enables live updates without requiring clients to continuously poll the API.
    -   **`ConnectionManager` (internal class):** Manages active WebSocket client connections, including handling new connections, disconnections, and broadcasting messages to all connected clients or specific clients.
    -   **`MetricsCollector` (internal class or integration):** Gathers and aggregates metrics from various parts of the system. This might involve in-memory aggregation or integration with a time-series database like InfluxDB or a caching layer like Redis for more scalable metric collection.
    -   **`EventBroadcaster` (internal class or logic):** Subscribes to relevant events from the `PubSubManager` (e.g., `MarketDataL2Event`, `TradeSignalApprovedEvent`, `ExecutionReportEvent`, `SystemStateEvent`, `LogEvent`). When these events are received, it formats them appropriately and pushes them to connected WebSocket clients via the `ConnectionManager`.
-   **Importance:** Serves as the data hub for the visual monitoring dashboard, enabling operators to see what the system is doing in real-time and access historical performance data.

### `dashboard_pages.py` (`EnhancedDashboardPages` class or similar)

-   **Purpose:** Responsible for generating the HTML content for various views or pages of the monitoring dashboard, especially if server-side rendering (e.g., using Jinja2 templates with FastAPI) is employed for some parts of the dashboard.
-   **Functionality:**
    -   Contains methods or classes that correspond to different dashboard sections (e.g., main overview, model management view, active experiments, reconciliation status, retraining job monitoring).
    -   Fetches data from the `DashboardService` (see below), and potentially directly from other system services like `PortfolioManager`, `model_lifecycle.Registry`, or `model_lifecycle.ExperimentManager`.
    -   Renders this data into HTML templates to create dynamic web pages.
    -   If the dashboard is primarily a Single Page Application (SPA) built with a JavaScript framework, this module might be less about full page rendering and more about serving the initial HTML shell or specific template snippets.
-   **Importance:** Provides the structure and content for the user-facing monitoring interface, making complex system data accessible and understandable.

### `dashboard_service.py` (`DashboardService` class)

-   **Purpose:** Acts as an intermediary service that aggregates, caches, and provides various system metrics and status information specifically for consumption by the `dashboard_backend.py` (API endpoints) and `dashboard_pages.py`.
-   **Functionality:**
    -   **System Health Aggregation:** Collects overall system health indicators, such as application uptime, CPU/memory usage (potentially using `psutil` or by querying `PerformanceMonitor`), and critical error counts.
    -   **Portfolio Summary:** (May have placeholder logic in some initial versions) Fetches and formats portfolio summaries from `PortfolioManager` (e.g., total equity, daily P&L, number of open positions).
    -   **Model Statistics:** Gathers information about deployed ML models from `model_lifecycle.Registry` (e.g., current production model versions, their registration dates, key performance indicators from training).
    -   **Experiment Data:** Retrieves status and performance of ongoing A/B tests from `model_lifecycle.ExperimentManager`.
    -   **WebSocket Connection Metrics:** Provides information about active WebSocket clients connected to the `dashboard_backend`.
    -   **Alert Summaries:** May provide a summary of recent or active alerts from the `AlertingSystem`.
-   **Importance:** Decouples the dashboard's data presentation layer from the direct complexities of querying multiple underlying services, providing a cleaner interface for data retrieval and potentially caching data to improve dashboard responsiveness.

### `alerting_system.py` (`AlertingSystem` class, `Alert` dataclass, `AlertRecipient` dataclass, `AlertDeliveryChannel` implementations)

-   **Purpose:** Manages the generation, filtering, and dispatch of alerts to notify operators or developers of important system events, errors, or predefined conditions.
-   **Key Components & Functionality:**
    -   **`Alert` (Dataclass):** Defines the structure of an alert, including severity (e.g., INFO, WARNING, ERROR, CRITICAL), message, source, timestamp, and optional context.
    -   **`AlertRecipient` (Dataclass):** Defines a recipient for alerts, including their contact details for various channels.
    -   **`AlertDeliveryChannel` (ABC/Interface and Implementations):**
        -   Defines a common interface for sending alerts.
        -   Concrete implementations for different notification channels:
            -   **Email:** Using services like SendGrid or standard SMTP.
            -   **SMS:** Using services like Twilio.
            -   **Discord:** Posting messages to a Discord channel via webhooks or a bot.
            -   **Slack:** Posting messages to a Slack channel via webhooks or a bot.
    -   **`AlertingSystem` Class:**
        -   **Configuration:** Loads alert rules, recipient lists, channel configurations, quiet hours, and deduplication settings from `ConfigManager`.
        -   **Alert Processing:** Receives alert triggers (e.g., from `MonitoringService` when a threshold is breached, or directly from other services publishing `AlertEvent`s or calling an `alert` method).
        -   **Filtering & Deduplication:** Filters alerts based on severity, configured quiet hours, and implements deduplication logic to avoid alert fatigue (e.g., suppress identical alerts within a certain time window).
        -   **Dispatching:** Routes processed alerts to the appropriate `AlertDeliveryChannel` implementations for delivery to configured `AlertRecipient`s.
-   **Importance:** Ensures that key stakeholders are promptly notified of critical system events, malfunctions, or significant trading outcomes, enabling timely intervention if required.

### `performance_monitor.py` (`PerformanceMonitor` class, `MetricCollector`, `SystemMonitor`, `PerformanceTimer`)

-   **Purpose:** Dedicated to tracking, analyzing, and reporting on the performance of the Gal-Friday application itself and the underlying system resources it uses.
-   **Key Components & Functionality:**
    -   **`MetricCollector` Class:**
        -   A utility for collecting and storing time-series samples of specific metrics (e.g., event processing latency, queue sizes, API call durations).
        -   Often uses a rolling window or fixed-size buffer for storing recent samples.
        -   May calculate basic statistics on these samples (average, min, max, percentiles).
    -   **`PerformanceTimer` Class (Context Manager):**
        -   A utility (e.g., `with PerformanceTimer("my_function_execution_time", self.metric_collector):`) to easily time the execution of specific blocks of code.
        -   Automatically records the duration into a `MetricCollector` or logs it.
    -   **`SystemMonitor` Class:**
        -   Collects OS-level system metrics using `psutil`:
            -   CPU utilization (overall and per-core).
            -   Memory usage (total, available, percent used).
            -   Disk I/O (read/write bytes, operations).
            -   Network I/O (bytes sent/received, connections).
        -   May also monitor the health and responsiveness of the asyncio event loop (e.g., by measuring loop iteration lag).
    -   **`PerformanceMonitor` Class (Orchestrator):**
        -   Initializes and manages `MetricCollector`s and `SystemMonitor`.
        -   Orchestrates the periodic collection or aggregation of performance data.
        -   Can check collected metrics against predefined performance thresholds and trigger alerts (via `AlertingSystem`) if thresholds are breached (e.g., high CPU, excessive event loop lag).
        -   May subscribe to `PubSubManager` events (e.g., before and after processing certain critical events) to calculate end-to-end processing times.
        -   Can provide an interface (e.g., an API endpoint via `dashboard_backend` or a Prometheus-compatible scrape endpoint) for exporting collected performance metrics.
-   **Importance:** Helps in identifying performance bottlenecks, ensuring efficient resource utilization, understanding system latencies, and maintaining the overall responsiveness and stability of the trading application.

### `auth.py`

-   **Purpose:** Provides authentication mechanisms for securing access to the monitoring dashboard and its backend API.
-   **Functionality:**
    -   Implements API key-based authentication, commonly using FastAPI's `HTTPBearer` security scheme or similar token-based approaches.
    -   Defines functions or dependencies to verify provided API keys against a stored list of valid keys (which should be securely configured, e.g., via `ConfigManager` and environment variables).
    -   Unauthorized access attempts are rejected with appropriate HTTP error codes.
-   **Importance:** Protects sensitive operational data and control functions exposed through the monitoring dashboard from unauthorized access.

## Interactions and Importance

The components within the `gal_friday/monitoring` folder are crucial for providing **observability** into the Gal-Friday system, which is a cornerstone of operational excellence and reliability, especially for an automated trading bot.

-   **Real-Time Insight:** The **dashboard** (powered by `dashboard_backend.py`, `dashboard_pages.py`, and `dashboard_service.py`) offers operators a visual, real-time window into the bot's activities, current state, market conditions, and performance.
-   **Proactive Issue Detection:** The **alerting system** (`alerting_system.py`) ensures that any significant events, errors, or breaches of operational thresholds are promptly communicated to the relevant personnel, enabling quick responses.
-   **Performance Optimization:** The **performance monitoring** tools (`performance_monitor.py`) help in identifying system bottlenecks, understanding resource consumption, and ensuring that the application runs efficiently and meets its latency requirements.
-   **Data-Driven Operations:** These monitoring components collect and present data that is vital for:
    -   Diagnosing problems when they occur.
    -   Understanding the system's behavior under different market conditions.
    -   Making informed decisions about strategy adjustments, resource allocation, or system scaling.
-   **Dependency on Core Services:** The monitoring tools rely heavily on data and events from many other core services:
    -   `PortfolioManager`: For portfolio value, P&L, positions.
    -   `model_lifecycle.Registry` and `ExperimentManager`: For model and experiment status.
    -   `PubSubManager`: For receiving a wide array of system events that are broadcast to the dashboard or trigger alerts.
    -   `MonitoringService` (the one at `gal_friday/monitoring_service.py`): For the global HALT status and other system-level health indicators.
    -   `LoggerService`: Logs from all services provide detailed context for diagnostics.

Without a robust monitoring suite, operating an automated trading system like Gal-Friday would be akin to flying blind, significantly increasing operational risk.

## Adherence to Standards

The development of a dedicated `monitoring` folder with components for dashboards, alerting, and performance tracking reflects a commitment to operational best practices. These tools are essential for maintaining system reliability, availability, and performance, aligning with principles often found in Site Reliability Engineering (SRE) and robust system operations. Secure access via `auth.py` is also a standard requirement for protecting sensitive operational interfaces.
