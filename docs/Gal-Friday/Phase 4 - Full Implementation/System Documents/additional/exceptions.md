# Gal-Friday Custom Exceptions (`gal_friday/exceptions.py`)

## Module Overview

The `gal_friday/exceptions.py` module defines a custom exception hierarchy specifically for the Gal-Friday trading application. The primary purpose of this hierarchy is to provide a structured and specific way to handle errors that can occur in various parts of the system. By defining custom exceptions, the application can differentiate between various error conditions, allowing for more precise error handling, logging, and debugging. This leads to a more robust and maintainable system.

## Base Exception

All custom exceptions defined for the Gal-Friday application inherit from a single base exception.

-   **`GalFridayError(Exception)`**:
    -   This is the root exception class for all application-specific errors.
    -   Catching `GalFridayError` can be used to handle any error originating from within the Gal-Friday application logic, distinguishing them from standard Python exceptions or third-party library exceptions.
    -   It typically takes a message and an optional `original_exception` or `details` argument to wrap the root cause.

## Key Exception Categories and Specific Exceptions

The custom exceptions are organized into logical categories to better reflect the area or nature of the error.

### Setup Errors (`SetupError(GalFridayError)`)

Errors that occur during the initial setup, configuration, or initialization phases of the application or its components.

-   **`DependencyMissingError(SetupError)`**:
    -   Raised when a required component, service, or configuration item is not available or has not been properly injected during the setup of another component.
    -   Example: `PortfolioManager` requires `MarketPriceService`, but it's not provided.
-   **`ComponentInitializationError(SetupError)`**:
    -   A general error indicating that a component or service failed to initialize correctly for reasons not covered by more specific setup exceptions.
-   **`CriticalExit(SystemExit)`**:
    -   This is a special category of setup errors that are so critical that the application cannot reasonably continue to operate. These exceptions inherit directly from `SystemExit` (and may also inherit from `SetupError` or `GalFridayError` for classification).
    -   When raised, they are intended to cause a graceful but immediate termination of the application, usually with a specific exit code.
    -   **Subclasses of `CriticalExit`:**
        -   `ServiceInstantiationFailedExit(CriticalExit)`: Raised when a core service (e.g., `PredictionService`, `RiskManager`) fails to instantiate during application startup.
        -   `ConfigurationLoadingFailedExit(CriticalExit)`: Raised if the `ConfigManager` fails to load the essential `config.yaml` file.
        -   `PubSubManagerStartFailedExit(CriticalExit)`: Raised if the `PubSubManager` fails to start its internal processes.
        -   `ExecutionHandlerInstantiationFailedExit(CriticalExit)`: If the specified `ExecutionHandler` cannot be created.
        -   `MarketPriceServiceCriticalFailureExit(CriticalExit)`: If the `MarketPriceService` encounters a fatal error during setup or cannot connect to its data source.
        -   `PortfolioManagerInstantiationFailedExit(CriticalExit)`: If `PortfolioManager` or its sub-components fail to initialize.
        -   `LoggerServiceInstantiationFailedExit(CriticalExit)`: If the `LoggerService` cannot be set up (e.g., cannot open log files or connect to the log database).

### Operational Errors (`OperationalError(GalFridayError)`)

Errors related to the general operation or runtime behavior of the system, not fitting into more specific categories like API or Database errors.

-   **`UnsupportedModeError(OperationalError)`**:
    -   Raised if an invalid or unsupported operational mode is specified for the application or a particular service (e.g., providing "live_trading" to a component that only supports "simulation").

### Configuration Errors (`ConfigurationError(GalFridayError)`)

Errors specifically related to invalid or missing application configuration values discovered after the initial loading phase.

-   **`InvalidLoggerTableNameError(ConfigurationError)`**:
    -   A specific error raised if the configured table name for database logging is invalid or not found (though typically the table name is fixed by the ORM model). This might be more relevant if table names were highly dynamic.

### API Interaction Errors (`APIError(GalFridayError)`)

Errors encountered while interacting with external APIs, such as those from cryptocurrency exchanges.

-   **`APIError(GalFridayError)`**:
    -   A general exception for issues related to external API communication.
    -   Typically includes attributes like `service_name` (e.g., "Kraken"), `http_status_code`, `api_error_code`, and `error_message` from the API response.
    -   Example: Used by `KrakenExecutionHandler` or `KrakenMarketPriceService` when an API request fails.

### Data Validation Errors (`DataValidationError(GalFridayError)`)

Errors that occur when input data, processed data, or data retrieved from external sources fails validation checks.

-   **`DataValidationError(GalFridayError)`**:
    -   Raised when data does not conform to expected formats, types, ranges, or business rules.
    -   Example: An incoming `ExecutionReportEvent` payload is missing critical fields or has incorrectly formatted numbers.

### Execution Errors (`ExecutionError(GalFridayError)`, `ExecutionHandlerError(ExecutionError)`)

Errors specifically related to the trade execution process or the operation of an `ExecutionHandler`.

-   **`ExecutionHandlerError(ExecutionError)`**: Base class for errors from an `ExecutionHandler`.
-   **`ExecutionHandlerAuthenticationError(ExecutionHandlerError)`**:
    -   Raised when authentication with the exchange API fails (e.g., invalid API key, incorrect signature).
-   **`ExecutionHandlerNetworkError(ExecutionHandlerError, NetworkError)`**:
    -   Raised for network-related issues encountered specifically during execution operations (e.g., placing an order, fetching balances). Inherits from both `ExecutionHandlerError` and `NetworkError`.
-   **`ExecutionHandlerCriticalError(ExecutionHandlerError)`**:
    -   For other critical, unrecoverable failures within an `ExecutionHandler` that are not related to authentication or generic network issues (e.g., unexpected API behavior, internal state corruption).

### Network Errors (`NetworkError(GalFridayError)`)

General errors related to network connectivity problems, not specific to a particular service like an `ExecutionHandler`.

-   **`NetworkError(GalFridayError)`**:
    -   Can be raised by any component that makes network calls if it encounters issues like DNS failures, connection timeouts (if not covered by `GalFridayTimeoutError`), or other transport-level problems.

### Database Errors (`DatabaseError(GalFridayError)`)

Errors related to database operations, whether using SQLAlchemy for a relational database or an InfluxDB client for time-series data.

-   **`DatabaseError(GalFridayError)`**:
    -   Wraps exceptions from database client libraries (e.g., `SQLAlchemyError` from SQLAlchemy, errors from `influxdb-client`).
    -   Provides a common exception type for issues like connection failures, query execution errors, constraint violations, or transaction rollbacks.

### General Operational Failures (`OperationalError` or direct `GalFridayError` subclasses)

A category for various runtime issues that can occur.

-   **`GalFridayTimeoutError(OperationalError)`**:
    -   Raised when a specific operation or request exceeds its allocated timeout period.
    -   Example: A request to an external API times out, or an internal asynchronous task does not complete in time.
-   **`AuthenticationError(GalFridayError)`**:
    -   A more general authentication error that could be used for internal system authentication or if a more specific `ExecutionHandlerAuthenticationError` is not appropriate.
-   **`GalFridayPermissionError(GalFridayError)`**:
    -   Raised when an operation is attempted without the necessary permissions (e.g., file system access, restricted API endpoint).
-   **`RateLimitError(APIError)`**: (Often a subclass of `APIError`)
    -   Raised specifically when an external API indicates that a rate limit has been exceeded.
    -   Services should handle this by implementing backoff strategies.
-   **`ServiceUnavailableError(OperationalError)`**:
    -   Raised if a required internal or external service is temporarily unavailable or not responding.
-   **`PriceNotAvailableError(OperationalError)`**:
    -   Raised by `MarketPriceService` or other components when a required market price for an asset cannot be obtained.
-   **`InsufficientFundsError(OperationalError)`**:
    -   Raised by `PortfolioManager`, `FundsManager`, or `RiskManager` when an operation (like placing a trade) cannot be completed due to a lack of available funds.

### Type Errors (`GalFridayTypeError(TypeError, GalFridayError)`)

Custom type errors for more specific type mismatch scenarios within the application logic.

-   **`UnsupportedParamsTypeError(GalFridayTypeError)`**:
    -   A specialized `TypeError` that can be raised when function or method parameters do not match expected types, particularly useful in adapters or interface implementations where type consistency is critical.

## Importance and Usage

Employing a custom exception hierarchy like the one defined in `gal_friday/exceptions.py` offers several significant advantages for the development and maintenance of a complex application like Gal-Friday:

1.  **Improved Error Identification:** Custom exceptions allow developers to quickly pinpoint both the location (which part of the system) and the nature (what kind of error) of a problem. This is far more informative than relying solely on generic built-in exceptions like `ValueError` or `RuntimeError`.
2.  **Granular Error Handling:** Callers can write more specific `try...except` blocks to handle different error conditions in different ways. For example, a `RateLimitError` might trigger a backoff-and-retry mechanism, while a `DataValidationError` might lead to logging the bad data and skipping the current item.
    ```python
    try:
        # some operation
        pass
    except RateLimitError:
        # handle rate limit
    except DataValidationError:
        # handle bad data
    except APIError:
        # handle other API issues
    except GalFridayError:
        # handle any other application-specific error
    except Exception:
        # handle unexpected errors
    ```
3.  **Standardized Error Reporting and Logging:** When logging errors, the type of the custom exception immediately provides context. This can be used to categorize errors, trigger specific alerts, or generate more meaningful error messages for users or operators.
4.  **Clearer Intent:** Defining exceptions like `InsufficientFundsError` or `PriceNotAvailableError` makes the code's intent and potential failure modes much clearer to anyone reading it.
5.  **Centralized Error Definitions:** Having all custom exceptions in one module makes it easier to manage and understand the types of errors the system can produce.
6.  **Controlled Application Termination:** `CriticalExit` exceptions provide a standardized way to terminate the application gracefully when unrecoverable errors occur during startup, ensuring that appropriate messages are logged.

## Adherence to Standards

The use of a well-defined custom exception hierarchy is a widely adopted best practice in software engineering for building robust and maintainable applications. It aligns with principles of clear error communication, structured error handling, and improved diagnosability. This approach helps in creating a more resilient system that can gracefully handle a variety of failure scenarios.
