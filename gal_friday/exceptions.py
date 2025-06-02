"""Standard exceptions for the Gal Friday application."""

from typing import Any


class GalFridayError(Exception):
    """Base class for Gal-Friday specific errors."""


class SetupError(GalFridayError):
    """Base class for errors during application setup."""


class DependencyMissingError(SetupError):
    """Errors when a required dependency or component is unavailable for a setup step."""

    def __init__(self, component: str, dependency: str, message: str | None = None) -> None:
        """Initialize DependencyMissingError.

        Args:
            component: The component that has a missing dependency
            dependency: The name of the missing dependency
            message: Optional custom error message
        """
        self.component = component
        self.dependency = dependency
        if message is None:
            message = (
                f"Dependency '{dependency}' is missing or "
                f"unavailable for component '{component}'."
            )
        super().__init__(message)


class ComponentInitializationError(SetupError):
    """Errors during the initialization of a critical component."""

    def __init__(
        self,
        component_name: str,
        details: str | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize ComponentInitializationError.

        Args:
            component_name: Name of the component that failed to initialize
            details: Optional detailed error information
            message: Optional custom error message
        """
        self.component_name = component_name
        self.details = details
        if message is None:
            message = f"Failed to initialize component: '{component_name}'."
            if details:
                message += f" Details: {details}"
        super().__init__(message)


class OperationalError(GalFridayError):
    """Base class for errors during application operation."""


class PositionNotFoundError(OperationalError):
    """Exception raised when a trading position is not found."""

    def __init__(
        self,
        trading_pair: str | None = None,
        position_id: str | None = None,
        message: str | None = None
    ) -> None:
        """Initialize PositionNotFoundError."""
        if message is None:
            if trading_pair:
                message = f"Position not found for trading pair: {trading_pair}"
            elif position_id:
                message = f"Position not found with ID: {position_id}"
            else:
                message = "Position not found"
        super().__init__(message)
        self.trading_pair = trading_pair
        self.position_id = position_id


class UnsupportedModeError(OperationalError, ValueError):
    """Error when an unsupported operational mode is encountered.

    Inherits from ValueError for semantic compatibility where applicable.
    """

    def __init__(
        self,
        mode: str,
        supported_modes: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize UnsupportedModeError.

        Args:
            mode: The mode that was not supported
            supported_modes: Optional list of supported modes
            message: Optional custom error message
        """
        self.mode = mode
        self.supported_modes = supported_modes
        if message is None:
            message = f"Unsupported operational mode: '{mode}'."
            if supported_modes:
                modes = ", ".join(supported_modes)
                message += f" Supported modes are: {modes}."
            else:
                message += " Please ensure the mode is valid."
        super().__init__(message)


# class ServiceInitializationError(ComponentInitializationError):
#     """Errors during the initialization phase of a specific service."""
#     pass


class ConfigurationError(GalFridayError):
    """Exception raised for errors in the configuration."""


class InvalidLoggerTableNameError(ConfigurationError):
    """Exception raised when an invalid table name is provided for logging."""

    def __init__(self, table_name: str, allowed_tables: set[str]) -> None:
        """Initialize InvalidLoggerTableNameError.

        Args:
            table_name: The invalid table name that was provided
            allowed_tables: Set of valid table names
        """
        message = f"Table name '{table_name}' not in allowed list: {allowed_tables}"
        super().__init__(message)
        self.table_name = table_name
        self.allowed_tables = allowed_tables


class APIError(GalFridayError):
    """Exception raised for errors when interacting with external APIs."""

    def __init__(
        self,
        message: str,
        service_name: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an APIError.

        Args:
        ----
            message: Description of the error
            service_name: Name of the API service that raised the error
            status_code: Optional HTTP status code returned by the API
            details: Optional dictionary with additional error details
        """
        details = details or {}
        details.update(
            {
                "service_name": service_name,
                "status_code": status_code,
            },
        )
        super().__init__(message, details)


class DataValidationError(GalFridayError):
    """Exception raised for errors in data validation."""


class ExecutionError(GalFridayError):
    """Exception raised for errors during order execution."""


class ExecutionHandlerError(ExecutionError):
    """Base exception for ExecutionHandler errors."""


class ExecutionHandlerAuthenticationError(ExecutionHandlerError):
    """Exception raised for authentication errors in execution handlers."""


class NetworkError(GalFridayError):
    """Exception raised for network-related errors."""


class ExecutionHandlerNetworkError(ExecutionHandlerError, NetworkError):
    """Exception raised for network errors in execution handlers."""


class ExecutionHandlerCriticalError(ExecutionHandlerError):
    """Exception raised for critical failures in execution handlers."""



class DatabaseError(GalFridayError):
    """Exception raised for database-related errors."""


class GalFridayTimeoutError(GalFridayError):
    """Exception raised for timeout errors."""


class AuthenticationError(GalFridayError):
    """Exception raised for authentication errors."""


class GalFridayPermissionError(GalFridayError):
    """Exception raised for permission errors."""


class RateLimitError(GalFridayError):
    """Exception raised when rate limits are exceeded."""


class ServiceUnavailableError(GalFridayError):
    """Exception raised when a service is unavailable."""


class PriceNotAvailableError(GalFridayError):
    """Exception raised when a price or rate is not available."""


class InsufficientFundsError(GalFridayError):
    """Exception raised when there are insufficient funds for an operation."""


class UnsupportedParamsTypeError(TypeError):
    """Exception raised when an unsupported parameter type is provided to an adapter."""

    def __init__(self, params_type: type) -> None:
        """Initialize UnsupportedParamsTypeError.

        Args:
            params_type: The unsupported parameter type that was provided
        """
        message = f"Unsupported params type: {params_type}"
        super().__init__(message)
        self.params_type = params_type


class CriticalExit(SystemExit):
    """Base class for custom SystemExit exceptions indicating critical failures."""

    def __init__(self, message: str, *args: object) -> None:
        """Initialize CriticalExit.

        Args:
            message: The error message
            *args: Additional arguments to pass to parent class
        """
        super().__init__(message, *args)


class ServiceInstantiationFailedExit(CriticalExit):
    """SystemExit raised when service instantiation fails broadly."""

    def __init__(self, _original_exception: Exception | None = None) -> None:
        """Initialize ServiceInstantiationFailedExit.

        Args:
            _original_exception: Optional original exception that caused the failure
        """
        super().__init__("Service instantiation failed. Application exiting.")


class PubSubManagerStartFailedExit(CriticalExit):
    """SystemExit raised when the PubSubManager fails to start."""

    def __init__(self, _original_exception: Exception | None = None) -> None:
        """Initialize PubSubManagerStartFailedExit.

        Args:
            _original_exception: Optional original exception that caused the failure
        """
        # The original exception cause is handled by 'from e' at the raise site
        super().__init__("PubSubManager failed to start. Application exiting.")


class ExecutionHandlerInstantiationFailedExit(CriticalExit):
    """SystemExit raised when the ExecutionHandler fails to instantiate for a given mode."""

    def __init__(self, mode: str, _original_exception: Exception | None = None) -> None:
        """Initialize ExecutionHandlerInstantiationFailedExit.

        Args:
            mode: The execution mode that failed
            _original_exception: Optional original exception that caused the failure
        """
        message = (
            f"Execution Handler failed to instantiate for mode: '{mode}'. " "Application exiting."
        )
        super().__init__(message)


class RiskManagerInstantiationFailedExit(CriticalExit):
    """SystemExit raised when the RiskManager fails to instantiate."""

    def __init__(
        self,
        component_name: str = "RiskManager",
        _original_exception: Exception | None = None,
    ) -> None:
        """Initialize RiskManagerInstantiationFailedExit.

        Args:
            component_name: Name of the component that failed to initialize
            _original_exception: Optional original exception that caused the failure
        """
        message = f"{component_name} instantiation failed. Application exiting."
        super().__init__(message)


class MarketPriceServiceUnsupportedModeError(UnsupportedModeError):
    """Error when MarketPriceService instantiation fails due to an unsupported mode."""

    def __init__(
        self,
        mode: str,
        supported_modes: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        """Initialize MarketPriceServiceUnsupportedModeError.

        Args:
            mode: The mode that is not supported by MarketPriceService
            supported_modes: Optional list of supported modes
            message: Optional custom error message
        """
        if message is None:
            message = (
                f"Cannot instantiate MarketPriceService for mode '{mode}'. "
                "Missing or incompatible implementation."
            )
        super().__init__(mode=mode, supported_modes=supported_modes, message=message)


class MarketPriceServiceCriticalFailureExit(CriticalExit):
    """SystemExit raised when MarketPriceService instantiation fails critically."""

    def __init__(self, _original_exception: Exception | None = None) -> None:
        """Initialize MarketPriceServiceCriticalFailureExit.

        Args:
            _original_exception: Optional original exception that caused the failure
        """
        super().__init__(
            "MarketPriceService instantiation failed critically. " "Application exiting.",
        )


class PortfolioManagerInstantiationFailedExit(CriticalExit):
    """SystemExit raised when PortfolioManager instantiation fails."""

    def __init__(self, _original_exception: Exception | None = None) -> None:
        """Initialize PortfolioManagerInstantiationFailedExit.

        Args:
            _original_exception: Optional original exception that caused the failure
        """
        super().__init__(
            "PortfolioManager instantiation failed. Application exiting.",
        )


class LoggerServiceInstantiationFailedExit(CriticalExit):
    """SystemExit raised when LoggerService instantiation fails."""

    def __init__(self, _original_exception: Exception | None = None) -> None:
        """Initialize LoggerServiceInstantiationFailedExit.

        Args:
            _original_exception: Optional original exception that caused the failure
        """
        super().__init__(
            "LoggerService instantiation failed. Application exiting.",
        )


class PubSubManagerInstantiationFailedExit(CriticalExit):
    """SystemExit raised when PubSubManager instantiation (not start) fails."""

    def __init__(self, _original_exception: Exception | None = None) -> None:
        """Initialize PubSubManagerInstantiationFailedExit.

        Args:
            _original_exception: Optional original exception that caused the failure
        """
        super().__init__(
            "PubSubManager core instantiation failed. Application exiting.",
        )


class ConfigurationLoadingFailedExit(CriticalExit):
    """SystemExit raised when loading the main application configuration fails."""

    def __init__(self, _original_exception: Exception | None = None) -> None:
        """Initialize ConfigurationLoadingFailedExit.

        Args:
            _original_exception: Optional original exception that caused the failure
        """
        super().__init__(
            "Configuration loading failed. Application exiting.",
        )
