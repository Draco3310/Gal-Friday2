"""Standard exceptions for the Gal Friday application."""

from typing import Any, Dict, Optional


class GalFridayError(Exception):
    """Base class for all Gal Friday exceptions."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize a GalFridayError.

        Args
        ----
            message: Description of the error
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ConfigurationError(GalFridayError):
    """Exception raised for errors in the configuration."""

    pass


class APIError(GalFridayError):
    """Exception raised for errors when interacting with external APIs."""

    def __init__(
        self,
        message: str,
        service_name: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an APIError.

        Args
        ----
            message: Description of the error
            service_name: Name of the API service that raised the error
            status_code: Optional HTTP status code returned by the API
            details: Optional dictionary with additional error details
        """
        details = details or {}
        details.update({
            "service_name": service_name,
            "status_code": status_code,
        })
        super().__init__(message, details)


class DataValidationError(GalFridayError):
    """Exception raised for errors in data validation."""

    pass


class ExecutionError(GalFridayError):
    """Exception raised for errors during order execution."""

    pass


class NetworkError(GalFridayError):
    """Exception raised for network-related errors."""

    pass


class DatabaseError(GalFridayError):
    """Exception raised for database-related errors."""

    pass


class TimeoutError(GalFridayError):
    """Exception raised for timeout errors."""

    pass


class AuthenticationError(GalFridayError):
    """Exception raised for authentication errors."""

    pass


class PermissionError(GalFridayError):
    """Exception raised for permission errors."""

    pass


class RateLimitError(GalFridayError):
    """Exception raised when rate limits are exceeded."""

    pass


class ServiceUnavailableError(GalFridayError):
    """Exception raised when a service is unavailable."""

    pass


class PriceNotAvailableError(GalFridayError):
    """Exception raised when a price or rate is not available."""

    pass


class InsufficientFundsError(GalFridayError):
    """Exception raised when there are insufficient funds for an operation."""

    pass
