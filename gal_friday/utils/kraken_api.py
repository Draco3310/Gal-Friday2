"""Shared Kraken API utilities for signature generation and common functionality.

This module provides common utilities for interacting with the Kraken API,
eliminating duplication across different Kraken-specific modules.
"""

import base64
import hashlib
import hmac
import urllib.parse
from typing import Any


class KrakenAPIError(Exception):
    """Base exception for Kraken API utilities."""


class InvalidAPISecretError(KrakenAPIError):
    """Error raised when API secret is invalid or improperly formatted."""

    def __init__(self, message: str = "Invalid API secret format") -> None:
        """Initialize the InvalidAPISecretError with an optional custom message.

        Args:
            message: Custom error message (default: "Invalid API secret format")
        """
        super().__init__(message)


def generate_kraken_signature(
    uri_path: str, data: dict[str, Any], nonce: int, api_secret: str) -> str:
    """Generate the API-Sign header required by Kraken private endpoints.

    This function creates the HMAC-SHA512 signature that Kraken requires for
    authenticated API requests.

    Args:
        uri_path: API endpoint path (e.g., "/0/private/AddOrder")
        data: Request parameters as a dictionary
        nonce: Unique nonce value (typically current timestamp in milliseconds)
        api_secret: Base64-encoded API secret from Kraken

    Returns:
        Base64-encoded HMAC-SHA512 signature

    Raises:
        InvalidAPISecretError: If the API secret is not valid base64
    """
    if not api_secret:
        raise InvalidAPISecretError("API secret cannot be empty")

    # URL-encode the request data
    post_data_str = urllib.parse.urlencode(data)

    # Create the message to be signed: nonce + postdata
    nonce_postdata = str(nonce) + post_data_str

    # Create the signature input: uri_path + sha256(nonce + postdata)
    encoded_nonce_postdata = nonce_postdata.encode()
    sha256_hash = hashlib.sha256(encoded_nonce_postdata).digest()
    message = uri_path.encode() + sha256_hash

    try:
        # Decode the API secret from base64
        secret_decoded = base64.b64decode(api_secret)
    except Exception as e:
        raise InvalidAPISecretError(f"API secret must be valid base64: {e}") from e

    # Generate HMAC-SHA512 signature
    mac = hmac.new(secret_decoded, message, hashlib.sha512)

    # Return base64-encoded signature
    return base64.b64encode(mac.digest()).decode()


def prepare_kraken_request_data(
    data: dict[str, Any], nonce: int | None = None) -> dict[str, Any]:
    """Prepare request data for Kraken API by adding nonce if not present.

    Args:
        data: Original request parameters
        nonce: Optional nonce value; if None, current timestamp in milliseconds is used

    Returns:
        Request data with nonce added
    """
    import time

    request_data = data.copy()
    if nonce is None:
        nonce = int(time.time() * 1000)
    request_data["nonce"] = nonce

    return request_data