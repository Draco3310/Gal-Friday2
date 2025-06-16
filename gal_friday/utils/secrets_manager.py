"""Secrets Manager for secure credential handling.

This module provides a centralized way to manage sensitive credentials
with support for multiple backends (environment variables, files, cloud services).
"""

import base64
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from typing import Any

# Constants
MAX_ACCESS_LOG_ENTRIES = 1000


class SecretsBackend:
    """Base class for secrets storage backends."""

    def get_secret(self, key: str) -> str | None:
        """Retrieve a secret value by key."""
        raise NotImplementedError

    def set_secret(self, key: str, value: str) -> bool:
        """Store a secret value."""
        raise NotImplementedError

    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        raise NotImplementedError


class EnvironmentBackend(SecretsBackend):
    """Environment variable backend for secrets."""

    def get_secret(self, key: str) -> str | None:
        """Get secret from environment variable."""
        return os.environ.get(key)

    def set_secret(self, key: str, value: str) -> bool:
        """Set environment variable (for current process only)."""
        os.environ[key] = value
        return True

    def delete_secret(self, key: str) -> bool:
        """Remove environment variable."""
        if key in os.environ:
            del os.environ[key]
            return True
        return False


class EncryptedFileBackend(SecretsBackend):
    """Encrypted file backend for secrets."""

    def __init__(self, file_path: Path, password: str) -> None:
        """Initialize the encrypted file backend.

        Args:
            file_path: Path to the encrypted secrets file
            password: Password used for encryption/decryption
        """
        self.file_path = file_path
        self._cipher = self._create_cipher(password)
        self._secrets: dict[str, str] = {}
        self._load_secrets()

    def _create_cipher(self, password: str) -> Fernet:
        """Create encryption cipher from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"gal-friday-salt",  # In production, use random salt
            iterations=100000)
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)

    def _load_secrets(self) -> None:
        """Load and decrypt secrets from file."""
        if self.file_path.exists():
            try:
                encrypted_data = self.file_path.read_bytes()
                decrypted_data = self._cipher.decrypt(encrypted_data)
                self._secrets = json.loads(decrypted_data.decode())
            except Exception:
                # If decryption fails, start with empty secrets
                self._secrets = {}

    def _save_secrets(self) -> None:
        """Encrypt and save secrets to file."""
        data = json.dumps(self._secrets).encode()
        encrypted_data = self._cipher.encrypt(data)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_bytes(encrypted_data)
        # Set restrictive permissions
        self.file_path.chmod(0o600)

    def get_secret(self, key: str) -> str | None:
        """Get secret from encrypted file."""
        return self._secrets.get(key)

    def set_secret(self, key: str, value: str) -> bool:
        """Store secret in encrypted file."""
        self._secrets[key] = value
        self._save_secrets()
        return True

    def delete_secret(self, key: str) -> bool:
        """Delete secret from encrypted file."""
        if key in self._secrets:
            del self._secrets[key]
            self._save_secrets()
            return True
        return False


class GCPSecretsBackend(SecretsBackend):
    """Google Cloud Platform Secrets Manager backend."""

    def __init__(self, project_id: str, logger: LoggerService) -> None:
        """Initialize the GCP Secrets Manager backend.

        Args:
            project_id: GCP project ID
            logger: Logger service instance
        """
        self.project_id = project_id
        self.logger = logger
        self._source_module = self.__class__.__name__

        try:
            from google.cloud import secretmanager  # type: ignore
            self.client = secretmanager.SecretManagerServiceClient()
        except ImportError:
            raise ImportError("google-cloud-secret-manager package not installed") from None

    def get_secret(self, key: str) -> str | None:
        """Retrieve secret from GCP Secrets Manager."""
        try:
            # Build the resource name
            name = f"projects/{self.project_id}/secrets/{key}/versions/latest"

            # Access the secret version
            response = self.client.access_secret_version(request={"name": name})

            # Return the secret value
            payload_data: bytes = response.payload.data  # type: ignore[attr-defined]
            return payload_data.decode("UTF-8")

        except Exception as e:
            if "NOT_FOUND" in str(e):
                return None
            self.logger.error(
                f"Error retrieving secret from GCP: {e}",
                source_module=self._source_module)
            return None

    def set_secret(self, key: str, value: str) -> bool:
        """Store secret in GCP Secrets Manager."""
        try:
            parent = f"projects/{self.project_id}"

            # Check if secret exists
            try:
                secret_name = f"{parent}/secrets/{key}"
                self.client.get_secret(request={"name": secret_name})
                # Secret exists, add new version
                self.client.add_secret_version(
                    request={
                        "parent": secret_name,
                        "payload": {"data": value.encode("UTF-8")},
                    })
            except Exception:
                # Secret doesn't exist, create it
                self.client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": key,
                        "secret": {"replication": {"automatic": {}}},
                    })
                # Add the secret version
                self.client.add_secret_version(
                    request={
                        "parent": f"{parent}/secrets/{key}",
                        "payload": {"data": value.encode("UTF-8")},
                    })

            return True

        except Exception as e:
            self.logger.error(
                f"Error storing secret in GCP: {e}",
                source_module=self._source_module)
            return False

    def delete_secret(self, key: str) -> bool:
        """Delete secret from GCP Secrets Manager."""
        try:
            name = f"projects/{self.project_id}/secrets/{key}"
            self.client.delete_secret(request={"name": name})
            return True
        except Exception as e:
            self.logger.error(
                f"Error deleting secret from GCP: {e}",
                source_module=self._source_module)
            return False


class SecretsManager:
    """Centralized secrets management with multiple backend support."""

    def __init__(self, config_manager: ConfigManager, logger_service: LoggerService) -> None:
        """Initialize the Secrets Manager.

        Args:
            config_manager: Configuration manager instance
            logger_service: Logger service instance
        """
        self.config = config_manager
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self._backends: list[SecretsBackend] = []
        self._access_log: list[dict[str, Any]] = []  # Audit log
        self._rotation_schedule: dict[str, datetime] = {}  # Key rotation tracking
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize configured secret backends in priority order."""
        # 1. Always check environment variables first
        self._backends.append(EnvironmentBackend())

        # 2. Add encrypted file backend if configured
        secrets_file = self.config.get("secrets.encrypted_file_path")
        if secrets_file:
            # Get master password from env or config
            master_password = os.environ.get("GAL_FRIDAY_MASTER_PASSWORD")
            if master_password:
                try:
                    file_backend = EncryptedFileBackend(
                        Path(secrets_file),
                        master_password)
                    self._backends.append(file_backend)
                    self.logger.info(
                        "Initialized encrypted file backend",
                        source_module=self._source_module)
                except Exception:
                    self.logger.exception(
                        "Failed to initialize encrypted file backend",
                        source_module=self._source_module)

        # 3. Add GCP Secrets Manager backend if configured
        gcp_project = self.config.get("secrets.gcp_project_id")
        if gcp_project:
            try:
                gcp_backend = GCPSecretsBackend(gcp_project, self.logger)
                self._backends.append(gcp_backend)
                self.logger.info(
                    "Initialized GCP Secrets Manager backend",
                    source_module=self._source_module)
            except Exception:
                self.logger.exception(
                    "Failed to initialize GCP Secrets Manager backend",
                    source_module=self._source_module)

    def get_secret(self, key: str) -> str | None:
        """Retrieve secret from first available backend.

        Args:
            key: Secret identifier

        Returns:
            Secret value or None if not found
        """
        # Log access for audit
        self._log_access(key, "GET")

        for backend in self._backends:
            try:
                value = backend.get_secret(key)
                if value is not None:
                    self.logger.debug(
                        f"Retrieved secret '{key}' from {backend.__class__.__name__}",
                        source_module=self._source_module)
                    return value
            except Exception:
                self.logger.exception(
                    f"Error retrieving secret from {backend.__class__.__name__}",
                    source_module=self._source_module)

        self.logger.warning(
            f"Secret '{key}' not found in any backend",
            source_module=self._source_module)
        return None

    def get_required_secret(self, key: str) -> str:
        """Retrieve required secret, raise exception if not found.

        Args:
            key: Secret identifier

        Returns:
            Secret value

        Raises:
            ValueError: If secret not found
        """
        value = self.get_secret(key)
        if value is None:
            raise ValueError(f"Required secret '{key}' not found")
        return value

    def set_secret(self, key: str, value: str, backend_name: str | None = None) -> bool:
        """Store secret in specified backend or first writable backend.

        Args:
            key: Secret identifier
            value: Secret value
            backend_name: Optional specific backend to use

        Returns:
            Success status
        """
        # Log access for audit
        self._log_access(key, "SET")

        if backend_name:
            # Store in specific backend
            for backend in self._backends:
                if backend.__class__.__name__ == backend_name:
                    return backend.set_secret(key, value)
            self.logger.error(
                f"Backend '{backend_name}' not found",
                source_module=self._source_module)
            return False
        # Store in first writable backend (skip environment)
        for backend in self._backends[1:]:  # Skip EnvironmentBackend
            try:
                if backend.set_secret(key, value):
                    self.logger.info(
                        f"Stored secret '{key}' in {backend.__class__.__name__}",
                        source_module=self._source_module)
                    return True
            except Exception:
                self.logger.exception(
                    f"Error storing secret in {backend.__class__.__name__}",
                    source_module=self._source_module)

        return False

    def rotate_secret(self, key: str, new_value: str) -> bool:
        """Rotate a secret by updating it in all backends where it exists.

        Args:
            key: Secret identifier
            new_value: New secret value

        Returns:
            Success status
        """
        # Log rotation for audit
        self._log_access(key, "ROTATE")

        success_count = 0
        for backend in self._backends:
            try:
                # Check if secret exists in this backend and set new value
                if backend.get_secret(key) is not None and backend.set_secret(key, new_value):
                    success_count += 1
                    self.logger.info(
                        f"Rotated secret '{key}' in {backend.__class__.__name__}",
                        source_module=self._source_module)
            except Exception:
                self.logger.exception(
                    f"Error rotating secret in {backend.__class__.__name__}",
                    source_module=self._source_module)

        # Update rotation schedule
        self._rotation_schedule[key] = datetime.now(UTC)

        return success_count > 0

    def validate_credentials(self) -> dict[str, bool]:
        """Validate all required credentials are available.

        Returns:
            Dict mapping credential names to availability status
        """
        required_credentials = [
            "KRAKEN_API_KEY",
            "KRAKEN_API_SECRET",
            "INFLUXDB_TOKEN",
        ]

        results = {}
        for cred in required_credentials:
            results[cred] = self.get_secret(cred) is not None

        missing = [k for k, v in results.items() if not v]
        if missing:
            self.logger.error(
                f"Missing required credentials: {missing}",
                source_module=self._source_module)

        return results

    def _log_access(self, key: str, action: str) -> None:
        """Log secret access for audit trail."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "key": key,
            "action": action,
            "source": self._source_module,
        }
        self._access_log.append(log_entry)

        # Keep only recent entries (last 1000)
        if len(self._access_log) > MAX_ACCESS_LOG_ENTRIES:
            self._access_log = self._access_log[-MAX_ACCESS_LOG_ENTRIES:]

    def get_access_log(self, key: str | None = None) -> list[dict[str, Any]]:
        """Get audit log of secret access.

        Args:
            key: Optional filter by specific key

        Returns:
            List of access log entries
        """
        if key:
            return [entry for entry in self._access_log if entry["key"] == key]
        return self._access_log.copy()

    def get_rotation_schedule(self) -> dict[str, datetime]:
        """Get secret rotation schedule.

        Returns:
            Dict mapping keys to last rotation time
        """
        return self._rotation_schedule.copy()