"""Configuration validation utilities for Gal-Friday.

This module provides comprehensive validation for configuration values to ensure
all settings are properly managed and no hardcoded values are used inappropriately.
Supports NFR-804 requirements for configuration management.
"""

import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""


class ConfigValidator:
    """Validates configuration values and ensures no inappropriate hardcoding."""

    # Known required configuration sections
    REQUIRED_SECTIONS: ClassVar[set[str]] = {
        "exchange",
        "trading",
        "portfolio",
        "risk",
        "monitoring",
        "logging",
        "prediction_service",
    }

    # URLs that should be configurable, not hardcoded
    CONFIGURABLE_URLS: ClassVar[dict[str, str]] = {
        "kraken.api_url": "https://api.kraken.com",
        "kraken.ws_url": "wss://ws.kraken.com/v2",
        "influx.url": "http://localhost:8086",
    }

    # Sensitive values that should come from environment
    SENSITIVE_KEYS: ClassVar[set[str]] = {
        "kraken.api_key",
        "kraken.api_secret",
        "database.password",
        "influx.token",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize validator with configuration data.

        Args:
            config: The configuration dictionary to validate
        """
        self.config = config
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks.

        Returns:
            True if configuration is valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()

        self._validate_required_sections()
        self._validate_url_configuration()
        self._validate_sensitive_values()
        self._validate_risk_parameters()
        self._validate_trading_parameters()
        self._validate_logging_configuration()
        self._validate_prediction_service_config()

        return len(self.errors) == 0

    def get_errors(self) -> list[str]:
        """Get all validation errors."""
        return self.errors.copy()

    def get_warnings(self) -> list[str]:
        """Get all validation warnings."""
        return self.warnings.copy()

    def _validate_required_sections(self) -> None:
        """Validate that all required configuration sections exist."""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.config:
                self.errors.append(f"Missing required configuration section: {section}")
            elif not isinstance(self.config[section], dict):
                self.errors.append(f"Configuration section '{section}' must be a dictionary")

    def _validate_url_configuration(self) -> None:
        """Validate that URLs are properly configured, not hardcoded."""
        for key_path, default_url in self.CONFIGURABLE_URLS.items():
            sections = key_path.split(".")
            current = self.config

            # Navigate to the nested key
            try:
                for section in sections[:-1]:
                    current = current[section]
                final_key = sections[-1]

                if final_key not in current:
                    self.warnings.append(
                        f"URL not configured for '{key_path}', will use default: {default_url}",
                    )
                else:
                    configured_url = current[final_key]
                    if not isinstance(configured_url, str):
                        self.errors.append(
                            f"URL configuration '{key_path}' must be a string",
                        )
                    elif not self._is_valid_url(configured_url):
                        self.errors.append(
                            f"Invalid URL format for '{key_path}': {configured_url}",
                        )

            except KeyError:
                self.warnings.append(
                    f"Section missing for URL configuration '{key_path}', "
                    f"will use default: {default_url}",
                )

    def _validate_sensitive_values(self) -> None:
        """Validate that sensitive values are properly handled."""
        for key_path in self.SENSITIVE_KEYS:
            sections = key_path.split(".")
            current = self.config

            # Check if sensitive value is directly in config (should be avoided)
            try:
                for section in sections[:-1]:
                    current = current[section]
                final_key = sections[-1]

                if final_key in current:
                    value = current[final_key]
                    if isinstance(value, str) and value.startswith("YOUR_"):
                        self.warnings.append(
                            f"Placeholder value detected for sensitive key '{key_path}'. "
                            f"Should be set via environment variable.",
                        )
                    elif isinstance(value, str) and not value.startswith("${"):
                        # Check if it looks like a real secret vs environment variable reference
                        env_var = f"{key_path.upper().replace('.', '_')}"
                        if env_var not in os.environ:
                            self.warnings.append(
                                f"Sensitive value '{key_path}' appears to be hardcoded. "
                                f"Consider using environment variable ${env_var}",
                            )

            except KeyError:
                # Sensitive value not configured - this might be OK for some deployments
                pass

    def _validate_risk_parameters(self) -> None:
        """Validate risk management parameters."""
        if "risk" not in self.config:
            return

        risk_config = self.config["risk"]

        if "limits" in risk_config:
            limits = risk_config["limits"]

            # Validate percentage values
            percentage_keys = [
                "max_total_drawdown_pct",
                "max_daily_drawdown_pct",
                "max_weekly_drawdown_pct",
                "risk_per_trade_pct",
                "max_position_size_pct_equity",
                "max_total_exposure_pct_equity",
            ]

            for key in percentage_keys:
                if key in limits:
                    try:
                        max_risk_percent = 100
                        value = Decimal(str(limits[key]))
                        if value < 0:
                            self.errors.append(
                                f"Risk parameter '{key}' cannot be negative: {value}",
                            )
                        elif value > max_risk_percent:
                            self.warnings.append(
                                f"Risk parameter '{key}' is very high: {value}%",
                            )
                    except (ValueError, TypeError):
                        self.errors.append(f"Risk parameter '{key}' must be a valid number")

            # Validate consecutive losses
            if "max_consecutive_losses" in limits:
                try:
                    consecutive_losses_value = int(limits["max_consecutive_losses"])
                    if consecutive_losses_value < 1:
                        self.errors.append("max_consecutive_losses must be at least 1")
                except (ValueError, TypeError):
                    self.errors.append("max_consecutive_losses must be a valid integer")

    def _validate_trading_parameters(self) -> None:
        """Validate trading configuration parameters."""
        if "trading" not in self.config:
            return

        trading_config = self.config["trading"]

        # Validate trading pairs
        if "pairs" in trading_config:
            pairs = trading_config["pairs"]
            if not isinstance(pairs, list):
                self.errors.append("Trading pairs must be a list")
            elif len(pairs) == 0:
                self.errors.append("At least one trading pair must be configured")
            else:
                for pair in pairs:
                    if not isinstance(pair, str):
                        self.errors.append(
                            f"Trading pair must be a string: {pair}",
                        )
                    elif "/" not in pair:
                        self.errors.append(
                            f"Invalid trading pair format: {pair} "
                            "(expected format: BASE/QUOTE)",
                        )

    def _validate_logging_configuration(self) -> None:
        """Validate logging configuration."""
        if "logging" not in self.config:
            return

        logging_config = self.config["logging"]

        # Validate log level
        if "level" in logging_config:
            valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            level = logging_config["level"].upper()
            if level not in valid_levels:
                self.errors.append(f"Invalid log level: {level}. Must be one of {valid_levels}")

        # Validate database configuration if enabled
        if "database" in logging_config and logging_config["database"].get("enabled"):
            db_config = logging_config["database"]
            if "connection_string" in db_config:
                conn_str = db_config["connection_string"]
                if "YOUR_DB_PASSWORD" in conn_str:
                    self.warnings.append(
                        "Database connection string contains placeholder password. "
                        "Update with actual credentials or use environment variables.",
                    )

    def _validate_prediction_service_config(self) -> None:
        """Validate prediction service configuration."""
        if "prediction_service" not in self.config:
            return

        pred_config = self.config["prediction_service"]

        # Validate ensemble strategy
        if "ensemble_strategy" in pred_config:
            valid_strategies = {"none", "average", "weighted_average"}
            strategy = pred_config["ensemble_strategy"]
            if strategy not in valid_strategies:
                self.errors.append(
                    f"Invalid ensemble strategy: {strategy}. Must be one of {valid_strategies}",
                )

        # Validate model configurations
        if "models" in pred_config:
            models = pred_config["models"]
            if not isinstance(models, list):
                self.errors.append("Prediction service models must be a list")
            else:
                for i, model in enumerate(models):
                    self._validate_model_config(model, i)

    def _validate_model_config(self, model: dict[str, Any], index: int) -> None:
        """Validate individual model configuration."""
        required_fields = ["model_id", "predictor_type", "model_path", "prediction_target"]

        for field in required_fields:
            if field not in model:
                self.errors.append(f"Model {index}: Missing required field '{field}'")

        # Validate predictor type
        if "predictor_type" in model:
            valid_types = {"xgboost", "sklearn", "lstm"}
            pred_type = model["predictor_type"]
            if pred_type not in valid_types:
                self.errors.append(
                    f"Model {index}: Invalid predictor type '{pred_type}'. "
                    f"Must be one of {valid_types}",
                )

        # Validate LSTM-specific config
        if model.get("predictor_type") == "lstm":
            if "framework" not in model:
                self.errors.append(
                    f"Model {index}: LSTM models must specify 'framework' "
                    "(tensorflow or pytorch)",
                )
            elif model["framework"] not in {"tensorflow", "pytorch"}:
                self.errors.append(
                    f"Model {index}: Invalid LSTM framework '{model['framework']}'. "
                    "Must be 'tensorflow' or 'pytorch'",
                )

            if "sequence_length" not in model:
                self.errors.append(f"Model {index}: LSTM models must specify 'sequence_length'")
            elif not isinstance(model["sequence_length"], int) or model["sequence_length"] < 1:
                self.errors.append(f"Model {index}: sequence_length must be a positive integer")

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL format is valid."""
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$", re.IGNORECASE)
        return bool(url_pattern.match(url))


def validate_config(config: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """Validate configuration and return results.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = ConfigValidator(config)
    is_valid = validator.validate_all()
    return is_valid, validator.get_errors(), validator.get_warnings()


def check_for_hardcoded_values(file_path: str) -> list[str]:
    """Scan a Python file for potential hardcoded values that should be configurable.

    Args:
        file_path: Path to the Python file to scan

    Returns:
        List of potential issues found
    """
    issues = []

    # Patterns that might indicate hardcoded values
    patterns = [
        (r'"https?://[^"]+\.com[^"]*"', "Hardcoded URL"),
        (r'"wss?://[^"]+\.com[^"]*"', "Hardcoded WebSocket URL"),
        (r"timeout\s*=\s*\d+", "Hardcoded timeout value"),
        (r"max_retries\s*=\s*\d+", "Hardcoded retry count"),
        (r"sleep\(\s*\d+\s*\)", "Hardcoded sleep duration"),
        (r"range\(\s*\d+\s*,\s*\d+\s*\)", "Hardcoded range values"),
    ]

    try:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        for line_num, line in enumerate(content.split("\n"), 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith(("#", '"""', "'''")):
                continue

            for pattern, description in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(f"Line {line_num}: {description} - {line.strip()}")

    except Exception as e:
        issues.append(f"Error scanning file {file_path}: {e!s}")

    return issues
