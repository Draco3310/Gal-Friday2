"""Model lifecycle specific exception helpers.

This module provides exception helper functions to resolve TRY301 linting issues
by abstracting raise statements into dedicated functions. Following enterprise
error handling patterns with structured logging and context preservation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from gal_friday.exceptions import OperationalError
from gal_friday.utils.error_helpers import log_and_reraise

if TYPE_CHECKING:
    from pathlib import Path

    from gal_friday.logger_service import LoggerService
    from gal_friday.model_lifecycle.enums import ModelStage


class ModelLifecycleError(OperationalError):
    """Base exception for model lifecycle operations."""


class ModelNotFoundError(ModelLifecycleError):
    """Exception raised when a model is not found."""


class ModelFileNotFoundError(ModelLifecycleError):
    """Exception raised when a model file is not found."""


class ModelPathError(ModelLifecycleError):
    """Exception raised when model artifact path is invalid."""


class ModelPromotionError(ModelLifecycleError):
    """Exception raised for invalid model stage promotions."""


class ProductionModelProtectionError(ModelLifecycleError):
    """Exception raised when attempting to modify production models without authorization."""


def _raise_model_file_not_found_error(
    model_path: Path,
    logger: LoggerService | None = None,
    source_module: str | None = None,
) -> NoReturn:
    """Raise a structured model file not found error.

    Args:
        model_path: Path to the missing model file
        logger: Optional logger for error logging
        source_module: Optional source module for context

    Raises:
        ModelFileNotFoundError: Always raised with contextual information
    """
    error_msg = f"Model file not found at {model_path}"

    if logger:
        log_and_reraise(
            logger,
            error_msg,
            source_module=source_module,
            context={"model_path": str(model_path), "path_exists": model_path.exists()},
        )

    raise ModelFileNotFoundError(error_msg)


def _raise_model_not_found_error(
    model_name: str,
    version: str | None = None,
    stage: ModelStage | None = None,
    logger: LoggerService | None = None,
    source_module: str | None = None,
) -> NoReturn:
    """Raise a structured model not found error.

    Args:
        model_name: Name of the model that wasn't found
        version: Optional version identifier
        stage: Optional model stage
        logger: Optional logger for error logging
        source_module: Optional source module for context

    Raises:
        ModelNotFoundError: Always raised with contextual information
    """
    error_msg = f"Model not found: {model_name}"
    if version:
        error_msg += f" (version={version}"
        if stage:
            error_msg += f", stage={stage}"
        error_msg += ")"

    context = {"model_name": model_name}
    if version:
        context["version"] = version
    if stage:
        context["stage"] = str(stage)

    if logger:
        log_and_reraise(
            logger,
            error_msg,
            source_module=source_module,
            context=context,
        )

    raise ModelNotFoundError(error_msg)


def _raise_model_path_error(
    model_name: str,
    logger: LoggerService | None = None,
    source_module: str | None = None,
) -> NoReturn:
    """Raise a structured model artifact path error.

    Args:
        model_name: Name of the model with invalid path
        logger: Optional logger for error logging
        source_module: Optional source module for context

    Raises:
        ModelPathError: Always raised with contextual information
    """
    error_msg = f"Model artifact path not found for: {model_name}"

    if logger:
        log_and_reraise(
            logger,
            error_msg,
            source_module=source_module,
            context={"model_name": model_name},
        )

    raise ModelPathError(error_msg)


def _raise_invalid_promotion_error(
    from_stage: ModelStage,
    to_stage: ModelStage,
    logger: LoggerService | None = None,
    source_module: str | None = None,
) -> NoReturn:
    """Raise a structured invalid promotion error.

    Args:
        from_stage: Current model stage
        to_stage: Target model stage
        logger: Optional logger for error logging
        source_module: Optional source module for context

    Raises:
        ModelPromotionError: Always raised with contextual information
    """
    error_msg = f"Invalid promotion: {from_stage.value} -> {to_stage.value}"

    if logger:
        log_and_reraise(
            logger,
            error_msg,
            source_module=source_module,
            context={"from_stage": from_stage.value, "to_stage": to_stage.value},
        )

    raise ModelPromotionError(error_msg)


def _raise_production_model_protection_error(
    logger: LoggerService | None = None,
    source_module: str | None = None,
) -> NoReturn:
    """Raise a structured production model protection error.

    Args:
        logger: Optional logger for error logging
        source_module: Optional source module for context

    Raises:
        ProductionModelProtectionError: Always raised with contextual information
    """
    error_msg = "Cannot delete/archive active production model without force=True. Demote first."

    if logger:
        log_and_reraise(
            logger,
            error_msg,
            source_module=source_module,
            context={"protection_type": "production_model_delete"},
        )

    raise ProductionModelProtectionError(error_msg)


def _raise_model_not_found_by_id_error(
    model_id_str: str,
    logger: LoggerService | None = None,
    source_module: str | None = None,
) -> NoReturn:
    """Raise a structured model not found by ID error.

    Args:
        model_id_str: String representation of the model ID
        logger: Optional logger for error logging
        source_module: Optional source module for context

    Raises:
        ModelNotFoundError: Always raised with contextual information
    """
    error_msg = f"Model not found: {model_id_str}"

    if logger:
        log_and_reraise(
            logger,
            error_msg,
            source_module=source_module,
            context={"model_id": model_id_str},
        )

    raise ModelNotFoundError(error_msg)
