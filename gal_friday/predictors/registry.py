"""Predictor plugin registry for dynamic predictor loading."""

from collections.abc import Callable
import importlib
from typing import Any, Protocol

from gal_friday.logger_service import LoggerService


class PredictorInterface(Protocol):
    """Protocol for predictor implementations."""

    @staticmethod
    def run_inference_in_process(
        model_path: str,
        scaler_path: str,
        features: dict[str, float],
        logger: LoggerService,
    ) -> dict[str, Any]:
        """Run inference in process."""
        ...


class PredictorRegistry:
    """Registry for predictor implementations with lazy loading."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the predictor registry.

        Args:
            logger: Logger service instance
        """
        self.logger = logger
        self._predictors: dict[str, tuple[type[PredictorInterface], Callable[..., Any]]] = {}
        self._errors: dict[str, ImportError] = {}
        self._attempted_loads: set[str] = set()

        # Define available predictors
        self._predictor_definitions = {
            "xgboost": ("gal_friday.predictors.xgboost_predictor", "XGBoostPredictor"),
            "sklearn": ("gal_friday.predictors.sklearn_predictor", "SKLearnPredictor"),
            "lstm": ("gal_friday.predictors.lstm_predictor", "LSTMPredictor"),
        }

    def _raise_attribute_error(self, error_msg: str) -> None:
        """Helper method to raise AttributeError, abstracting the raise from try block."""
        raise AttributeError(error_msg)

    def _try_load_predictor(self, name: str) -> bool:
        """Try to load a predictor by name.

        Args:
            name: Predictor type name

        Returns:
            True if loaded successfully, False otherwise
        """
        if name in self._attempted_loads:
            return name in self._predictors

        self._attempted_loads.add(name)

        if name not in self._predictor_definitions:
            self.logger.warning(
                f"Unknown predictor type: {name}",
                source_module="PredictorRegistry",
            )
            return False

        module_path, class_name = self._predictor_definitions[name]

        try:
            module = importlib.import_module(module_path)
            predictor_class = getattr(module, class_name)

            # Verify it has the required method
            if not hasattr(predictor_class, "run_inference_in_process"):
                self._raise_attribute_error(
                    f"{class_name} does not have run_inference_in_process method",
                )

            self._predictors[name] = (
                predictor_class,
                predictor_class.run_inference_in_process,
            )

            self.logger.info(
                f"Successfully loaded predictor: {name}",
                source_module="PredictorRegistry",
            )

        except ImportError as e:
            self._errors[name] = e
            self.logger.warning(
                f"Failed to import {name} predictor: {e}",
                source_module="PredictorRegistry",
            )
            return False
        except Exception as e:
            self._errors[name] = ImportError(f"Failed to load {name}: {e}")
            self.logger.exception(
                f"Error loading {name} predictor",
                source_module="PredictorRegistry",
            )
            return False
        else:
            return True

    def get_predictor(self, name: str) -> tuple[type[PredictorInterface], Callable[..., Any]]:
        """Get a predictor by name.

        Args:
            name: Predictor type name

        Returns:
            Tuple of (predictor_class, runner_method)

        Raises:
            ImportError: If predictor cannot be loaded
            KeyError: If predictor type is unknown
        """
        # Try to load if not already loaded
        if name not in self._predictors:
            self._try_load_predictor(name)

        if name in self._errors:
            raise self._errors[name]

        if name not in self._predictors:
            raise KeyError(f"Predictor '{name}' not available")

        return self._predictors[name]

    def get_available_predictors(self) -> dict[str, tuple[type[PredictorInterface], Callable[..., Any]]]:
        """Get all available predictors.

        Returns:
            Dictionary mapping predictor names to (class, runner) tuples
        """
        # Try to load all predictors
        for name in self._predictor_definitions:
            if name not in self._attempted_loads:
                self._try_load_predictor(name)

        return dict(self._predictors)

    def is_available(self, name: str) -> bool:
        """Check if a predictor is available.

        Args:
            name: Predictor type name

        Returns:
            True if available, False otherwise
        """
        if name not in self._attempted_loads:
            self._try_load_predictor(name)
        return name in self._predictors

    def register_predictor(
        self,
        name: str,
        predictor_class: type[PredictorInterface],
        runner_method: Callable[..., Any] | None = None,
    ) -> None:
        """Register a custom predictor.

        Args:
            name: Predictor type name
            predictor_class: Predictor class
            runner_method: Optional runner method (defaults to run_inference_in_process)
        """
        if runner_method is None:
            runner_method = predictor_class.run_inference_in_process

        self._predictors[name] = (predictor_class, runner_method)
        self._attempted_loads.add(name)

        self.logger.info(
            f"Registered custom predictor: {name}",
            source_module="PredictorRegistry",
        )
