import importlib
import logging
from typing import Type, Any
from src.core.interfaces import BaseStrategy
from src.predictors.base_predictor import BasePredictor

logger = logging.getLogger(__name__)

class PluginLoader:
    @staticmethod
    def load_strategy(module_path: str, class_name: str) -> Type[BaseStrategy]:
        """
        Dynamically loads a strategy class.
        """
        try:
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)

            if not issubclass(strategy_class, BaseStrategy):
                raise TypeError(f"{class_name} must inherit from BaseStrategy")

            logger.info(f"Loaded strategy: {class_name} from {module_path}")
            return strategy_class
        except (ImportError, AttributeError, TypeError) as e:
            logger.error(f"Failed to load strategy {class_name} from {module_path}: {e}")
            raise

    @staticmethod
    def load_predictor(module_path: str, class_name: str) -> Type[BasePredictor]:
        """
        Dynamically loads a predictor class.
        """
        try:
            module = importlib.import_module(module_path)
            predictor_class = getattr(module, class_name)

            # In a real scenario, we'd check issubclass(predictor_class, BasePredictor)
            # But I need to import BasePredictor first.
            # Assuming strict typing is desired.
            if not issubclass(predictor_class, BasePredictor):
               raise TypeError(f"{class_name} must inherit from BasePredictor")

            logger.info(f"Loaded predictor: {class_name} from {module_path}")
            return predictor_class
        except (ImportError, AttributeError, TypeError) as e:
            logger.error(f"Failed to load predictor {class_name} from {module_path}: {e}")
            raise
