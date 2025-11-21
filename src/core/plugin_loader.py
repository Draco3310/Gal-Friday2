import importlib
import logging
from typing import Type, Any
from src.core.interfaces import BaseStrategy

logger = logging.getLogger(__name__)

class PluginLoader:
    @staticmethod
    def load_strategy(module_path: str, class_name: str) -> Type[BaseStrategy]:
        """
        Dynamically loads a strategy class.

        Args:
            module_path: Dotted path to the module (e.g. 'src.strategies.donchian_volatility')
            class_name: Name of the class (e.g. 'DonchianStrategy')

        Returns:
            The strategy class (not instantiated).
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
