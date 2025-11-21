from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd

@dataclass
class Signal:
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: Optional[float] = None
    quantity: Optional[float] = None # Can be calculated later by execution logic
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def analyze(self, df: pd.DataFrame) -> Signal:
        """
        Analyzes the market data and returns a trading signal.

        Args:
            df: DataFrame containing OHLCV data.

        Returns:
            Signal: The trading signal.
        """
        pass

class BasePredictor(ABC):
    """
    Abstract base class for ML predictors.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def predict(self, df: pd.DataFrame) -> Any:
        """
        Generates a prediction based on input data.
        """
        pass
