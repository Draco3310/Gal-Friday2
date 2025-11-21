import pandas as pd
import pandas_ta as ta
from src.core.interfaces import BaseStrategy, Signal
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DonchianStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lower_length = config.get('lower_length', 20)
        self.upper_length = config.get('upper_length', 20)
        self.symbol = config.get('symbol', 'BTC/USD')
        self.ml_buy_threshold = config.get('ml_buy_threshold', 0.7)

    async def analyze(self, df: pd.DataFrame, ml_score: Optional[float] = None) -> Signal:
        """
        Donchian Channel Breakout Strategy with ML Filter.
        """
        if df.empty:
            return Signal(symbol=self.symbol, action='HOLD', reason="Empty DataFrame")

        # Calculate Donchian Channels
        donchian = df.ta.donchian(lower_length=self.lower_length, upper_length=self.upper_length)

        if donchian is None or donchian.empty:
             return Signal(symbol=self.symbol, action='HOLD', reason="Failed to calculate indicators")

        # Append to df
        df = pd.concat([df, donchian], axis=1)

        # Standard naming: DCL_{lower}_{upper}, DCU_{lower}_{upper}
        lower_col = f"DCL_{self.lower_length}_{self.upper_length}"
        upper_col = f"DCU_{self.lower_length}_{self.upper_length}"

        if lower_col not in df.columns or upper_col not in df.columns:
             return Signal(symbol=self.symbol, action='HOLD', reason="Indicator columns not found")

        current_price = df['close'].iloc[-1]
        upper_band = df[upper_col].iloc[-1]
        lower_band = df[lower_col].iloc[-1]

        # Logic
        action = 'HOLD'
        reason = "Inside channels"

        if current_price > upper_band:
            action = 'BUY'
            reason = f"Price {current_price} > Upper Band {upper_band}"
        elif current_price < lower_band:
            action = 'SELL'
            reason = f"Price {current_price} < Lower Band {lower_band}"

        # ML Filter
        if action == 'BUY' and ml_score is not None:
            if ml_score < self.ml_buy_threshold:
                action = 'HOLD'
                reason += f" BUT ML Score {ml_score:.2f} < {self.ml_buy_threshold}"
            else:
                reason += f" AND ML Score {ml_score:.2f} >= {self.ml_buy_threshold}"

        return Signal(
            symbol=self.symbol,
            action=action,
            price=float(current_price),
            reason=reason
        )
