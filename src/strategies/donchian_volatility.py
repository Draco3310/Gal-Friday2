import pandas as pd
import pandas_ta as ta
from src.core.interfaces import BaseStrategy, Signal
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DonchianStrategy(BaseStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lower_length = config.get('lower_length', 20)
        self.upper_length = config.get('upper_length', 20)
        self.symbol = config.get('symbol', 'BTC/USD')

    async def analyze(self, df: pd.DataFrame) -> Signal:
        """
        Donchian Channel Breakout Strategy.
        """
        if df.empty:
            return Signal(symbol=self.symbol, action='HOLD', reason="Empty DataFrame")

        # Calculate Donchian Channels
        # pandas_ta donchian returns lower, mid, upper columns
        # We need to be careful with column names

        # Using 'donchian' method
        # It usually returns columns like DCL_20_20, DCM_20_20, DCU_20_20
        donchian = df.ta.donchian(lower_length=self.lower_length, upper_length=self.upper_length)

        if donchian is None or donchian.empty:
             return Signal(symbol=self.symbol, action='HOLD', reason="Failed to calculate indicators")

        # Append to df
        df = pd.concat([df, donchian], axis=1)

        # Identify column names dynamically or assume standard naming
        # Standard naming: DCL_{lower}_{upper}, DCU_{lower}_{upper}
        lower_col = f"DCL_{self.lower_length}_{self.upper_length}"
        upper_col = f"DCU_{self.lower_length}_{self.upper_length}"

        if lower_col not in df.columns or upper_col not in df.columns:
             # Fallback or error
             return Signal(symbol=self.symbol, action='HOLD', reason="Indicator columns not found")

        current_price = df['close'].iloc[-1]
        upper_band = df[upper_col].iloc[-1]
        lower_band = df[lower_col].iloc[-1]

        # Simple Breakout Logic
        # We check the PREVIOUS candle's breakout to avoid repainting if using current open candle,
        # but usually strategies run on closed candles.
        # Assuming df contains closed candles + maybe current open one.
        # Let's look at the last row.

        if current_price > upper_band:
            return Signal(
                symbol=self.symbol,
                action='BUY',
                price=float(current_price),
                reason=f"Price {current_price} > Upper Band {upper_band}"
            )
        elif current_price < lower_band:
             return Signal(
                symbol=self.symbol,
                action='SELL',
                price=float(current_price),
                reason=f"Price {current_price} < Lower Band {lower_band}"
            )

        return Signal(symbol=self.symbol, action='HOLD', reason="Inside channels")
