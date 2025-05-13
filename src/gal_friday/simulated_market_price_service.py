"""Simulate market price data for backtesting trading strategies.

This module provides a service that simulates market price data for backtesting trading strategies.
It uses historical OHLCV data to provide price information,
bid-ask spreads, and simulated order book.
It also supports volatility-adjusted spread calculation and market depth simulation.

"""

import logging
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

# Import the base class
from .market_price_service import MarketPriceService

# Attempt to import the actual ConfigManager
if TYPE_CHECKING:
    from .config_manager import ConfigManager
else:
    try:
        from .config_manager import ConfigManager
    except ImportError:
        # Fallback for environments where ConfigManager might not be in the expected path
        log_temp = logging.getLogger(__name__)
        log_temp.warning(
            "Could not import ConfigManager from .config_manager. "
            "SimulatedMarketPriceService will use "
            "default config values if ConfigManager is not provided."
        )

        # Define a minimal interface for static type checking
        class _DummyConfigManager:
            """Minimal placeholder for ConfigManager."""

            def get(self, key: str, default: Any = None) -> Any:
                """Get a value from config."""
                return default

            def get_decimal(self, key: str, default: Decimal) -> Decimal:
                """Get a decimal value from config."""
                return default

            def get_int(self, key: str, default: int) -> int:
                """Get an integer value from config."""
                return default

        # Use the dummy class as a fallback for type checking
        ConfigManager = _DummyConfigManager  # type: Any

# Attempt to import pandas_ta for ATR calculation
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    log_temp = logging.getLogger(__name__)
    log_temp.warning(
        "pandas_ta library not found. ATR calculation for "
        "volatility-adjusted spread will be disabled."
    )


# Set Decimal precision context if needed elsewhere, but usually handled globally or per-module
# from decimal import getcontext
# getcontext().prec = 28

# Module-level logger, can be replaced by injected logger in __init__
# log = logging.getLogger(__name__) # Standard logger, to be used if no logger_service is injected
_SOURCE_MODULE = "SimulatedMarketPriceService"


class SimulatedMarketPriceService(MarketPriceService):  # Inherit from MarketPriceService
    """Provide access to the latest market prices.

    based on historical data during a backtest simulation.

    Aligns with the MarketPriceService ABC.
    """

    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        config_manager: Optional[Any] = None,  # Changed ConfigManager to Any
        logger: Optional[logging.Logger] = None,
    ):  # Standard Python Logger
        """Initialize the service with historical market data.

        Args
        ----
            historical_data: A dictionary where keys are trading pairs (e.g., "XRP/USD")
                             and values are pandas DataFrames containing OHLCV data
                             indexed by timestamp (UTC).
            config_manager: An instance of ConfigManager for accessing configuration.
            logger: An instance of logging.Logger. If None, a default logger is used.
        """
        self.historical_data = historical_data
        self._current_timestamp: Optional[datetime] = None

        # self.logger and self.config are set by the
        # MarketPriceService ABC if it has a base __init__
        # or set directly if no super().__init__ is called or if it's called appropriately.
        # The current Kraken implementation does not call super().__init__.
        # For now, we assume direct assignment or
        # specific handling in subclasses if base has __init__.
        self.logger = (
            logger if logger else logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        )
        self.config = config_manager
        self._source_module = _SOURCE_MODULE

        self._load_simulation_config()

        # Validate data format, including HLC columns if volatility is enabled
        for pair, df in historical_data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning(
                    f"Historical data for {pair} does not have a DatetimeIndex.",
                    extra={"source_module": self._source_module},
                )

            required_cols = {self._price_column}
            if self._volatility_enabled and ta is not None:  # Check ta availability too
                required_cols.add(self._atr_high_col)
                required_cols.add(self._atr_low_col)
                required_cols.add(self._atr_close_col)

            for col in required_cols:
                if col not in df.columns:
                    self.logger.warning(
                        f"Historical data for {pair} is missing required column '{col}'.",
                        extra={"source_module": self._source_module},
                    )

        self.logger.info(
            "SimulatedMarketPriceService initialized.",
            extra={"source_module": self._source_module},
        )

    def _load_simulation_config(self) -> None:
        """Load simulation-specific configurations.

        Uses defaults if config_manager is not available.
        """
        if self.config:  # ConfigManager is provided
            sim_config = self.config.get("simulation", {})
            self._price_column = sim_config.get("price_column", "close")

            spread_config = sim_config.get("spread", {})
            self._default_spread_pct = self.config.get_decimal(
                "simulation.spread.default_pct", Decimal("0.1")
            )
            self._pair_specific_spread_config = spread_config.get(
                "pairs", {}
            )  # Renamed from self._spread_config to avoid confusion
            self._volatility_multiplier = self.config.get_decimal(
                "simulation.spread.volatility_multiplier", Decimal("1.5")
            )

            vol_config = spread_config.get("volatility", {})
            self._volatility_enabled = vol_config.get(
                "enabled", True if ta is not None else False
            )  # Default true only if ta available
            self._volatility_lookback_period = vol_config.get("lookback_period", 14)
            self._min_volatility_data_points = vol_config.get(
                "min_data_points", self._volatility_lookback_period + 5
            )  # Ensure enough data for ATR
            self._atr_high_col = vol_config.get("atr_high_col", "high")
            self._atr_low_col = vol_config.get("atr_low_col", "low")
            # This is ATR's 'close', not necessarily self._price_column
            self._atr_close_col = vol_config.get("atr_close_col", "close")
            self._max_volatility_adjustment_factor = self.config.get_decimal(
                "simulation.spread.volatility.max_adjustment_factor", Decimal("2.0")
            )

            depth_config = sim_config.get("depth", {})
            self._depth_simulation_enabled = depth_config.get("enabled", True)
            self._depth_num_levels = depth_config.get("num_levels", 5)
            self._depth_price_step_pct = self.config.get_decimal(
                "simulation.depth.price_step_pct", Decimal("0.001")
            )
            self._depth_base_volume = self.config.get_decimal(
                "simulation.depth.base_volume", Decimal("10.0")
            )
            self._depth_volume_decay_factor = self.config.get_decimal(
                "simulation.depth.volume_decay_factor", Decimal("0.8")
            )
            self._depth_price_precision = self.config.get_int(
                "simulation.depth.price_precision", 8
            )
            self._depth_volume_precision = self.config.get_int(
                "simulation.depth.volume_precision", 4
            )

            self.logger.info(
                f"Loaded simulation config via ConfigManager: "
                f"price_column='{self._price_column}', "
                f"default_spread_pct={self._default_spread_pct}",
                extra={"source_module": self._source_module},
            )
            self.logger.info(
                f"Volatility params: enabled={self._volatility_enabled}, "
                f"lookback={self._volatility_lookback_period}, "
                f"multiplier={self._volatility_multiplier}, "
                f"max_factor={self._max_volatility_adjustment_factor}",
                extra={"source_module": self._source_module},
            )
            self.logger.info(
                f"Loaded depth params via ConfigManager: "
                f"enabled={self._depth_simulation_enabled}, "
                f"levels={self._depth_num_levels}, "
                f"price_step_pct={self._depth_price_step_pct}",
                extra={"source_module": self._source_module},
            )

        else:  # ConfigManager is NOT provided, use defaults
            self.logger.warning(
                "ConfigManager not provided. Using default simulation "
                "parameters for SimulatedMarketPriceService.",
                extra={"source_module": self._source_module},
            )
            self._price_column = "close"
            self._default_spread_pct = Decimal("0.1")
            self._pair_specific_spread_config = {}
            self._volatility_multiplier = Decimal("1.5")

            # Defaults for volatility
            self._volatility_enabled = True if ta is not None else False
            self._volatility_lookback_period = 14
            self._min_volatility_data_points = self._volatility_lookback_period + 5
            self._atr_high_col = "high"
            self._atr_low_col = "low"
            self._atr_close_col = "close"
            self._max_volatility_adjustment_factor = Decimal("2.0")

            # Defaults for market depth simulation
            self._depth_simulation_enabled = True
            self._depth_num_levels = 5
            self._depth_price_step_pct = Decimal("0.001")
            self._depth_base_volume = Decimal("10.0")
            self._depth_volume_decay_factor = Decimal("0.8")
            self._depth_price_precision = 8
            self._depth_volume_precision = 4

            self.logger.info(
                f"Using default simulation config: price_column='{self._price_column}', "
                f"default_spread_pct={self._default_spread_pct}",
                extra={"source_module": self._source_module},
            )
            self.logger.info(
                f"Using default volatility params: enabled={self._volatility_enabled}, "
                f"lookback={self._volatility_lookback_period}, "
                f"multiplier={self._volatility_multiplier}, "
                f"max_factor={self._max_volatility_adjustment_factor}",
                extra={"source_module": self._source_module},
            )
            self.logger.info(
                f"Using default depth params: enabled={self._depth_simulation_enabled}, "
                f"levels={self._depth_num_levels}, "
                f"price_step_pct={self._depth_price_step_pct}",
                extra={"source_module": self._source_module},
            )

    def _get_atr_dataframe_slice(
        self, trading_pair: str, pair_data_full: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """Get the relevant slice of data for ATR calculation."""
        if not isinstance(self._current_timestamp, datetime):
            self.logger.error(
                f"Internal error: _current_timestamp is not a datetime object for {trading_pair} "
                f"at {self._current_timestamp} before slicing for ATR.",
                extra={"source_module": self._source_module},
            )
            return None  # Should be unreachable

        try:
            timestamp_for_slice = pd.Timestamp(self._current_timestamp)
        except Exception as e:  # Handle potential errors during Timestamp conversion
            self.logger.error(
                f"Error converting _current_timestamp to pd.Timestamp for {trading_pair} "
                f"at {self._current_timestamp}: {e!r}",
                extra={"source_module": self._source_module},
            )
            return None

        df_slice = pair_data_full.loc[:timestamp_for_slice]

        if len(df_slice) < self._min_volatility_data_points:
            self.logger.debug(
                f"Not enough data points ({len(df_slice)} < {self._min_volatility_data_points}) "
                f"for {trading_pair} at {self._current_timestamp} to calculate ATR.",
                extra={"source_module": self._source_module},
            )
            return None
        return df_slice

    def _calculate_atr_from_slice(
        self, df_slice: pd.DataFrame, trading_pair: str
    ) -> Optional[Decimal]:
        """Calculate ATR from a given data slice."""
        required_atr_cols = {self._atr_high_col, self._atr_low_col, self._atr_close_col}
        missing_cols = required_atr_cols - set(df_slice.columns)
        if missing_cols:
            self.logger.warning(
                f"Missing columns {missing_cols} required for ATR calculation for {trading_pair}.",
                extra={"source_module": self._source_module},
            )
            return None

        try:
            high_series = pd.to_numeric(df_slice[self._atr_high_col], errors="coerce")
            low_series = pd.to_numeric(df_slice[self._atr_low_col], errors="coerce")
            close_series = pd.to_numeric(df_slice[self._atr_close_col], errors="coerce")

            if (
                high_series.isnull().any()
                or low_series.isnull().any()
                or close_series.isnull().any()
            ):
                self.logger.warning(
                    f"NaN values found in HLC columns after coercion for {trading_pair}, "
                    "cannot calculate ATR.",
                    extra={"source_module": self._source_module},
                )
                return None

            atr_series = ta.atr(
                high=high_series,
                low=low_series,
                close=close_series,
                length=self._volatility_lookback_period,
            )

            if (
                atr_series is None
                or atr_series.empty
                or atr_series.iloc[-1] is None
                or pd.isna(atr_series.iloc[-1])
            ):
                self.logger.debug(
                    f"ATR calculation returned None or NaN for {trading_pair} "
                    f"at {self._current_timestamp}.",
                    extra={"source_module": self._source_module},
                )
                return None
            return Decimal(str(atr_series.iloc[-1]))
        except Exception as e:
            self.logger.error(
                f"Error during ATR calculation for {trading_pair}: {e!r}",
                exc_info=True,
                extra={"source_module": self._source_module},
            )
            return None

    def _calculate_normalized_atr(self, trading_pair: str) -> Optional[Decimal]:
        """Calculate ATR and normalize it by the current close price."""
        if not self._volatility_enabled or ta is None:
            self.logger.debug(
                "Volatility adjustment or pandas_ta is disabled, cannot calculate ATR.",
                extra={"source_module": self._source_module},
            )
            return None

        if self._current_timestamp is None:
            self.logger.warning(
                f"Cannot calculate ATR for {trading_pair}: current_timestamp is not set.",
                extra={"source_module": self._source_module},
            )
            return None

        pair_data_full = self.historical_data.get(trading_pair)
        if pair_data_full is None or pair_data_full.empty:
            self.logger.debug(
                f"No historical data for {trading_pair} to calculate ATR.",
                extra={"source_module": self._source_module},
            )
            return None

        if not pair_data_full.index.is_monotonic_increasing:
            pair_data_full = pair_data_full.sort_index()
            self.historical_data[trading_pair] = pair_data_full

        if not isinstance(pair_data_full.index, pd.DatetimeIndex):
            self.logger.warning(
                f"Cannot calculate ATR for {trading_pair} at {self._current_timestamp}: "
                f"DataFrame index is type {type(pair_data_full.index).__name__}, "
                "not DatetimeIndex.",
                extra={"source_module": self._source_module},
            )
            return None

        df_slice = self._get_atr_dataframe_slice(trading_pair, pair_data_full)
        if df_slice is None:
            return None

        latest_atr = self._calculate_atr_from_slice(df_slice, trading_pair)
        if latest_atr is None or latest_atr <= Decimal(0):
            self.logger.debug(
                f"Calculated ATR is zero, negative, or None ({latest_atr}) for {trading_pair}.",
                extra={"source_module": self._source_module},
            )
            return None

        current_close_price = self._get_latest_price_at_current_time(trading_pair)
        if current_close_price is None or current_close_price <= Decimal(0):
            self.logger.debug(
                f"Could not get a positive current close price for {trading_pair} "
                "to normalize ATR.",
                extra={"source_module": self._source_module},
            )
            return None

        normalized_atr = latest_atr / current_close_price
        self.logger.debug(
            f"Calculated normalized ATR for {trading_pair}: {normalized_atr} "
            f"(ATR: {latest_atr}, Close: {current_close_price})",
            extra={"source_module": self._source_module},
        )
        return normalized_atr

    def update_time(self, timestamp: datetime) -> None:
        """Update the current simulation time."""
        self.logger.debug(
            f"Updating simulated time to: {timestamp}",
            extra={"source_module": self._source_module},
        )
        self._current_timestamp = timestamp

    def _get_latest_price_at_current_time(self, trading_pair: str) -> Optional[Decimal]:
        """Get the latest known price for a trading pair at the current simulation time.

        This is an internal method.

        Args
        ----
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns
        -------
            The configured price (e.g. 'close') as a Decimal, or None if unavailable.
        """
        if self._current_timestamp is None:
            self.logger.error(
                "Cannot get latest price: Simulation time not set.",
                extra={"source_module": self._source_module},
            )
            return None

        pair_data = self.historical_data.get(trading_pair)
        if pair_data is None:
            if trading_pair.count("/") == 1:
                base, quote = trading_pair.split("/")
                if base == quote:
                    return Decimal("1.0")
            self.logger.warning(
                f"No historical data found for trading pair: {trading_pair}",
                extra={"source_module": self._source_module},
            )
            return None

        try:
            price_column_to_use = self._price_column
            if price_column_to_use not in pair_data.columns:
                self.logger.error(
                    f"Configured price column '{price_column_to_use}' not found in data "
                    f"for {trading_pair}.",
                    extra={"source_module": self._source_module},
                )
                return None

            # Ensure data is sorted for asof
            if not pair_data.index.is_monotonic_increasing:
                self.logger.debug(
                    f"Data for {trading_pair} is not sorted by index. "
                    "Sorting now for price lookup.",
                    extra={"source_module": self._source_module},
                )
                pair_data = pair_data.sort_index()
                self.historical_data[trading_pair] = pair_data

            # Use asof for efficient lookup of the latest value at or before current_timestamp
            # This handles cases where exact timestamp might not be in the index or
            # falls between bars.
            price_series = pair_data[price_column_to_use]
            price_at_timestamp = price_series.asof(self._current_timestamp)

            if pd.isna(price_at_timestamp):
                self.logger.warning(
                    f"Could not find price for {trading_pair} using column "
                    f"'{price_column_to_use}' at or before {self._current_timestamp} "
                    "(asof returned NaN).",
                    extra={"source_module": self._source_module},
                )
                return None

            return Decimal(str(price_at_timestamp))

        except Exception as e:
            self.logger.error(
                f"Error retrieving latest price for {trading_pair} at {self._current_timestamp} "
                f"using column '{price_column_to_use}': {e!r}",
                exc_info=True,
                extra={"source_module": self._source_module},
            )
            return None

    # --- Interface Alignment Methods (as per MarketPriceService ABC) ---

    async def start(self) -> None:
        """Initialize the simulated service (no-op for simulation)."""
        self.logger.info(
            "SimulatedMarketPriceService started.", extra={"source_module": self._source_module}
        )
        # No external connections needed for simulation

    async def stop(self) -> None:
        """Stop the simulated service (no-op for simulation)."""
        self.logger.info(
            "SimulatedMarketPriceService stopped.", extra={"source_module": self._source_module}
        )
        # No external connections

    async def get_latest_price(
        self, trading_pair: str
    ) -> Optional[Decimal]:  # Changed return type
        """Get the latest known price at the current simulation time.

        Returns the price as a Decimal or None.
        """
        price = self._get_latest_price_at_current_time(trading_pair)
        return price  # Return Decimal or None directly

    async def get_bid_ask_spread(
        self, trading_pair: str
    ) -> Optional[Tuple[Decimal, Decimal]]:  # Changed return type
        """Get the simulated bid and ask prices at the current simulation time.

        Returns a tuple (bid, ask) or None.
        """
        close_price = self._get_latest_price_at_current_time(trading_pair)
        if close_price is None:
            return None

        try:
            pair_specific_cfg = self._pair_specific_spread_config.get(trading_pair, {})
            base_spread_pct_str = pair_specific_cfg.get("base_pct", str(self._default_spread_pct))
            base_spread_pct = Decimal(base_spread_pct_str)

            final_spread_pct = base_spread_pct

            if self._volatility_enabled:
                normalized_atr = self._calculate_normalized_atr(trading_pair)
                if normalized_atr is not None and normalized_atr > Decimal(0):
                    volatility_impact = normalized_atr * self._volatility_multiplier
                    spread_adjustment_factor = Decimal(1) + volatility_impact

                    # Cap the adjustment factor
                    if spread_adjustment_factor > self._max_volatility_adjustment_factor:
                        spread_adjustment_factor = self._max_volatility_adjustment_factor
                        self.logger.debug(
                            f"Spread adjustment factor for {trading_pair} capped at "
                            f"{self._max_volatility_adjustment_factor}.",
                            extra={"source_module": self._source_module},
                        )

                    final_spread_pct = base_spread_pct * spread_adjustment_factor
                    self.logger.debug(
                        f"Adjusted spread for {trading_pair} due to volatility: "
                        f"final_spread_pct={final_spread_pct:.6f} "
                        f"(base={base_spread_pct:.6f}, norm_atr={normalized_atr:.6f}, "
                        f"factor={spread_adjustment_factor:.4f})",
                        extra={"source_module": self._source_module},
                    )
                else:
                    self.logger.debug(
                        f"Could not calculate normalized ATR or it was non-positive for "
                        f"{trading_pair}. Using base spread.",
                        extra={"source_module": self._source_module},
                    )

            if final_spread_pct < Decimal(0):
                self.logger.warning(
                    f"Final spread_pct ({final_spread_pct}) is negative for {trading_pair} "
                    "after adjustments. Clamping to 0.",
                    extra={"source_module": self._source_module},
                )
                final_spread_pct = Decimal(0)

            half_spread_amount = close_price * (final_spread_pct / Decimal(200))

            bid = close_price - half_spread_amount
            ask = close_price + half_spread_amount

            if bid >= ask:
                if bid == ask:
                    self.logger.debug(
                        f"Calculated zero spread for {trading_pair} at {close_price} "
                        f"with final_spread_pct {final_spread_pct}. Returning as is.",
                        extra={"source_module": self._source_module},
                    )
                    return (bid, ask)  # Return as tuple
                else:
                    self.logger.warning(
                        f"Simulated spread resulted in bid ({bid}) >= ask ({ask}) for "
                        f"{trading_pair} with final_spread_pct {final_spread_pct}. "
                        "Returning None.",
                        extra={"source_module": self._source_module},
                    )
                    return None

            return (bid, ask)  # Return as tuple

        except Exception as e:
            self.logger.error(
                f"Error calculating simulated spread for {trading_pair}: {e!r}",
                exc_info=True,
                extra={"source_module": self._source_module},
            )
            return None

    async def get_price_timestamp(self, trading_pair: str) -> Optional[datetime]:
        """Get the simulation timestamp for which the current price is valid."""
        price = self._get_latest_price_at_current_time(trading_pair)
        if price is not None and self._current_timestamp is not None:
            # Ensure timestamp is timezone-aware if it originated with timezone
            if (
                self._current_timestamp.tzinfo is None
                and self._current_timestamp.utcoffset() is None
            ):
                # If a naive datetime is somehow set, log warning or make UTC
                # For simulation, _current_timestamp should consistently be UTC.
                # Example usage in file implies it is timezone-aware (pytz.UTC).
                pass  # Assuming _current_timestamp is already correctly UTC and aware
            return self._current_timestamp
        else:
            return None

    async def is_price_fresh(self, trading_pair: str, max_age_seconds: float = 60.0) -> bool:
        """Check if price data is available at the current simulation time."""
        price_ts = await self.get_price_timestamp(trading_pair)
        if price_ts is None:
            return False

        # In simulation, "freshness" means data exists for the current sim time.
        # The max_age_seconds could be interpreted as how far back from self._current_timestamp
        # the data point returned by .asof() could be, but .asof() returns the *latest* before
        # or at self._current_timestamp. So, if get_price_timestamp returns a value, it is
        # considered "fresh" for that simulation instant.
        return True

    async def get_order_book_snapshot(
        self, trading_pair: str
    ) -> Optional[Dict[str, List[List[str]]]]:
        """Generate a simulated order book snapshot with market depth.

        Args
        ----
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns
        -------
            A dictionary with 'bids' and 'asks' lists, or None if depth cannot be generated.
            Each inner list is [price_str, volume_str].
        """
        if not self._depth_simulation_enabled:
            self.logger.debug(
                f"Market depth simulation is disabled. Skipping for {trading_pair}.",
                extra={"source_module": self._source_module},
            )
            return None

        bbo = await self.get_bid_ask_spread(trading_pair)
        if bbo is None:
            self.logger.warning(
                f"Could not retrieve BBO for {trading_pair}. Cannot generate order book.",
                extra={"source_module": self._source_module},
            )
            return None

        best_bid_price = bbo[0]
        best_ask_price = bbo[1]

        if best_bid_price <= Decimal(0) or best_ask_price <= Decimal(0):
            self.logger.warning(
                f"BBO for {trading_pair} has non-positive price ({best_bid_price}, "
                f"{best_ask_price}). Cannot generate order book.",
                extra={"source_module": self._source_module},
            )
            return None

        if best_bid_price > best_ask_price:  # Should not happen if get_bid_ask_spread is correct
            self.logger.error(
                f"Best bid {best_bid_price} is greater than best ask {best_ask_price} "
                f"for {trading_pair}. Order book invalid.",
                extra={"source_module": self._source_module},
            )
            return None

        bids_levels: List[List[str]] = []
        asks_levels: List[List[str]] = []

        # Initialize current prices for stepping through levels
        current_level_bid_price = best_bid_price
        current_level_ask_price = best_ask_price
        current_level_volume = self._depth_base_volume

        price_format_str = f"{{:.{self._depth_price_precision}f}}"
        volume_format_str = f"{{:.{self._depth_volume_precision}f}}"

        for i in range(self._depth_num_levels):
            # Quantize price for consistent string formatting
            # and to avoid floating point artifacts in steps
            # This step is crucial if _depth_price_step_pct is
            # very small or prices are large
            quantizer = Decimal("1e-" + str(self._depth_price_precision))
            current_bid_for_level = current_level_bid_price.quantize(
                quantizer, rounding=ROUND_DOWN
            )
            current_ask_for_level = current_level_ask_price.quantize(quantizer, rounding=ROUND_UP)
            current_volume_for_level = current_level_volume.quantize(
                Decimal("1e-" + str(self._depth_volume_precision)), rounding=ROUND_DOWN
            )

            if current_volume_for_level <= Decimal(0):
                self.logger.debug(
                    f"Volume for level {i+1} for {trading_pair} became zero or negative. "
                    "Stopping depth generation.",
                    extra={"source_module": self._source_module},
                )
                break  # Stop if volume is zero or less

            if i == 0:  # BBO level
                bids_levels.append([
                    price_format_str.format(current_bid_for_level),
                    volume_format_str.format(current_volume_for_level),
                ])
                asks_levels.append([
                    price_format_str.format(current_ask_for_level),
                    volume_format_str.format(current_volume_for_level),
                ])
            else:
                # Ensure subsequent levels do not cross or create invalid book structure
                # (e.g. new bid > best_ask or new_ask < best_bid)
                # More importantly, ensure new_bid < new_ask for this level itself
                if current_bid_for_level >= current_ask_for_level and not (
                    current_bid_for_level == best_bid_price
                    and current_ask_for_level == best_ask_price
                    and best_bid_price == best_ask_price
                ):
                    self.logger.debug(
                        f"At level {i+1}, calculated bid {current_bid_for_level} "
                        f"crossed/met ask {current_ask_for_level} for {trading_pair}. "
                        "Stopping depth here.",
                        extra={"source_module": self._source_module},
                    )
                    break
                if current_bid_for_level <= Decimal(0):
                    self.logger.debug(
                        f"At level {i+1}, calculated bid {current_bid_for_level} "
                        f"is zero/negative for {trading_pair}. Stopping bid depth here.",
                        extra={"source_module": self._source_module},
                    )
                    # Allow asks to continue if bids hit zero
                else:
                    bids_levels.append([
                        price_format_str.format(current_bid_for_level),
                        volume_format_str.format(current_volume_for_level),
                    ])

                asks_levels.append([
                    price_format_str.format(current_ask_for_level),
                    volume_format_str.format(current_volume_for_level),
                ])

            # Calculate prices for the *next* level (i+1)
            if current_level_bid_price > Decimal(0):  # Avoid issues if bid is already zero
                step_bid = current_level_bid_price * self._depth_price_step_pct
                current_level_bid_price = current_level_bid_price - step_bid

            step_ask = current_level_ask_price * self._depth_price_step_pct
            current_level_ask_price = current_level_ask_price + step_ask
            current_level_volume *= self._depth_volume_decay_factor

        # Filter out levels where bid became non-positive if asks continued
        bids_levels = [level for level in bids_levels if Decimal(level[0]) > Decimal(0)]

        if not bids_levels and not asks_levels:
            self.logger.info(
                f"Generated an empty order book for {trading_pair} (e.g. BBO was zero spread "
                "and only one level requested, or other edge case).",
                extra={"source_module": self._source_module},
            )
            # Return empty book as per spec, or None if this is an error condition for the caller
            # For now, returning empty lists within the dict structure.

        return {"bids": bids_levels, "asks": asks_levels}

    # --- End Interface Alignment Methods ---

    async def _get_direct_or_reverse_price(
        self, from_currency: str, to_currency: str
    ) -> Optional[Tuple[Decimal, bool]]:  # Returns (price, is_direct_rate)
        """Get direct or reverse conversion rate."""
        # Direct conversion: from_currency/to_currency
        pair1 = f"{from_currency}/{to_currency}"
        price1 = await self.get_latest_price(pair1)  # Returns Optional[Decimal]
        if price1 is not None and price1 > 0:
            return price1, True

        # Reverse conversion: to_currency/from_currency
        pair2 = f"{to_currency}/{from_currency}"
        price2 = await self.get_latest_price(pair2)  # Returns Optional[Decimal]
        if price2 is not None and price2 > 0:
            return price2, False
        return None

    async def _get_cross_conversion_price(
        self, from_amount: Decimal, from_currency: str, to_currency: str, intermediary: str
    ) -> Optional[Decimal]:
        """Get cross-conversion via an intermediary currency."""
        # Path: from_currency -> intermediary -> to_currency
        from_to_intermediary_rate_info = await self._get_direct_or_reverse_price(
            from_currency, intermediary
        )

        if from_to_intermediary_rate_info:
            rate1, is_direct1 = from_to_intermediary_rate_info
            amount_in_intermediary = from_amount * rate1 if is_direct1 else from_amount / rate1

            intermediary_to_target_rate_info = await self._get_direct_or_reverse_price(
                intermediary, to_currency
            )
            if intermediary_to_target_rate_info:
                rate2, is_direct2 = intermediary_to_target_rate_info
                if is_direct2:
                    return amount_in_intermediary * rate2
                else:
                    return amount_in_intermediary / rate2
        return None

    async def convert_amount(
        self, from_amount: Decimal, from_currency: str, to_currency: str
    ) -> Optional[Decimal]:
        """Convert an amount from one currency to another using available market data."""
        if not isinstance(from_amount, Decimal):
            self.logger.warning(
                f"convert_amount received non-Decimal from_amount: {type(from_amount)}. "
                "Attempting conversion.",
                extra={"source_module": self._source_module},
            )
            try:
                from_amount = Decimal(str(from_amount))
            except Exception:
                self.logger.error(
                    "Could not convert from_amount to Decimal in convert_amount.",
                    exc_info=True,
                    extra={"source_module": self._source_module},
                )
                return None

        if from_currency == to_currency:
            return from_amount

        direct_or_reverse_info = await self._get_direct_or_reverse_price(
            from_currency, to_currency
        )
        if direct_or_reverse_info:
            rate, is_direct = direct_or_reverse_info
            try:
                return from_amount * rate if is_direct else from_amount / rate
            except ZeroDivisionError:
                self.logger.error(
                    f"Zero price encountered for pair during direct/reverse conversion "
                    f"({from_currency}/{to_currency} or {to_currency}/{from_currency}).",
                    extra={"source_module": self._source_module},
                )
                return None

        # Cross-conversion via USD (or a common intermediary from config)
        intermediary_currency = "USD"  # Could be made configurable
        if from_currency != intermediary_currency and to_currency != intermediary_currency:
            try:
                converted_amount = await self._get_cross_conversion_price(
                    from_amount, from_currency, to_currency, intermediary_currency
                )
                if converted_amount is not None:
                    return converted_amount
            except ZeroDivisionError:
                self.logger.error(
                    f"Zero price encountered during {intermediary_currency}-mediated "
                    f"cross-conversion from {from_currency} to {to_currency}.",
                    extra={"source_module": self._source_module},
                )
                return None

        self.logger.warning(
            f"Could not convert {from_amount} {from_currency} to {to_currency}. "
            f"No direct, reverse, or {intermediary_currency}-mediated path found.",
            extra={"source_module": self._source_module},
        )
        return None


# Example Usage
async def main() -> None:  # Made async
    """Run example demonstrating the SimulatedMarketPriceService functionality."""
    # Basic logging setup for example
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(source_module)s] - %(message)s",
    )
    main_logger = logging.getLogger("SimulatedMarketPriceServiceExample")

    # Create dummy historical data
    # Ensure index is sorted and values are Decimal for consistency
    idx1 = pd.to_datetime(
        [
            "2023-01-01 00:00:00",
            "2023-01-01 00:01:00",
            "2023-01-01 00:02:00",
            "2023-01-01 00:03:00",
            "2023-01-01 00:04:00",
        ],
        utc=True,
    )
    data1 = pd.DataFrame(
        {
            "open": [
                Decimal("10.0"),
                Decimal("11.0"),
                Decimal("12.0"),
                Decimal("11.0"),
                Decimal("13.0"),
            ],
            "close": [
                Decimal("11.0"),
                Decimal("12.0"),
                Decimal("11.5"),
                Decimal("13.0"),
                Decimal("14.0"),
            ],
            "high": [  # Added for ATR calculation
                Decimal("11.5"),
                Decimal("12.5"),
                Decimal("12.0"),
                Decimal("13.5"),
                Decimal("14.0"),
            ],
            "low": [  # Added for ATR calculation
                Decimal("9.5"),
                Decimal("10.5"),
                Decimal("11.0"),
                Decimal("10.5"),
                Decimal("12.5"),
            ],
        },
        index=idx1,
    )

    idx2 = pd.to_datetime(
        [
            "2023-01-01 00:00:00",
            "2023-01-01 00:01:00",
            "2023-01-01 00:02:00",
            "2023-01-01 00:03:00",
            "2023-01-01 00:04:00",
        ],
        utc=True,
    )
    data2 = pd.DataFrame(
        {
            "open": [
                Decimal("1.0"),
                Decimal("1.1"),
                Decimal("1.2"),
                Decimal("1.1"),
                Decimal("1.0"),
            ],
            "close": [
                Decimal("1.1"),
                Decimal("1.2"),
                Decimal("1.15"),
                Decimal("1.1"),
                Decimal("1.05"),
            ],
            "high": [  # Added for ATR calculation
                Decimal("1.15"),
                Decimal("1.25"),
                Decimal("1.20"),
                Decimal("1.15"),
                Decimal("1.10"),
            ],
            "low": [  # Added for ATR calculation
                Decimal("0.95"),
                Decimal("1.05"),
                Decimal("1.10"),
                Decimal("1.05"),
                Decimal("1.00"),
            ],
        },
        index=idx2,
    )

    hist_data = {"BTC/USD": data1, "ETH/USD": data2}

    # Initialize without ConfigManager to test fallback
    price_service = SimulatedMarketPriceService(hist_data, logger=main_logger)
    await price_service.start()

    # Test 1: Get price at an exact timestamp
    # No need to import pytz if pandas datetime index is already timezone-aware UTC
    ts1 = datetime(2023, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
    price_service.update_time(ts1)

    btc_price_info1 = await price_service.get_latest_price("BTC/USD")
    eth_price_info1 = await price_service.get_latest_price("ETH/USD")
    usd_price_info = await price_service.get_latest_price("USD/USD")
    btc_spread_info1 = await price_service.get_bid_ask_spread("BTC/USD")

    btc_price1 = btc_price_info1  # Now Optional[Decimal]
    eth_price1 = eth_price_info1  # Now Optional[Decimal]
    usd_price = usd_price_info  # Now Optional[Decimal]

    main_logger.info(f"Prices at {ts1}: BTC={btc_price1}, ETH={eth_price1}, USD/USD={usd_price}")
    if btc_spread_info1:
        main_logger.info(
            f"BTC Spread at {ts1}: Bid={btc_spread_info1[0]:.2f}, "  # Added formatting
            f"Ask={btc_spread_info1[1]:.2f}"
        )
    else:
        main_logger.info(f"BTC Spread at {ts1}: None")

    # Test 2: Get price between timestamps (should get previous close)
    ts2 = datetime(2023, 1, 1, 0, 1, 30, tzinfo=timezone.utc)
    price_service.update_time(ts2)
    btc_price2 = await price_service.get_latest_price("BTC/USD")  # Directly Decimal | None
    main_logger.info(f"Prices at {ts2}: BTC={btc_price2} (Should be same as {ts1} close: 12.0)")

    # Test 3: Get price before data starts
    ts3 = datetime(2022, 12, 31, 23, 59, 0, tzinfo=timezone.utc)
    price_service.update_time(ts3)
    btc_price3 = await price_service.get_latest_price("BTC/USD")  # Directly Decimal | None
    main_logger.info(f"Prices at {ts3}: BTC={btc_price3} (Should be None)")

    # Test 4: Unknown pair
    price_service.update_time(ts1)
    unknown_price = await price_service.get_latest_price("LTC/USD")  # Directly Decimal | None
    main_logger.info(f"Prices at {ts1}: LTC={unknown_price} (Should be None)")

    # Test 5: Price timestamp and freshness
    ts_btc = await price_service.get_price_timestamp("BTC/USD")
    is_fresh_btc = await price_service.is_price_fresh("BTC/USD")
    main_logger.info(f"BTC Price Timestamp at {ts1}: {ts_btc}, Fresh: {is_fresh_btc}")

    ts_ltc = await price_service.get_price_timestamp("LTC/USD")  # For an unknown pair
    is_fresh_ltc = await price_service.is_price_fresh("LTC/USD")
    main_logger.info(
        f"LTC Price Timestamp at {ts1}: {ts_ltc}, Fresh: {is_fresh_ltc} " "(Should be None, False)"
    )

    # Test Order Book Snapshot
    btc_order_book = await price_service.get_order_book_snapshot("BTC/USD")
    if btc_order_book:
        main_logger.info(f"BTC/USD Order Book at {ts1}:")
        main_logger.info(f"  Bids: {btc_order_book.get('bids')}")  # Use .get for safety
        main_logger.info(f"  Asks: {btc_order_book.get('asks')}")  # Use .get for safety
    else:
        main_logger.info(f"Could not generate BTC/USD order book at {ts1}.")

    # Test with a zero-spread scenario if possible by manipulating data or config
    # For now, this test relies on the default 0.1% spread which likely won't be zero.

    # Test conversion
    amount_to_convert = Decimal("2.0")
    # Add ETH/BTC pair for testing cross-conversion if BTC/ETH is not direct
    idx_eth_btc = pd.to_datetime(["2023-01-01 00:01:00"], utc=True)
    data_eth_btc = pd.DataFrame(
        {"close": [Decimal("0.05")], "high": [Decimal("0.05")], "low": [Decimal("0.05")]},
        # Added H/L
        index=idx_eth_btc,
    )  # 1 ETH = 0.05 BTC
    price_service.historical_data["ETH/BTC"] = data_eth_btc

    # BTC -> USD (direct) -> ETH (reverse of ETH/USD)
    # If BTC/USD = 12, ETH/USD = 1.2, then 2 BTC = 24 USD. 24 USD / 1.2 (ETH/USD) = 20 ETH
    converted_btc_to_eth_via_usd = await price_service.convert_amount(
        amount_to_convert, "BTC", "ETH"
    )
    main_logger.info(
        f"Converting {amount_to_convert} BTC to ETH via USD: {converted_btc_to_eth_via_usd}"
    )

    # Add direct pair for BTC/ETH
    idx_btc_eth = pd.to_datetime(["2023-01-01 00:01:00"], utc=True)
    # If 1 ETH = 0.05 BTC, then 1 BTC = 1/0.05 = 20 ETH
    data_btc_eth = pd.DataFrame(
        {"close": [Decimal("20.0")], "high": [Decimal("20.0")], "low": [Decimal("20.0")]},
        # Added H/L
        index=idx_btc_eth,
    )
    price_service.historical_data["BTC/ETH"] = data_btc_eth
    converted_btc_to_eth_direct = await price_service.convert_amount(
        amount_to_convert, "BTC", "ETH"
    )
    main_logger.info(
        f"Converting {amount_to_convert} BTC to ETH (direct): {converted_btc_to_eth_direct}"
    )

    await price_service.stop()


if __name__ == "__main__":
    import asyncio

    # Note: Need pandas installed (`pip install pandas`)
    # No longer need pytz directly if using timezone.utc and pd.to_datetime with utc=True
    try:
        import pandas  # noqa

        # import pandas_ta # noqa - ensure this is also handled if used directly in main
        # import asyncio # Already imported if needed for main
        asyncio.run(main())  # Run the async main function
    except ImportError:
        print("Could not run example: pandas not installed.")
    except Exception as e:
        print(f"An error occurred during example execution: {e!r}")
