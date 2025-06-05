"""Simulate market price data for backtesting trading strategies.

This module provides a service that simulates market price data for backtesting trading strategies.
It uses historical OHLCV data to provide price information,
bid-ask spreads, and simulated order book.
It also supports volatility-adjusted spread calculation and market depth simulation.

"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
import numpy as np

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
            "default config values if ConfigManager is not provided.",
        )

        # Define a minimal interface for static type checking
        class _DummyConfigManager:
            """Minimal placeholder for ConfigManager."""

            def get(self, _key: str, default: object | None = None) -> object | None:
                """Get a value from config."""
                return default

            def get_decimal(self, _key: str, default: Decimal) -> Decimal:
                """Get a decimal value from config."""
                return default

            def get_int(self, _key: str, default: int) -> int:
                """Get an integer value from config."""
                return default

        # Use the dummy class as a fallback for type checking
        ConfigManager = _DummyConfigManager  # type: Optional[ConfigManager]

# Attempt to import pandas_ta for ATR calculation
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    log_temp = logging.getLogger(__name__)
    log_temp.warning(
        "pandas_ta library not found. ATR calculation for "
        "volatility-adjusted spread will be disabled.",
    )

_SOURCE_MODULE = "SimulatedMarketPriceService"


@dataclass
class BookLevelConstructionContext:
    """Context for constructing order book levels."""

    best_bid_price_bbo: Decimal
    best_ask_price_bbo: Decimal
    price_format_str: str
    volume_format_str: str
    trading_pair: str


class SimulatedMarketPriceService(MarketPriceService):  # Inherit from MarketPriceService
    """Provide access to the latest market prices.

    based on historical data during a backtest simulation.

    Aligns with the MarketPriceService ABC.
    """

    def __init__(
        self,
        historical_data: dict[str, pd.DataFrame],
        config_manager: ConfigManager | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the service with historical market data.

        Args:
        ----
            historical_data: A dictionary where keys are trading pairs (e.g., "XRP/USD")
                             and values are pandas DataFrames containing OHLCV data
                             indexed by timestamp (UTC).
            config_manager: An instance of ConfigManager for accessing configuration.
            logger: An instance of logging.Logger. If None, a default logger is used.
        """
        self.historical_data = historical_data
        self._current_timestamp: datetime | None = None

        # Properly handle logger and config initialization
        # Check if MarketPriceService ABC has set these attributes
        if hasattr(self, 'logger') and self.logger is not None:
            # Logger was set by parent class
            pass
        elif logger is not None:
            # Use provided logger
            self.logger = logger
        else:
            # Create default logger
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
            
        if hasattr(self, 'config') and self.config is not None:
            # Config was set by parent class
            pass
        elif config_manager is not None:
            # Use provided config
            self.config = config_manager
        else:
            # Create dummy config manager for defaults
            self.config = _DummyConfigManager()
            
        self._source_module = _SOURCE_MODULE

        self._load_simulation_config()

        # Validate data format, including HLC columns if volatility is enabled
        self._validate_historical_data(historical_data)
        
        self.logger.info(
            "SimulatedMarketPriceService initialized.",
            extra={"source_module": self._source_module},
        )

    def _validate_historical_data(self, historical_data: dict[str, pd.DataFrame]) -> None:
        """Validate the format and content of historical data."""
        for pair, df in historical_data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning(
                    "Historical data for %s does not have a DatetimeIndex. "
                    "This may cause issues with time-based lookups.",
                    pair,
                    extra={"source_module": self._source_module},
                )
                # Attempt to convert index to DatetimeIndex if possible
                try:
                    df.index = pd.to_datetime(df.index, utc=True)
                    historical_data[pair] = df
                    self.logger.info(
                        "Successfully converted index to DatetimeIndex for %s",
                        pair,
                        extra={"source_module": self._source_module},
                    )
                except Exception as e:
                    self.logger.error(
                        "Failed to convert index to DatetimeIndex for %s: %s",
                        pair,
                        str(e),
                        extra={"source_module": self._source_module},
                    )

            required_cols = {self._price_column}
            if self._volatility_enabled and ta is not None:  # Check ta availability too
                required_cols.update({self._atr_high_col, self._atr_low_col, self._atr_close_col})

            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                self.logger.warning(
                    "Historical data for %s is missing required columns: %s. "
                    "Available columns: %s",
                    pair,
                    missing_cols,
                    list(df.columns),
                    extra={"source_module": self._source_module},
                )
                # Disable features that require missing columns
                if self._price_column in missing_cols:
                    self.logger.error(
                        "Critical: Price column '%s' missing for %s. "
                        "This pair may not function properly.",
                        self._price_column,
                        pair,
                        extra={"source_module": self._source_module},
                    )
                if any(col in missing_cols for col in [self._atr_high_col, self._atr_low_col, self._atr_close_col]):
                    self.logger.warning(
                        "Disabling volatility features for %s due to missing HLC columns",
                        pair,
                        extra={"source_module": self._source_module},
                    )

    def _apply_config_values_from_manager(self) -> None:
        """Apply configuration values from the ConfigManager."""
        if self.config is None:
            # Use default values if config is not available
            self._apply_default_config_values()
            return

        sim_config = self.config.get("simulation", {})
        self._price_column = sim_config.get("price_column", "close")

        spread_config = sim_config.get("spread", {})
        self._default_spread_pct = self.config.get_decimal(
            "simulation.spread.default_pct",
            Decimal("0.1"),
        )
        self._pair_specific_spread_config = spread_config.get("pairs", {})
        self._volatility_multiplier = self.config.get_decimal(
            "simulation.spread.volatility_multiplier",
            Decimal("1.5"),
        )

        vol_config = spread_config.get("volatility", {})
        self._volatility_enabled = vol_config.get(
            "enabled",
            ta is not None,
        )  # Default true only if ta available
        self._volatility_lookback_period = vol_config.get("lookback_period", 14)
        self._min_volatility_data_points = vol_config.get(
            "min_data_points",
            self._volatility_lookback_period + 5,
        )
        self._atr_high_col = vol_config.get("atr_high_col", "high")
        self._atr_low_col = vol_config.get("atr_low_col", "low")
        self._atr_close_col = vol_config.get("atr_close_col", "close")
        self._max_volatility_adjustment_factor = self.config.get_decimal(
            "simulation.spread.volatility.max_adjustment_factor",
            Decimal("2.0"),
        )

        depth_config = sim_config.get("depth", {})
        self._depth_simulation_enabled = depth_config.get("enabled", True)
        self._depth_num_levels = depth_config.get("num_levels", 5)
        self._depth_price_step_pct = self.config.get_decimal(
            "simulation.depth.price_step_pct",
            Decimal("0.001"),
        )
        self._depth_base_volume = self.config.get_decimal(
            "simulation.depth.base_volume",
            Decimal("10.0"),
        )
        self._depth_volume_decay_factor = self.config.get_decimal(
            "simulation.depth.volume_decay_factor",
            Decimal("0.8"),
        )
        self._depth_price_precision = self.config.get_int("simulation.depth.price_precision", 8)
        self._depth_volume_precision = self.config.get_int("simulation.depth.volume_precision", 4)

        conv_config = sim_config.get("conversion", {})
        self._intermediary_conversion_currency = conv_config.get("intermediary_currency", "USD")

        self.logger.info(
            "Loaded simulation config via ConfigManager: price_column='%s', default_spread_pct=%s",
            self._price_column,
            self._default_spread_pct,
            extra={"source_module": self._source_module},
        )
        self.logger.info(
            "Volatility params: enabled=%s, lookback=%s, multiplier=%s, max_factor=%s",
            self._volatility_enabled,
            self._volatility_lookback_period,
            self._volatility_multiplier,
            self._max_volatility_adjustment_factor,
            extra={"source_module": self._source_module},
        )
        self.logger.info(
            "Loaded depth params via ConfigManager: enabled=%s, levels=%s, price_step_pct=%s",
            self._depth_simulation_enabled,
            self._depth_num_levels,
            self._depth_price_step_pct,
            extra={"source_module": self._source_module},
        )
        self.logger.info(
            "Intermediary currency for conversion: '%s'",
            self._intermediary_conversion_currency,
            extra={"source_module": self._source_module},
        )

    def _apply_default_config_values(self) -> None:
        """Apply default simulation configuration values."""
        self.logger.warning(
            "ConfigManager not provided. Using default simulation "
            "parameters for SimulatedMarketPriceService.",
            extra={"source_module": self._source_module},
        )
        self._price_column = "close"
        self._default_spread_pct = Decimal("0.1")
        self._pair_specific_spread_config = {}
        self._volatility_multiplier = Decimal("1.5")

        self._volatility_enabled = ta is not None
        self._volatility_lookback_period = 14
        self._min_volatility_data_points = self._volatility_lookback_period + 5
        self._atr_high_col = "high"
        self._atr_low_col = "low"
        self._atr_close_col = "close"
        self._max_volatility_adjustment_factor = Decimal("2.0")

        self._depth_simulation_enabled = True
        self._depth_num_levels = 5
        self._depth_price_step_pct = Decimal("0.001")
        self._depth_base_volume = Decimal("10.0")
        self._depth_volume_decay_factor = Decimal("0.8")
        self._depth_price_precision = 8
        self._depth_volume_precision = 4
        self._intermediary_conversion_currency = "USD"

        self.logger.info(
            "Using default simulation config: price_column='%s', default_spread_pct=%s",
            self._price_column,
            self._default_spread_pct,
            extra={"source_module": self._source_module},
        )
        self.logger.info(
            "Using default volatility params: enabled=%s, "
            "lookback=%s, "
            "multiplier=%s, "
            "max_factor=%s",
            self._volatility_enabled,
            self._volatility_lookback_period,
            self._volatility_multiplier,
            self._max_volatility_adjustment_factor,
            extra={"source_module": self._source_module},
        )
        self.logger.info(
            "Using default depth params: enabled=%s, levels=%s, price_step_pct=%s",
            self._depth_simulation_enabled,
            self._depth_num_levels,
            self._depth_price_step_pct,
            extra={"source_module": self._source_module},
        )
        self.logger.info(
            "Using default intermediary currency for conversion: '%s'",
            self._intermediary_conversion_currency,
            extra={"source_module": self._source_module},
        )

    def _load_simulation_config(self) -> None:
        """Load simulation-specific configurations.

        Uses defaults if config_manager is not available.
        """
        if self.config:  # ConfigManager is provided
            self._apply_config_values_from_manager()
        else:  # ConfigManager is NOT provided, use defaults
            self._apply_default_config_values()

    def _get_atr_dataframe_slice(
        self,
        trading_pair: str,
        pair_data_full: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Get the relevant slice of data for ATR calculation."""
        if not isinstance(self._current_timestamp, datetime):
            self.logger.error(
                "Internal error: _current_timestamp is not a datetime object for %s "
                "at %s before slicing for ATR.",
                trading_pair,
                self._current_timestamp,
                extra={"source_module": self._source_module},
            )
            return None  # Should be unreachable

        try:
            timestamp_for_slice = pd.Timestamp(self._current_timestamp)
        except Exception:  # Handle potential errors during Timestamp conversion
            self.logger.exception(
                "Error converting _current_timestamp to pd.Timestamp for %s at %s",
                trading_pair,
                self._current_timestamp,
                extra={"source_module": self._source_module},
            )
            return None

        df_slice = pair_data_full.loc[:timestamp_for_slice]

        if len(df_slice) < self._min_volatility_data_points:
            self.logger.debug(
                "Not enough data points (%s < %s) for %s at %s to calculate ATR.",
                len(df_slice),
                self._min_volatility_data_points,
                trading_pair,
                self._current_timestamp,
                extra={"source_module": self._source_module},
            )
            return None
        return df_slice

    def _calculate_atr_from_slice(
        self,
        df_slice: pd.DataFrame,
        trading_pair: str,
    ) -> Decimal | None:
        """Calculate ATR from a given data slice."""
        required_atr_cols = {self._atr_high_col, self._atr_low_col, self._atr_close_col}
        missing_cols = required_atr_cols - set(df_slice.columns)
        if missing_cols:
            self.logger.warning(
                "Missing columns %s required for ATR calculation for %s.",
                missing_cols,
                trading_pair,
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
                    "NaN values found in HLC columns after coercion for %s, cannot calculate ATR.",
                    trading_pair,
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
                    "ATR calculation returned None or NaN for %s at %s.",
                    trading_pair,
                    self._current_timestamp,
                    extra={"source_module": self._source_module},
                )
                return None
            return Decimal(str(atr_series.iloc[-1]))
        except Exception:
            self.logger.exception(
                "Error during ATR calculation for %s",
                trading_pair,
                extra={"source_module": self._source_module},
            )
            return None

    def _get_raw_atr_for_pair(self, trading_pair: str) -> Decimal | None:
        """Calculate the raw ATR for a given trading pair at the current time."""
        atr_to_return: Decimal | None = None

        if not self._volatility_enabled or ta is None:
            self.logger.debug(
                "Volatility adjustment or pandas_ta is disabled, cannot calculate raw ATR.",
                extra={"source_module": self._source_module},
            )
        elif self._current_timestamp is None:
            self.logger.warning(
                "Cannot calculate raw ATR for %s: current_timestamp is not set.",
                trading_pair,
                extra={"source_module": self._source_module},
            )
        else:
            pair_data_full = self.historical_data.get(trading_pair)
            if pair_data_full is None or pair_data_full.empty:
                self.logger.debug(
                    "No historical data for %s to calculate raw ATR.",
                    trading_pair,
                    extra={"source_module": self._source_module},
                )
            elif not isinstance(pair_data_full.index, pd.DatetimeIndex):
                self.logger.warning(
                    "Cannot calculate raw ATR for %s at %s: DataFrame index is type %s, "
                    "not DatetimeIndex.",
                    trading_pair,
                    self._current_timestamp,
                    type(pair_data_full.index).__name__,
                    extra={"source_module": self._source_module},
                )
            else:
                if not pair_data_full.index.is_monotonic_increasing:
                    self.logger.debug(
                        "Data for %s is not sorted by index. Sorting now for ATR calculation.",
                        trading_pair,
                        extra={"source_module": self._source_module},
                    )
                    pair_data_full = pair_data_full.sort_index()
                    self.historical_data[trading_pair] = pair_data_full

                df_slice = self._get_atr_dataframe_slice(trading_pair, pair_data_full)
                if df_slice is not None:
                    raw_atr_calculated = self._calculate_atr_from_slice(df_slice, trading_pair)
                    if raw_atr_calculated is not None and raw_atr_calculated > Decimal(0):
                        atr_to_return = raw_atr_calculated
                    else:
                        # Logging for zero/negative/None ATR is done in _calculate_atr_from_slice
                        # or the conditions leading to it (e.g. not enough data).
                        # This debug log might be redundant or could be made more specific.
                        self.logger.debug(
                            "Calculated raw ATR is zero, negative, or None for %s. "
                            "Final value: %s",
                            trading_pair,
                            raw_atr_calculated,  # Log the value for clarity
                            extra={"source_module": self._source_module},
                        )
                # If df_slice is None, logging is done in _get_atr_dataframe_slice

        if atr_to_return is not None:
            self.logger.debug(
                "Successfully calculated raw ATR for %s: %s",
                trading_pair,
                atr_to_return,
                extra={"source_module": self._source_module},
            )

        return atr_to_return

    def _calculate_normalized_atr(self, trading_pair: str) -> Decimal | None:
        """Calculate ATR and normalize it by the current close price."""
        raw_atr = self._get_raw_atr_for_pair(trading_pair)

        if raw_atr is None:
            # If raw_atr is None, logging about why would have occurred in _get_raw_atr_for_pair.
            # It also implies raw_atr was not positive if it was calculable.
            return None

        current_close_price = self._get_latest_price_at_current_time(trading_pair)
        if current_close_price is None or current_close_price <= Decimal(0):
            self.logger.debug(
                "Could not get a positive current close price for %s to normalize ATR. "
                "Raw ATR was %s.",
                trading_pair,
                raw_atr,
                extra={"source_module": self._source_module},
            )
            return None

        normalized_atr = raw_atr / current_close_price
        self.logger.debug(
            "Calculated normalized ATR for %s: %s (Raw ATR: %s, Close: %s)",
            trading_pair,
            normalized_atr,
            raw_atr,
            current_close_price,
            extra={"source_module": self._source_module},
        )
        return normalized_atr

    def update_time(self, timestamp: datetime) -> None:
        """Update the current simulation time."""
        self.logger.debug(
            "Updating simulated time to: %s",
            timestamp,
            extra={"source_module": self._source_module},
        )
        self._current_timestamp = timestamp

    def _get_price_from_dataframe_asof(
        self,
        df: pd.DataFrame,
        trading_pair: str,  # Needed for updating self.historical_data if sorted
        timestamp_to_lookup: datetime,
    ) -> Decimal | None:
        """Extract price from a DataFrame using asof, handling column checks and sorting."""
        price_col = self._price_column
        if price_col not in df.columns:
            self.logger.error(
                "Configured price column '%s' not found in data for %s.",
                price_col,
                trading_pair,
                extra={"source_module": self._source_module},
            )
            return None

        if not df.index.is_monotonic_increasing:
            self.logger.debug(
                "Data for %s is not sorted by index. Sorting now for price lookup.",
                trading_pair,
                extra={"source_module": self._source_module},
            )
            df = df.sort_index()
            self.historical_data[trading_pair] = df  # Update the stored DataFrame

        price_series = df[price_col]
        price_at_timestamp = price_series.asof(timestamp_to_lookup)

        if pd.isna(price_at_timestamp):
            self.logger.warning(
                "Could not find price for %s using column '%s' at or before %s "
                "(asof returned NaN).",
                trading_pair,
                price_col,
                timestamp_to_lookup,
                extra={"source_module": self._source_module},
            )
            return None

        return Decimal(str(price_at_timestamp))

    def _get_latest_price_at_current_time(self, trading_pair: str) -> Decimal | None:
        """Get the latest known price for a trading pair at the current simulation time."""
        current_ts = self._current_timestamp

        if current_ts is None:
            self.logger.error(
                "Cannot get latest price: Simulation time not set.",
                extra={"source_module": self._source_module},
            )
            return None

        # Handle self-referential pairs like "USD/USD"
        if trading_pair.count("/") == 1:
            base, quote = trading_pair.split("/")
            if base == quote:
                return Decimal("1.0")

        # General lookup for all other cases
        pair_data_df = self.historical_data.get(trading_pair)

        if pair_data_df is None:
            self.logger.warning(
                "No historical data found for trading pair: %s",
                trading_pair,
                extra={"source_module": self._source_module},
            )
            return None

        # At this point, data exists; attempt to process it using the helper
        try:
            return self._get_price_from_dataframe_asof(pair_data_df, trading_pair, current_ts)
        except Exception:
            # Use self._price_column for logging in except block
            self.logger.exception(
                "Error retrieving latest price for %s at %s using column '%s'",
                trading_pair,
                current_ts,
                self._price_column,
                extra={"source_module": self._source_module},
            )
            return None

    # --- Interface Alignment Methods (as per MarketPriceService ABC) ---

    async def start(self) -> None:
        """Initialize the simulated service (no-op for simulation)."""
        self.logger.info(
            "SimulatedMarketPriceService started.",
            extra={"source_module": self._source_module},
        )
        # No external connections needed for simulation

    async def stop(self) -> None:
        """Stop the simulated service (no-op for simulation)."""
        self.logger.info(
            "SimulatedMarketPriceService stopped.",
            extra={"source_module": self._source_module},
        )
        # No external connections

    async def get_latest_price(
        self,
        trading_pair: str,
    ) -> Decimal | None:  # Changed return type
        """Get the latest known price at the current simulation time.

        Returns the price as a Decimal or None.
        """
        return self._get_latest_price_at_current_time(trading_pair)

    async def get_bid_ask_spread(
        self,
        trading_pair: str,
    ) -> tuple[Decimal, Decimal] | None:  # Changed return type
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
                            "Spread adjustment factor for %s capped at %s.",
                            trading_pair,
                            self._max_volatility_adjustment_factor,
                            extra={"source_module": self._source_module},
                        )

                    final_spread_pct = base_spread_pct * spread_adjustment_factor
                    self.logger.debug(
                        "Adjusted spread for %s due to volatility: "
                        "final_spread_pct=%.6f (base=%.6f, norm_atr=%.6f, factor=%.4f)",
                        trading_pair,
                        final_spread_pct,
                        base_spread_pct,
                        normalized_atr,
                        spread_adjustment_factor,
                        extra={"source_module": self._source_module},
                    )
                else:
                    self.logger.debug(
                        "Could not calculate normalized ATR or it was non-positive for %s. "
                        "Using base spread.",
                        trading_pair,
                        extra={"source_module": self._source_module},
                    )

            if final_spread_pct < Decimal(0):
                self.logger.warning(
                    "Final spread_pct (%s) is negative for %s after adjustments. Clamping to 0.",
                    final_spread_pct,
                    trading_pair,
                    extra={"source_module": self._source_module},
                )
                final_spread_pct = Decimal(0)

            # Call the helper method for the final bid/ask calculation and comparison
            return self._calculate_bid_ask_tuple(close_price, final_spread_pct, trading_pair)

        except Exception:  # pylint: disable=broad-except
            self.logger.exception(
                "Error calculating simulated spread for %s",
                trading_pair,
                extra={"source_module": self._source_module},
            )
            return None

    async def get_price_timestamp(self, trading_pair: str) -> datetime | None:
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
        return None

    async def get_raw_atr(self, trading_pair: str) -> Decimal | None:
        """Get the latest calculated raw Average True Range (ATR) for the trading pair.

        This value can be used by other services (e.g., SimulatedExecutionHandler)
        to implement volatility-based slippage models.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns:
        -------
            The raw ATR as a Decimal, or None if it cannot be calculated.
        """
        # This method is async to align with the MarketPriceService interface,
        # but the underlying calls are currently synchronous.
        return self._get_raw_atr_for_pair(trading_pair)

    async def is_price_fresh(self, trading_pair: str, _max_age_seconds: float = 60.0) -> bool:
        """Check if price data is available at the current simulation time."""
        price_ts = await self.get_price_timestamp(trading_pair)
        return price_ts is not None

    def _calculate_bid_ask_tuple(
        self,
        close_price: Decimal,
        final_spread_pct: Decimal,
        trading_pair: str,
    ) -> tuple[Decimal, Decimal] | None:
        """Calculate bid/ask tuple from close price and final spread percentage."""
        half_spread_amount = close_price * (final_spread_pct / Decimal(200))
        bid = close_price - half_spread_amount
        ask = close_price + half_spread_amount

        if bid > ask:
            self.logger.warning(
                "Simulated spread: bid (%s) > ask (%s) for %s with final_spread_pct %s. "
                "Returning None.",
                bid,
                ask,
                trading_pair,
                final_spread_pct,
                extra={"source_module": self._source_module},
            )
            return None
        if bid == ask:
            self.logger.debug(
                "Calculated zero spread for %s at %s with final_spread_pct %s. Returning as is.",
                trading_pair,
                close_price,
                final_spread_pct,
                extra={"source_module": self._source_module},
            )
            return (bid, ask)
        # bid < ask
        return (bid, ask)

    def _create_book_level_entries(
        self,
        current_bid_for_level: Decimal,
        current_ask_for_level: Decimal,
        current_volume_for_level: Decimal,
        level_index: int,
        context: BookLevelConstructionContext,
    ) -> tuple[list[str] | None, list[str] | None, bool]:  # (bid_entry, ask_entry, stop_gen)
        """Create bid/ask entries for a single order book level and check for termination."""
        bid_entry: list[str] | None = None
        ask_entry: list[str] | None = None
        stop_generation = False

        if level_index == 0:  # BBO level
            bid_entry = [
                context.price_format_str.format(current_bid_for_level),
                context.volume_format_str.format(current_volume_for_level),
            ]
            ask_entry = [
                context.price_format_str.format(current_ask_for_level),
                context.volume_format_str.format(current_volume_for_level),
            ]
        else:
            # Check for crossed/invalid book before creating entries for non-BBO levels
            if current_bid_for_level >= current_ask_for_level and not (
                current_bid_for_level == context.best_bid_price_bbo
                and current_ask_for_level == context.best_ask_price_bbo
                and context.best_bid_price_bbo == context.best_ask_price_bbo
            ):
                self.logger.debug(
                    "At level %s, calculated bid %s crossed/met ask %s for %s. "
                    "Stopping depth here.",
                    level_index + 1,
                    current_bid_for_level,
                    current_ask_for_level,
                    context.trading_pair,
                    extra={"source_module": self._source_module},
                )
                stop_generation = True
                return bid_entry, ask_entry, stop_generation

            if current_bid_for_level <= Decimal(0):
                self.logger.debug(
                    "At level %s, calculated bid %s is zero/negative for %s. "
                    "Stopping bid depth here.",
                    level_index + 1,
                    current_bid_for_level,
                    context.trading_pair,
                    extra={"source_module": self._source_module},
                )
                # Bid entry remains None, but asks can continue if valid
            else:
                bid_entry = [
                    context.price_format_str.format(current_bid_for_level),
                    context.volume_format_str.format(current_volume_for_level),
                ]

            # Ask entry is always attempted if not stopped by crossed book,
            # as a zero bid doesn't preclude valid asks.
            ask_entry = [
                context.price_format_str.format(current_ask_for_level),
                context.volume_format_str.format(current_volume_for_level),
            ]

        return bid_entry, ask_entry, stop_generation

    async def get_order_book_snapshot(
        self,
        trading_pair: str,
    ) -> dict[str, list[list[str]]] | None:
        """Generate a simulated order book snapshot with market depth.

        Args:
        ----
            trading_pair: The trading_pair symbol (e.g., "XRP/USD").

        Returns:
        -------
            A dictionary with 'bids' and 'asks' lists, or None if depth cannot be generated.
            Each inner list is [price_str, volume_str].
        """
        if not self._depth_simulation_enabled:
            self.logger.debug(
                "Market depth simulation is disabled. Skipping for %s.",
                trading_pair,
                extra={"source_module": self._source_module},
            )
            return None

        bbo = await self.get_bid_ask_spread(trading_pair)
        if bbo is None:
            self.logger.warning(
                "Could not retrieve BBO for %s. Cannot generate order book.",
                trading_pair,
                extra={"source_module": self._source_module},
            )
            return None

        best_bid_price_bbo = bbo[0]  # Store initial BBO for reference
        best_ask_price_bbo = bbo[1]

        if best_bid_price_bbo <= Decimal(0) or best_ask_price_bbo <= Decimal(0):
            self.logger.warning(
                "BBO for %s has non-positive price (%s, %s). Cannot generate order book.",
                trading_pair,
                best_bid_price_bbo,
                best_ask_price_bbo,
                extra={"source_module": self._source_module},
            )
            return None

        if best_bid_price_bbo > best_ask_price_bbo:
            self.logger.error(
                "Best bid %s is greater than best ask %s for %s. Order book invalid.",
                best_bid_price_bbo,
                best_ask_price_bbo,
                trading_pair,
                extra={"source_module": self._source_module},
            )
            return None

        bids_levels: list[list[str]] = []
        asks_levels: list[list[str]] = []

        current_level_bid_price = best_bid_price_bbo
        current_level_ask_price = best_ask_price_bbo
        current_level_volume = self._depth_base_volume

        price_format_str = f"{{:.{self._depth_price_precision}f}}"
        volume_format_str = f"{{:.{self._depth_volume_precision}f}}"
        quantizer_price = Decimal("1e-" + str(self._depth_price_precision))
        quantizer_volume = Decimal("1e-" + str(self._depth_volume_precision))

        # Create context object for _create_book_level_entries
        construction_context = BookLevelConstructionContext(
            best_bid_price_bbo=best_bid_price_bbo,
            best_ask_price_bbo=best_ask_price_bbo,
            price_format_str=price_format_str,
            volume_format_str=volume_format_str,
            trading_pair=trading_pair,
        )

        for i in range(self._depth_num_levels):
            quantized_bid = current_level_bid_price.quantize(quantizer_price, rounding=ROUND_DOWN)
            quantized_ask = current_level_ask_price.quantize(quantizer_price, rounding=ROUND_UP)
            quantized_volume = current_level_volume.quantize(quantizer_volume, rounding=ROUND_DOWN)

            if quantized_volume <= Decimal(0):
                self.logger.debug(
                    "Volume for level %s for %s became zero or negative. "
                    "Stopping depth generation.",
                    i + 1,
                    trading_pair,
                    extra={"source_module": self._source_module},
                )
                break

            bid_entry, ask_entry, stop_generation = self._create_book_level_entries(
                quantized_bid,
                quantized_ask,
                quantized_volume,
                i,
                construction_context,
            )

            if bid_entry:
                bids_levels.append(bid_entry)
            if ask_entry:
                asks_levels.append(ask_entry)

            if stop_generation:
                break

            # Calculate prices for the *next* level (i+1)
            if current_level_bid_price > Decimal(0):
                step_bid = current_level_bid_price * self._depth_price_step_pct
                current_level_bid_price -= step_bid

            step_ask = current_level_ask_price * self._depth_price_step_pct
            current_level_ask_price += step_ask
            current_level_volume *= self._depth_volume_decay_factor

        # This is implicitly handled now if bid_entry from helper is None for those cases.
        # Re-affirm by explicit filtering if needed, but helper logic should suffice.

        if not bids_levels and not asks_levels:
            self.logger.info(
                "Generated an empty order book for %s (e.g. BBO was zero spread and "
                "only one level requested, or other edge case).",
                trading_pair,
                extra={"source_module": self._source_module},
            )

        return {"bids": bids_levels, "asks": asks_levels}

    # --- End Interface Alignment Methods ---

    async def _get_direct_or_reverse_price(
        self,
        from_currency: str,
        to_currency: str,
    ) -> tuple[Decimal, bool] | None:  # Returns (price, is_direct_rate)
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
        self,
        from_amount: Decimal,
        from_currency: str,
        to_currency: str,
        intermediary: str,
    ) -> Decimal | None:
        """Get cross-conversion via an intermediary currency."""
        # Path: from_currency -> intermediary -> to_currency
        from_to_intermediary_rate_info = await self._get_direct_or_reverse_price(
            from_currency,
            intermediary,
        )

        if from_to_intermediary_rate_info:
            rate1, is_direct1 = from_to_intermediary_rate_info
            amount_in_intermediary = from_amount * rate1 if is_direct1 else from_amount / rate1

            intermediary_to_target_rate_info = await self._get_direct_or_reverse_price(
                intermediary,
                to_currency,
            )
            if intermediary_to_target_rate_info:
                rate2, is_direct2 = intermediary_to_target_rate_info
                if is_direct2:
                    return amount_in_intermediary * rate2
                return amount_in_intermediary / rate2
        return None

    async def convert_amount(
        self,
        from_amount: Decimal,
        from_currency: str,
        to_currency: str,
    ) -> Decimal | None:
        """Convert an amount from one currency to another using available market data."""
        # 1. Ensure from_amount is Decimal
        if not isinstance(from_amount, Decimal):
            self.logger.warning(
                "convert_amount received non-Decimal from_amount: %s. Attempting conversion.",
                type(from_amount),
                extra={"source_module": self._source_module},
            )
            try:
                from_amount = Decimal(str(from_amount))
            except Exception:
                self.logger.exception(
                    "Could not convert from_amount to Decimal in convert_amount.",
                    extra={"source_module": self._source_module},
                )
                return None  # Early exit if input is not convertible

        # 2. Handle same currency
        if from_currency == to_currency:
            return from_amount

        # 3. Try direct or reverse conversion
        direct_or_reverse_info = await self._get_direct_or_reverse_price(
            from_currency,
            to_currency,
        )
        if direct_or_reverse_info:
            rate, is_direct = direct_or_reverse_info
            try:
                return from_amount * rate if is_direct else from_amount / rate
            except ZeroDivisionError:
                self.logger.exception(
                    "Zero price for direct/reverse conversion (%s/%s or %s/%s).",
                    from_currency,  # Argument for first %s
                    to_currency,  # Argument for second %s
                    to_currency,  # Argument for third %s
                    from_currency,  # Argument for fourth %s
                    extra={"source_module": self._source_module},
                )
                # Fall through to try cross-conversion or fail if ZeroDivisionError occurs

        # 4. Try cross-conversion
        intermediary_currency = self._intermediary_conversion_currency
        # Ensure cross-conversion is applicable (not converting to/from intermediary
        # if direct failed, and ensure currencies are different from intermediary
        # to avoid redundant steps)
        if all(c != intermediary_currency for c in (from_currency, to_currency)):
            try:
                converted_amount = await self._get_cross_conversion_price(
                    from_amount,
                    from_currency,
                    to_currency,
                    intermediary_currency,
                )
                if converted_amount is not None:
                    return converted_amount  # Successful cross-conversion
            except ZeroDivisionError:
                self.logger.exception(
                    "Zero price encountered during %s-mediated cross-conversion from %s to %s.",
                    intermediary_currency,
                    from_currency,
                    to_currency,
                    extra={"source_module": self._source_module},
                )
                # Fall through to final failure if ZeroDivisionError occurs here

        # 5. Log failure if no path found or previous attempts failed through to here
        self.logger.warning(
            "Could not convert %s %s to %s. No direct, reverse, or %s-mediated path found.",
            from_amount,
            from_currency,
            to_currency,
            intermediary_currency,
            extra={"source_module": self._source_module},
        )
        return None

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        timeframe: str,  # - Required for API compatibility
        since: datetime,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Fetch historical OHLCV data for a trading pair from the stored historical data.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").
            timeframe: The timeframe for the candles (e.g., "1m", "1h", "1d").
            since: Python datetime object indicating the start time for fetching data (UTC).
            limit: The maximum number of candles to return.

        Returns:
        -------
            A list of dictionaries, where each dictionary represents an OHLCV candle:
            {'timestamp': datetime_obj, 'open': Decimal, 'high': Decimal,
             'low': Decimal, 'close': Decimal, 'volume': Decimal},
            or None if data is unavailable or an error occurs. Timestamps are UTC.
        """
        # Check if we have data for this trading pair
        if trading_pair not in self.historical_data:
            self.logger.warning(
                "No historical data available for trading pair %s",
                trading_pair,
                extra={"source_module": self._source_module},
            )
            return None

        df = self.historical_data[trading_pair]

        # Filter data from the provided start time
        if since is not None:
            df = df[df.index >= since]

        # Apply limit if specified
        if limit is not None and limit > 0:
            df = df.head(limit)

        # If no data available after filtering
        if df.empty:
            return None

        # Convert DataFrame to list of dictionaries
        result = []
        for timestamp, row in df.iterrows():
            candle = {
                "timestamp": timestamp,
                "open": Decimal(str(row.get("open", 0))),
                "high": Decimal(str(row.get("high", 0))),
                "low": Decimal(str(row.get("low", 0))),
                "close": Decimal(str(row.get("close", 0))),
                "volume": Decimal(str(row.get("volume", 0))),
            }
            result.append(candle)

        return result

    async def get_volatility(
        self,
        trading_pair: str,
        lookback_hours: int = 24,
    ) -> float | None:
        """Calculate the price volatility for a trading pair.

        For the simulated service, this returns the normalized ATR if available,
        which represents volatility as a percentage of the current price.
        """
        # Try to calculate volatility using ATR
        try:
            normalized_atr = self._calculate_normalized_atr(trading_pair)
            if normalized_atr is not None:
                # Convert to percentage
                volatility_pct = float(normalized_atr * Decimal("100"))
                self.logger.debug(
                    "Calculated volatility for %s: %.2f%% (lookback: %d hours)",
                    trading_pair,
                    volatility_pct,
                    lookback_hours,
                    extra={"source_module": self._source_module},
                )
                return volatility_pct
        except Exception as e:
            self.logger.error(
                "Error calculating volatility for %s: %s",
                trading_pair,
                str(e),
                extra={"source_module": self._source_module},
                exc_info=True,
            )
        
        # Fallback: calculate simple standard deviation of returns
        try:
            if self._current_timestamp is None:
                self.logger.warning(
                    "Cannot calculate volatility: current timestamp not set",
                    extra={"source_module": self._source_module},
                )
                return None
                
            pair_data = self.historical_data.get(trading_pair)
            if pair_data is None or pair_data.empty:
                self.logger.warning(
                    "No historical data available for %s to calculate volatility",
                    trading_pair,
                    extra={"source_module": self._source_module},
                )
                return None
            
            # Get data for lookback period
            lookback_start = self._current_timestamp - timedelta(hours=lookback_hours)
            recent_data = pair_data.loc[lookback_start:self._current_timestamp]
            
            if len(recent_data) < 2:
                self.logger.warning(
                    "Insufficient data points for volatility calculation for %s",
                    trading_pair,
                    extra={"source_module": self._source_module},
                )
                return None
            
            # Calculate returns
            prices = recent_data[self._price_column]
            returns = prices.pct_change().dropna()
            
            if len(returns) == 0:
                return None
                
            # Calculate annualized volatility (assuming 365 days for crypto)
            # Hourly returns -> annualize by sqrt(24 * 365)
            hourly_volatility = returns.std()
            annualized_volatility = hourly_volatility * np.sqrt(24 * 365)
            
            return float(annualized_volatility * 100)  # Return as percentage
            
        except Exception as e:
            self.logger.error(
                "Error in fallback volatility calculation for %s: %s",
                trading_pair,
                str(e),
                extra={"source_module": self._source_module},
                exc_info=True,
            )
            return None


# Example Usage


async def _setup_service_and_data(
    main_logger: logging.Logger,
) -> tuple[SimulatedMarketPriceService, datetime]:
    """Set up historical data, price service, and a common timestamp for tests."""
    # Create dummy historical data
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
            "high": [
                Decimal("11.5"),
                Decimal("12.5"),
                Decimal("12.0"),
                Decimal("13.5"),
                Decimal("14.0"),
            ],
            "low": [
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
            "high": [
                Decimal("1.15"),
                Decimal("1.25"),
                Decimal("1.20"),
                Decimal("1.15"),
                Decimal("1.10"),
            ],
            "low": [
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
    price_service = SimulatedMarketPriceService(hist_data, logger=main_logger)

    # Common timestamp for many tests
    ts1 = datetime(2023, 1, 1, 0, 1, 0, tzinfo=UTC)
    return price_service, ts1


async def _test_price_queries(
    price_service: SimulatedMarketPriceService,
    main_logger: logging.Logger,
    ts1: datetime,
) -> None:
    """Test various price retrieval scenarios."""
    main_logger.info("--- Testing Price Queries ---")
    # Test 1: Get price at an exact timestamp
    price_service.update_time(ts1)

    btc_price_info1 = await price_service.get_latest_price("BTC/USD")
    eth_price_info1 = await price_service.get_latest_price("ETH/USD")
    usd_price_info = await price_service.get_latest_price("USD/USD")
    btc_spread_info1 = await price_service.get_bid_ask_spread("BTC/USD")

    btc_price1 = btc_price_info1
    eth_price1 = eth_price_info1
    usd_price = usd_price_info

    main_logger.info(
        "Prices at %s: BTC=%s, ETH=%s, USD/USD=%s",
        ts1,
        btc_price1,
        eth_price1,
        usd_price,
    )
    if btc_spread_info1:
        main_logger.info(
            "BTC Spread at %s: Bid=%.2f, Ask=%.2f",
            ts1,
            btc_spread_info1[0],
            btc_spread_info1[1],
        )
    else:
        main_logger.info("BTC Spread at %s: None", ts1)

    # Test 2: Get price between timestamps (should get previous close)
    ts2 = datetime(2023, 1, 1, 0, 1, 30, tzinfo=UTC)
    price_service.update_time(ts2)
    btc_price2 = await price_service.get_latest_price("BTC/USD")
    main_logger.info(
        "Prices at %s: BTC=%s (Should be same as %s close: 12.0)",
        ts2,
        btc_price2,
        ts1,
    )

    # Test 3: Get price before data starts
    ts3 = datetime(2022, 12, 31, 23, 59, 0, tzinfo=UTC)
    price_service.update_time(ts3)
    btc_price3 = await price_service.get_latest_price("BTC/USD")
    main_logger.info("Prices at %s: BTC=%s (Should be None)", ts3, btc_price3)

    # Test 4: Unknown pair
    price_service.update_time(ts1)  # Reset time for consistency if other tests modify it
    unknown_price = await price_service.get_latest_price("LTC/USD")
    main_logger.info("Prices at %s: LTC=%s (Should be None)", ts1, unknown_price)


async def _test_price_metadata(
    price_service: SimulatedMarketPriceService,
    main_logger: logging.Logger,
    ts1: datetime,
) -> None:
    """Test retrieval of price timestamp and freshness."""
    main_logger.info("--- Testing Price Metadata ---")
    price_service.update_time(ts1)  # Ensure current time context

    ts_btc = await price_service.get_price_timestamp("BTC/USD")
    is_fresh_btc = await price_service.is_price_fresh("BTC/USD")
    main_logger.info("BTC Price Timestamp at %s: %s, Fresh: %s", ts1, ts_btc, is_fresh_btc)

    ts_ltc = await price_service.get_price_timestamp("LTC/USD")
    is_fresh_ltc = await price_service.is_price_fresh("LTC/USD")
    main_logger.info(
        "LTC Price Timestamp at %s: %s, Fresh: %s (Should be None, False)",
        ts1,
        ts_ltc,
        is_fresh_ltc,
    )


async def _test_order_book_snapshot(
    price_service: SimulatedMarketPriceService,
    main_logger: logging.Logger,
    ts1: datetime,
) -> None:
    """Test order book snapshot generation."""
    main_logger.info("--- Testing Order Book Snapshot ---")
    price_service.update_time(ts1)  # Ensure current time context

    btc_order_book = await price_service.get_order_book_snapshot("BTC/USD")
    if btc_order_book:
        main_logger.info("BTC/USD Order Book at %s:", ts1)
        main_logger.info("  Bids: %s", btc_order_book.get("bids"))
        main_logger.info("  Asks: %s", btc_order_book.get("asks"))
    else:
        main_logger.info("Could not generate BTC/USD order book at %s.", ts1)


async def _test_currency_conversions(
    price_service: SimulatedMarketPriceService,
    main_logger: logging.Logger,
    ts1: datetime,
) -> None:
    """Test currency conversion functionality."""
    main_logger.info("--- Testing Currency Conversions ---")
    price_service.update_time(ts1)  # Ensure current time context

    amount_to_convert = Decimal("2.0")

    # Add ETH/BTC pair for testing cross-conversion if BTC/ETH is not direct
    idx_eth_btc = pd.to_datetime(["2023-01-01 00:01:00"], utc=True)
    data_eth_btc = pd.DataFrame(
        {"close": [Decimal("0.05")], "high": [Decimal("0.05")], "low": [Decimal("0.05")]},
        index=idx_eth_btc,
    )
    price_service.historical_data["ETH/BTC"] = data_eth_btc

    converted_btc_to_eth_via_usd = await price_service.convert_amount(
        amount_to_convert,
        "BTC",
        "ETH",
    )
    main_logger.info(
        "Converting %s BTC to ETH via USD: %s",
        amount_to_convert,
        converted_btc_to_eth_via_usd,
    )

    # Add direct pair for BTC/ETH
    idx_btc_eth = pd.to_datetime(["2023-01-01 00:01:00"], utc=True)
    data_btc_eth = pd.DataFrame(
        {"close": [Decimal("20.0")], "high": [Decimal("20.0")], "low": [Decimal("20.0")]},
        index=idx_btc_eth,
    )
    price_service.historical_data["BTC/ETH"] = data_btc_eth
    converted_btc_to_eth_direct = await price_service.convert_amount(
        amount_to_convert,
        "BTC",
        "ETH",
    )
    main_logger.info(
        "Converting %s BTC to ETH (direct): %s",
        amount_to_convert,
        converted_btc_to_eth_direct,
    )


async def main() -> None:  # Made async
    """Run example demonstrating the SimulatedMarketPriceService functionality."""
    # Basic logging setup for example
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(source_module)s] - %(message)s",
    )
    main_logger = logging.getLogger("SimulatedMarketPriceServiceExample")

    price_service, ts1 = await _setup_service_and_data(main_logger)
    await price_service.start()

    await _test_price_queries(price_service, main_logger, ts1)
    await _test_price_metadata(price_service, main_logger, ts1)
    await _test_order_book_snapshot(price_service, main_logger, ts1)
    await _test_currency_conversions(price_service, main_logger, ts1)

    # Test zero-spread scenario
    main_logger.info("--- Testing Zero-Spread Scenario ---")
    # Temporarily set spread to zero for a specific pair
    original_spread = price_service._default_spread_pct
    price_service._default_spread_pct = Decimal("0")
    price_service._pair_specific_spread_config["BTC/USD"] = Decimal("0")
    
    zero_spread_result = await price_service.get_bid_ask_spread("BTC/USD")
    if zero_spread_result:
        bid, ask = zero_spread_result
        if bid == ask:
            main_logger.info("Zero spread confirmed: Bid=%s, Ask=%s", bid, ask)
        else:
            main_logger.warning("Expected zero spread but got: Bid=%s, Ask=%s", bid, ask)
    else:
        main_logger.error("Failed to get bid/ask spread for zero-spread test")
    
    # Restore original spread
    price_service._default_spread_pct = original_spread
    price_service._pair_specific_spread_config.pop("BTC/USD", None)

    # Test edge cases
    main_logger.info("--- Testing Edge Cases ---")
    
    # Test with future timestamp
    future_ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    price_service.update_time(future_ts)
    future_price = await price_service.get_latest_price("BTC/USD")
    main_logger.info("Price at future time %s: %s", future_ts, future_price)
    
    # Test volatility calculation
    if hasattr(price_service, 'get_volatility'):
        volatility = await price_service.get_volatility("BTC/USD", lookback_hours=1)
        main_logger.info("BTC/USD volatility (1hr lookback): %s%%", volatility)

    await price_service.stop()


if __name__ == "__main__":
    # Set up basic logging configuration
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Check for required dependencies
    missing_deps = []
    try:
        import pandas  # noqa
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy  # noqa
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import asyncio
    except ImportError:
        missing_deps.append("asyncio")
    
    if missing_deps:
        logger.error(
            "Missing required dependencies: %s. Install with: pip install %s",
            ", ".join(missing_deps),
            " ".join(missing_deps)
        )
    else:
        try:
            asyncio.run(main())  # Run the async main function
        except KeyboardInterrupt:
            logger.info("Example interrupted by user")
        except Exception:
            logger.exception("An error occurred during example execution")
