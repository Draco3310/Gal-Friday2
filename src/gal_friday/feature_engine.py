"""Module for creating and managing features from market data for trading strategies."""

# Feature Engine Module

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pandas_ta as ta  # Required for technical indicators (RSI, ROC, etc.)

# Event imports
from .core.events import SystemStateEvent  # Assuming SystemStateEvent is defined in core.events
from .core.events import EventType, FeatureEvent, MarketDataL2Event, MarketDataOHLCVEvent

# Import PubSubManager
from .core.pubsub import PubSubManager

# Import custom exceptions
from .exceptions import APIError, DataValidationError, GalFridayError, NetworkError, TimeoutError

# Import HistoricalDataService
from .historical_data_service import HistoricalDataService

# Import LoggerService
from .logger_service import LoggerService

# from .data_ingestor import (
#     MarketDataL2Event,
#     MarketDataOHLCVEvent,
#     SystemStatusEvent, # Remove import from data_ingestor
# )

# Replace debug print with proper logging.
# print("Feature Engine Loaded")


# REMOVED Local FeatureEvent definition - Import from core.events

# --- FeatureEngine Class ---


@dataclass
class FeaturePayload:
    """Payload for feature events."""

    trading_pair: str
    exchange: str
    timestamp_features_for: datetime
    features: dict[str, str]  # Feature name to string value mapping


class FeatureEngine:
    """
    Consume market data events and calculate features.

    Processes L2 and OHLCV data, calculates features (L2 & TA
    based on MVP requirements), and publishes feature events.
    Implements refined triggering: L2 features on L2 events, TA on OHLCV events.
    Uses pandas-ta for TA calculations.
    """

    def __init__(
        self,
        config: dict,
        # event_bus_in: asyncio.Queue, # Replace with PubSubManager
        # event_bus_out: asyncio.Queue,
        pubsub_manager: PubSubManager,
        logger_service: LoggerService,
        historical_data_service: Optional[HistoricalDataService] = None,
    ):
        """
        Initialize the FeatureEngine.

        Args:
            config (dict): Configuration settings. Expected structure:
                feature_engine:
                  ohlcv_history_size: 100
                  feature_configs:
                    book_imbalance:
                      depth: 5
                    rsi:
                      period: 14
                    roc:
                      period: 1 # Rate of Change period
            pubsub_manager (PubSubManager): PubSubManager for subscribing and publishing.
            logger_service (LoggerService): The shared logger instance.
            historical_data_service (HistoricalDataService, optional):
            Service for accessing historical data.
        """
        self._config = config.get("feature_engine", {})  # Get relevant sub-config
        # self._event_bus_in = event_bus_in
        # self._event_bus_out = event_bus_out
        self.pubsub = pubsub_manager  # Store PubSubManager
        self.logger = logger_service  # Assigned injected logger
        self.historical_data_service = historical_data_service
        # Store historical data service
        self._is_running = False
        self._main_task = None
        self._source_module = self.__class__.__name__

        # Log initialization instead of using print
        self.logger.info("Feature Engine module initialized", source_module=self._source_module)

        # Store handlers for unsubscribing
        self._l2_handler = self._handle_l2_event
        self._ohlcv_handler = self._handle_ohlcv_event
        self._status_handler = self._handle_system_status_event

        # Feature calculation parameters
        self._feature_configs = self._config.get("feature_configs", {})
        self._ohlcv_history_size = self._config.get("ohlcv_history_size", 100)

        # Internal state to store recent market data for feature calculation
        # L2 Book Cache: Store the latest L2 state per pair
        self._latest_l2_data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"bids": [], "asks": [], "timestamp": None}
        )

        # OHLCV History Cache: Store recent candles per pair/interval
        # Key: (trading_pair, interval_str), Value: deque of dicts (OHLCV)
        self._ohlcv_history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=self._ohlcv_history_size)
        )

        # DataFrame Cache: Store dataframes created from OHLCV history
        # Key: (trading_pair, interval_str), Value: (DataFrame, history_length)
        self._dataframe_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, int]] = {}

        # Cache for latest calculated features (to combine L2/TA)
        # Key: trading_pair, Value: dict[feature_name, value_str]
        self._latest_features: Dict[str, Dict[str, str]] = defaultdict(dict)

        # Track system status
        self._system_status = "unknown"  # Don't calculate if not online

    async def start(self) -> None:
        """Start listening for market data events."""
        if self._is_running:
            self.logger.warning(
                "FeatureEngine already running.", source_module=self.__class__.__name__
            )
            return
        self._is_running = True
        # Clear state from previous runs if any
        self._latest_l2_data.clear()
        self._ohlcv_history.clear()
        self._latest_features.clear()

        # Subscribe to input events
        self.pubsub.subscribe(EventType.MARKET_DATA_L2, self._l2_handler)
        self.pubsub.subscribe(EventType.MARKET_DATA_OHLCV, self._ohlcv_handler)
        self.pubsub.subscribe(EventType.SYSTEM_STATE_CHANGE, self._status_handler)
        self.logger.info(
            "Subscribed to input market data and system state events.",
            source_module=self._source_module,
        )

        # self._main_task = asyncio.create_task(self._run_event_loop()) #
        # Remove loop if directly handling via subscribe
        self.logger.info("FeatureEngine started.", source_module=self.__class__.__name__)

    async def stop(self) -> None:
        """Stop the event processing loop."""
        if not self._is_running:
            return
        self._is_running = False

        # Unsubscribe from events
        try:
            self.pubsub.unsubscribe(EventType.MARKET_DATA_L2, self._l2_handler)
            self.pubsub.unsubscribe(EventType.MARKET_DATA_OHLCV, self._ohlcv_handler)
            self.pubsub.unsubscribe(EventType.SYSTEM_STATE_CHANGE, self._status_handler)
            self.logger.info("Unsubscribed from input events.", source_module=self._source_module)
        except (ValueError, RuntimeError, GalFridayError) as e:
            self.logger.error(
                f"Error unsubscribing FeatureEngine handlers: {e}",
                exc_info=True,
                source_module=self._source_module,
            )

        # Cancel task if it was used (now likely not needed)
        # if self._main_task:
        #     self._main_task.cancel()
        #     try:
        #         await self._main_task
        #     except asyncio.CancelledError:
        #         pass  # Expected
        #     self._main_task = None
        self.logger.info("FeatureEngine stopped.", source_module=self.__class__.__name__)

    # Remove _run_event_loop as handlers are called directly by PubSubManager
    # async def _run_event_loop(self):
    #     ...

    async def _handle_l2_event(self, event: MarketDataL2Event) -> None:
        """Handle L2 market data event and calculate features."""
        # Check type
        if not isinstance(event, MarketDataL2Event):
            self.logger.warning(
                f"Received non-MarketDataL2Event: {type(event)}", source_module=self._source_module
            )
            return

        # Validate L2 data
        if not self._validate_l2_data(event):
            return

        # Extract payload fields directly (assuming payload structure matches
        # core.events)
        trading_pair = event.trading_pair
        trigger_timestamp = event.timestamp_exchange or event.timestamp

        # Update L2 cache
        self._latest_l2_data[trading_pair]["bids"] = event.bids
        self._latest_l2_data[trading_pair]["asks"] = event.asks
        self._latest_l2_data[trading_pair]["timestamp"] = trigger_timestamp
        self.logger.debug(
            f"Updated L2 cache for {trading_pair}",
            source_module=self._source_module,
        )

        # Calculate L2 features
        l2_features = self._calculate_l2_features(trading_pair)
        if l2_features is not None:
            # Update latest features cache
            self._latest_features[trading_pair].update(l2_features)
            # Publish combined features
            await self._publish_feature_event(trading_pair, trigger_timestamp)
        else:
            self.logger.debug(f"No L2 features calculated for {trading_pair}")

    def _validate_l2_prices_quantities(
        self, side_name: str, orders: list, trading_pair: str
    ) -> bool:
        """Validate prices and quantities for a side of the order book."""
        try:
            for i, (price_str, qty_str) in enumerate(orders):
                price = Decimal(price_str)
                qty = Decimal(qty_str)

                if price <= Decimal(0) or qty <= Decimal(0):
                    self.logger.warning(
                        f"L2 data validation: Non-positive {side_name} price/qty at index {i} "
                        f"for {trading_pair}: {price_str}/{qty_str}",
                        source_module=self._source_module,
                    )
                    return False
        except (ValueError, InvalidOperation, TypeError) as e:
            self.logger.warning(
                f"L2 data validation: Invalid numeric format in {trading_pair} "
                f"order book for {side_name}: {e}",
                source_module=self._source_module,
            )
            return False
        return True

    def _validate_l2_bid_sorting(self, bids: list, trading_pair: str) -> bool:
        """Validate sorting of L2 bid data."""
        try:
            prev_price = None
            for i, (price_str, _) in enumerate(bids):
                price = Decimal(price_str)
                if prev_price is not None and price > prev_price:
                    self.logger.warning(
                        f"L2 data validation: Bids not properly sorted (descending) "
                        f"at index {i} for {trading_pair}",
                        source_module=self._source_module,
                    )
                    return False
                prev_price = price
        except (ValueError, InvalidOperation, TypeError) as e:
            self.logger.warning(
                f"L2 data validation: Error while checking bid order book sorting "
                f"for {trading_pair}: {e}",
                source_module=self._source_module,
            )
            return False
        return True

    def _validate_l2_ask_sorting(self, asks: list, trading_pair: str) -> bool:
        """Validate sorting of L2 ask data."""
        try:
            prev_price = None
            for i, (price_str, _) in enumerate(asks):
                price = Decimal(price_str)
                if prev_price is not None and price < prev_price:
                    self.logger.warning(
                        f"L2 data validation: Asks not properly sorted (ascending) "
                        f"at index {i} for {trading_pair}",
                        source_module=self._source_module,
                    )
                    return False
                prev_price = price
        except (ValueError, InvalidOperation, TypeError) as e:
            self.logger.warning(
                f"L2 data validation: Error while checking ask order book sorting "
                f"for {trading_pair}: {e}",
                source_module=self._source_module,
            )
            return False
        return True

    def _validate_l2_data(self, event: MarketDataL2Event) -> bool:
        """Validate incoming L2 data for correctness and completeness."""
        if not event.bids or not event.asks:
            self.logger.warning(
                f"L2 data validation: Empty bids or asks for {event.trading_pair}",
                source_module=self._source_module,
            )
            return False

        if not self._validate_l2_prices_quantities("bids", event.bids, event.trading_pair):
            return False
        if not self._validate_l2_prices_quantities("asks", event.asks, event.trading_pair):
            return False

        if self._config.get("validate_orderbook_sorting", True):
            if not self._validate_l2_bid_sorting(event.bids, event.trading_pair):
                return False
            if not self._validate_l2_ask_sorting(event.asks, event.trading_pair):
                return False
        return True

    async def _handle_ohlcv_event(self, event: MarketDataOHLCVEvent) -> None:
        """Handle OHLCV event, update history, and calculate TA features."""
        # Check type
        if not isinstance(event, MarketDataOHLCVEvent):
            self.logger.warning(
                f"Received non-MarketDataOHLCVEvent: {type(event)}",
                source_module=self._source_module,
            )
            return

        # Validate OHLCV data
        if not self._validate_ohlcv_data(event):
            return

        trading_pair = event.trading_pair
        interval = event.interval
        history_key = (trading_pair, interval)
        trigger_timestamp = event.timestamp_bar_start  # Timestamp feature relates to

        # Convert OHLCV string data from event to a dict suitable for deque/DataFrame
        # Assuming Decimal conversion happens later if needed
        ohlcv_dict = {
            "timestamp": event.timestamp_bar_start,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        }

        # Append to history
        self._ohlcv_history[history_key].append(ohlcv_dict)
        self.logger.debug(
            f"Appended OHLCV data for {trading_pair} interval {interval}",
            source_module=self._source_module,
        )

        # Check if enough history exists for TA calculation
        if len(self._ohlcv_history[history_key]) < self._get_min_history_required():
            self.logger.debug(
                f"Not enough OHLCV history yet for {trading_pair} {interval}",
                source_module=self._source_module,
            )
            return

        # Calculate TA features
        ta_features = self._calculate_ta_features(trading_pair, interval)
        if ta_features is not None:
            # Update latest features cache
            self._latest_features[trading_pair].update(ta_features)

            # Calculate historical features if HistoricalDataService is available
            if self.historical_data_service:
                historical_features = await self._calculate_historical_features(
                    trading_pair, trigger_timestamp, interval
                )
                if historical_features:
                    self._latest_features[trading_pair].update(historical_features)

            # Publish combined features
            await self._publish_feature_event(trading_pair, trigger_timestamp)
        else:
            self.logger.debug(f"No TA features calculated for {trading_pair} {interval}")

    def _validate_ohlcv_data(self, event: MarketDataOHLCVEvent) -> bool:
        """Validate incoming OHLCV data for correctness and completeness."""
        # Check for required fields
        required_fields = ["open", "high", "low", "close", "volume"]
        for field in required_fields:
            if not hasattr(event, field) or getattr(event, field) is None:
                self.logger.warning(
                    f"OHLCV data validation: Missing required field '{field}' for "
                    f"{event.trading_pair} {event.interval}",
                    source_module=self._source_module,
                )
                return False

        # Check for valid numeric values
        try:
            # Convert to Decimal for validation
            ohlc_values = [
                Decimal(str(getattr(event, field))) for field in ["open", "high", "low", "close"]
            ]
            volume = Decimal(str(event.volume))

            # Check for negative values
            if any(val < Decimal(0) for val in ohlc_values) or volume < Decimal(0):
                self.logger.warning(
                    f"OHLCV data validation: Negative value found for "
                    f"{event.trading_pair} {event.interval}",
                    source_module=self._source_module,
                )
                return False

            # Check for OHLC relationship (high >= open,close >= low)
            open_val, high_val, low_val, close_val = ohlc_values
            if not (
                high_val >= open_val
                and high_val >= close_val
                and open_val >= low_val
                and close_val >= low_val
            ):
                self.logger.warning(
                    f"OHLCV data validation: Invalid OHLC relationship for "
                    f"{event.trading_pair} {event.interval}: O={open_val}, H={high_val}, "
                    f"L={low_val}, C={close_val}",
                    source_module=self._source_module,
                )
                return False

        except (ValueError, InvalidOperation, TypeError) as e:
            self.logger.warning(
                f"OHLCV data validation: Invalid numeric format in "
                f"{event.trading_pair} {event.interval}: {e}",
                source_module=self._source_module,
            )
            return False

        # Check for stale data if timestamp_bar_start is available
        if hasattr(event, "timestamp_bar_start") and event.timestamp_bar_start is not None:
            current_time = datetime.utcnow()
            max_staleness = self._config.get(
                "max_ohlcv_staleness_seconds", 300
            )  # 5 minutes default

            if (current_time - event.timestamp_bar_start).total_seconds() > max_staleness:
                self.logger.warning(
                    f"OHLCV data validation: Stale data detected for "
                    f"{event.trading_pair} {event.interval}, bar start: "
                    f"{event.timestamp_bar_start}, current: {current_time}",
                    source_module=self._source_module,
                )
                # We still return True as stale data might still be usable, just with a warning
        return True

    async def _handle_system_status_event(self, event: SystemStateEvent) -> None:
        """Handle system status events to track system state."""
        # Check type
        if not isinstance(event, SystemStateEvent):
            self.logger.warning(
                f"Received non-SystemStateEvent: {type(event)}",
                source_module=self._source_module,
            )
            return

        self._system_status = event.new_state
        self.logger.info(
            f"System status updated to: {self._system_status}",
            source_module=self.__class__.__name__,
        )

    # Removed handle_event method as handlers are called directly via subscribe
    # async def handle_event(self, event: Event):
    #    ...

    async def _publish_feature_event(self, trading_pair: str, trigger_timestamp: datetime) -> None:
        """Publish the latest combined features for a trading pair."""
        if self._system_status != "online":  # Or relevant status
            # Maybe check SystemStateEvent.new_state == "RUNNING"?
            self.logger.debug("System not online, skipping feature publish.")
            return

        combined_features = self._latest_features.get(trading_pair)
        if not combined_features:
            self.logger.debug(f"No combined features available to publish for {trading_pair}")
            return

        # Create a FeaturePayload instead of FeatureEvent directly
        payload = FeaturePayload(
            trading_pair=trading_pair,
            exchange="kraken",  # Assuming kraken, get from config?
            timestamp_features_for=trigger_timestamp,
            features=combined_features,  # Pass the dict directly
        )

        await self._publish_features(payload)

    async def _publish_features(self, payload: FeaturePayload) -> None:
        """Publish features from a FeaturePayload object."""
        if not payload.features:  # Don't publish if no features were calculated
            return

        # Create feature event directly instead of using .create() method
        feature_event = FeatureEvent(
            source_module=self._source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            trading_pair=payload.trading_pair,
            exchange=payload.exchange,
            timestamp_features_for=payload.timestamp_features_for,
            features=payload.features,
        )

        # Use pubsub manager instead of _event_bus_out
        await self.pubsub.publish(feature_event)

        self.logger.info(
            f"Published FeatureEvent for {payload.trading_pair}",
            source_module=self.__class__.__name__,
        )

    def _calculate_l2_basic_spread_metrics(
        self, bids: list, asks: list, trading_pair: str
    ) -> dict[str, str]:
        """Calculate best bid/ask, mid price, and spread."""
        features = {}
        best_bid_str = bids[0][0]
        best_ask_str = asks[0][0]
        best_bid = Decimal(best_bid_str)
        best_ask = Decimal(best_ask_str)

        features["best_bid"] = best_bid_str
        features["best_ask"] = best_ask_str

        if best_ask > best_bid:
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_pct = (spread / mid_price) * 100 if mid_price > 0 else Decimal(0)

            features["mid_price"] = str(mid_price)
            features["spread"] = str(spread)
            features["spread_pct"] = f"{spread_pct:.6f}"
        else:
            self.logger.warning(
                f"L2 Features: Book crossed or zero spread for {trading_pair}? "
                f"Bid={best_bid_str}, Ask={best_ask_str}",
                source_module=self.__class__.__name__,
            )
            features["mid_price"] = str(best_ask)  # Fallback mid
            features["spread"] = "0"
            features["spread_pct"] = "0.000000"
        return features

    def _calculate_l2_book_imbalance(
        self, bids: list, asks: list, trading_pair: str, imb_cfg: dict
    ) -> dict[str, str]:
        """Calculate book imbalance feature."""
        features = {}
        imb_depth = imb_cfg.get("depth")
        if isinstance(imb_depth, int) and imb_depth > 0:
            feature_name = f"book_imbalance_{imb_depth}"
            try:
                bid_vol = sum(Decimal(b[1]) for b in bids[:imb_depth])
                ask_vol = sum(Decimal(a[1]) for a in asks[:imb_depth])
                total_vol = bid_vol + ask_vol
                imbalance = (bid_vol / total_vol) if total_vol > 0 else Decimal(0.5)
                features[feature_name] = f"{imbalance:.4f}"
            except (IndexError, ValueError, InvalidOperation) as calc_error:
                self.logger.error(
                    f"Error calculating {feature_name} for {trading_pair}: {calc_error}",
                    source_module=self.__class__.__name__,
                )
            except Exception as imb_error:
                msg = (
                    f"Unexpected error calculating {feature_name} for {trading_pair}: "
                    f"{imb_error}"
                )
                self.logger.error(msg, source_module=self.__class__.__name__, exc_info=True)
        return features

    def _calculate_l2_depth_level_features(
        self, bids: list, asks: list, trading_pair: str, level: int
    ) -> dict[str, str]:
        """Calculate cumulative volume and WAP for a given depth level."""
        features = {}
        try:
            bid_vol_cum = sum(Decimal(b[1]) for b in bids[:level])
            ask_vol_cum = sum(Decimal(a[1]) for a in asks[:level])
            features[f"bid_vol_cum_{level}"] = f"{bid_vol_cum:.8f}"
            features[f"ask_vol_cum_{level}"] = f"{ask_vol_cum:.8f}"

            if bid_vol_cum > Decimal(0):
                bid_wap = sum(Decimal(b[0]) * Decimal(b[1]) for b in bids[:level]) / bid_vol_cum
                features[f"bid_wap_{level}"] = f"{bid_wap:.8f}"
            if ask_vol_cum > Decimal(0):
                ask_wap = sum(Decimal(a[0]) * Decimal(a[1]) for a in asks[:level]) / ask_vol_cum
                features[f"ask_wap_{level}"] = f"{ask_wap:.8f}"
        except (IndexError, ValueError, InvalidOperation) as calc_error:
            self.logger.error(
                f"Error calculating depth/WAP features at level {level} "
                f"for {trading_pair}: {calc_error}",
                source_module=self._source_module,
            )
        except Exception as depth_error:
            self.logger.error(
                f"Unexpected error calculating depth/WAP features at level {level} "
                f"for {trading_pair}: {depth_error}",
                source_module=self._source_module,
                exc_info=True,
            )
        return features

    def _calculate_l2_features(self, trading_pair: str) -> dict[str, str] | None:
        """Calculate features based on the latest L2 book data."""
        if self._system_status != "online":
            msg = f"L2 Features: Status {self._system_status}, skipping."
            self.logger.debug(msg, source_module=self.__class__.__name__)
            return None

        l2_data = self._latest_l2_data.get(trading_pair)
        if not l2_data or not l2_data["timestamp"]:
            msg = f"No L2 data for {trading_pair}"
            self.logger.debug(msg, source_module=self.__class__.__name__)
            return None

        features = {}
        bids = l2_data["bids"]
        asks = l2_data["asks"]

        try:
            if not bids or not asks:
                msg = f"Empty bids or asks for {trading_pair}"
                self.logger.debug(msg, source_module=self.__class__.__name__)
                return None

            basic_metrics = self._calculate_l2_basic_spread_metrics(bids, asks, trading_pair)
            features.update(basic_metrics)

            imb_cfg = self._feature_configs.get("book_imbalance", {})
            imbalance_feature = self._calculate_l2_book_imbalance(
                bids, asks, trading_pair, imb_cfg
            )
            features.update(imbalance_feature)

            depth_levels = self._feature_configs.get("l2_depth_levels", [1, 5, 10])
            for level in depth_levels:
                if not isinstance(level, int) or level <= 0:
                    continue
                depth_features = self._calculate_l2_depth_level_features(
                    bids, asks, trading_pair, level
                )
                features.update(depth_features)

            self.logger.debug(
                f"Calculated L2 features for {trading_pair}: {list(features.keys())}",
                source_module=self.__class__.__name__,
            )
            return features

        except (IndexError, ValueError, InvalidOperation) as calc_error:
            msg = f"L2 feature calculation error: {calc_error}"
            self.logger.error(msg, source_module=self.__class__.__name__, exc_info=True)
            return None

    def _prepare_ohlcv_dataframe(self, history_key: tuple[str, str]) -> pd.DataFrame | None:
        """Prepare OHLCV data as a pandas DataFrame."""
        history = self._ohlcv_history.get(history_key)
        if not history:
            return None

        history_len = len(history)
        cached_df, cached_len = self._dataframe_cache.get(history_key, (None, 0))

        # If cache exists and history length hasn't changed, return cached DataFrame
        if cached_df is not None and cached_len == history_len:
            self.logger.debug(
                f"Using cached DataFrame for {history_key[0]} {history_key[1]}",
                source_module=self._source_module,
            )
            return cached_df

        try:
            # Convert deque to pandas DataFrame
            df = pd.DataFrame(list(history))
            df = df.set_index("timestamp")
            # Ensure correct dtypes for pandas_ta
            df = df.astype(
                {
                    "open": "float64",
                    "high": "float64",
                    "low": "float64",
                    "close": "float64",
                    "volume": "float64",
                }
            )

            # Update cache
            self._dataframe_cache[history_key] = (df, history_len)

            return df
        except (ValueError, KeyError, TypeError, DataValidationError) as df_error:
            self.logger.warning(
                f"Error preparing DataFrame for {history_key}: {df_error}",
                source_module=self._source_module,
            )
            return None

    def _calculate_rsi_feature(
        self, df: pd.DataFrame, trading_pair: str, interval: str
    ) -> dict[str, str]:
        """Calculate RSI feature if configured."""
        features = {}
        rsi_cfg = self._feature_configs.get("rsi", {})
        rsi_period = rsi_cfg.get("period")

        if isinstance(rsi_period, int) and rsi_period > 0:
            feature_name = f"rsi_{rsi_period}_{interval}"
            if len(df) >= rsi_period + 1:
                try:
                    rsi_series = ta.rsi(df["close"], length=rsi_period)
                    last_rsi = rsi_series.iloc[-1]
                    if not pd.isna(last_rsi):
                        features[feature_name] = f"{last_rsi:.2f}"
                    else:
                        msg = f"RSI is NaN for {trading_pair} {interval}"
                        self.logger.debug(msg, source_module=self.__class__.__name__)
                except (ValueError, KeyError, TypeError, DataValidationError) as rsi_error:
                    self.logger.warning(
                        f"Error calculating RSI for {trading_pair}/{interval}: {rsi_error}",
                        source_module=self._source_module,
                    )
                    return {}
            else:
                msg = "Not enough data for RSI calculation"
                self.logger.debug(msg, source_module=self.__class__.__name__)
        return features

    def _calculate_roc_feature(
        self, df: pd.DataFrame, trading_pair: str, interval: str
    ) -> dict[str, str]:
        """Calculate ROC feature if configured."""
        features = {}
        roc_cfg = self._feature_configs.get("roc", {})
        roc_period = roc_cfg.get("period")

        if isinstance(roc_period, int) and roc_period > 0:
            feature_name = f"roc_{roc_period}_{interval}"
            if len(df) >= roc_period + 1:
                try:
                    roc_series = ta.roc(df["close"], length=roc_period)
                    last_roc = roc_series.iloc[-1]
                    if not pd.isna(last_roc):
                        features[feature_name] = f"{last_roc:.6f}"
                    else:
                        msg = f"ROC is NaN for {trading_pair} {interval}"
                        self.logger.debug(msg, source_module=self.__class__.__name__)
                except (ValueError, KeyError, TypeError, DataValidationError) as roc_error:
                    self.logger.warning(
                        f"Error calculating ROC for {trading_pair}/{interval}: {roc_error}",
                        source_module=self._source_module,
                    )
                    return {}
            else:
                msg = "Not enough data for ROC calculation"
                self.logger.debug(msg, source_module=self.__class__.__name__)
        return features

    def _calculate_macd_feature(
        self, df: pd.DataFrame, trading_pair: str, interval: str
    ) -> dict[str, str]:
        """Calculate MACD feature if configured."""
        features: dict[str, str] = {}
        cfg = self._feature_configs.get("macd", {})  # Get specific config
        fast = cfg.get("fast_period", 12)
        slow = cfg.get("slow_period", 26)
        signal = cfg.get("signal_period", 9)

        if not all(isinstance(p, int) and p > 0 for p in [fast, slow, signal]):
            self.logger.warning(
                f"Invalid MACD periods configured for {trading_pair} {interval}",
                source_module=self._source_module,
            )
            return features

        feature_prefix = f"macd_{fast}_{slow}_{signal}_{interval}"
        min_len = slow + signal - 1  # Approx minimum length needed

        if len(df) >= min_len:
            try:
                macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
                if macd_df is not None and not macd_df.empty:
                    last_row = macd_df.iloc[-1]
                    macd_line = last_row.get(f"MACD_{fast}_{slow}_{signal}")
                    signal_line = last_row.get(f"MACDs_{fast}_{slow}_{signal}")
                    hist = last_row.get(f"MACDh_{fast}_{slow}_{signal}")

                    if pd.notna(macd_line):
                        features[f"{feature_prefix}_line"] = f"{macd_line:.8f}"
                    if pd.notna(signal_line):
                        features[f"{feature_prefix}_signal"] = f"{signal_line:.8f}"
                    if pd.notna(hist):
                        features[f"{feature_prefix}_hist"] = f"{hist:.8f}"
                else:
                    self.logger.debug(
                        f"MACD result is None or empty for {trading_pair} {interval}",
                        source_module=self._source_module,
                    )
            except Exception as e:
                self.logger.error(
                    f"MACD calculation failed for {trading_pair} {interval}: {e}",
                    source_module=self._source_module,
                    exc_info=True,
                )
        else:
            self.logger.debug(
                f"Not enough data for MACD ({len(df)} < {min_len}) for {trading_pair} {interval}",
                source_module=self._source_module,
            )
        return features

    def _calculate_bbands_feature(
        self, df: pd.DataFrame, trading_pair: str, interval: str
    ) -> dict[str, str]:
        """Calculate Bollinger Bands feature if configured."""
        features: dict[str, str] = {}
        cfg = self._feature_configs.get("bbands", {})
        length = cfg.get("length", 20)
        std_dev = cfg.get("std_dev", 2.0)

        if not (
            isinstance(length, int)
            and length > 0
            and isinstance(std_dev, (float, int))
            and std_dev > 0
        ):
            self.logger.warning(
                f"Invalid BBands params configured for {trading_pair} {interval}",
                source_module=self._source_module,
            )
            return features

        feature_prefix = f"bbands_{length}_{std_dev:.1f}_{interval}"
        if len(df) >= length:
            try:
                bbands_df = ta.bbands(df["close"], length=length, std=std_dev)
                if bbands_df is not None and not bbands_df.empty:
                    last_row = bbands_df.iloc[-1]
                    lower = last_row.get(f"BBL_{length}_{std_dev}")
                    middle = last_row.get(f"BBM_{length}_{std_dev}")  # SMA
                    upper = last_row.get(f"BBU_{length}_{std_dev}")
                    bandwidth = last_row.get(f"BBB_{length}_{std_dev}")  # Bandwidth
                    percent = last_row.get(f"BBP_{length}_{std_dev}")  # %B

                    if pd.notna(lower):
                        features[f"{feature_prefix}_lower"] = f"{lower:.8f}"
                    if pd.notna(middle):
                        features[f"{feature_prefix}_middle"] = f"{middle:.8f}"
                    if pd.notna(upper):
                        features[f"{feature_prefix}_upper"] = f"{upper:.8f}"
                    if pd.notna(bandwidth):
                        features[f"{feature_prefix}_bandwidth"] = f"{bandwidth:.6f}"
                    if pd.notna(percent):
                        features[f"{feature_prefix}_percent"] = f"{percent:.6f}"
                else:
                    self.logger.debug(
                        f"BBands result is None or empty for {trading_pair} {interval}",
                        source_module=self._source_module,
                    )
            except Exception as e:
                self.logger.error(
                    f"BBands calculation failed for {trading_pair} {interval}: {e}",
                    source_module=self._source_module,
                    exc_info=True,
                )
        else:
            self.logger.debug(
                f"Not enough data for BBands ({len(df)} < {length}) for {trading_pair} {interval}",
                source_module=self._source_module,
            )
        return features

    def _calculate_vwap_feature(
        self, df: pd.DataFrame, trading_pair: str, interval: str
    ) -> dict[str, str]:
        """Calculate VWAP feature (requires high, low, close, volume)."""
        # Note: VWAP is typically calculated intraday, resetting daily.
        # For simplicity, we'll calculate rolling VWAP if configured.
        features: dict[str, str] = {}
        cfg = self._feature_configs.get("vwap", {})
        length = cfg.get("length", 14)  # Example: Rolling VWAP over 14 periods

        if not (isinstance(length, int) and length > 0):
            self.logger.warning(
                f"Invalid VWAP length configured for {trading_pair} {interval}",
                source_module=self._source_module,
            )
            return features

        feature_name = f"vwap_{length}_{interval}"
        # VWAP requires H, L, C, V columns
        required_cols = ["high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(
                f"Missing required columns for VWAP for {trading_pair} {interval}",
                source_module=self._source_module,
            )
            return features

        if len(df) >= length:
            try:
                # Use pandas-ta vwap function
                vwap_series = ta.vwap(
                    df["high"], df["low"], df["close"], df["volume"], length=length
                )
                if vwap_series is not None and not vwap_series.empty:
                    last_vwap = vwap_series.iloc[-1]
                    if pd.notna(last_vwap):
                        features[feature_name] = f"{last_vwap:.8f}"
                    else:
                        self.logger.debug(
                            f"VWAP is NaN for {trading_pair} {interval}",
                            source_module=self._source_module,
                        )
                else:
                    self.logger.debug(
                        f"VWAP result is None or empty for {trading_pair} {interval}",
                        source_module=self._source_module,
                    )
            except Exception as e:
                self.logger.error(
                    f"VWAP calculation failed for {trading_pair} {interval}: {e}",
                    source_module=self._source_module,
                    exc_info=True,
                )
        else:
            self.logger.debug(
                f"Not enough data for VWAP ({len(df)} < {length}) for {trading_pair} {interval}",
                source_module=self._source_module,
            )
        return features

    def _calculate_atr_from_df(
        self, df: pd.DataFrame, trading_pair: str, interval: str, atr_len: int
    ) -> dict[str, str]:
        """Calculate ATR from a DataFrame."""
        features: dict[str, str] = {}
        feature_name = f"atr_{atr_len}_{interval}"
        required_cols = ["high", "low", "close"]
        if all(col in df.columns for col in required_cols) and len(df) >= atr_len:
            try:
                atr_series = ta.atr(df["high"], df["low"], df["close"], length=atr_len)
                if atr_series is not None and not atr_series.empty:
                    last_atr = atr_series.iloc[-1]
                    if pd.notna(last_atr):
                        features[feature_name] = f"{last_atr:.8f}"
                else:
                    self.logger.debug(
                        f"ATR result None/empty for {trading_pair} {interval}",
                        source_module=self._source_module,
                    )
            except Exception as e:
                self.logger.error(
                    f"ATR calc failed for {trading_pair} {interval}: {e}",
                    source_module=self._source_module,
                    exc_info=True,
                )
        else:
            self.logger.debug(
                f"Not enough data/cols for ATR ({len(df)} < {atr_len}) "
                f"for {trading_pair} {interval}",
                source_module=self._source_module,
            )
        return features

    def _calculate_stdev_from_df(
        self, df: pd.DataFrame, trading_pair: str, interval: str, stdev_len: int
    ) -> dict[str, str]:
        """Calculate Standard Deviation feature."""
        features: dict[str, str] = {}
        feature_name = f"stdev_{stdev_len}"

        if len(df) >= stdev_len and "close" in df.columns:
            try:
                # Using pandas_ta for standard deviation calculation
                stdev_result = ta.stdev(df["close"], length=stdev_len)
                if stdev_result is not None and not stdev_result.empty:
                    last_stdev = stdev_result.iloc[-1]
                    if pd.notna(last_stdev):
                        features[feature_name] = f"{last_stdev:.8f}"
                else:
                    self.logger.debug(
                        f"Stdev result None/empty for {trading_pair} {interval}",
                        source_module=self._source_module,
                    )
            except (ValueError, TypeError, KeyError, DataValidationError) as e:
                self.logger.error(
                    f"Stdev calc failed for {trading_pair} {interval}: {e}",
                    source_module=self._source_module,
                    exc_info=True,
                )
        else:
            self.logger.debug(
                f"Not enough data/cols for Stdev ({len(df)} < {stdev_len}) "
                f"for {trading_pair} {interval}",
                source_module=self._source_module,
            )
        return features

    def _calculate_volatility_feature(
        self, df: pd.DataFrame, trading_pair: str, interval: str
    ) -> dict[str, str]:
        """Calculate ATR and/or Stdev volatility features if configured."""
        features: dict[str, str] = {}
        atr_cfg = self._feature_configs.get("atr", {})
        stdev_cfg = self._feature_configs.get("stdev", {})
        atr_len = atr_cfg.get("length", 14)
        stdev_len = stdev_cfg.get("length", 14)

        if isinstance(atr_len, int) and atr_len > 0:
            atr_features = self._calculate_atr_from_df(df, trading_pair, interval, atr_len)
            features.update(atr_features)

        if isinstance(stdev_len, int) and stdev_len > 0:
            stdev_features = self._calculate_stdev_from_df(df, trading_pair, interval, stdev_len)
            features.update(stdev_features)
        return features

    def _calculate_ta_features(self, trading_pair: str, interval: str) -> dict[str, str] | None:
        """Calculate TA features based on the latest OHLCV history."""
        if self._system_status != "online":
            msg = f"TA Features: Status {self._system_status}, skipping."
            self.logger.debug(msg, source_module=self.__class__.__name__)
            return None

        history_key = (trading_pair, interval)
        df = self._prepare_ohlcv_dataframe(history_key)
        if df is None:
            msg = f"No OHLCV history for {trading_pair} {interval}"
            self.logger.debug(msg, source_module=self.__class__.__name__)
            return None

        features = {}
        # Calculate RSI features
        rsi_features = self._calculate_rsi_feature(df, trading_pair, interval)
        features.update(rsi_features)

        # Calculate ROC features
        roc_features = self._calculate_roc_feature(df, trading_pair, interval)
        features.update(roc_features)

        # Calculate MACD features
        macd_features = self._calculate_macd_feature(df, trading_pair, interval)
        features.update(macd_features)

        # Calculate Bollinger Bands features
        bbands_features = self._calculate_bbands_feature(df, trading_pair, interval)
        features.update(bbands_features)

        # Calculate VWAP features
        vwap_features = self._calculate_vwap_feature(df, trading_pair, interval)
        features.update(vwap_features)

        # Calculate volatility features
        volatility_features = self._calculate_volatility_feature(df, trading_pair, interval)
        features.update(volatility_features)

        if not features:
            return None

        return features

    def _get_period_from_config(
        self, feature_name: str, field_name: str, default_value: int
    ) -> int:
        """Helper method to get period from config for a specific feature."""
        feature_cfg = self._feature_configs.get(feature_name, {})
        period_value = feature_cfg.get(field_name, default_value)

        return (
            period_value if isinstance(period_value, int) and period_value > 0 else default_value
        )

    def _get_min_history_required(self) -> int:
        """Get the minimum required history size for TA calculations."""
        min_size = 1  # Minimum baseline

        # Check various indicator requirements
        periods = [
            self._get_period_from_config("rsi", "period", 14),
            self._get_period_from_config("roc", "period", 1),
            self._get_period_from_config("bbands", "length", 20),
            self._get_period_from_config("vwap", "length", 14),
            self._get_period_from_config("atr", "length", 14),
            self._get_period_from_config("stdev", "length", 14),
        ]

        # Special handling for MACD which requires multiple periods
        macd_slow = self._get_period_from_config("macd", "slow_period", 26)
        macd_signal = self._get_period_from_config("macd", "signal_period", 9)
        macd_min = macd_slow + macd_signal - 1
        periods.append(macd_min)

        # Get the maximum required period
        max_period = max(periods)
        if max_period > min_size:
            min_size = max_period

        # Add 1 for better reliability in calculations
        return min_size + 1

    async def _calculate_historical_atr(
        self, trading_pair: str, timestamp: datetime, atr_period: int
    ) -> dict[str, str]:
        """Fetch and calculate historical ATR."""
        features: dict[str, str] = {}
        if not self.historical_data_service:
            return features  # Or log warning

        try:
            atr_value = self.historical_data_service.get_atr(
                trading_pair, timestamp, period=atr_period
            )
            if atr_value is not None:
                # Example: features[f"atr_{atr_period}_historical"] = f"{atr_value:.8f}"
                # Actual feature naming and usage would depend on requirements.
                # For now, let's assume it's not directly added to features here
                # or needs a specific key.
                pass  # Placeholder for actual feature assignment
            else:
                # Placeholder for logging if atr_value is None
                pass
        except (ValueError, TypeError, KeyError, DataValidationError, APIError, NetworkError) as e:
            self.logger.error(
                f"Error calculating historical ATR for {trading_pair}: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
        return features

    async def _fetch_and_process_historical_ohlcv(
        self, trading_pair: str, start_time: datetime, end_time: datetime, interval: str
    ) -> dict[str, str]:
        """Fetch historical OHLCV and calculate features from it."""
        features: dict[str, str] = {}
        if not self.historical_data_service:
            return features  # Or log warning

        try:
            hist_df = await self.historical_data_service.get_historical_ohlcv(
                trading_pair, start_time, end_time, interval
            )

            if hist_df is not None and not hist_df.empty:
                # Calculate more complex indicators that need more data
                # than what's available in the real-time cache.
                # Example:
                # if "close" in hist_df.columns:
                #    sma_200 = ta.sma(hist_df["close"], length=200)
                #    if sma_200 is not None and not sma_200.empty and pd.notna(sma_200.iloc[-1]):
                #        features[f"sma_200_historical_{interval}"] = f"{sma_200.iloc[-1]:.8f}"
                pass  # Placeholder for actual feature calculations
            else:
                # Placeholder for logging if hist_df is None or empty
                pass
        except (
            ValueError,
            TypeError,
            KeyError,
            DataValidationError,
            APIError,
            NetworkError,
            TimeoutError,
        ) as e:
            self.logger.error(
                f"Error fetching/processing historical OHLCV for {trading_pair}: {e}",
                source_module=self._source_module,
                exc_info=True,
            )
        return features

    async def _calculate_historical_features(
        self, trading_pair: str, timestamp: datetime, interval: str = "1h"
    ) -> dict[str, str]:
        """Calculate features based on historical data from HistoricalDataService."""
        if not self.historical_data_service:
            self.logger.warning(
                "Cannot calculate historical features - no HistoricalDataService provided.",
                source_module=self._source_module,
            )
            return {}

        features = {}

        atr_cfg = self._feature_configs.get("atr", {})
        atr_period = atr_cfg.get("period", 14)
        if atr_period > 0:
            historical_atr_features = await self._calculate_historical_atr(
                trading_pair, timestamp, atr_period
            )
            features.update(historical_atr_features)

        lookback_days = self._config.get("historical_lookback_days", 30)
        start_time = timestamp - timedelta(days=lookback_days)

        processed_ohlcv_features = await self._fetch_and_process_historical_ohlcv(
            trading_pair, start_time, timestamp, interval
        )
        features.update(processed_ohlcv_features)

        return features


# Example Usage Placeholder (Remove or guard in final app)
# if __name__ == "__main__": ...
