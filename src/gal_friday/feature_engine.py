# Feature Engine Module

import asyncio
from dataclasses import dataclass, field
import pandas as pd
import pandas_ta as ta  # Required for technical indicators (RSI, ROC, etc.)
from collections import deque, defaultdict
from decimal import Decimal, InvalidOperation
import uuid
from datetime import datetime
from typing import Dict, Tuple, Any

# Event imports
from .core.events import (
    Event, 
    EventType, 
    FeatureEvent, 
    MarketDataL2Event, 
    MarketDataOHLCVEvent, 
    SystemStateEvent # Assuming SystemStateEvent is defined in core.events
)
# from .data_ingestor import (
#     MarketDataL2Event,
#     MarketDataOHLCVEvent,
#     SystemStatusEvent, # Remove import from data_ingestor
# )

# Import PubSubManager
from .core.pubsub import PubSubManager 
# Import LoggerService
from .logger_service import LoggerService

print("Feature Engine Loaded")


# REMOVED Local FeatureEvent definition - Import from core.events

# --- FeatureEngine Class ---

@dataclass
class FeaturePayload:
    """Payload for feature events"""
    trading_pair: str
    exchange: str
    timestamp_features_for: datetime
    features: dict[str, str]  # Feature name to string value mapping

class FeatureEngine:
    """
    Consumes market data events (L2, OHLCV), calculates features (L2 & TA
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
    ):
        """
        Initializes the FeatureEngine.

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
        """
        self._config = config.get("feature_engine", {})  # Get relevant sub-config
        # self._event_bus_in = event_bus_in
        # self._event_bus_out = event_bus_out
        self.pubsub = pubsub_manager # Store PubSubManager
        self.logger = logger_service  # Assigned injected logger
        self._is_running = False
        self._main_task = None
        self._source_module = self.__class__.__name__
        
        # Store handlers for unsubscribing
        self._l2_handler = self._handle_l2_event
        self._ohlcv_handler = self._handle_ohlcv_event
        self._status_handler = self._handle_system_status_event

        # Feature calculation parameters
        self._feature_configs = self._config.get("feature_configs", {})
        self._ohlcv_history_size = self._config.get("ohlcv_history_size", 100)

        # Internal state to store recent market data for feature calculation
        # L2 Book Cache: Store the latest L2 state per pair
        self._latest_l2_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"bids": [], "asks": [], "timestamp": None})

        # OHLCV History Cache: Store recent candles per pair/interval
        # Key: (trading_pair, interval_str), Value: deque of dicts (OHLCV)
        self._ohlcv_history: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=self._ohlcv_history_size))

        # Cache for latest calculated features (to combine L2/TA)
        # Key: trading_pair, Value: dict[feature_name, value_str]
        self._latest_features: Dict[str, Dict[str, str]] = defaultdict(dict)

        # Track system status
        self._system_status = "unknown"  # Don't calculate if not online

    async def start(self) -> None:
        """Starts listening for market data events."""
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
        self.logger.info("Subscribed to input market data and system state events.", source_module=self._source_module)
        
        # self._main_task = asyncio.create_task(self._run_event_loop()) # Remove loop if directly handling via subscribe
        self.logger.info("FeatureEngine started.", source_module=self.__class__.__name__)

    async def stop(self) -> None:
        """Stops the event processing loop."""
        if not self._is_running:
            return
        self._is_running = False
        
        # Unsubscribe from events
        try:
            self.pubsub.unsubscribe(EventType.MARKET_DATA_L2, self._l2_handler)
            self.pubsub.unsubscribe(EventType.MARKET_DATA_OHLCV, self._ohlcv_handler)
            self.pubsub.unsubscribe(EventType.SYSTEM_STATE_CHANGE, self._status_handler)
            self.logger.info("Unsubscribed from input events.", source_module=self._source_module)
        except Exception as e:
             self.logger.error(f"Error unsubscribing FeatureEngine handlers: {e}", exc_info=True, source_module=self._source_module)
             
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
            self.logger.warning(f"Received non-MarketDataL2Event: {type(event)}", source_module=self._source_module)
            return
            
        # Extract payload fields directly (assuming payload structure matches core.events)
        trading_pair = event.trading_pair
        trigger_timestamp = event.timestamp_exchange or event.timestamp

        # Update L2 cache
        self._latest_l2_data[trading_pair]["bids"] = event.bids
        self._latest_l2_data[trading_pair]["asks"] = event.asks
        self._latest_l2_data[trading_pair]["timestamp"] = trigger_timestamp
        self.logger.debug(
            "Updated L2 cache for {pair}".format(pair=trading_pair),
            source_module=self.__class__.__name__
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

    async def _handle_ohlcv_event(self, event: MarketDataOHLCVEvent) -> None:
        """Handle OHLCV event, update history, and calculate TA features."""
         # Check type
        if not isinstance(event, MarketDataOHLCVEvent):
            self.logger.warning(f"Received non-MarketDataOHLCVEvent: {type(event)}", source_module=self._source_module)
            return
            
        trading_pair = event.trading_pair
        interval = event.interval
        history_key = (trading_pair, interval)
        trigger_timestamp = event.timestamp_bar_start # Timestamp feature relates to

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
            "Appended OHLCV data for {pair} interval {interval}".format(
                pair=trading_pair, interval=interval
            ),
            source_module=self.__class__.__name__,
        )

        # Check if enough history exists for TA calculation
        if len(self._ohlcv_history[history_key]) < self._get_min_history_required():
            self.logger.debug(
                "Not enough OHLCV history yet for {pair} {interval}".format(
                    pair=trading_pair, interval=interval
                )
            )
            return

        # Calculate TA features
        ta_features = self._calculate_ta_features(trading_pair, interval)
        if ta_features is not None:
            # Update latest features cache
            self._latest_features[trading_pair].update(ta_features)
            # Publish combined features
            await self._publish_feature_event(trading_pair, trigger_timestamp)
        else:
            self.logger.debug(f"No TA features calculated for {trading_pair} {interval}")

    async def _handle_system_status_event(self, event: SystemStateEvent) -> None:
        """Handle system status changes."""
         # Check type
        if not isinstance(event, SystemStateEvent):
            self.logger.warning(f"Received non-SystemStateEvent: {type(event)}", source_module=self._source_module)
            return
            
        self._system_status = event.new_state
        self.logger.info(
            "System status updated to: {status}".format(status=self._system_status),
            source_module=self.__class__.__name__,
        )

    # Removed handle_event method as handlers are called directly via subscribe
    # async def handle_event(self, event: Event):
    #    ...

    async def _publish_feature_event(self, trading_pair: str, trigger_timestamp: datetime) -> None:
        """Publish the latest combined features for a trading pair."""
        if self._system_status != "online": # Or relevant status
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
            exchange="kraken", # Assuming kraken, get from config?
            timestamp_features_for=trigger_timestamp,
            features=combined_features # Pass the dict directly
        )

        await self._publish_features(payload)

    async def _publish_features(self, payload: FeaturePayload) -> None:
        """Helper to publish features; refactored from _publish_feature_event."""
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
            features=payload.features
        )
        
        # Use pubsub manager instead of _event_bus_out
        await self.pubsub.publish(feature_event)
        
        self.logger.info(
            f"Published FeatureEvent for {payload.trading_pair}",
            source_module=self.__class__.__name__
        )

    def _calculate_l2_features(self, trading_pair: str) -> dict[str, str] | None:
        """Calculates features based on the latest L2 book data."""
        if self._system_status != "online":
            msg = (
                "L2 Features: Status {status}, skipping."
            ).format(status=self._system_status)
            self.logger.debug(msg, source_module=self.__class__.__name__)
            return None

        l2_data = self._latest_l2_data.get(trading_pair)
        if not l2_data or not l2_data["timestamp"]:
            msg = "No L2 data for {pair}".format(pair=trading_pair)
            self.logger.debug(msg, source_module=self.__class__.__name__)
            return None

        features = {}
        bids = l2_data["bids"]
        asks = l2_data["asks"]

        try:
            if not bids or not asks:
                msg = "Empty bids or asks for {pair}".format(pair=trading_pair)
                self.logger.debug(msg, source_module=self.__class__.__name__)
                return None

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
                    "L2 Features: Book crossed or zero spread for {pair}? "
                    "Bid={bid}, Ask={ask}".format(
                        pair=trading_pair,
                        bid=best_bid_str,
                        ask=best_ask_str
                    ),
                    source_module=self.__class__.__name__
                )
                features["mid_price"] = str(best_ask)  # Use ask as fallback mid?
                features["spread"] = "0"
                features["spread_pct"] = "0.000000"

            # --- Book Imbalance --- #
            imb_cfg = self._feature_configs.get("book_imbalance", {})
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
                    msg = "Error calculating {feat} for {pair}: {err}".format(
                        feat=feature_name,
                        pair=trading_pair,
                        err=calc_error
                    )
                    self.logger.error(msg, source_module=self.__class__.__name__)
                except Exception as imb_error:
                    msg = (
                        "Unexpected error calculating {feat} for {pair}: {err}"
                    ).format(
                        feat=feature_name,
                        pair=trading_pair,
                        err=imb_error
                    )
                    self.logger.error(
                        msg,
                        source_module=self.__class__.__name__,
                        exc_info=True
                    )

            # --- Add other L2 features here if needed --- #

            self.logger.debug(
                "Calculated L2 features for {pair}: {features}".format(
                    pair=trading_pair,
                    features=list(features.keys())
                ),
                source_module=self.__class__.__name__
            )
            return features

        except (IndexError, ValueError, InvalidOperation) as calc_error:
            # Use the caught error in the message
            msg = (
                "L2 feature calculation error: {error}"
            ).format(error=calc_error)
            self.logger.error(
                msg,
                source_module=self.__class__.__name__,
                exc_info=True
            )
            return None

    def _prepare_ohlcv_dataframe(self, history_key: tuple[str, str]) -> pd.DataFrame | None:
        """Prepare OHLCV data as a pandas DataFrame."""
        history = self._ohlcv_history.get(history_key)
        if not history:
            return None

        try:
            # Convert deque to pandas DataFrame
            df = pd.DataFrame(list(history))
            df = df.set_index("timestamp")
            # Ensure correct dtypes for pandas_ta
            df = df.astype({
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            })
            return df
        except Exception as df_error:
            msg = (
                "Error creating DataFrame: {error}"
            ).format(error=df_error)
            self.logger.error(
                msg,
                source_module=self.__class__.__name__,
                exc_info=True
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
                        msg = (
                            "RSI is NaN for {pair} {int}"
                        ).format(pair=trading_pair, int=interval)
                        self.logger.debug(msg, source_module=self.__class__.__name__)
                except Exception as rsi_error:
                    msg = (
                        "RSI calculation failed: {error}"
                    ).format(error=rsi_error)
                    self.logger.error(msg, source_module=self.__class__.__name__)
            else:
                msg = (
                    "Not enough data for RSI calculation"
                )
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
                        msg = (
                            "ROC is NaN for {pair} {int}"
                        ).format(pair=trading_pair, int=interval)
                        self.logger.debug(msg, source_module=self.__class__.__name__)
                except Exception as roc_error:
                    msg = (
                        "ROC calculation failed: {error}"
                    ).format(error=roc_error)
                    self.logger.error(msg, source_module=self.__class__.__name__)
            else:
                msg = (
                    "Not enough data for ROC calculation"
                )
                self.logger.debug(msg, source_module=self.__class__.__name__)
        return features

    def _calculate_ta_features(self, trading_pair: str, interval: str) -> dict[str, str] | None:
        """Calculates TA features based on the latest OHLCV history."""
        if self._system_status != "online":
            # Split long line into multiple lines for readability
            msg = (
                "TA Features: Status {status}, skipping."
            ).format(status=self._system_status)
            self.logger.debug(msg, source_module=self.__class__.__name__)
            return None

        history_key = (trading_pair, interval)
        df = self._prepare_ohlcv_dataframe(history_key)
        if df is None:
            # Split long line into multiple lines for readability
            msg = (
                "No OHLCV history for {pair} {interval}"
            ).format(pair=trading_pair, interval=interval)
            self.logger.debug(msg, source_module=self.__class__.__name__)
            return None

        features = {}
        # Calculate RSI features
        rsi_features = self._calculate_rsi_feature(df, trading_pair, interval)
        features.update(rsi_features)

        # Calculate ROC features
        roc_features = self._calculate_roc_feature(df, trading_pair, interval)
        features.update(roc_features)

        if not features:
            return None

        return features

    def _get_min_history_required(self) -> int:
        """Get the minimum required history size for TA calculations."""
        min_size = 1  # Minimum baseline
        
        # Check RSI requirements
        rsi_cfg = self._feature_configs.get("rsi", {})
        rsi_period = rsi_cfg.get("period")
        if isinstance(rsi_period, int) and rsi_period > min_size:
            min_size = rsi_period
            
        # Check ROC requirements
        roc_cfg = self._feature_configs.get("roc", {})
        roc_period = roc_cfg.get("period")
        if isinstance(roc_period, int) and roc_period > min_size:
            min_size = roc_period
            
        # Add 1 for better reliability in calculations
        return min_size + 1


# Example Usage Placeholder (Remove or guard in final app)
# if __name__ == "__main__": ...
