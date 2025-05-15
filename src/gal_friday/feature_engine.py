"""Feature engineering implementation for Gal-Friday.

This module provides the FeatureEngine class that handles computation of technical
indicators and other features used in prediction models.
"""

from collections import defaultdict, deque
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Optional, Union
import uuid

import pandas as pd
import pandas_ta as ta

from src.gal_friday.core.events import EventType
from src.gal_friday.core.pubsub import PubSubManager
from src.gal_friday.interfaces.historical_data_service_interface import (
    HistoricalDataService,
)
from src.gal_friday.logger_service import LoggerService


class FeatureEngine:
    """Processes market data to compute technical indicators and other features.

    The FeatureEngine is responsible for converting raw market data into features
    that can be used for machine learning models, including technical indicators,
    derived features, and potentially other types of features.
    """

    _EXPECTED_L2_LEVEL_LENGTH = 2

    def __init__(
        self,
        config: dict[str, Any],
        pubsub_manager: PubSubManager,
        logger_service: LoggerService,
        historical_data_service: Optional[HistoricalDataService] = None,
    ) -> None:
        """Initialize the FeatureEngine with configuration and required services.

        Args
        ----
            config: Dictionary containing configuration settings
            pubsub_manager: Instance of the pub/sub event manager
            logger_service: Logging service instance
            historical_data_service: Optional historical data service for initial data loading
        """
        self.config = config
        self.pubsub_manager = pubsub_manager
        self.logger = logger_service
        self.historical_data_service = historical_data_service
        self._source_module = self.__class__.__name__

        # Feature configuration derived from config
        self._feature_configs: dict[str, dict[str, Any]] = {}
        self._extract_feature_configs()

        # Initialize feature handlers dispatcher
        self._feature_handlers: dict[
            str, Callable[[dict[str, Any], dict[str, Any]], Optional[dict[str, str]]]
        ] = {
            "rsi": self._process_rsi_feature,
            "macd": self._process_macd_feature,
            "bbands": self._process_bbands_feature,
            "vwap": self._process_vwap_feature,  # Handles OHLCV & trade-based VWAP
            "roc": self._process_roc_feature,
            "atr": self._process_atr_feature,
            "stdev": self._process_stdev_feature,
            "spread": self._process_l2_spread_feature,
            "imbalance": self._process_l2_imbalance_feature,
            "wap": self._process_l2_wap_feature,
            "depth": self._process_l2_depth_feature,
            "volume_delta": self._process_volume_delta_feature,
            # Note: vwap_ohlcv and vwap_trades are handled by _process_vwap_feature
        }

        # Initialize data storage
        # OHLCV data will be stored in a DataFrame per trading pair
        # Columns: timestamp_bar_start (index), open, high, low, close, volume
        self.ohlcv_history: dict[str, pd.DataFrame] = defaultdict(
            lambda: pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            ).astype(
                {
                    "open": "object", # Store as Decimal initially
                    "high": "object",
                    "low": "object",
                    "close": "object",
                    "volume": "object",
                }
            )
        )

        # L2 order book data (latest snapshot)
        self.l2_books: dict[str, dict[str, Any]] = defaultdict(dict)

        # Store recent trades for calculating true Volume Delta and trade-based VWAP
        # deque stores: {"ts": datetime, "price": Decimal, "vol": Decimal, "side": "buy"/"sell"}
        trade_history_maxlen = config.get("feature_engine", {}).get("trade_history_maxlen", 2000)
        self.trade_history: dict[str, deque] = \
            defaultdict(lambda: deque(maxlen=trade_history_maxlen))

        self.logger.info("FeatureEngine initialized.", source_module=self._source_module)

    def _handle_ohlcv_update(self, trading_pair: str, ohlcv_payload: dict[str, Any]) -> None:
        """Parse and store an OHLCV update."""
        try:
            # Extract and convert data from payload
            # Timestamp parsing (ISO 8601 string to datetime object)
            # According to inter_module_comm.md: payload.timestamp_bar_start (ISO 8601)
            timestamp_str = ohlcv_payload.get("timestamp_bar_start")
            if not timestamp_str:
                self.logger.warning(
                    "Missing 'timestamp_bar_start' in OHLCV payload for %s",
                    trading_pair,
                    source_module=self._source_module,
                    context={"payload": ohlcv_payload},
                )
                return

            # Convert to datetime. PANDAS will handle timezone if present in string
            # Forcing UTC if not specified, adjust if local timezone is expected/preferred
            bar_timestamp = pd.to_datetime(timestamp_str, utc=True)

            # Price/Volume conversion (string to Decimal for precision)
            open_price = Decimal(ohlcv_payload["open"])
            high_price = Decimal(ohlcv_payload["high"])
            low_price = Decimal(ohlcv_payload["low"])
            close_price = Decimal(ohlcv_payload["close"])
            volume = Decimal(ohlcv_payload["volume"])

            # Prepare new row as a dictionary
            new_bar_data = {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }

            # Get the DataFrame for the trading pair
            df = self.ohlcv_history[trading_pair]

            # Create a new DataFrame for the new row with the correct index
            new_row_df = pd.DataFrame([new_bar_data], index=[bar_timestamp])
            new_row_df.index.name = "timestamp_bar_start"

            # Ensure new_row_df columns match df columns and types are compatible
            # This is important if df was empty or had different types initially
            for col in df.columns:
                if col not in new_row_df:
                    new_row_df[col] = pd.NA # Or appropriate default
            new_row_df = new_row_df[df.columns] # Ensure column order

            # If df is empty, new_row_df types might not match self.ohlcv_history default astype.
            # Re-apply astype or ensure compatible types for concat.
            # Assuming concat handles type promotion or columns are compatible.

            # Append new data
            # Check if timestamp already exists to avoid duplicates, update if it does
            if bar_timestamp not in df.index:
                df = pd.concat([df, new_row_df])
            else:
                # Update existing row
                df.loc[bar_timestamp] = new_row_df.iloc[0]
                self.logger.debug(
                    "Updated existing OHLCV bar for %s at %s",
                    trading_pair, bar_timestamp,
                    source_module=self._source_module,
                )

            # Sort by timestamp (index)
            df.sort_index(inplace=True)

            # Prune old data - keep a bit more than strictly required for safety margin
            min_hist = self._get_min_history_required()
            required_length = min_hist + 50  # Keep 50 extra bars as buffer
            if len(df) > required_length:
                df = df.iloc[-required_length:]

            self.ohlcv_history[trading_pair] = df
            self.logger.debug(
                "Processed OHLCV update for %s. History size: %s",
                trading_pair, len(df),
                source_module=self._source_module,
            )

        except KeyError:
            self.logger.exception(
                "Missing key in OHLCV payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )
        except (ValueError, TypeError):
            self.logger.exception(
                "Data conversion error in OHLCV payload for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )
        except Exception:
            self.logger.exception(
                "Unexpected error handling OHLCV update for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )

    def _handle_l2_update(self, trading_pair: str, l2_payload: dict[str, Any]) -> None:
        """Parse and store an L2 order book update."""
        try:
            # Extract bids and asks. Ensure they are lists of lists/tuples as expected.
            # inter_module_comm.md: bids/asks: List of lists [[price_str, volume_str], ...]
            raw_bids = l2_payload.get("bids")
            raw_asks = l2_payload.get("asks")

            if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
                self.logger.warning(
                    "L2 bids/asks are not lists for %s. Payload: %s",
                    trading_pair, l2_payload,
                    source_module=self._source_module,
                )
                # Decide if we should clear the book or keep stale data.
                # For now, we'll just return and not update if format is wrong.
                return

            # Convert price/volume strings to Decimal for precision
            # Bids: List of [Decimal(price), Decimal(volume)]
            # Asks: List of [Decimal(price), Decimal(volume)]
            # We expect bids to be sorted highest first, asks lowest first as per doc.

            processed_bids = []
            for i, bid_level in enumerate(raw_bids):
                if (
                    isinstance(bid_level, (list, tuple)) and
                    len(bid_level) == self._EXPECTED_L2_LEVEL_LENGTH
                ):
                    try:
                        processed_bids.append([Decimal(bid_level[0]), Decimal(bid_level[1])])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            "Error converting L2 bid level %s for %s: %s - %s",
                            i, trading_pair, bid_level, e,
                            source_module=self._source_module
                        )
                        # Optionally skip this level or use NaN/None
                        continue # Skip malformed level
                else:
                    self.logger.warning(
                        "Malformed L2 bid level %s for %s: %s",
                        i, trading_pair, bid_level, source_module=self._source_module
                    )

            processed_asks = []
            for i, ask_level in enumerate(raw_asks):
                if (
                    isinstance(ask_level, (list, tuple)) and
                    len(ask_level) == self._EXPECTED_L2_LEVEL_LENGTH
                ):
                    try:
                        processed_asks.append([Decimal(ask_level[0]), Decimal(ask_level[1])])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            "Error converting L2 ask level %s for %s: %s - %s",
                            i, trading_pair, ask_level, e, source_module=self._source_module
                        )
                        continue # Skip malformed level
                else:
                    self.logger.warning(
                        "Malformed L2 ask level %s for %s: %s",
                        i, trading_pair, ask_level, source_module=self._source_module
                    )

            # Store the processed L2 book data
            # The L2 book features will expect bids sorted high to low, asks low to high.
            self.l2_books[trading_pair] = {
                "bids": processed_bids, # Already sorted highest bid first from source
                "asks": processed_asks, # Already sorted lowest ask first from source
                "timestamp": pd.to_datetime(
                    l2_payload.get("timestamp_exchange") or datetime.utcnow(), utc=True
                )
            }
            self.logger.debug(
                "Processed L2 update for %s. Num bids: %s, Num asks: %s",
                trading_pair, len(processed_bids), len(processed_asks),
                source_module=self._source_module,
            )

        except KeyError:
            self.logger.exception(
                "Missing key in L2 payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context={"payload": l2_payload},
            )
        except Exception:
            self.logger.exception(
                "Unexpected error handling L2 update for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": l2_payload},
            )

    def _handle_trade_event(self, event_dict: dict[str, Any]) -> None:
        """Handle incoming raw trade events and store them."""
        # This method will be called by pubsub, so it takes the full event_dict
        payload = event_dict.get("payload")
        if not payload:
            self.logger.warning(
                "Trade event missing payload.",
                context=event_dict,
                source_module=self._source_module
            )
            return

        trading_pair = payload.get("trading_pair")
        if not trading_pair:
            self.logger.warning(
                "Trade event payload missing trading_pair.",
                context=payload, source_module=self._source_module
            )
            return

        try:
            trade_timestamp_str = payload.get("timestamp_exchange")
            price_str = payload.get("price")
            volume_str = payload.get("volume")
            side = payload.get("side") # "buy" or "sell"

            if not all([trade_timestamp_str, price_str, volume_str, side]):
                self.logger.warning(
                    "Trade event for %s is missing required fields "
                    "(timestamp, price, volume, or side).",
                    trading_pair,
                    context=payload, source_module=self._source_module
                )
                return

            trade_data = {
                "timestamp": pd.to_datetime(trade_timestamp_str, utc=True),
                "price": Decimal(price_str),
                "volume": Decimal(volume_str),
                "side": side.lower() # Ensure lowercase for consistency
            }

            if trade_data["side"] not in ["buy", "sell"]:
                self.logger.warning(
                    "Invalid trade side '%s' for %s.", trade_data["side"], trading_pair,
                    context=payload, source_module=self._source_module
                )
                return

            self.trade_history[trading_pair].append(trade_data)
            self.logger.debug(
                "Stored trade for %s: P=%s, V=%s, Side=%s",
                trading_pair, trade_data["price"], trade_data["volume"], trade_data["side"],
                source_module=self._source_module
            )

        except KeyError:
            self.logger.exception(
                "Missing key in trade event payload for %s.", trading_pair,
                source_module=self._source_module, context=payload
            )
        except (ValueError, TypeError):
            self.logger.exception(
                "Data conversion error in trade event payload for %s",
                trading_pair,
                source_module=self._source_module, context=payload
            )
        except Exception:
            self.logger.exception(
                "Unexpected error handling trade event for %s",
                trading_pair,
                source_module=self._source_module, context=payload
            )

    def _extract_feature_configs(self) -> None:
        """Extract feature-specific configurations from the main config."""
        features_config = self.config.get("features", {})
        if isinstance(features_config, dict):
            self._feature_configs = features_config

    def _get_min_history_required(self) -> int:
        """Determine the minimum required history size for TA calculations."""
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

        if periods:
            min_size = max(periods) * 3  # Multiply by 3 for a safe margin

        return max(100, min_size)  # At least 100 bars for good measure

    def _get_period_from_config(
        self, feature_name: str, field_name: str, default_value: int
    ) -> int:
        """Retrieve the period from config for a specific feature."""
        feature_cfg = self._feature_configs.get(feature_name, {})
        period_value = feature_cfg.get(field_name, default_value)

        return (
            period_value if isinstance(period_value, int) and period_value > 0 else default_value
        )

    async def start(self) -> None:
        """Start the feature engine and subscribe to relevant events."""
        try:
            # Subscribe process_market_data to handle both OHLCV and L2 updates
            await self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_OHLCV, self.process_market_data
            )
            await self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_L2, self.process_market_data
            )
            await self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_TRADE, self._handle_trade_event # New subscription
            )
            self.logger.info(
                "FeatureEngine started and subscribed to MARKET_DATA_OHLCV, "
                "MARKET_DATA_L2, and MARKET_DATA_TRADE events.",
                source_module=self._source_module
            )
        except Exception:
            self.logger.exception(
                "Error during FeatureEngine start and subscription",
                source_module=self._source_module
            )
            # Depending on desired behavior, might re-raise or handle to prevent full stop

    async def stop(self) -> None:
        """Stop the feature engine and clean up resources."""
        try:
            await self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_OHLCV, self.process_market_data
            )
            await self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_L2, self.process_market_data
            )
            await self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_TRADE, self._handle_trade_event # New unsubscription
            )
            self.logger.info(
                "FeatureEngine stopped and unsubscribed from market data events.",
                source_module=self._source_module
            )
        except Exception:
            self.logger.exception(
                "Error during FeatureEngine stop and unsubscription",
                source_module=self._source_module
            )

    async def process_market_data(self, market_data_event_dict: dict[str, Any]) -> None:
        """Process market data to generate features.

        Args
        ----
            market_data_event_dict: Market data event dictionary
        """
        # Assuming market_data_event_dict is the full event object including
        # event_type, payload, source_module, etc. as per inter_module_comm.md

        event_type = market_data_event_dict.get("event_type")
        payload = market_data_event_dict.get("payload")
        source_module = market_data_event_dict.get("source_module") # For logging/context

        if not event_type or not payload:
            self.logger.warning(
                "Received market data event with missing event_type or payload.",
                source_module=self._source_module, # Log from FeatureEngine itself
                context={"original_event": market_data_event_dict}
            )
            return

        trading_pair = payload.get("trading_pair")
        if not trading_pair:
            self.logger.warning(
                "Market data event (type: %s) missing trading_pair.",
                event_type,
                source_module=self._source_module,
                context={"original_event": market_data_event_dict}
            )
            return

        self.logger.debug(
            "Processing event %s for %s from %s",
            event_type, trading_pair, source_module,
            source_module=self._source_module
        )

        if event_type == "MARKET_DATA_OHLCV":
            self._handle_ohlcv_update(trading_pair, payload)
            # OHLCV update is the trigger for calculating all features for that bar's timestamp
            timestamp_bar_start = payload.get("timestamp_bar_start")
            if timestamp_bar_start:
                # We'll define _calculate_and_publish_features as async shortly
                await self._calculate_and_publish_features(trading_pair, timestamp_bar_start)
            else:
                self.logger.warning(
                    "OHLCV event for %s missing 'timestamp_bar_start', "
                    "cannot calculate features.",
                    trading_pair,
                    source_module=self._source_module,
                    context={"payload": payload}
                )
        elif event_type == "MARKET_DATA_L2":
            self._handle_l2_update(trading_pair, payload)
            # L2 updates typically don't trigger a full feature calculation on their own
            # in this design, as features are aligned with OHLCV bar closures.
            # L2 data is stored and used when an OHLCV bar triggers calculation.
        elif event_type == "MARKET_DATA_TRADE":
            self._handle_trade_event(market_data_event_dict)
        else:
            self.logger.warning(
                "Received unknown market data event type: %s for %s",
                event_type, trading_pair,
                source_module=self._source_module,
                context={"original_event": market_data_event_dict}
            )

    # --- Stubs for individual feature calculation methods ---
    def _calculate_rsi(self, close_series_decimal: pd.Series, period: int) -> Optional[Decimal]:
        self.logger.debug(
            "Calculating RSI with period %s using pandas-TA",
            period,
            source_module=self._source_module
        )
        if len(close_series_decimal) < period:
            self.logger.warning(
                "Not enough data for RSI period %s. Have %s points.",
                period,
                len(close_series_decimal),
                source_module=self._source_module
            )
            return None
        try:
            # pandas-ta expects float Series
            close_series_float = close_series_decimal.astype(float)
            rsi_series = close_series_float.ta.rsi(length=period)
            if rsi_series is None or rsi_series.empty or pd.isna(rsi_series.iloc[-1]):
                return None
            return Decimal(str(rsi_series.iloc[-1]))
        except Exception:
            self.logger.exception(
                "Error calculating RSI with pandas-TA", source_module=self._source_module
            )
            return None

    def _calculate_macd(
        self, close_series_decimal: pd.Series, fast: int, slow: int, signal: int
    ) -> Optional[dict[str, Decimal]]:
        self.logger.debug(
            "Calculating MACD with fast=%s, slow=%s, signal=%s using pandas-TA",
            fast, slow, signal,
            source_module=self._source_module
        )
        if len(close_series_decimal) < slow: # Need at least `slow` periods for MACD
            self.logger.warning(
                "Not enough data for MACD slow period %s. Have %s points.",
                slow,
                len(close_series_decimal),
                source_module=self._source_module
            )
            return None
        try:
            close_series_float = close_series_decimal.astype(float)
            macd_df = close_series_float.ta.macd(fast=fast, slow=slow, signal=signal)
            if macd_df is None or macd_df.empty:
                return None

            # MACD DataFrame columns are typically:
            # MACD_fast_slow_signal, MACDH_fast_slow_signal, MACDS_fast_slow_signal
            # Example: macd_df.columns might be ['MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9']
            macd_line_col = f"MACD_{fast}_{slow}_{signal}"
            signal_line_col = f"MACDS_{fast}_{slow}_{signal}"
            hist_col = f"MACDH_{fast}_{slow}_{signal}"

            if not all(
                col in macd_df.columns for col in [macd_line_col, signal_line_col, hist_col]
            ):
                self.logger.error(
                    "MACD columns not found in pandas-TA output. Columns: %s",
                    macd_df.columns,
                    source_module=self._source_module
                )
                return None

            latest_macd = macd_df[macd_line_col].iloc[-1]
            latest_signal = macd_df[signal_line_col].iloc[-1]
            latest_hist = macd_df[hist_col].iloc[-1]

            if pd.isna(latest_macd) or pd.isna(latest_signal) or pd.isna(latest_hist):
                return None

            return {
                "macd": Decimal(str(latest_macd)),
                "macdsignal": Decimal(str(latest_signal)),
                "macdhist": Decimal(str(latest_hist))
            }
        except Exception:
            self.logger.exception(
                "Error calculating MACD with pandas-TA",
                source_module=self._source_module
            )
            return None

    def _calculate_bollinger_bands(
        self, close_series_decimal: pd.Series, length: int, std_dev: float
    ) -> Optional[dict[str, Decimal]]:
        self.logger.debug(
            "Calculating Bollinger Bands with length=%s, std_dev=%s using pandas-TA",
            length,
            std_dev,
            source_module=self._source_module
        )
        if len(close_series_decimal) < length:
            self.logger.warning(
                "Not enough data for Bollinger Bands period %s. Have %s points.",
                length,
                len(close_series_decimal),
                source_module=self._source_module
            )
            return None
        try:
            close_series_float = close_series_decimal.astype(float)
            bbands_df = close_series_float.ta.bbands(length=length, std=std_dev)
            if bbands_df is None or bbands_df.empty:
                return None

            # Columns: BBL_length_std, BBM_length_std, BBU_length_std
            # (Lower, Middle, Upper Band)
            # Assuming pandas-ta uses 1 decimal for std in column names
            lower_col = f"BBL_{length}_{std_dev:.1f}"
            middle_col = f"BBM_{length}_{std_dev:.1f}"
            upper_col = f"BBU_{length}_{std_dev:.1f}"

            # Check for potential variations in column naming (e.g. if std_dev is integer)
            if not all(col in bbands_df.columns for col in [lower_col, middle_col, upper_col]):
                # Try with integer std_dev if it was an int
                if isinstance(std_dev, int) or std_dev == int(std_dev):
                    lower_col = f"BBL_{length}_{int(std_dev)}"
                    middle_col = f"BBM_{length}_{int(std_dev)}"
                    upper_col = f"BBU_{length}_{int(std_dev)}"
                if not all(col in bbands_df.columns for col in [lower_col, middle_col, upper_col]):
                    self.logger.error(
                        "Bollinger Bands columns not found. Expected BBL_%s_%.1f, etc. Got: %s",
                        length, std_dev, bbands_df.columns,
                        source_module=self._source_module
                    )
                    return None

            latest_lower = bbands_df[lower_col].iloc[-1]
            latest_middle = bbands_df[middle_col].iloc[-1]
            latest_upper = bbands_df[upper_col].iloc[-1]

            if pd.isna(latest_lower) or pd.isna(latest_middle) or pd.isna(latest_upper):
                return None

            return {
                "lowerband": Decimal(str(latest_lower)),
                "middleband": Decimal(str(latest_middle)),
                "upperband": Decimal(str(latest_upper))
            }
        except Exception:
            self.logger.exception(
                "Error calculating Bollinger Bands with pandas-TA",
                source_module=self._source_module
            )
            return None

    def _calculate_vwap(self, ohlcv_df_decimal: pd.DataFrame, length: int) -> Optional[Decimal]:
        # VWAP is often calculated per session or needs specific anchoring.
        # pandas-ta has vwap, but we'll do a rolling one here for simplicity as per previous stub.
        # If a specific library version of VWAP is desired, that can be used.
        self.logger.debug(
            "Calculating rolling VWAP with length=%s", length, source_module=self._source_module
        )
        if len(ohlcv_df_decimal) < length:
            self.logger.warning(
                "Not enough data for VWAP period %s. Have %s points.",
                length,
                len(ohlcv_df_decimal),
                source_module=self._source_module
            )
            return None
        try:
            relevant_df = ohlcv_df_decimal.iloc[-length:].copy() # Work on a copy
            # Ensure columns are of a numeric type that supports arithmetic (Decimal is fine here)
            relevant_df["typical_price"] = (
                relevant_df["high"] + relevant_df["low"] + relevant_df["close"].astype(Decimal)
            ) / Decimal("3.0")

            sum_tp_vol = (relevant_df["typical_price"] * relevant_df["volume"]).sum()
            sum_vol = relevant_df["volume"].sum()

            if sum_vol == Decimal("0"): # Avoid division by zero
                return None
            return sum_tp_vol / sum_vol
        except Exception:
            self.logger.exception(
                "Error calculating VWAP", source_module=self._source_module
            )
            return None

    def _calculate_roc(self, close_series_decimal: pd.Series, period: int) -> Optional[Decimal]:
        self.logger.debug(
            "Calculating ROC with period %s using pandas-TA",
            period,
            source_module=self._source_module
        )
        if len(close_series_decimal) <= period: # Needs more than 'period' points to compare
            self.logger.warning(
                "Not enough data for ROC period %s. Have %s points.",
                period,
                len(close_series_decimal),
                source_module=self._source_module
            )
            return None
        try:
            close_series_float = close_series_decimal.astype(float)
            roc_series = close_series_float.ta.roc(length=period)
            if roc_series is None or roc_series.empty or pd.isna(roc_series.iloc[-1]):
                return None
            return Decimal(str(roc_series.iloc[-1]))
        except Exception:
            self.logger.exception(
                "Error calculating ROC with pandas-TA", source_module=self._source_module
            )
            return None

    def _calculate_atr(self, ohlcv_df_decimal: pd.DataFrame, length: int) -> Optional[Decimal]:
        self.logger.debug(
            "Calculating ATR with length=%s using pandas-TA",
            length,
            source_module=self._source_module
        )
        # ATR needs high, low, close series
        if len(ohlcv_df_decimal) < length:
            self.logger.warning(
                "Not enough data for ATR period %s. Have %s points.",
                length,
                len(ohlcv_df_decimal),
                source_module=self._source_module
            )
            return None
        try:
            # pandas-ta expects float columns
            high_float = ohlcv_df_decimal["high"].astype(float)
            low_float = ohlcv_df_decimal["low"].astype(float)
            close_float = ohlcv_df_decimal["close"].astype(float)

            atr_series = ta.atr(high=high_float, low=low_float, close=close_float, length=length)
            if atr_series is None or atr_series.empty or pd.isna(atr_series.iloc[-1]):
                return None
            return Decimal(str(atr_series.iloc[-1]))
        except Exception:
            self.logger.exception(
                "Error calculating ATR with pandas-TA", source_module=self._source_module
            )
            return None

    def _calculate_stdev(self, close_series_decimal: pd.Series, length: int) -> Optional[Decimal]:
        self.logger.debug(
            "Calculating StDev with length=%s using pandas-TA (via .std())",
            length,
            source_module=self._source_module
        )
        if len(close_series_decimal) < length:
            self.logger.warning(
                "Not enough data for StDev period %s. Have %s points.",
                length,
                len(close_series_decimal),
                source_module=self._source_module
            )
            return None
        try:
            close_series_float = close_series_decimal.astype(float)
            # pandas-ta .stdev() is just a wrapper for pandas .std()
            stdev_val = close_series_float.rolling(window=length).std().iloc[-1]
            if pd.isna(stdev_val):
                return None
            return Decimal(str(stdev_val))
        except Exception:
            self.logger.exception(
                "Error calculating StDev",
                source_module=self._source_module
            )
            return None

    def _calculate_bid_ask_spread(self, l2_book: dict[str, Any]) -> Optional[dict[str, Decimal]]:
        self.logger.debug("Calculating Bid-Ask Spread", source_module=self._source_module)
        if not l2_book or not l2_book.get("bids") or not l2_book.get("asks"):
            return None
        best_bid = l2_book["bids"][0][0] # Price of first bid
        best_ask = l2_book["asks"][0][0] # Price of first ask
        if best_bid and best_ask:
            abs_spread = best_ask - best_bid
            pct_spread = (abs_spread / ((best_bid + best_ask) / Decimal(2))) * Decimal(100)
            return {"abs_spread": abs_spread, "pct_spread": pct_spread}
        return None

    def _calculate_order_book_imbalance(
        self, l2_book: dict[str, Any], levels: int = 5
    ) -> Optional[Decimal]:
        self.logger.debug(
            "Calculating Order Book Imbalance for %s levels",
            levels,
            source_module=self._source_module
        )
        # Placeholder
        if not l2_book or not l2_book.get("bids") or not l2_book.get("asks"):
            return None
        bid_vol_at_levels = sum(vol for price, vol in l2_book["bids"][:levels])
        ask_vol_at_levels = sum(vol for price, vol in l2_book["asks"][:levels])
        if bid_vol_at_levels + ask_vol_at_levels > 0:
            imbalance = (
                (bid_vol_at_levels - ask_vol_at_levels) /
                (bid_vol_at_levels + ask_vol_at_levels)
            )
            return Decimal(str(imbalance))
        return Decimal("0.0") # Or None if preferred for no volume

    def _calculate_wap(self, l2_book: dict[str, Any], levels: int = 1) -> Optional[Decimal]:
        self.logger.debug(
            "Calculating WAP for %s levels", levels, source_module=self._source_module
        )
        # This usually refers to WAP at the very top of the book (level 1)
        if not l2_book or \
           not l2_book.get("bids") or \
           not l2_book.get("asks") or \
           not l2_book["bids"] or \
           not l2_book["asks"]:
            return None
        best_bid_price = l2_book["bids"][0][0]
        best_bid_vol = l2_book["bids"][0][1]
        best_ask_price = l2_book["asks"][0][0]
        best_ask_vol = l2_book["asks"][0][1]
        if best_bid_vol + best_ask_vol > 0:
            wap_numerator = (best_bid_price * best_ask_vol) + (best_ask_price * best_bid_vol)
            wap_denominator = best_bid_vol + best_ask_vol
            wap = wap_numerator / wap_denominator
            return Decimal(str(wap))
        return None

    def _calculate_depth(
        self, l2_book: dict[str, Any], levels: int = 5
    ) -> Optional[dict[str, Decimal]]:
        self.logger.debug(
            "Calculating Depth for %s levels", levels,
            source_module=self._source_module
        )
        if not l2_book or not l2_book.get("bids") or not l2_book.get("asks"):
            return None
        bid_depth = sum(vol for price, vol in l2_book["bids"][:levels] if isinstance(vol, Decimal))
        ask_depth = sum(vol for price, vol in l2_book["asks"][:levels] if isinstance(vol, Decimal))
        return {"bid_depth": bid_depth, "ask_depth": ask_depth}

    def _calculate_true_volume_delta_from_trades(
        self,
        trades: deque,
        current_bar_start_time: datetime,
        bar_interval_seconds: int = 60
    ) -> Optional[Decimal]:
        """Calculate true Volume Delta from a deque of recent trades.

        Relevant to the current OHLCV bar.
        """
        self.logger.debug(
            "Calculating True Volume Delta from trades.",
            source_module=self._source_module
        )
        if not trades:
            return Decimal("0.0")

        # Calculate the end time for the current bar
        bar_end_time = current_bar_start_time + pd.Timedelta(seconds=bar_interval_seconds)

        relevant_trades = [
            trade for trade in trades
            if current_bar_start_time <= trade["timestamp"] < bar_end_time
        ]

        if not relevant_trades:
            # Could also return None if no trades in interval is distinct from zero delta
            return Decimal("0.0")

        buy_volume = sum(trade["volume"] for trade in relevant_trades if trade["side"] == "buy")
        sell_volume = sum(trade["volume"] for trade in relevant_trades if trade["side"] == "sell")

        return buy_volume - sell_volume

    def _calculate_vwap_from_trades(
        self, trades: deque, current_bar_start_time: datetime, bar_interval_seconds: int = 60
    ) -> Optional[Decimal]:
        """Calculate VWAP from a deque of recent trades relevant to the current OHLCV bar."""
        self.logger.debug("Calculating VWAP from trades.", source_module=self._source_module)
        if not trades:
            return None

        # Calculate the end time for the current bar
        bar_end_time = current_bar_start_time + pd.Timedelta(seconds=bar_interval_seconds)

        relevant_trades = [
            trade for trade in trades
            if current_bar_start_time <= trade["timestamp"] < bar_end_time
        ]

        if not relevant_trades:
            return None

        sum_price_volume = sum(trade["price"] * trade["volume"] for trade in relevant_trades)
        sum_volume = sum(trade["volume"] for trade in relevant_trades)

        if sum_volume == Decimal("0"):
            return None

        return sum_price_volume / sum_volume

    async def _calculate_and_publish_features(
        self, trading_pair: str, timestamp_features_for: str
    ) -> None:
        """Calculate all configured features and publish them as a FeatureEvent."""
        ohlcv_df = self.ohlcv_history.get(trading_pair)

        min_history = self._get_min_history_required()
        if ohlcv_df is None or len(ohlcv_df) < min_history:
            self.logger.info(
                (
                    "Not enough OHLCV data for %s to calculate features. "
                    "Need %s, have %s."
                ),
                trading_pair,
                min_history,
                len(ohlcv_df) if ohlcv_df is not None else 0,
                source_module=self._source_module
            )
            return

        current_l2_book = self.l2_books.get(trading_pair)
        if current_l2_book and \
           (not current_l2_book.get("bids") or not current_l2_book.get("asks")):
            self.logger.debug(
                "L2 book for %s is present but empty or missing bids/asks. "
                "L2 features may be skipped.",
                trading_pair,
                source_module=self._source_module
            )
            # L2 features might be skipped by their handlers if book is not suitable.

        # Prepare common data sources for feature handlers
        close_series_decimal = ohlcv_df["close"]
        ohlcv_df_decimal = ohlcv_df
        trades_for_pair = self.trade_history.get(trading_pair)
        current_bar_start_dt = pd.to_datetime(timestamp_features_for, utc=True)

        data_sources: dict[str, Any] = {
            "ohlcv": ohlcv_df_decimal,
            "close": close_series_decimal,
            "l2": current_l2_book,
            "trades": trades_for_pair,
            "bar_start": current_bar_start_dt,
        }

        calculated_features_dict: dict[str, str] = {}

        for feature_key, params in self._feature_configs.items():
            handler = self._feature_handlers.get(feature_key)
            if handler:
                try:
                    feature_results = handler(params, data_sources)
                    if feature_results: # Handler returns a dict of features or None
                        calculated_features_dict.update(feature_results)
                except Exception:
                    self.logger.exception(
                        "Error processing feature '%s' for %s using its handler",
                        feature_key,
                        trading_pair,
                        source_module=self._source_module
                    )
            else:
                self.logger.warning(
                    "No handler found for feature configuration key: '%s'",
                    feature_key,
                    source_module=self._source_module
                )

        if not calculated_features_dict:
            self.logger.info(
                "No features were successfully calculated for %s at %s. Not publishing event.",
                trading_pair,
                timestamp_features_for,
                source_module=self._source_module
            )
            return

        # Construct and publish FeatureEvent
        event_payload = {
            "trading_pair": trading_pair,
            "exchange": self.config.get("exchange_name", "kraken"),
            "timestamp_features_for": timestamp_features_for,
            "features": calculated_features_dict,
        }

        full_feature_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": EventType.FEATURES_CALCULATED.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_module": self._source_module,
            "payload": event_payload,
        }

        try:
            await self.pubsub_manager.publish(full_feature_event)
            self.logger.info(
                "Published FEATURES_CALCULATED event for %s at %s",
                trading_pair,
                timestamp_features_for,
                source_module=self._source_module,
                context={
                    "event_id": full_feature_event["event_id"],
                    "num_features": len(calculated_features_dict)
                }
            )
        except Exception:
            self.logger.exception(
                "Failed to publish FEATURES_CALCULATED event for %s",
                trading_pair,
                source_module=self._source_module,
            )

    def _format_feature_value(self, value: Union[Decimal, float, object]) -> str:
        """Format a feature value to string. Decimal/float to 8 decimal places."""
        if isinstance(value, (Decimal, float)):
            return f"{value:.8f}"
        return str(value)

    # --- Placeholder Feature Processing Methods ---
    def _process_rsi_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        close_series_decimal = data_sources.get("close")
        if close_series_decimal is None:
            self.logger.warning(
                "RSI calculation skipped: close data not available.",
                source_module=self._source_module
            )
            return None

        period = params.get("period", 14)
        result_val = self._calculate_rsi(close_series_decimal, period)

        if result_val is None:
            # _calculate_rsi already logs warnings for insufficient data
            return None

        feature_name = f"rsi_{period}"
        return {feature_name: self._format_feature_value(result_val)}

    def _process_macd_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        close_series_decimal = data_sources.get("close")
        if close_series_decimal is None:
            self.logger.warning(
                "MACD calculation skipped: close data not available.",
                source_module=self._source_module
            )
            return None

        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        result_dict = self._calculate_macd(close_series_decimal, fast, slow, signal)

        if result_dict is None:
            # _calculate_macd already logs warnings
            return None

        output_feature_base_name = f"macd_{fast}_{slow}_{signal}"
        formatted_features: dict[str, str] = {}
        for sub_key, sub_val in result_dict.items():
            if sub_val is not None: # Should always be non-None if result_dict is not None
                feature_full_name = f"{output_feature_base_name}_{sub_key}"
                formatted_features[feature_full_name] = self._format_feature_value(sub_val)

        return formatted_features if formatted_features else None

    def _process_bbands_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        close_series_decimal = data_sources.get("close")
        if close_series_decimal is None:
            self.logger.warning(
                "Bollinger Bands calculation skipped: close data not available.",
                source_module=self._source_module
            )
            return None

        length = params.get("length", 20)
        std_dev = float(params.get("std_dev", 2.0)) # _calculate_bollinger_bands expects float
        result_dict = self._calculate_bollinger_bands(close_series_decimal, length, std_dev)

        if result_dict is None:
            return None

        output_feature_base_name = f"bbands_{length}_{std_dev:.1f}"
        formatted_features: dict[str, str] = {}
        for sub_key, sub_val in result_dict.items():
            feature_full_name = f"{output_feature_base_name}_{sub_key}"
            formatted_features[feature_full_name] = self._format_feature_value(sub_val)

        return formatted_features if formatted_features else None

    def _process_vwap_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        source = params.get("source", "ohlcv")
        result_val: Optional[Decimal] = None
        feature_name: Optional[str] = None

        if source == "ohlcv":
            ohlcv_df_decimal = data_sources.get("ohlcv")
            if ohlcv_df_decimal is None:
                self.logger.warning(
                    "OHLCV VWAP calculation skipped: OHLCV data not available.",
                    source_module=self._source_module
                )
            else:
                length = params.get("length", 14)
                result_val = self._calculate_vwap(ohlcv_df_decimal, length)
                if result_val is not None:
                    feature_name = f"vwap_ohlcv_{length}"

        elif source == "trades":
            trades = data_sources.get("trades")
            bar_start_time = data_sources.get("bar_start")
            if trades is None or bar_start_time is None:
                self.logger.warning(
                    "Trade-based VWAP calculation skipped: "
                    "trade data or bar start time not available.",
                    source_module=self._source_module
                )
            else:
                bar_interval_seconds = params.get("length_seconds", 60)
                result_val = self._calculate_vwap_from_trades(
                    trades, bar_start_time, bar_interval_seconds
                )
                if result_val is not None:
                    feature_name = f"vwap_trades_{bar_interval_seconds}s"
        else:
            self.logger.warning(
                "Unknown VWAP source specified: %s. Skipping VWAP calculation.",
                source,
                source_module=self._source_module
            )

        if result_val is not None and feature_name is not None:
            return {feature_name: self._format_feature_value(result_val)}

        return None # Handles all cases where result_val or feature_name is None

    def _process_roc_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        close_series_decimal = data_sources.get("close")
        if close_series_decimal is None:
            self.logger.warning(
                "ROC calculation skipped: close data not available.",
                source_module=self._source_module
            )
            return None

        period = params.get("period", 10)
        result_val = self._calculate_roc(close_series_decimal, period)

        if result_val is None:
            return None

        feature_name = f"roc_{period}"
        return {feature_name: self._format_feature_value(result_val)}

    def _process_atr_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        ohlcv_df_decimal = data_sources.get("ohlcv")
        if ohlcv_df_decimal is None:
            self.logger.warning(
                "ATR calculation skipped: OHLCV data not available.",
                source_module=self._source_module
            )
            return None

        length = params.get("length", 14)
        result_val = self._calculate_atr(ohlcv_df_decimal, length)

        if result_val is None:
            return None

        feature_name = f"atr_{length}"
        return {feature_name: self._format_feature_value(result_val)}

    def _process_stdev_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        close_series_decimal = data_sources.get("close")
        if close_series_decimal is None:
            self.logger.warning(
                "StDev calculation skipped: close data not available.",
                source_module=self._source_module
            )
            return None

        length = params.get("length", 20)
        result_val = self._calculate_stdev(close_series_decimal, length)

        if result_val is None:
            return None

        feature_name = f"stdev_{length}"
        return {feature_name: self._format_feature_value(result_val)}

    def _process_l2_spread_feature(
        self, _params: dict[str, Any], data_sources: dict[str, Any] # params marked as unused
    ) -> Optional[dict[str, str]]:
        l2_book = data_sources.get("l2")
        if not l2_book or not l2_book.get("bids") or not l2_book.get("asks"):
            self.logger.debug(
                "L2 Spread calculation skipped: L2 book not suitable.",
                source_module=self._source_module
            )
            return None

        result_dict = self._calculate_bid_ask_spread(l2_book)
        if result_dict is None:
            return None

        # Result dict has 'abs_spread' and 'pct_spread'
        output_feature_base_name = "spread"
        formatted_features: dict[str, str] = {}
        for sub_key, sub_val in result_dict.items():
            formatted_features[
                f"{output_feature_base_name}_{sub_key}"
            ] = self._format_feature_value(sub_val)
        return formatted_features if formatted_features else None

    def _process_l2_imbalance_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        l2_book = data_sources.get("l2")
        if not l2_book or not l2_book.get("bids") or not l2_book.get("asks"):
            self.logger.debug(
                "L2 Imbalance calculation skipped: L2 book not suitable.",
                source_module=self._source_module
            )
            return None

        levels = params.get("levels", 5)
        result_val = self._calculate_order_book_imbalance(l2_book, levels)

        if result_val is None:
            return None

        feature_name = f"imbalance_{levels}"
        return {feature_name: self._format_feature_value(result_val)}

    def _process_l2_wap_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        l2_book = data_sources.get("l2")
        if not l2_book or not l2_book.get("bids") or not l2_book.get("asks"):
            self.logger.debug(
                "L2 WAP calculation skipped: L2 book not suitable.",
                source_module=self._source_module
            )
            return None

        levels = params.get("levels", 1) # WAP usually for top level
        result_val = self._calculate_wap(l2_book, levels)

        if result_val is None:
            return None

        feature_name = f"wap_{levels}"
        return {feature_name: self._format_feature_value(result_val)}

    def _process_l2_depth_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        l2_book = data_sources.get("l2")
        if not l2_book or not l2_book.get("bids") or not l2_book.get("asks"):
            self.logger.debug(
                "L2 Depth calculation skipped: L2 book not suitable.",
                source_module=self._source_module
            )
            return None

        levels = params.get("levels", 5)
        result_dict = self._calculate_depth(l2_book, levels)

        if result_dict is None:
            return None

        output_feature_base_name = f"depth_{levels}"
        formatted_features: dict[str, str] = {}
        for sub_key, sub_val in result_dict.items():
            formatted_features[
                f"{output_feature_base_name}_{sub_key}"
            ] = self._format_feature_value(sub_val)
        return formatted_features if formatted_features else None

    def _process_volume_delta_feature(
        self, params: dict[str, Any], data_sources: dict[str, Any]
    ) -> Optional[dict[str, str]]:
        trades = data_sources.get("trades")
        bar_start_time = data_sources.get("bar_start")

        if trades is None or bar_start_time is None:
            self.logger.warning(
                "Volume Delta calculation skipped: trade data or bar start time not available.",
                source_module=self._source_module
            )
            return None

        # Default interval for trade aggregation, make configurable if needed from params
        bar_interval_seconds = params.get("bar_interval_seconds", 60)
        result_val = self._calculate_true_volume_delta_from_trades(
            trades, bar_start_time, bar_interval_seconds
        )

        # _calculate_true_volume_delta_from_trades returns Decimal("0.0") if no trades
        if result_val is None:
            # but could theoretically be adapted to return None if desired for "no data"
            return None

        # Construct the feature name dynamically using the bar_interval_seconds from params
        feature_name = f"volume_delta_{bar_interval_seconds}s"
        return {feature_name: self._format_feature_value(result_val)}
