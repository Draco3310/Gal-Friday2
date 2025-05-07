# Strategy Arbitrator Module

import uuid
from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Optional, Tuple

# Event imports
from .core.events import EventType, PredictionEvent, TradeSignalProposedEvent

# Import PubSubManager
from .core.pubsub import PubSubManager

# Import LoggerService
from .logger_service import LoggerService

# Import MarketPriceService
from .market_price_service import MarketPriceService

# print("Strategy Arbitrator Loaded") # Removed this debug print


# Define PredictionPayload for typing
# @dataclass
# class PredictionPayload:
#     """Payload for prediction data (used for type hints)"""

#     trading_pair: str
#     exchange: str
#     model_id: str
#     prediction_target: str
#     prediction_value: float
#     confidence: Optional[float] = None
#     timestamp_prediction_for: Optional[datetime] = None


# --- StrategyArbitrator Class ---
class StrategyArbitrator:
    """
    Consumes prediction events, applies configurable trading strategy logic,
    and publishes proposed trade signal events.
    """

    def __init__(
        self,
        config: dict,
        pubsub_manager: PubSubManager,
        logger_service: LoggerService,
        market_price_service: MarketPriceService,
    ):
        """
        Initializes the StrategyArbitrator.

        Args:
            config (dict): Configuration settings. Expected structure:
                strategy_arbitrator:
                  strategies:
                    - id: "mvp_threshold_v1"
                      buy_threshold: 0.65
                      sell_threshold: 0.35
                      entry_type: "MARKET"
                      sl_pct: 0.2
                      tp_pct: 0.4
                      # Example for new configs:
                      # confirmation_rules:
                      #   - feature: "momentum_5"
                      #     condition: "gt"
                      #     threshold: 0
                      # limit_offset_pct: 0.01
                      # prediction_interpretation: "prob_up"
                      # default_reward_risk_ratio: 2.0
            pubsub_manager (PubSubManager): For subscribing/publishing events.
            logger_service (LoggerService): The shared logger instance.
            market_price_service (MarketPriceService): Service to get market prices.
        """
        self._config = config.get("strategy_arbitrator", {})
        self.pubsub = pubsub_manager
        self.logger = logger_service
        self.market_price_service = market_price_service
        self._is_running = False
        self._main_task = None
        self._source_module = self.__class__.__name__

        self._prediction_handler = self.handle_prediction_event

        self._strategies = self._config.get("strategies", [])
        if not self._strategies:
            # Log error and raise a custom exception or handle gracefully
            err_msg = "No strategies configured for StrategyArbitrator."
            self.logger.error(err_msg, source_module=self._source_module)
            raise StrategyConfigurationError(err_msg)

        # For now, continue with MVP using the first strategy.
        # Robust implementation would iterate or select based on criteria.
        self._mvp_strategy_config = self._strategies[0]
        self._strategy_id = self._mvp_strategy_config.get("id", "default_strategy")

        try:
            self._buy_threshold = Decimal(str(self._mvp_strategy_config["buy_threshold"]))
            self._sell_threshold = Decimal(str(self._mvp_strategy_config["sell_threshold"]))
            self._entry_type = self._mvp_strategy_config.get("entry_type", "MARKET").upper()

            sl_pct_conf = self._mvp_strategy_config.get("sl_pct")
            tp_pct_conf = self._mvp_strategy_config.get("tp_pct")
            self._sl_pct = Decimal(str(sl_pct_conf)) if sl_pct_conf is not None else None
            self._tp_pct = Decimal(str(tp_pct_conf)) if tp_pct_conf is not None else None

            # New config options from whiteboard
            self._confirmation_rules = self._mvp_strategy_config.get("confirmation_rules", [])
            self._limit_offset_pct = Decimal(
                str(self._mvp_strategy_config.get("limit_offset_pct", "0.0001"))
            )  # e.g. 0.01% = 0.0001
            self._prediction_interpretation = self._mvp_strategy_config.get(
                "prediction_interpretation", "prob_up"
            )
            default_rr_ratio_str = self._mvp_strategy_config.get(
                "default_reward_risk_ratio", "2.0"
            )
            self._default_reward_risk_ratio = Decimal(str(default_rr_ratio_str))

            self._validate_configuration()

        except KeyError as key_error:
            err_msg = f"Missing required strategy parameter: {key_error}"
            self.logger.error(err_msg, source_module=self._source_module)
            raise StrategyConfigurationError(err_msg)
        except (InvalidOperation, TypeError) as value_error:
            err_msg = f"Invalid parameter format in strategy configuration: {value_error}"
            self.logger.error(err_msg, source_module=self._source_module)
            raise StrategyConfigurationError(err_msg)

    def _validate_configuration(self) -> None:
        """Validates loaded strategy configuration."""
        if self._entry_type not in ["MARKET", "LIMIT"]:
            raise StrategyConfigurationError(f"Invalid entry_type: {self._entry_type}")
        if self._buy_threshold <= self._sell_threshold:
            raise StrategyConfigurationError(
                f"buy_threshold ({self._buy_threshold}) must be greater than "
                f"sell_threshold ({self._sell_threshold})"
            )
        if self._sl_pct is not None and (self._sl_pct <= 0 or self._sl_pct >= 1):
            raise StrategyConfigurationError(
                f"sl_pct ({self._sl_pct}) must be between 0 and 1 (exclusive)."
            )
        if self._tp_pct is not None and (self._tp_pct <= 0 or self._tp_pct >= 1):
            raise StrategyConfigurationError(
                f"tp_pct ({self._tp_pct}) must be between 0 and 1 (exclusive)."
            )
        if self._limit_offset_pct < 0:
            raise StrategyConfigurationError(
                f"limit_offset_pct ({self._limit_offset_pct}) cannot be negative."
            )
        if self._prediction_interpretation not in [
            "prob_up",
            "prob_down",
            "price_change_pct",
        ]:  # Example interpretations
            raise StrategyConfigurationError(
                f"Invalid prediction_interpretation: {self._prediction_interpretation}"
            )

        # Validate confirmation rules structure (basic example)
        for rule in self._confirmation_rules:
            if not all(k in rule for k in ["feature", "condition", "threshold"]):
                raise StrategyConfigurationError(
                    f"Invalid confirmation rule, missing keys: {rule}"
                )
        self.logger.info(
            "Strategy configuration validated successfully.", source_module=self._source_module
        )

    def _validate_prediction_event(self, event: PredictionEvent) -> bool:
        """Validates the incoming PredictionEvent."""
        if not hasattr(event, "prediction_value") or event.prediction_value is None:
            self.logger.warning(
                f"PredictionEvent {event.event_id} missing prediction_value.",
                source_module=self._source_module,
            )
            return False
        if not hasattr(event, "trading_pair") or not event.trading_pair:
            self.logger.warning(
                f"PredictionEvent {event.event_id} missing trading_pair.",
                source_module=self._source_module,
            )
            return False
        # Example: Check if probability is between 0 and 1 if that's the interpretation
        if self._prediction_interpretation in ["prob_up", "prob_down"]:
            try:
                val = float(event.prediction_value)
                if not (0 <= val <= 1):
                    self.logger.warning(
                        f"Prediction_value {val} for {event.trading_pair} is outside [0,1] range.",
                        source_module=self._source_module,
                    )
                    return False
            except ValueError:
                self.logger.warning(
                    f"Prediction_value {event.prediction_value} is not a valid float.",
                    source_module=self._source_module,
                )
                return False
        return True

    async def _calculate_sl_tp_prices(
        self, side: str, current_price: Decimal, trading_pair: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculates SL/TP prices based on configuration and current price."""
        if self._sl_pct is None or self._tp_pct is None:
            self.logger.error(
                f"SL/TP percentages are not configured for strategy {self._strategy_id} "
                f"on {trading_pair}.",
                source_module=self._source_module,
            )
            return None, None
        if current_price <= 0:
            self.logger.error(
                f"Cannot calculate SL/TP for {trading_pair}: Invalid current_price "
                f"{current_price}",
                source_module=self._source_module,
            )
            return None, None

        try:
            if side == "BUY":
                sl_price = current_price * (Decimal("1") - self._sl_pct)
                tp_price = current_price * (Decimal("1") + self._tp_pct)
            elif side == "SELL":
                sl_price = current_price * (Decimal("1") + self._sl_pct)
                tp_price = current_price * (Decimal("1") - self._tp_pct)
            else:
                self.logger.error(
                    f"Invalid side '{side}' for SL/TP calculation.",
                    source_module=self._source_module,
                )
                return None, None  # Should not happen

            # Placeholder for rounding based on pair precision
            # sl_price = self._round_price(sl_price, trading_pair)
            # tp_price = self._round_price(tp_price, trading_pair)

            return sl_price, tp_price
        except Exception as e:
            self.logger.error(
                f"Error calculating SL/TP prices for {trading_pair}: {e}",
                exc_info=True,
                source_module=self._source_module,
            )
            return None, None

    async def _determine_entry_price(
        self, side: str, current_price: Decimal, trading_pair: str
    ) -> Optional[Decimal]:
        """Determines the proposed entry price based on order type."""
        if self._entry_type == "MARKET":
            return None  # No specific price for market orders
        elif self._entry_type == "LIMIT":
            try:
                # Fetch current spread
                spread_data = await self.market_price_service.get_bid_ask_spread(trading_pair)
                if (
                    spread_data
                    is None
                    # or spread_data.get("bid") is None # spread_data is now a tuple or None
                    # or spread_data.get("ask") is None
                ):
                    self.logger.warning(
                        f"Cannot determine limit price for {trading_pair}: "
                        "Bid/Ask unavailable. Falling back to current price.",
                        source_module=self._source_module,
                    )
                    return current_price  # Fallback as per whiteboard suggestion

                # best_bid = Decimal(str(spread_data["bid"]))
                # best_ask = Decimal(str(spread_data["ask"]))
                best_bid, best_ask = spread_data  # Unpack tuple

                if side == "BUY":
                    # Place limit slightly below current ask or at ask
                    limit_price = best_ask * (Decimal(1) - self._limit_offset_pct)
                    # Ensure buy limit is not above current_price (or ask)
                    # significantly if offset is large
                    limit_price = min(limit_price, best_ask)
                elif side == "SELL":
                    # Place limit slightly above current bid or at bid
                    limit_price = best_bid * (Decimal(1) + self._limit_offset_pct)
                    # Ensure sell limit is not below current_price (or bid) significantly
                    limit_price = max(limit_price, best_bid)
                else:  # Should not happen
                    self.logger.error(
                        f"Invalid side '{side}' for limit price determination.",
                        source_module=self._source_module,
                    )
                    return None

                # Placeholder for rounding
                # limit_price = self._round_price(limit_price, trading_pair)
                return limit_price
            except Exception as e:
                self.logger.error(
                    f"Error determining limit price for {trading_pair}: {e}. "
                    "Falling back to current price.",
                    exc_info=True,
                    source_module=self._source_module,
                )
                return current_price  # Fallback on error
        else:
            self.logger.error(
                f"Unsupported entry type for price determination: {self._entry_type}",
                source_module=self._source_module,
            )
            return None

    def _validate_confirmation_rule(
        self, rule: dict, features: dict, trading_pair: str, primary_side: str
    ) -> bool:
        """Validates a single confirmation rule against event features."""
        feature_name = rule.get("feature")
        condition = rule.get("condition")
        threshold_str = rule.get("threshold")

        if not all([feature_name, condition, threshold_str]):
            self.logger.warning(
                f"Skipping invalid confirmation rule (missing component): {rule} "
                f"for {trading_pair}",
                source_module=self._source_module,
            )
            return True
        # Or False, depending on how strict we want to be. Let's assume skip means not failed.

        if feature_name not in features:
            self.logger.info(
                f"Secondary confirmation failed for {primary_side} signal on {trading_pair}: "
                f"Required feature '{feature_name}' not found in event features.",
                source_module=self._source_module,
            )
            return False

        try:
            feature_value = Decimal(str(features[feature_name]))
            threshold = Decimal(str(threshold_str))

            passes = False
            if condition == "gt" and feature_value > threshold:
                passes = True
            elif condition == "lt" and feature_value < threshold:
                passes = True
            elif condition == "eq" and feature_value == threshold:
                passes = True
            elif condition == "gte" and feature_value >= threshold:
                passes = True
            elif condition == "lte" and feature_value <= threshold:
                passes = True
            elif condition == "ne" and feature_value != threshold:
                passes = True
            else:
                self.logger.warning(
                    f"Unsupported condition '{condition}' in confirmation rule: {rule} "
                    f"for {trading_pair}",
                    source_module=self._source_module,
                )
                return True  # Skip rule with unsupported condition

            if not passes:
                self.logger.info(
                    f"Secondary confirmation failed for {primary_side} signal on "
                    f"{trading_pair}: Rule {feature_name} {condition} {threshold} "
                    f"(Value: {feature_value}) not met.",
                    source_module=self._source_module,
                )
                return False
            return True
        except (InvalidOperation, TypeError, KeyError) as e:
            self.logger.error(
                f"Error applying confirmation rule {rule} for {trading_pair}: {e}",
                exc_info=True,
                source_module=self._source_module,
            )
            return False

    def _apply_secondary_confirmation(
        self, prediction_event: PredictionEvent, primary_side: str
    ) -> bool:
        """Checks if secondary confirmation rules pass."""
        if not self._confirmation_rules:
            return True  # No rules defined, confirmation passes by default

        features = getattr(prediction_event, "associated_features", None)
        if not features:
            self.logger.warning(
                f"No associated features in PredictionEvent {prediction_event.event_id} "
                f"for secondary confirmation of {primary_side} signal on "
                f"{prediction_event.trading_pair}.",
                source_module=self._source_module,
            )
            return False  # Cannot confirm without features

        for rule in self._confirmation_rules:
            if not self._validate_confirmation_rule(
                rule, features, prediction_event.trading_pair, primary_side
            ):
                return False

        self.logger.debug(
            f"All secondary confirmation rules passed for {primary_side} signal "
            f"on {prediction_event.trading_pair}.",
            source_module=self._source_module,
        )
        return True

    async def _evaluate_strategy(
        self, prediction_event: PredictionEvent
    ) -> Optional[TradeSignalProposedEvent]:
        """
        Evaluates trading strategy based on prediction probabilities.
        Returns TradeSignalProposedEvent if strategy triggers, None otherwise.
        """
        try:
            trading_pair = prediction_event.trading_pair
            prediction_val = Decimal(str(prediction_event.prediction_value))  # Ensure Decimal

            # --- Interpret Prediction ---
            prob_up = None
            # Assuming binary prediction for now based on original logic,
            # but this part needs to be flexible based on self._prediction_interpretation
            if self._prediction_interpretation == "prob_up":
                prob_up = prediction_val
                prob_down = Decimal(1) - prob_up
            elif self._prediction_interpretation == "prob_down":
                prob_down = prediction_val
                prob_up = Decimal(1) - prob_down
            # elif self._prediction_interpretation == "price_change_pct":
            #     # Logic for price change percentage would be different,
            #     # e.g. positive for buy, negative for sell.
            #     # This part needs full implementation if used.
            #     self.logger.warning(f"price_change_pct interpretation not fully implemented for
            #     {trading_pair}.", source_module=self._source_module)
            #     return None
            else:  # Default to original interpretation if not set or unknown
                prob_up = prediction_val
                prob_down = Decimal(1) - prob_up

            if prob_up is None:  # Should be caught by config validation, but as safeguard
                self.logger.error(
                    f"Could not determine prob_up based on prediction_value for "
                    f"{trading_pair} with interpretation {self._prediction_interpretation}",
                    source_module=self._source_module,
                )
                return None

            # --- Primary Signal Logic ---
            side = None
            if prob_up >= self._buy_threshold:
                side = "BUY"
            elif (
                prob_down >= self._buy_threshold
            ):  # Whiteboard used buy_threshold for sell side too
                side = "SELL"
            # Original logic: elif prob_down >= self._sell_threshold
            # (if sell_threshold is different for P(down))
            # The original code used self._buy_threshold for P(down) threshold as well.
            # The whiteboard implies the primary signal is based on
            # prediction_value (assumed P_up) vs buy_threshold
            # and (1-prediction_value) vs sell_threshold (or similar).
            # For now, sticking to: prob_up >= buy_thresh for BUY,
            # prob_down >= buy_thresh for SELL (as in current code)
            # This means sell_threshold is effectively not used if P(down) uses buy_threshold.
            # Clarification: The original code compared `prob_down >=
            # float(self._buy_threshold)` for SELL.
            # The config has `self._sell_threshold`. If the intent is symmetric:
            # if prob_up >= self._buy_threshold: side = "BUY"
            # if prob_down >= (1 - self._sell_threshold)
            # if sell_threshold is P(up) for sell
            # OR if sell_threshold is P(down) for sell: if prob_down >= self._sell_threshold
            # Given the config `buy_threshold` and `sell_threshold` for P(up) values,
            # a common pattern is: P(up) > buy_thresh for BUY; P(up) < sell_thresh for SELL.
            # Let's use the existing pattern from the code:
            # P(up) >= buy_threshold => BUY
            # P(down) >= buy_threshold => SELL
            # This implies sell_threshold is not directly used unless interpretation changes.
            # Sticking to the provided original logic for primary
            # signal generation based on thresholds.

            if not side:
                return None  # No primary signal

            # --- Apply Secondary Confirmation ---
            if not self._apply_secondary_confirmation(prediction_event, side):
                self.logger.info(
                    f"Primary signal {side} for {trading_pair} "
                    f"(PredID: {prediction_event.event_id}) failed secondary confirmation.",
                    source_module=self._source_module,
                )
                return None

            # --- Get Current Price ---
            # Assuming market_price_service.get_latest_price returns a Decimal or compatible type
            current_price = await self.market_price_service.get_latest_price(trading_pair)
            if current_price is None:
                self.logger.warning(
                    f"Cannot generate signal for {trading_pair} "
                    f"(PredID: {prediction_event.event_id}): Failed to get current price.",
                    source_module=self._source_module,
                )
                return None

            # --- Calculate SL/TP ---
            sl_price, tp_price = await self._calculate_sl_tp_prices(
                side, current_price, trading_pair
            )
            if sl_price is None or tp_price is None:
                self.logger.warning(
                    f"Failed to calculate SL/TP for {side} signal on {trading_pair} "
                    f"(PredID: {prediction_event.event_id}). Current price: {current_price}",
                    source_module=self._source_module,
                )
                return None  # Error logged in helper

            # --- Determine Entry Price ---
            proposed_entry = await self._determine_entry_price(side, current_price, trading_pair)
            # Note: _determine_entry_price can return None for MARKET,
            # or fallback to current_price for LIMIT on error

            signal_id = uuid.uuid4()
            proposed_event = TradeSignalProposedEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                signal_id=signal_id,
                trading_pair=trading_pair,
                exchange=prediction_event.exchange,
                side=side,
                entry_type=self._entry_type,
                proposed_sl_price=sl_price,  # Use calculated value
                proposed_tp_price=tp_price,  # Use calculated value
                strategy_id=self._strategy_id,
                proposed_entry_price=proposed_entry,  # Use calculated value
                triggering_prediction_event_id=prediction_event.event_id,
            )

            self.logger.info(
                f"Generated {side} signal proposal ({signal_id}) for {trading_pair} "
                f"from PredID {prediction_event.event_id}. SL: {sl_price}, TP: {tp_price}, "
                f"Entry: {proposed_entry or 'MARKET'}",
                source_module=self._source_module,
            )
            return proposed_event

        except (
            InvalidOperation,
            TypeError,
            AttributeError,
            Exception,
        ) as e:  # Catch generic Exception too
            event_id_str = getattr(prediction_event, "event_id", "UNKNOWN_EVENT_ID")
            pair_str = getattr(prediction_event, "trading_pair", "UNKNOWN_PAIR")
            self.logger.error(
                f"Error evaluating strategy for prediction {event_id_str} on {pair_str}: {e}",
                exc_info=True,
                source_module=self._source_module,
            )
        return None

    async def start(self) -> None:
        """Starts listening for prediction events."""
        if self._is_running:
            self.logger.warning(
                "StrategyArbitrator already running.",
                source_module=self._source_module,
            )
            return
        self._is_running = True

        # Subscribe to PredictionEvent
        self.pubsub.subscribe(EventType.PREDICTION_GENERATED, self._prediction_handler)

        self.logger.info("StrategyArbitrator started.", source_module=self._source_module)

    async def stop(self) -> None:
        """Stops the event processing loop."""
        if not self._is_running:
            return
        self._is_running = False

        # Unsubscribe
        try:
            self.pubsub.unsubscribe(EventType.PREDICTION_GENERATED, self._prediction_handler)
            self.logger.info(
                "Unsubscribed from PREDICTION_GENERATED.", source_module=self._source_module
            )
        except Exception as e:
            self.logger.error(
                f"Error unsubscribing StrategyArbitrator: {e}",
                exc_info=True,
                source_module=self._source_module,
            )

        self.logger.info("StrategyArbitrator stopped.", source_module=self._source_module)

    async def handle_prediction_event(self, event: PredictionEvent) -> None:
        """Handles incoming prediction events directly."""
        if not isinstance(event, PredictionEvent):
            self.logger.warning(
                f"Received non-PredictionEvent: {type(event)}", source_module=self._source_module
            )
            return

        # Validate prediction event
        if not self._validate_prediction_event(event):
            # Validation failed, error logged in helper
            return

        if not self._is_running:
            self.logger.debug(
                "StrategyArbitrator is not running, skipping prediction event.",
                source_module=self._source_module,
            )
            return

        # Evaluate strategy based on the prediction event
        proposed_signal_event = await self._evaluate_strategy(event)

        # Publish the proposed signal event if generated
        if proposed_signal_event:
            await self._publish_trade_signal_proposed(proposed_signal_event)

    async def _publish_trade_signal_proposed(self, event: TradeSignalProposedEvent) -> None:
        """Publishes the TradeSignalProposedEvent."""
        try:
            await self.pubsub.publish(event)
            self.logger.debug(
                f"Published TradeSignalProposedEvent: {event.signal_id} for {event.trading_pair}",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.error(
                f"Failed to publish TradeSignalProposedEvent {event.signal_id} for "
                f"{event.trading_pair}: {e}",
                exc_info=True,
                source_module=self._source_module,
            )


# Custom Exception for configuration errors
class StrategyConfigurationError(ValueError):
    """Custom exception for strategy configuration errors."""

    pass
