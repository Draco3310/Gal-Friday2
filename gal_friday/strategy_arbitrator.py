"""Strategy arbitration for trading signal generation.

This module contains the StrategyArbitrator, which consumes prediction events from models,
applies trading strategy logic, and produces proposed trade signals. The arbitrator
supports configurable threshold strategies with secondary confirmation rules.
"""

# Strategy Arbitrator Module

import operator  # Added for condition dispatch
import uuid
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import ClassVar

# Event imports
from .core.events import EventType, PredictionEvent, TradeSignalProposedEvent

# Import PubSubManager
from .core.pubsub import PubSubManager

# Import LoggerService
from .logger_service import LoggerService

# Import MarketPriceService
from .market_price_service import MarketPriceService

# Import FeatureRegistryClient
from .core.feature_registry_client import FeatureRegistryClient


# --- StrategyArbitrator Class ---
class StrategyArbitrator:
    """
    Consumes prediction events from models, applies trading strategy logic, and
    produces proposed trade signals.

    The arbitrator supports configurable threshold-based strategies with secondary
    confirmation rules. These rules can leverage features published by the
    `FeatureEngine`. The `StrategyArbitrator` uses an injected `FeatureRegistryClient`
    to validate that feature names specified in confirmation rules exist in the
    Feature Registry during its configuration validation phase.

    During live operation, it expects `PredictionEvent`s to carry the necessary
    `triggering_features` (as a `dict[str, float]`) within their
    `associated_features` payload, which are then used to evaluate the
    confirmation rules.
    """

    def __init__(
        self,
        config: dict,
        pubsub_manager: PubSubManager,
        logger_service: LoggerService,
        market_price_service: MarketPriceService,
        feature_registry_client: FeatureRegistryClient, # Added parameter
    ) -> None:
        """Initialize the StrategyArbitrator.

        Args:
            config (dict): Configuration settings. Expected structure:
                strategy_arbitrator:
                  strategies:
                    - id: "mvp_threshold_v1"
                      buy_threshold: 0.65
                      # ... other strategy params ...
                      confirmation_rules: # Rules use feature names from Feature Registry
                        - feature: "rsi_14_default"
                          condition: "lt"
                          threshold: 30
            pubsub_manager (PubSubManager): For subscribing/publishing events.
            logger_service (LoggerService): The shared logger instance.
            market_price_service (MarketPriceService): Service to get market prices.
            feature_registry_client (FeatureRegistryClient): Instance of the feature registry client.
        """
        self._config = config.get("strategy_arbitrator", {})
        self.pubsub = pubsub_manager
        self.logger = logger_service
        self.market_price_service = market_price_service
        self.feature_registry_client = feature_registry_client # Use passed instance
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
                str(self._mvp_strategy_config.get("limit_offset_pct", "0.0001")),
            )  # e.g. 0.01% = 0.0001
            self._prediction_interpretation = self._mvp_strategy_config.get(
                "prediction_interpretation",
                "prob_up",
            )
            default_rr_ratio_str = self._mvp_strategy_config.get(
                "default_reward_risk_ratio",
                "2.0",
            )
            self._default_reward_risk_ratio = Decimal(str(default_rr_ratio_str))

            # Specific config for price_change_pct interpretation
            if self._prediction_interpretation == "price_change_pct":
                self._price_change_buy_threshold_pct = Decimal(
                    str(self._mvp_strategy_config["price_change_buy_threshold_pct"]),
                )
                self._price_change_sell_threshold_pct = Decimal(
                    str(self._mvp_strategy_config["price_change_sell_threshold_pct"]),
                )

            self._validate_configuration()

        except KeyError as key_error:
            self.logger.exception(
                "Missing required strategy parameter.",
                source_module=self._source_module,
            )
            raise StrategyConfigurationError from key_error
        except (InvalidOperation, TypeError) as value_error:
            self.logger.exception(
                "Invalid parameter format in strategy configuration.",
                source_module=self._source_module,
            )
            raise StrategyConfigurationError from value_error

    def _validate_core_parameters(self) -> None:
        """Validate core strategy parameters like entry type, thresholds, SL/TP percentages."""
        if self._entry_type not in ["MARKET", "LIMIT"]:
            self.logger.error(
                "Invalid entry_type in strategy: %s. Must be 'MARKET' or 'LIMIT'.",
                self._entry_type,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError
        if self._buy_threshold <= self._sell_threshold:
            self.logger.error(
                "Buy threshold (%s) must be greater than sell threshold (%s).",
                self._buy_threshold,
                self._sell_threshold,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError
        if self._sl_pct is not None and (self._sl_pct <= 0 or self._sl_pct >= 1):
            self.logger.error(
                "Stop-loss percentage (%s) must be between 0 and 1 (exclusive).",
                self._sl_pct,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError
        if self._tp_pct is not None and (self._tp_pct <= 0 or self._tp_pct >= 1):
            self.logger.error(
                "Take-profit percentage (%s) must be between 0 and 1 (exclusive).",
                self._tp_pct,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError
        if self._limit_offset_pct < 0:
            self.logger.error(
                "Limit offset percentage (%s) cannot be negative.",
                self._limit_offset_pct,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError

    def _validate_prediction_interpretation_config(self) -> None:
        """Validate prediction interpretation settings and related thresholds."""
        if self._prediction_interpretation not in [
            "prob_up",
            "prob_down",
            "price_change_pct",
        ]:  # Example interpretations
            self.logger.error(
                "Invalid prediction_interpretation in strategy: %s",
                self._prediction_interpretation,
                source_module=self._source_module,
            )
            raise StrategyConfigurationError

        if self._prediction_interpretation == "price_change_pct":
            if not hasattr(self, "_price_change_buy_threshold_pct") or not hasattr(
                self,
                "_price_change_sell_threshold_pct",
            ):
                self.logger.error(
                    "Missing price_change_pct thresholds for '%s' interpretation.",
                    self._prediction_interpretation,
                    source_module=self._source_module,
                )
                raise StrategyConfigurationError
            if self._price_change_buy_threshold_pct <= self._price_change_sell_threshold_pct:
                self.logger.error(
                    "Price change buy_threshold_pct (%s) must be > sell_threshold_pct (%s).",
                    self._price_change_buy_threshold_pct,
                    self._price_change_sell_threshold_pct,
                    source_module=self._source_module,
                )
                raise StrategyConfigurationError
            if self._price_change_buy_threshold_pct <= 0:
                self.logger.warning(
                    "price_change_buy_threshold_pct is not positive. This might be unintentional.",
                    source_module=self._source_module,
                )
            if self._price_change_sell_threshold_pct >= 0:
                self.logger.warning(
                    "price_change_sell_threshold_pct is not negative. "
                    "This might be unintentional.",
                    source_module=self._source_module,
                )

    def _validate_confirmation_rules_config(self) -> None:
        """
        Validate the structure of confirmation rules and check if feature names
        exist in the Feature Registry.
        """
        if not self.feature_registry_client or not self.feature_registry_client.is_loaded():
            self.logger.warning(
                "FeatureRegistryClient not available or not loaded. "
                "Skipping feature name validation in confirmation rules for strategy '%s'.",
                self._strategy_id,
                source_module=self._source_module
            )
            # Perform only structural validation if registry is not available
            for rule in self._confirmation_rules:
                if not all(k in rule for k in ["feature", "condition", "threshold"]):
                    self.logger.error(
                        "Invalid confirmation rule structure for strategy '%s': %s",
                        self._strategy_id, rule,
                        source_module=self._source_module,
                    )
                    raise StrategyConfigurationError(f"Invalid confirmation rule structure for strategy {self._strategy_id}: {rule}")
            return

        for rule in self._confirmation_rules:
            if not all(k in rule for k in ["feature", "condition", "threshold"]):
                self.logger.error(
                    "Invalid confirmation rule structure for strategy '%s': %s",
                     self._strategy_id, rule,
                    source_module=self._source_module,
                )
                raise StrategyConfigurationError(f"Invalid confirmation rule structure for strategy {self._strategy_id}: {rule}")

            feature_name = rule.get("feature")
            if feature_name: # feature_name is present in the rule structure
                definition = self.feature_registry_client.get_feature_definition(str(feature_name))
                if definition is None:
                    self.logger.warning(
                        "Confirmation rule for strategy '%s' references feature_key '%s' "
                        "which is not found in the Feature Registry. This rule may be "
                        "ineffective or cause errors during secondary confirmation.",
                        self._strategy_id, feature_name,
                        source_module=self._source_module
                    )
                else:
                    self.logger.debug(
                        "Confirmation rule feature '%s' for strategy '%s' validated against Feature Registry.",
                        feature_name, self._strategy_id,
                        source_module=self._source_module
                    )
            # If feature_name is None or empty, the structural check above would have already caught it
            # if 'feature' was a required key with a non-empty value.
            # The current structural check `all(k in rule for k in ["feature", ...])` ensures 'feature' key exists.
            # An additional check for empty feature_name string might be useful if desired.

    def _validate_configuration(self) -> None:
        """Validate loaded strategy configuration by calling specific validators."""
        self._validate_core_parameters()
        self._validate_prediction_interpretation_config()
        self._validate_confirmation_rules_config()

        self.logger.info(
            "Strategy configuration validated successfully.",
            source_module=self._source_module,
        )

    def _validate_prediction_event(self, event: PredictionEvent) -> bool:
        """Validate the incoming PredictionEvent."""
        if not hasattr(event, "prediction_value") or event.prediction_value is None:
            self.logger.warning(
                "PredictionEvent %s missing prediction_value.",
                event.event_id,
                source_module=self._source_module,
            )
            return False
        if not hasattr(event, "trading_pair") or not event.trading_pair:
            self.logger.warning(
                "PredictionEvent %s missing trading_pair.",
                event.event_id,
                source_module=self._source_module,
            )
            return False
        # Example: Check if probability is between 0 and 1 if that's the interpretation
        if self._prediction_interpretation in ["prob_up", "prob_down"]:
            try:
                val = float(event.prediction_value)
                if not (0 <= val <= 1):
                    self.logger.warning(
                        "Prediction_value %s for %s is outside [0,1] range.",
                        val,
                        event.trading_pair,
                        source_module=self._source_module,
                    )
                    return False
            except ValueError:
                self.logger.warning(
                    "Prediction_value %s is not a valid float.",
                    event.prediction_value,
                    source_module=self._source_module,
                )
                return False
        return True

    def _calculate_stop_loss_price_and_risk(
        self,
        side: str,
        current_price: Decimal,
        # Optionally, tp_price can be provided to derive SL if sl_pct is not set
        tp_price_for_rr_calc: Decimal | None = None,
    ) -> tuple[Decimal | None, Decimal | None]:  # Returns (sl_price, risk_amount_per_unit)
        """Calculate stop-loss price and risk amount per unit."""
        sl_price: Decimal | None = None
        risk_amount_per_unit: Decimal | None = None

        if self._sl_pct is not None and self._sl_pct > 0:
            if side == "BUY":
                sl_price = current_price * (Decimal("1") - self._sl_pct)
                risk_amount_per_unit = current_price - sl_price
            elif side == "SELL":
                sl_price = current_price * (Decimal("1") + self._sl_pct)
                risk_amount_per_unit = sl_price - current_price
        elif (
            tp_price_for_rr_calc is not None  # Check if TP is provided for derivation
            and self._default_reward_risk_ratio is not None
            and self._default_reward_risk_ratio > 0
        ):
            if side == "BUY":
                reward_amount_per_unit = tp_price_for_rr_calc - current_price
                if reward_amount_per_unit > 0:  # Ensure positive reward
                    risk_amount_per_unit = reward_amount_per_unit / self._default_reward_risk_ratio
                    sl_price = current_price - risk_amount_per_unit
            elif side == "SELL":
                reward_amount_per_unit = current_price - tp_price_for_rr_calc
                if reward_amount_per_unit > 0:  # Ensure positive reward
                    risk_amount_per_unit = reward_amount_per_unit / self._default_reward_risk_ratio
                    sl_price = current_price + risk_amount_per_unit
        return sl_price, risk_amount_per_unit

    def _calculate_take_profit_price(
        self,
        side: str,
        current_price: Decimal,
        # Optionally, sl_price and risk can be provided to derive TP
        sl_price_for_rr_calc: Decimal | None = None,
        risk_amount_for_rr_calc: Decimal | None = None,
    ) -> Decimal | None:
        """Calculate take-profit price."""
        tp_price: Decimal | None = None

        if self._tp_pct is not None and self._tp_pct > 0:
            if side == "BUY":
                tp_price = current_price * (Decimal("1") + self._tp_pct)
            elif side == "SELL":
                tp_price = current_price * (Decimal("1") - self._tp_pct)
        elif (
            sl_price_for_rr_calc is not None
            and risk_amount_for_rr_calc is not None
            and risk_amount_for_rr_calc > 0
            and self._default_reward_risk_ratio is not None
            and self._default_reward_risk_ratio > 0
        ):
            reward_adjustment = risk_amount_for_rr_calc * self._default_reward_risk_ratio
            if side == "BUY":
                tp_price = current_price + reward_adjustment
            elif side == "SELL":
                tp_price = current_price - reward_adjustment
        return tp_price

    async def _calculate_sl_tp_prices(
        self,
        side: str,
        current_price: Decimal,
        trading_pair: str,
    ) -> tuple[Decimal | None, Decimal | None]:
        """Calculate SL/TP prices based on configuration and current price."""
        if current_price <= 0:
            self.logger.error(
                "Cannot calculate SL/TP for %s: Invalid current_price %s",
                trading_pair,
                current_price,
                source_module=self._source_module,
            )
            return None, None

        sl_price: Decimal | None = None
        tp_price: Decimal | None = None
        risk_amount_per_unit: Decimal | None = None

        # Attempt 1: Calculate SL directly, then TP
        sl_price, risk_amount_per_unit = self._calculate_stop_loss_price_and_risk(
            side,
            current_price,
        )
        if sl_price and risk_amount_per_unit:
            tp_price = self._calculate_take_profit_price(
                side,
                current_price,
                sl_price,
                risk_amount_per_unit,
            )

        # Attempt 2: If SL failed but TP might be possible directly, calculate TP then derive SL
        if not (sl_price and tp_price):  # If first attempt didn't yield both
            tp_price_direct = self._calculate_take_profit_price(side, current_price)
            if tp_price_direct:
                # Try to derive SL using this TP
                derived_sl, derived_risk = self._calculate_stop_loss_price_and_risk(
                    side,
                    current_price,
                    tp_price_for_rr_calc=tp_price_direct,
                )
                if derived_sl and derived_risk:
                    sl_price = derived_sl
                    risk_amount_per_unit = derived_risk
                    tp_price = tp_price_direct  # Use the directly calculated TP

        # Logging for failure if still no SL or TP
        if not sl_price:
            self.logger.error(
                "SL params error for %s on %s. Need sl_pct or (tp_pct & RR).",
                self._strategy_id,
                trading_pair,
                source_module=self._source_module,
            )
        if not tp_price:
            self.logger.error(
                "TP params error for %s on %s. Need tp_pct or (sl_pct & RR).",
                self._strategy_id,
                trading_pair,
                source_module=self._source_module,
            )

        if not (sl_price and tp_price):
            self.logger.error(
                "Failed to calculate both SL and TP prices for %s.",
                trading_pair,
                source_module=self._source_module,
            )
            return None, None

        # Validate that prices make sense
        valid_prices = True
        if (side == "BUY" and (sl_price >= current_price or tp_price <= current_price)) or (
            side == "SELL" and (sl_price <= current_price or tp_price >= current_price)
        ):
            valid_prices = False

        if not valid_prices:
            self.logger.error(
                "Invalid SL/TP for %s on %s: SL=%s, TP=%s, Cur=%s",
                side,
                trading_pair,
                sl_price,
                tp_price,
                current_price,
                source_module=self._source_module,
            )
            return None, None

        try:
            return sl_price, tp_price
        except Exception:  # Should be rare as calculations are done, but for safety
            self.logger.exception(
                "Error during final SL/TP return for %s",
                trading_pair,
                source_module=self._source_module,
            )
            return None, None

    async def _determine_entry_price(
        self,
        side: str,
        current_price: Decimal,
        trading_pair: str,
    ) -> Decimal | None:
        """Determine the proposed entry price based on order type."""
        if self._entry_type == "MARKET":
            return None  # No specific price for market orders
        if self._entry_type == "LIMIT":
            try:
                # Fetch current spread
                spread_data = await self.market_price_service.get_bid_ask_spread(trading_pair)
                if (
                    spread_data is None
                    # or spread_data.get("bid") is None # spread_data is now a tuple or None
                    # or spread_data.get("ask") is None
                ):
                    self.logger.warning(
                        "Cannot determine limit price for %s: "
                        "Bid/Ask unavailable. Falling back to current price.",
                        trading_pair,
                        source_module=self._source_module,
                    )
                    return current_price  # Fallback as per whiteboard suggestion

                # Removed commented-out lines for ERA001
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
                        "Invalid side '%s' for limit price determination.",  # G004 fix
                        side,
                        source_module=self._source_module,
                    )
                    return None

                # No rounding here, handled by downstream modules
            except Exception:
                self.logger.exception(
                    "Error determining limit price for %s. Falling back to current price.",
                    trading_pair,
                    source_module=self._source_module,
                )
                return current_price  # Fallback on error
            else:
                return limit_price  # TRY300 fix: moved from try block
        else:
            self.logger.error(
                "Unsupported entry type for price determination: %s",  # G004 fix
                self._entry_type,
                source_module=self._source_module,
            )
            return None

    _CONDITION_OPERATORS: ClassVar[dict[str, Callable]] = {
        "gt": operator.gt,
        "lt": operator.lt,
        "eq": operator.eq,
        "gte": operator.ge,
        "lte": operator.le,
        "ne": operator.ne,
    }

    def _validate_confirmation_rule(
        self,
        rule: dict,
        features: dict,
        trading_pair: str,
        primary_side: str,
    ) -> bool:
        """
        Validate a single confirmation rule against the provided features.

        Args:
            rule: A dictionary defining the confirmation rule (feature, condition, threshold).
            features: A dictionary of feature names to their float values, typically
                      `triggering_features` from a `PredictionEvent`.
            trading_pair: The trading pair for which the rule is being validated.
            primary_side: The primary signal side ("BUY" or "SELL") being considered.

        Returns:
            True if the rule passes or is skipped (due to invalid structure or
            unsupported condition), False if the rule condition is not met or an
            error occurs during processing.
        """
        feature_name = rule.get("feature")
        condition_key = rule.get("condition")
        threshold_str = rule.get("threshold")

        if not all([feature_name, condition_key, threshold_str is not None]): # threshold can be 0
            self.logger.warning(
                "Skipping invalid confirmation rule (missing component): %s for %s",
                rule,
                trading_pair,
                source_module=self._source_module,
            )
            return True # Skip invalid rule, effectively passing it by not blocking

        # Optional: Validate feature_name against registry if desired for stricter checks
        # if not self.feature_registry_client.get_feature_definition(feature_name):
        #     self.logger.warning(f"Feature '{feature_name}' in rule not found in registry. Rule may fail.")

        feature_value_float = features.get(feature_name)

        if feature_value_float is None:
            self.logger.info(
                "Sec confirm failed for %s on %s: Feature '%s' not in triggering_features.",
                primary_side,
                trading_pair,
                feature_name,
                source_module=self._source_module,
            )
            return False

        rule_passes = False
        try:
            # Convert feature (float) and threshold (str) to Decimal for precise comparison
            feature_value_decimal = Decimal(str(feature_value_float))
            threshold = Decimal(str(threshold_str))

            op = self._CONDITION_OPERATORS.get(str(condition_key))

            if op:
                condition_met = op(feature_value, threshold)
                if condition_met:
                    rule_passes = True
                else:
                    self.logger.info(
                        "Secondary confirm failed for %s on %s: Rule %s %s %s (Val: %s) not met.",
                        primary_side,
                        trading_pair,
                        feature_name,
                        condition_key,  # Use renamed variable
                        threshold,
                        feature_value,
                        source_module=self._source_module,
                    )
                    rule_passes = False
            else:
                self.logger.warning(
                    "Unsupported condition '%s' in rule: %s for %s",
                    condition_key,  # Use renamed variable
                    rule,
                    trading_pair,
                    source_module=self._source_module,
                )
                rule_passes = True  # Skip rule with unsupported condition (treat as passed)

        except (InvalidOperation, TypeError, KeyError):
            self.logger.exception(
                "Error applying confirmation rule %s for %s",
                rule,
                trading_pair,
                source_module=self._source_module,
            )
            rule_passes = False  # Error in processing means rule fails

        return rule_passes

    def _apply_secondary_confirmation(
        self,
        prediction_event: PredictionEvent,
        primary_side: str,
    ) -> bool:
        """
        Check if secondary confirmation rules pass using features from the PredictionEvent.

        Args:
            prediction_event: The `PredictionEvent` containing associated features.
            primary_side: The primary signal side ("BUY" or "SELL") being considered.

        Returns:
            True if all confirmation rules pass or if no rules are defined.
            False if any rule fails or if necessary features are missing.

        Features are expected in `prediction_event.associated_features['triggering_features']`
        as a `dict[str, float]`.
        """
        if not self._confirmation_rules:
            return True  # No rules defined, confirmation passes by default

        associated_payload = getattr(prediction_event, "associated_features", None)
        raw_features: Optional[Dict[str, float]] = None
        if isinstance(associated_payload, dict):
            raw_features = associated_payload.get("triggering_features")

        if not raw_features or not isinstance(raw_features, dict): # Check if raw_features is None, empty or not a dict
            self.logger.warning(
                "No valid 'triggering_features' (dict[str, float]) found in PredictionEvent %s for %s on %s.",
                prediction_event.event_id,
                primary_side,
                prediction_event.trading_pair,
                source_module=self._source_module,
            )
            return False # Cannot confirm without features

        for rule in self._confirmation_rules:
            if not self._validate_confirmation_rule(
                rule,
                raw_features, # Pass the dict[str, float]
                prediction_event.trading_pair,
                primary_side,
            ):
                return False # Rule failed

        self.logger.debug(
            "All secondary confirmation rules passed for %s signal on %s.",
            primary_side,
            prediction_event.trading_pair,
            source_module=self._source_module,
        )
        return True

    def _get_side_from_prob_up(self, prob_up: Decimal) -> str | None:
        """Determine signal side based on probability of price increase."""
        if prob_up >= self._buy_threshold:
            return "BUY"
        if prob_up < self._sell_threshold:  # sell_threshold is upper bound for prob_up to sell
            return "SELL"
        return None

    def _get_side_from_prob_down(self, prob_down: Decimal, trading_pair: str) -> str | None:
        """Determine signal side based on probability of price decrease."""
        buy_signal: str | None = None
        # BUY condition based on effective P(up)
        effective_prob_up = Decimal(1) - prob_down
        if effective_prob_up >= self._buy_threshold:
            buy_signal = "BUY"

        sell_signal: str | None = None
        # SELL condition based on P(down) vs buy_threshold (Whiteboard rule)
        if prob_down >= self._buy_threshold:
            sell_signal = "SELL"

        if buy_signal and sell_signal:
            # Implies buy_thresh <= 0.5; rare for typical probability thresholds. (E501 fix)
            self.logger.warning(
                "Conflicting signals for %s using prob_down: BUY and SELL triggered. "
                "P(down)=%s, buy_threshold=%s. No signal generated.",
                trading_pair,
                prob_down,
                self._buy_threshold,
                source_module=self._source_module,
            )
            return None

        return buy_signal or sell_signal

    def _get_side_from_price_change_pct(self, price_change_pct: Decimal) -> str | None:
        """Determine signal side based on predicted price change percentage."""
        if price_change_pct >= self._price_change_buy_threshold_pct:
            return "BUY"
        if price_change_pct <= self._price_change_sell_threshold_pct:
            return "SELL"
        return None

    def _calculate_signal_side(self, prediction_event: PredictionEvent) -> str | None:
        """Interpret prediction and determine the primary signal side."""
        trading_pair = prediction_event.trading_pair
        try:
            prediction_val = Decimal(str(prediction_event.prediction_value))
        except (InvalidOperation, TypeError):
            self.logger.warning(
                "Invalid prediction_value: '%s' for %s. Cannot determine side.",
                prediction_event.prediction_value,
                trading_pair,
                source_module=self._source_module,
            )
            return None

        interpretation = self._prediction_interpretation

        if interpretation == "prob_up":
            return self._get_side_from_prob_up(prediction_val)
        if interpretation == "prob_down":
            return self._get_side_from_prob_down(prediction_val, trading_pair)
        if interpretation == "price_change_pct":
            return self._get_side_from_price_change_pct(prediction_val)

        # Default for unknown interpretation
        self.logger.warning(
            "Unknown prediction_interpretation '%s'. Defaulting to 'prob_up' for %s.",
            interpretation,
            trading_pair,
            source_module=self._source_module,
        )
        return self._get_side_from_prob_up(prediction_val)

    async def _evaluate_strategy(
        self,
        prediction_event: PredictionEvent,
    ) -> TradeSignalProposedEvent | None:
        """Evaluate trading strategy based on prediction probabilities.

        Returns TradeSignalProposedEvent if strategy triggers, None otherwise.
        """
        # Init to None. Return value if any step fails or exception.
        generated_event: TradeSignalProposedEvent | None = None

        try:
            trading_pair = prediction_event.trading_pair
            side = self._calculate_signal_side(prediction_event)

            if not side:
                self.logger.debug(
                    "No primary signal for %s (PredID: %s, Val: %s, Interpret: %s)",
                    trading_pair,
                    prediction_event.event_id,
                    prediction_event.prediction_value,
                    self._prediction_interpretation,
                    source_module=self._source_module,
                )
                # generated_event remains None, will be returned by the 'else' block below
            elif not self._apply_secondary_confirmation(prediction_event, side):
                self.logger.info(
                    "Primary signal %s for %s (PredID: %s) failed secondary confirmation.",
                    side,
                    trading_pair,
                    prediction_event.event_id,
                    source_module=self._source_module,
                )
                # generated_event remains None
            else:
                current_price = await self.market_price_service.get_latest_price(trading_pair)
                if current_price is None:
                    self.logger.warning(
                        "Cannot generate signal for %s (PredID: %s): Failed to get current price.",
                        trading_pair,
                        prediction_event.event_id,
                        source_module=self._source_module,
                    )
                    # generated_event remains None
                else:
                    sl_price, tp_price = await self._calculate_sl_tp_prices(
                        side,
                        current_price,
                        trading_pair,
                    )
                    if sl_price is None or tp_price is None:
                        self.logger.warning(
                            "Failed to calc SL/TP for %s on %s (PredID: %s). Price: %s",
                            side,
                            trading_pair,
                            prediction_event.event_id,
                            current_price,
                            source_module=self._source_module,
                        )
                        # generated_event remains None
                    else:
                        # All checks passed, proceed to create the event
                        proposed_entry = await self._determine_entry_price(
                            side,
                            current_price,
                            trading_pair,
                        )
                        signal_id = uuid.uuid4()
                        generated_event = TradeSignalProposedEvent(
                            source_module=self._source_module,
                            event_id=uuid.uuid4(),
                            timestamp=datetime.utcnow(),
                            signal_id=signal_id,
                            trading_pair=trading_pair,
                            exchange=prediction_event.exchange,
                            side=side,
                            entry_type=self._entry_type,
                            proposed_sl_price=sl_price,
                            proposed_tp_price=tp_price,
                            strategy_id=self._strategy_id,
                            proposed_entry_price=proposed_entry,
                            triggering_prediction_event_id=prediction_event.event_id,
                        )
                        self.logger.info(
                            "Signal: %s (%s) for %s from PredID %s. SL:%s TP:%s Entry:%s",
                            side,
                            signal_id,
                            trading_pair,
                            prediction_event.event_id,
                            sl_price,
                            tp_price,
                            proposed_entry or "MARKET",
                            source_module=self._source_module,
                        )
            # No early returns in the try block for guard clauses.
            # generated_event is either a TradeSignalProposedEvent or None.

        except (
            InvalidOperation,
            TypeError,
            AttributeError,
            Exception,
        ):  # Catch generic Exception too
            event_id_str = getattr(prediction_event, "event_id", "UNKNOWN_EVENT_ID")
            pair_str = getattr(prediction_event, "trading_pair", "UNKNOWN_PAIR")
            self.logger.exception(
                "Error evaluating strategy for prediction %s on %s",
                event_id_str,
                pair_str,
                source_module=self._source_module,
            )
            return None  # Explicit return None for clarity on error in exception case
        else:
            # This block executes if the try block completed without exceptions.
            # It returns the generated_event, which is either the event or None if checks failed.
            return generated_event

    async def start(self) -> None:
        """Start listening for prediction events."""
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
        """Stop the event processing loop."""
        if not self._is_running:
            return
        self._is_running = False

        # Unsubscribe
        try:
            self.pubsub.unsubscribe(EventType.PREDICTION_GENERATED, self._prediction_handler)
            self.logger.info(
                "Unsubscribed from PREDICTION_GENERATED.",
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception(
                "Error unsubscribing StrategyArbitrator",
                source_module=self._source_module,
            )

        self.logger.info("StrategyArbitrator stopped.", source_module=self._source_module)

    async def handle_prediction_event(self, event: PredictionEvent) -> None:
        """Handle incoming prediction events directly."""
        if not isinstance(event, PredictionEvent):
            self.logger.warning(
                "Received non-PredictionEvent: %s",
                type(event),
                source_module=self._source_module,
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
        """Publish the TradeSignalProposedEvent."""
        try:
            await self.pubsub.publish(event)
            self.logger.debug(
                "Published TradeSignalProposedEvent: %s for %s",
                event.signal_id,
                event.trading_pair,
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception(
                "Failed to publish TradeSignalProposedEvent %s for %s",
                event.signal_id,
                event.trading_pair,
                source_module=self._source_module,
            )


# Custom Exception for configuration errors
class StrategyConfigurationError(ValueError):
    """Custom exception for strategy configuration errors."""
