# Strategy Arbitrator Module

import asyncio
import uuid
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

# Event imports
from .core.events import (
    Event, 
    EventType, 
    PredictionEvent, 
    TradeSignalProposedEvent
)

# Import PubSubManager
from .core.pubsub import PubSubManager
# Import LoggerService
from .logger_service import LoggerService


print("Strategy Arbitrator Loaded")


# Define PredictionPayload for typing
@dataclass
class PredictionPayload:
    """Payload for prediction data (used for type hints)"""
    trading_pair: str
    exchange: str
    model_id: str
    prediction_target: str
    prediction_value: float
    confidence: Optional[float] = None
    timestamp_prediction_for: Optional[datetime] = None


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
            pubsub_manager (PubSubManager): For subscribing/publishing events.
            logger_service (LoggerService): The shared logger instance.
        """
        self._config = config.get("strategy_arbitrator", {})
        self.pubsub = pubsub_manager # Store PubSubManager
        self.logger = logger_service
        self._is_running = False
        self._main_task = None
        self._source_module = self.__class__.__name__

        # Store handler for unsubscribing
        self._prediction_handler = self.handle_prediction_event

        # Load strategy configurations
        self._strategies = self._config.get("strategies", [])
        if not self._strategies:
            raise ValueError(
                "At least one strategy configuration is required."
            )

        # Select the first strategy config for MVP
        self._mvp_strategy_config = self._strategies[0]
        self._strategy_id = self._mvp_strategy_config.get("id", "default_strategy")

        # Get MVP strategy parameters & validate
        try:
            self._buy_threshold = Decimal(
                str(self._mvp_strategy_config["buy_threshold"])
            )
            self._sell_threshold = Decimal(
                str(self._mvp_strategy_config["sell_threshold"])
            )
            self._entry_type = self._mvp_strategy_config.get(
                "entry_type", "MARKET"
            ).upper()
            sl_pct_conf = self._mvp_strategy_config.get("sl_pct")
            tp_pct_conf = self._mvp_strategy_config.get("tp_pct")
            self._sl_pct = (
                Decimal(str(sl_pct_conf)) if sl_pct_conf is not None else None
            )
            self._tp_pct = (
                Decimal(str(tp_pct_conf)) if tp_pct_conf is not None else None
            )
        except KeyError as key_error:
            raise ValueError(
                "Missing required parameter '{key}'".format(key=key_error)
            )
        except (InvalidOperation, TypeError) as value_error:
            raise ValueError(
                "Invalid parameter format: {error}".format(error=value_error)
            )

        if self._entry_type not in ["MARKET", "LIMIT"]:
            raise ValueError(
                "Invalid entry_type '{type}'".format(type=self._entry_type)
            )
        if self._buy_threshold <= self._sell_threshold:
            msg = (
                "buy_threshold ({buy}) must be greater than "
                "sell_threshold ({sell})"
            ).format(
                buy=self._buy_threshold,
                sell=self._sell_threshold
            )
            raise ValueError(msg)

    def _evaluate_strategy(
        self,
        prediction_event: PredictionEvent # Use PredictionEvent directly
    ) -> Optional[TradeSignalProposedEvent]:
        """
        Evaluates trading strategy based on prediction probabilities.
        Returns TradeSignalProposedEvent if strategy triggers, None otherwise.
        """
        # Extract prediction probabilities
        try:
            # Access fields directly from the event object
            prob_up = prediction_event.prediction_value # Assuming this is P(up)
            prob_down = 1.0 - prob_up # Assuming binary prediction target
            trading_pair = prediction_event.trading_pair

            # --- MVP Strategy Logic --- #
            side = None
            if prob_up >= float(self._buy_threshold):
                side = "BUY"
            elif prob_down >= float(self._buy_threshold): # Using same threshold for down
                 side = "SELL"

            if side:
                signal_id = uuid.uuid4()
                # For MVP, assume MARKET order, SL/TP defined by % but calculated later
                
                proposed_event = TradeSignalProposedEvent(
                    source_module=self._source_module,
                    event_id=uuid.uuid4(), # ID for this *proposed* event
                    timestamp=datetime.utcnow(),
                    signal_id=signal_id, # ID linking proposal->approval->execution
                    trading_pair=trading_pair,
                    exchange=prediction_event.exchange,
                    side=side,
                    entry_type=self._entry_type,
                    proposed_sl_price=Decimal("0"), # Placeholder - Must be calculated later!
                    proposed_tp_price=Decimal("0"), # Placeholder - Must be calculated later!
                    strategy_id=self._strategy_id,
                    proposed_entry_price=None,
                    triggering_prediction_event_id=prediction_event.event_id
                )
                
                self.logger.info(
                    f"Generated {side} signal proposal ({signal_id}) for {trading_pair}"
                )
                return proposed_event

        except (InvalidOperation, TypeError, AttributeError) as e:
            self.logger.error(
                f"Error evaluating strategy from prediction {prediction_event}: {e}",
                exc_info=True,
            )

        return None

    async def start(self) -> None:
        """Starts listening for prediction events."""
        if self._is_running:
            self.logger.warning(
                "StrategyArbitrator already running.",
                source_module=self.__class__.__name__,
            )
            return
        self._is_running = True
        
        # Subscribe to PredictionEvent
        self.pubsub.subscribe(EventType.PREDICTION_GENERATED, self._prediction_handler)
        
        self.logger.info(
            "StrategyArbitrator started.", source_module=self.__class__.__name__
        )

    async def stop(self) -> None:
        """Stops the event processing loop."""
        if not self._is_running:
            return
        self._is_running = False
        
        # Unsubscribe
        try:
            self.pubsub.unsubscribe(EventType.PREDICTION_GENERATED, self._prediction_handler)
            self.logger.info("Unsubscribed from PREDICTION_GENERATED.", source_module=self._source_module)
        except Exception as e:
             self.logger.error(f"Error unsubscribing StrategyArbitrator: {e}", exc_info=True, source_module=self._source_module)
             
        self.logger.info(
            "StrategyArbitrator stopped.", source_module=self.__class__.__name__
        )

    async def handle_prediction_event(self, event: PredictionEvent) -> None:
        """Handles incoming prediction events directly."""
        if not isinstance(event, PredictionEvent):
            self.logger.warning(f"Received non-PredictionEvent: {type(event)}", source_module=self._source_module)
            return

        if not self._is_running:
            return # Don't process if stopped

        # Evaluate strategy based on the prediction event
        proposed_signal_event = self._evaluate_strategy(event)

        # Publish the proposed signal event if generated
        if proposed_signal_event:
            await self._publish_trade_signal_proposed(proposed_signal_event)

    async def _publish_trade_signal_proposed(self, event: TradeSignalProposedEvent) -> None:
        """Publishes the TradeSignalProposedEvent."""
        try:
            await self.pubsub.publish(event)
            self.logger.debug(
                f"Published TradeSignalProposedEvent: {event.signal_id}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to publish TradeSignalProposedEvent {event.signal_id}: {e}",
                exc_info=True,
            )
