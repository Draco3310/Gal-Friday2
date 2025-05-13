"""Core event definitions for the Gal-Friday trading system.

This module defines the complete event hierarchy used for communication between
system components, including market data events, trading signals, and execution reports.
All events are implemented as immutable dataclasses with comprehensive validation.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

"""
Core Event Definitions for Gal-Friday

Design Notes
 -----
- Events are implemented as frozen dataclasses for immutability.
- Financial values (prices, quantities, commissions) in trade-related events
  (signals, execution reports) use decimal.Decimal for precision.
- Market data events (L2, OHLCV) use strings for numeric fields to accurately
  represent the raw data format received from exchanges, which may be needed
  for checksums or initial parsing consistency. Downstream modules are
  responsible for converting these strings to numeric types (Decimal/float)
  as needed for calculations.
- Timestamps should generally be UTC. Exchange timestamps are included where available.
- UUIDs are used for unique event identification and signal correlation.
- Factory methods (.create()) are provided for consistent event creation with validation.
"""


class EventType(Enum):
    """Enumeration of possible event types within the system."""

    # Data Flow Events
    MARKET_DATA_L2 = auto()  # L2 Order Book Update (bids/asks)
    MARKET_DATA_OHLCV = auto()  # OHLCV Bar Update
    FEATURES_CALCULATED = auto()  # New features calculated by FeatureEngine
    PREDICTION_GENERATED = auto()  # New prediction from PredictionService
    TRADE_SIGNAL_PROPOSED = auto()  # Proposed trade signal from StrategyArbitrator
    TRADE_SIGNAL_APPROVED = auto()  # Approved trade signal from RiskManager
    TRADE_SIGNAL_REJECTED = auto()  # Rejected trade signal from RiskManager
    EXECUTION_REPORT = auto()  # Report from ExecutionHandler (fill, error, etc.)

    # System & Operational Events
    SYSTEM_STATE_CHANGE = auto()  # Change in global system state (HALTED, RUNNING)
    LOG_ENTRY = auto()  # Log message event (for potential event-based logging)
    # Potential Halt signal (can be published by multiple modules)
    POTENTIAL_HALT_TRIGGER = auto()


# --- Base Event ---


@dataclass(frozen=True)
class Event:
    """Base class for all events, containing common metadata.

    NOTE: Defaults removed from base class to resolve mypy field ordering issues.
    Base fields (source_module, event_id, timestamp) must now be provided
    during instantiation of specific event subclasses.
    """

    source_module: str
    event_id: uuid.UUID
    timestamp: datetime


# --- Specific Event Definitions ---


@dataclass(frozen=True)
class SystemStateEvent(Event):
    """Event representing a change in the global system state."""

    new_state: str
    reason: str
    halt_action: Optional[str] = None  # e.g., "LIQUIDATE_POSITIONS"
    event_type: EventType = field(default=EventType.SYSTEM_STATE_CHANGE, init=False)


@dataclass(frozen=True)
class PotentialHaltTriggerEvent(Event):
    """Event indicating a condition that might warrant a system HALT."""

    reason: str
    event_type: EventType = field(default=EventType.POTENTIAL_HALT_TRIGGER, init=False)


@dataclass(frozen=True)
class MarketDataL2Event(Event):
    """Event carrying L2 order book updates."""

    trading_pair: str
    exchange: str
    # Use Decimal for price/volume internally if possible, converting from str early.
    # Keep as str if exact representation from exchange is critical for
    # checksums etc.
    bids: List[Tuple[str, str]]  # [[price_str, volume_str], ...]
    asks: List[Tuple[str, str]]  # [[price_str, volume_str], ...]
    is_snapshot: bool
    timestamp_exchange: Optional[datetime] = None
    event_type: EventType = field(default=EventType.MARKET_DATA_L2, init=False)


@dataclass(frozen=True)
class MarketDataOHLCVEvent(Event):
    """Event carrying OHLCV bar updates."""

    trading_pair: str
    exchange: str
    interval: str  # e.g., "1m", "5m"
    timestamp_bar_start: datetime
    open: str  # Using string representation from inter_module_comm doc
    high: str
    low: str
    close: str
    volume: str
    event_type: EventType = field(default=EventType.MARKET_DATA_OHLCV, init=False)


@dataclass(frozen=True)
class FeatureEvent(Event):
    """Event carrying calculated features."""

    trading_pair: str
    exchange: str
    timestamp_features_for: datetime
    # Feature values (consider specific types if known)
    features: Dict[str, Any]
    event_type: EventType = field(default=EventType.FEATURES_CALCULATED, init=False)


@dataclass(frozen=True)
class PredictionEvent(Event):
    """Event carrying model predictions."""

    trading_pair: str
    exchange: str
    timestamp_prediction_for: datetime
    model_id: str
    prediction_target: str  # e.g., "prob_price_up_0.1pct_5min"
    # Use float or Decimal for probability/value
    prediction_value: float
    confidence: Optional[float] = None
    associated_features: Optional[Dict[str, Any]] = None
    event_type: EventType = field(default=EventType.PREDICTION_GENERATED, init=False)


@dataclass(frozen=True)
class TradeSignalProposedEvent(Event):
    """Event carrying a proposed trade signal from the strategy."""

    signal_id: uuid.UUID  # Unique ID for this proposal
    trading_pair: str
    exchange: str
    side: str  # "BUY" or "SELL"
    entry_type: str  # "LIMIT" or "MARKET"
    # Use Decimal for prices
    proposed_sl_price: Decimal
    proposed_tp_price: Decimal
    strategy_id: str
    proposed_entry_price: Optional[Decimal] = None
    triggering_prediction_event_id: Optional[uuid.UUID] = None
    triggering_prediction: Optional[Dict[str, Any]] = None  # Added full prediction data
    event_type: EventType = field(default=EventType.TRADE_SIGNAL_PROPOSED, init=False)

    @classmethod
    def _validate_input(
        cls,
        side: str,
        entry_type: str,
        proposed_sl_price: Decimal,
        proposed_tp_price: Decimal,
        proposed_entry_price: Optional[Decimal] = None,
    ) -> None:
        """Validate inputs for trade signal proposal."""
        # Validate basic parameters
        cls._validate_basic_params(side, entry_type, proposed_entry_price)

        # Validate price values
        cls._validate_price_values(proposed_sl_price, proposed_tp_price, proposed_entry_price)

        # Validate SL/TP positions if entry price is provided
        if proposed_entry_price is not None:
            cls._validate_sl_tp_positions(
                side, proposed_sl_price, proposed_tp_price, proposed_entry_price
            )

    @staticmethod
    def _validate_basic_params(
        side: str, entry_type: str, proposed_entry_price: Optional[Decimal]
    ) -> None:
        """Validate basic parameters for trade signal."""
        if side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")

        if entry_type not in ["LIMIT", "MARKET"]:
            raise ValueError(f"Invalid entry_type: {entry_type}. Must be 'LIMIT' or 'MARKET'.")

        if entry_type == "LIMIT" and proposed_entry_price is None:
            raise ValueError("proposed_entry_price must be provided for LIMIT entry type.")

    @staticmethod
    def _validate_price_values(
        sl_price: Decimal, tp_price: Decimal, entry_price: Optional[Decimal]
    ) -> None:
        """Validate that price values are positive."""
        if sl_price <= Decimal(0):
            raise ValueError(f"Stop loss price must be positive: {sl_price}")

        if tp_price <= Decimal(0):
            raise ValueError(f"Take profit price must be positive: {tp_price}")

        if entry_price is not None and entry_price <= Decimal(0):
            raise ValueError(f"Entry price must be positive: {entry_price}")

    @staticmethod
    def _validate_sl_tp_positions(
        side: str, sl_price: Decimal, tp_price: Decimal, entry_price: Decimal
    ) -> None:
        """Validate stop loss and take profit positions relative to entry price and side."""
        if side == "BUY":
            if sl_price >= entry_price:
                raise ValueError(
                    f"For BUY orders, stop loss price ({sl_price}) "
                    f"must be below entry price ({entry_price})"
                )

            if tp_price <= entry_price:
                raise ValueError(
                    f"For BUY orders, take profit price ({tp_price}) "
                    f"must be above entry price ({entry_price})"
                )
        else:  # side == "SELL"
            if sl_price <= entry_price:
                raise ValueError(
                    f"For SELL orders, stop loss price ({sl_price}) "
                    f"must be above entry price ({entry_price})"
                )

            if tp_price >= entry_price:
                raise ValueError(
                    f"For SELL orders, take profit price ({tp_price}) "
                    f"must be below entry price ({entry_price})"
                )

    @classmethod
    def create(
        cls,
        source_module: str,
        trading_pair: str,
        exchange: str,
        side: str,
        entry_type: str,
        proposed_sl_price: Decimal,
        proposed_tp_price: Decimal,
        strategy_id: str,
        proposed_entry_price: Optional[Decimal] = None,
        triggering_prediction_event_id: Optional[uuid.UUID] = None,
        triggering_prediction: Optional[Dict[str, Any]] = None,
    ) -> "TradeSignalProposedEvent":
        """Create a validated TradeSignalProposedEvent instance.

        Args
        ----
            source_module: The module creating this event
            trading_pair: Symbol pair to trade (e.g., "BTC/USDT")
            exchange: Exchange to execute on (e.g., "kraken")
            side: Trade direction, must be "BUY" or "SELL"
            entry_type: Entry type, must be "LIMIT" or "MARKET"
            proposed_sl_price: Proposed stop-loss price
            proposed_tp_price: Proposed take-profit price
            strategy_id: Identifier for the strategy proposing this signal
            proposed_entry_price: Proposed entry price (required for LIMIT orders)
            triggering_prediction_event_id: UUID of the prediction event that triggered this signal
            triggering_prediction: Full data of the prediction that triggered this signal

        Returns
        -------
            A validated TradeSignalProposedEvent instance

        Raises
        ------
            ValueError: If any validation check fails
        """
        # Validate inputs
        cls._validate_input(
            side=side,
            entry_type=entry_type,
            proposed_sl_price=proposed_sl_price,
            proposed_tp_price=proposed_tp_price,
            proposed_entry_price=proposed_entry_price,
        )

        # Create and return instance
        return cls(
            source_module=source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            signal_id=uuid.uuid4(),  # Generate a new UUID for this signal
            trading_pair=trading_pair,
            exchange=exchange,
            side=side,
            entry_type=entry_type,
            proposed_sl_price=proposed_sl_price,
            proposed_tp_price=proposed_tp_price,
            strategy_id=strategy_id,
            proposed_entry_price=proposed_entry_price,
            triggering_prediction_event_id=triggering_prediction_event_id,
            triggering_prediction=triggering_prediction,
        )


@dataclass(frozen=True)
class TradeSignalApprovedEvent(Event):
    """Event carrying an approved trade signal from the risk manager."""

    signal_id: uuid.UUID  # Corresponds to proposed event
    trading_pair: str
    exchange: str
    side: str
    order_type: str  # "LIMIT" or "MARKET"
    # Use Decimal for quantity/prices
    quantity: Decimal
    sl_price: Decimal
    tp_price: Decimal
    risk_parameters: Dict[str, Any]  # Parameters used by RiskManager for approval
    limit_price: Optional[Decimal] = None
    event_type: EventType = field(default=EventType.TRADE_SIGNAL_APPROVED, init=False)

    @classmethod
    def _validate_input(
        cls,
        side: str,
        order_type: str,
        quantity: Decimal,
        sl_price: Decimal,
        tp_price: Decimal,
        limit_price: Optional[Decimal] = None,
    ) -> None:
        """Validate inputs for trade signal approval."""
        # Validate basic parameters
        cls._validate_basic_params(side, order_type, limit_price)

        # Validate prices and quantity
        cls._validate_prices_and_quantity(quantity, sl_price, tp_price, limit_price)

        # Validate SL/TP positions if limit price is provided
        if limit_price is not None:
            cls._validate_sl_tp_positions(side, sl_price, tp_price, limit_price)

    @staticmethod
    def _validate_basic_params(side: str, order_type: str, limit_price: Optional[Decimal]) -> None:
        """Validate basic parameters for trade signal."""
        if side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")

        if order_type not in ["LIMIT", "MARKET"]:
            raise ValueError(f"Invalid order_type: {order_type}. Must be 'LIMIT' or 'MARKET'.")

        if order_type == "LIMIT" and limit_price is None:
            raise ValueError("limit_price must be provided for LIMIT orders.")

    @staticmethod
    def _validate_prices_and_quantity(
        quantity: Decimal, sl_price: Decimal, tp_price: Decimal, limit_price: Optional[Decimal]
    ) -> None:
        """Validate that prices and quantity are positive."""
        if quantity <= Decimal(0):
            raise ValueError(f"Quantity must be positive: {quantity}")

        if sl_price <= Decimal(0):
            raise ValueError(f"Stop loss price must be positive: {sl_price}")

        if tp_price <= Decimal(0):
            raise ValueError(f"Take profit price must be positive: {tp_price}")

        if limit_price is not None and limit_price <= Decimal(0):
            raise ValueError(f"Limit price must be positive: {limit_price}")

    @staticmethod
    def _validate_sl_tp_positions(
        side: str, sl_price: Decimal, tp_price: Decimal, limit_price: Decimal
    ) -> None:
        """Validate stop loss and take profit positions relative to entry price and side."""
        if side == "BUY":
            if sl_price >= limit_price:
                raise ValueError(
                    f"For BUY orders, stop loss price ({sl_price}) "
                    f"must be below entry price ({limit_price})"
                )

            if tp_price <= limit_price:
                raise ValueError(
                    f"For BUY orders, take profit price ({tp_price}) "
                    f"must be above entry price ({limit_price})"
                )
        else:  # side == "SELL"
            if sl_price <= limit_price:
                raise ValueError(
                    f"For SELL orders, stop loss price ({sl_price}) "
                    f"must be above entry price ({limit_price})"
                )

            if tp_price >= limit_price:
                raise ValueError(
                    f"For SELL orders, take profit price ({tp_price}) "
                    f"must be below entry price ({limit_price})"
                )

    @classmethod
    def create(
        cls,
        source_module: str,
        signal_id: uuid.UUID,
        trading_pair: str,
        exchange: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        sl_price: Decimal,
        tp_price: Decimal,
        risk_parameters: Dict[str, Any],
        limit_price: Optional[Decimal] = None,
    ) -> "TradeSignalApprovedEvent":
        """Create a validated TradeSignalApprovedEvent instance.

        Args
        ----
            source_module: The module creating this event
            signal_id: UUID of the original trade signal proposal
            trading_pair: Symbol pair to trade (e.g., "BTC/USDT")
            exchange: Exchange to execute on (e.g., "kraken")
            side: Trade direction, must be "BUY" or "SELL"
            order_type: Order type, must be "LIMIT" or "MARKET"
            quantity: Amount to trade
            sl_price: Stop-loss price
            tp_price: Take-profit price
            risk_parameters: Risk parameters used by RiskManager for approval
            limit_price: Limit price (required for LIMIT orders)

        Returns
        -------
            A validated TradeSignalApprovedEvent instance

        Raises
        ------
            ValueError: If any validation check fails
        """
        # Validate inputs
        cls._validate_input(
            side=side,
            order_type=order_type,
            quantity=quantity,
            sl_price=sl_price,
            tp_price=tp_price,
            limit_price=limit_price,
        )

        # Create and return instance
        return cls(
            source_module=source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            signal_id=signal_id,
            trading_pair=trading_pair,
            exchange=exchange,
            side=side,
            order_type=order_type,
            quantity=quantity,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_parameters=risk_parameters,
            limit_price=limit_price,
        )


@dataclass(frozen=True)
class TradeSignalRejectedEvent(Event):
    """Event carrying a rejected trade signal from the risk manager."""

    signal_id: uuid.UUID  # Corresponds to proposed event
    trading_pair: str
    exchange: str
    side: str
    reason: str
    event_type: EventType = field(default=EventType.TRADE_SIGNAL_REJECTED, init=False)

    @classmethod
    def create(
        cls,
        source_module: str,
        signal_id: uuid.UUID,
        trading_pair: str,
        exchange: str,
        side: str,
        reason: str,
    ) -> "TradeSignalRejectedEvent":
        """Create a TradeSignalRejectedEvent instance.

        Args
        ----
            source_module: The module creating this event
            signal_id: UUID of the rejected trade signal proposal
            trading_pair: Symbol pair of the rejected trade
            exchange: Exchange of the rejected trade
            side: Trade direction that was rejected
            reason: Reason for rejection

        Returns
        -------
            A TradeSignalRejectedEvent instance
        """
        return cls(
            source_module=source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            signal_id=signal_id,
            trading_pair=trading_pair,
            exchange=exchange,
            side=side,
            reason=reason,
        )


@dataclass(frozen=True)
class ExecutionReportEvent(Event):
    """Event carrying updates on order execution from the exchange."""

    exchange_order_id: str
    trading_pair: str
    exchange: str
    order_status: str  # e.g., "NEW", "FILLED", "CANCELED", "REJECTED"
    order_type: str  # e.g., "LIMIT", "MARKET"
    side: str
    # Use Decimal for quantity/prices/commission
    quantity_ordered: Decimal
    signal_id: Optional[uuid.UUID] = None  # Originating signal ID
    client_order_id: Optional[str] = None  # Internal ID if used
    quantity_filled: Decimal = Decimal(0)
    average_fill_price: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    commission_asset: Optional[str] = None
    timestamp_exchange: Optional[datetime] = None
    error_message: Optional[str] = None
    event_type: EventType = field(default=EventType.EXECUTION_REPORT, init=False)

    @classmethod
    def _validate_input(
        cls,
        order_status: str,
        order_type: str,
        side: str,
        quantity_ordered: Decimal,
        quantity_filled: Decimal,
        average_fill_price: Optional[Decimal],
        limit_price: Optional[Decimal],
        commission: Optional[Decimal],
        commission_asset: Optional[str],
    ) -> None:
        """Validate execution report inputs."""
        # Validate order properties
        cls._validate_order_properties(order_status, order_type, side)

        # Validate quantities
        cls._validate_quantities(quantity_ordered, quantity_filled)

        # Validate prices
        cls._validate_prices(order_type, order_status, limit_price, average_fill_price)

        # Validate commission
        cls._validate_commission(commission, commission_asset)

    @staticmethod
    def _validate_order_properties(order_status: str, order_type: str, side: str) -> None:
        """Validate order status, type, and side."""
        valid_statuses = ["NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "REJECTED", "EXPIRED"]
        if order_status not in valid_statuses:
            raise ValueError(
                f"Invalid order_status: {order_status}. Must be one of {valid_statuses}"
            )

        valid_order_types = ["LIMIT", "MARKET", "STOP", "STOP_LIMIT"]
        if order_type not in valid_order_types:
            raise ValueError(
                f"Invalid order_type: {order_type}. Must be one of {valid_order_types}"
            )

        if side not in ["BUY", "SELL"]:
            raise ValueError(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")

    @staticmethod
    def _validate_quantities(quantity_ordered: Decimal, quantity_filled: Decimal) -> None:
        """Validate order quantities."""
        if quantity_ordered <= Decimal(0):
            raise ValueError(f"Quantity ordered must be positive: {quantity_ordered}")

        if quantity_filled < Decimal(0):
            raise ValueError(f"Quantity filled cannot be negative: {quantity_filled}")

        if quantity_filled > quantity_ordered:
            raise ValueError(
                f"Quantity filled ({quantity_filled}) cannot exceed "
                f"quantity ordered ({quantity_ordered})"
            )

    @staticmethod
    def _validate_prices(
        order_type: str,
        order_status: str,
        limit_price: Optional[Decimal],
        average_fill_price: Optional[Decimal],
    ) -> None:
        """Validate price values."""
        if order_type == "LIMIT" and limit_price is None:
            raise ValueError("limit_price must be provided for LIMIT orders")

        if limit_price is not None and limit_price <= Decimal(0):
            raise ValueError(f"Limit price must be positive: {limit_price}")

        if average_fill_price is not None and average_fill_price <= Decimal(0):
            raise ValueError(f"Average fill price must be positive: {average_fill_price}")

        filled_statuses = ["FILLED", "PARTIALLY_FILLED"]
        if order_status in filled_statuses and average_fill_price is None:
            raise ValueError(
                "average_fill_price must be provided for filled or partially filled orders"
            )

    @staticmethod
    def _validate_commission(
        commission: Optional[Decimal], commission_asset: Optional[str]
    ) -> None:
        """Validate commission details."""
        if commission is not None:
            if commission < Decimal(0):
                raise ValueError(f"Commission cannot be negative: {commission}")
            if commission_asset is None:
                raise ValueError("commission_asset must be provided when commission is specified")

    @classmethod
    def create(
        cls,
        source_module: str,
        exchange_order_id: str,
        trading_pair: str,
        exchange: str,
        order_status: str,
        order_type: str,
        side: str,
        quantity_ordered: Decimal,
        signal_id: Optional[uuid.UUID] = None,
        client_order_id: Optional[str] = None,
        quantity_filled: Decimal = Decimal(0),
        average_fill_price: Optional[Decimal] = None,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        commission: Optional[Decimal] = None,
        commission_asset: Optional[str] = None,
        timestamp_exchange: Optional[datetime] = None,
        error_message: Optional[str] = None,
    ) -> "ExecutionReportEvent":
        """Create a validated ExecutionReportEvent instance.

        Args
        ----
            source_module: The module creating this event
            exchange_order_id: Order ID assigned by the exchange
            trading_pair: Symbol pair for the trade (e.g., "BTC/USDT")
            exchange: Exchange where the order was executed
            order_status: Current status of the order (e.g., "NEW", "FILLED")
            order_type: Type of order (e.g., "LIMIT", "MARKET")
            side: Order side, must be "BUY" or "SELL"
            quantity_ordered: Total quantity ordered
            signal_id: UUID of the original trade signal (if applicable)
            client_order_id: Internal order ID (if used)
            quantity_filled: Quantity that has been executed so far
            average_fill_price: Average execution price for filled portion
            limit_price: Limit price (for LIMIT orders)
            stop_price: Stop price (for stop orders)
            commission: Commission charged by exchange
            commission_asset: Asset in which commission was charged
            timestamp_exchange: Timestamp reported by exchange
            error_message: Error message (if rejected/failed)

        Returns
        -------
            A validated ExecutionReportEvent instance

        Raises
        ------
            ValueError: If any validation check fails
        """
        # Validate inputs
        cls._validate_input(
            order_status=order_status,
            order_type=order_type,
            side=side,
            quantity_ordered=quantity_ordered,
            quantity_filled=quantity_filled,
            average_fill_price=average_fill_price,
            limit_price=limit_price,
            commission=commission,
            commission_asset=commission_asset,
        )

        # Create and return instance
        return cls(
            source_module=source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            exchange_order_id=exchange_order_id,
            trading_pair=trading_pair,
            exchange=exchange,
            order_status=order_status,
            order_type=order_type,
            side=side,
            quantity_ordered=quantity_ordered,
            signal_id=signal_id,
            client_order_id=client_order_id,
            quantity_filled=quantity_filled,
            average_fill_price=average_fill_price,
            limit_price=limit_price,
            stop_price=stop_price,
            commission=commission,
            commission_asset=commission_asset,
            timestamp_exchange=timestamp_exchange,
            error_message=error_message,
        )


@dataclass(frozen=True)
class LogEvent(Event):
    """Event carrying structured log information."""

    level: str  # e.g., "INFO", "ERROR"
    message: str
    context: Optional[Dict[str, Any]] = None
    event_type: EventType = field(default=EventType.LOG_ENTRY, init=False)


# Cleanup placeholder comment
# Removed the TODO section as definitions are now added.
# Removed the example creation comment.
