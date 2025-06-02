"""Core event definitions for the Gal-Friday trading system.

This module defines the complete event hierarchy used for communication between
system components, including market data events, trading signals, and execution reports.
All events are implemented as immutable dataclasses with comprehensive validation.
"""

import contextlib  # For SIM105
import datetime as dt  # For DTZ003
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Any

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


# --- Custom Exceptions ---


class ValidationError(ValueError):
    """Base class for all event validation errors."""


class CommissionAssetMissingError(ValidationError):
    """Error raised when commission is specified but commission_asset is missing."""

    def __init__(self) -> None:
        """Initialize CommissionAssetMissingError."""
        super().__init__("Commission asset must be provided when commission is specified")


class NegativeCommissionError(ValidationError):
    """Error raised when commission is negative."""

    def __init__(self, commission: Decimal) -> None:
        """Initialize NegativeCommissionError."""
        super().__init__(f"Commission cannot be negative: {commission}")


class MissingAverageFillPriceError(ValidationError):
    """Error raised when average_fill_price is missing for filled or partially filled orders."""

    def __init__(self) -> None:
        """Initialize MissingAverageFillPriceError."""
        super().__init__(
            "Average fill price must be provided for filled or partially filled orders",
        )


class MissingLimitPriceError(ValidationError):
    """Error raised when limit_price is missing for LIMIT orders."""

    def __init__(self) -> None:
        """Initialize MissingLimitPriceError."""
        super().__init__("Limit price must be provided for LIMIT orders")


class NonPositiveLimitPriceError(ValidationError):
    """Error raised when limit_price is not positive."""

    def __init__(self, limit_price: Decimal) -> None:
        """Initialize NonPositiveLimitPriceError."""
        super().__init__(f"Limit price must be positive: {limit_price}")


class NonPositiveAverageFillPriceError(ValidationError):
    """Error raised when average_fill_price is not positive."""

    def __init__(self, average_fill_price: Decimal) -> None:
        """Initialize NonPositiveAverageFillPriceError."""
        super().__init__(f"Average fill price must be positive: {average_fill_price}")


class QuantityFilledExceedsOrderedError(ValidationError):
    """Error raised when quantity_filled exceeds quantity_ordered."""

    def __init__(self, quantity_filled: Decimal, quantity_ordered: Decimal) -> None:
        """Initialize QuantityFilledExceedsOrderedError."""
        message = (
            f"Quantity filled ({quantity_filled}) cannot exceed "
            f"quantity ordered ({quantity_ordered})"
        )
        super().__init__(message)


class NonPositiveQuantityOrderedError(ValidationError):
    """Error raised when quantity_ordered is not positive."""

    def __init__(self, quantity_ordered: Decimal) -> None:
        """Initialize NonPositiveQuantityOrderedError."""
        super().__init__(f"Quantity ordered must be positive: {quantity_ordered}")


class NegativeQuantityFilledError(ValidationError):
    """Error raised when quantity_filled is negative."""

    def __init__(self, quantity_filled: Decimal) -> None:
        """Initialize NegativeQuantityFilledError."""
        super().__init__(f"Quantity filled cannot be negative: {quantity_filled}")


class InvalidOrderStatusError(ValidationError):
    """Error raised when order_status is invalid."""

    def __init__(self, order_status: str, valid_statuses: list[str]) -> None:
        """Initialize InvalidOrderStatusError."""
        super().__init__(f"Invalid order_status: {order_status}. Must be one of {valid_statuses}")


class InvalidOrderTypeError(ValidationError):
    """Error raised when order_type is invalid."""

    def __init__(self, order_type: str, valid_order_types: list[str]) -> None:
        """Initialize InvalidOrderTypeError."""
        super().__init__(f"Invalid order_type: {order_type}. Must be one of {valid_order_types}")


class InvalidSideError(ValidationError):
    """Error raised when side is invalid."""

    def __init__(self, side: str) -> None:
        """Initialize InvalidSideError."""
        super().__init__(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")


class SellTakeProfitNotBelowEntryError(ValidationError):
    """Error raised for SELL orders when take profit is not below entry price."""

    def __init__(self, tp_price: Decimal, limit_price: Decimal) -> None:
        """Initialize SellTakeProfitNotBelowEntryError."""
        message = (
            f"For SELL orders, take profit price ({tp_price}) must be below "
            f"entry price ({limit_price})"
        )
        super().__init__(message)


class SellStopLossNotAboveEntryError(ValidationError):
    """Error raised for SELL orders when stop loss is not above entry price."""

    def __init__(self, sl_price: Decimal, limit_price: Decimal) -> None:
        """Initialize SellStopLossNotAboveEntryError."""
        message = (
            f"For SELL orders, stop loss price ({sl_price}) must be above "
            f"entry price ({limit_price})"
        )
        super().__init__(message)


class InvalidTradeSignalSideError(ValidationError):
    """Error raised for invalid side in a trade signal."""

    def __init__(self, side: str) -> None:
        """Initialize InvalidTradeSignalSideError."""
        super().__init__(f"Invalid side: {side}. Must be 'BUY' or 'SELL'.")


class InvalidTradeSignalEntryTypeError(ValidationError):
    """Error raised for invalid entry_type in a trade signal."""

    def __init__(self, entry_type: str) -> None:
        """Initialize InvalidTradeSignalEntryTypeError."""
        super().__init__(f"Invalid entry_type: {entry_type}. Must be 'LIMIT' or 'MARKET'.")


class MissingProposedEntryPriceError(ValidationError):
    """Error raised when proposed_entry_price is missing for a LIMIT order."""

    def __init__(self) -> None:
        """Initialize MissingProposedEntryPriceError."""
        super().__init__("proposed_entry_price must be provided for LIMIT entry type.")


class NonPositiveStopLossPriceError(ValidationError):
    """Error raised when a stop loss price is not positive."""

    def __init__(self, sl_price: Decimal) -> None:
        """Initialize NonPositiveStopLossPriceError."""
        super().__init__(f"Stop loss price must be positive: {sl_price}")


class NonPositiveTakeProfitPriceError(ValidationError):
    """Error raised when a take profit price is not positive."""

    def __init__(self, tp_price: Decimal) -> None:
        """Initialize NonPositiveTakeProfitPriceError."""
        super().__init__(f"Take profit price must be positive: {tp_price}")


class NonPositiveProposedEntryPriceError(ValidationError):
    """Error raised when a proposed entry price is not positive."""

    def __init__(self, entry_price: Decimal) -> None:
        """Initialize NonPositiveProposedEntryPriceError."""
        super().__init__(f"Entry price must be positive: {entry_price}")


class BuyStopLossNotBelowEntryError(ValidationError):
    """Error for BUY orders when stop loss is not below entry price."""

    def __init__(self, sl_price: Decimal, entry_price: Decimal) -> None:
        """Initialize BuyStopLossNotBelowEntryError."""
        message = (
            f"For BUY orders, stop loss price ({sl_price}) must be below "
            f"entry price ({entry_price})"
        )
        super().__init__(message)


class BuyTakeProfitNotAboveEntryError(ValidationError):
    """Error for BUY orders when take profit is not above entry price."""

    def __init__(self, tp_price: Decimal, entry_price: Decimal) -> None:
        """Initialize BuyTakeProfitNotAboveEntryError."""
        message = (
            f"For BUY orders, take profit price ({tp_price}) must be above "
            f"entry price ({entry_price})"
        )
        super().__init__(message)


class SellStopLossNotAboveProposedEntryError(ValidationError):
    """Error for SELL orders when stop loss is not above proposed entry price."""

    def __init__(self, sl_price: Decimal, entry_price: Decimal) -> None:
        """Initialize SellStopLossNotAboveProposedEntryError."""
        message = (
            f"For SELL orders, stop loss price ({sl_price}) must be above "
            f"proposed entry price ({entry_price})"
        )
        super().__init__(message)


class SellTakeProfitNotBelowProposedEntryError(ValidationError):
    """Error for SELL orders when take profit is not below proposed entry price."""

    def __init__(self, tp_price: Decimal, entry_price: Decimal) -> None:
        """Initialize SellTakeProfitNotBelowProposedEntryError."""
        message = (
            f"For SELL orders, take profit price ({tp_price}) must be below "
            f"proposed entry price ({entry_price})"
        )
        super().__init__(message)


class NonPositiveApprovedQuantityError(ValidationError):
    """Error raised when quantity is not positive for an approved signal."""

    def __init__(self, quantity: Decimal) -> None:
        """Initialize NonPositiveApprovedQuantityError."""
        super().__init__(f"Quantity must be positive: {quantity}")


class NonPositiveApprovedStopLossPriceError(ValidationError):
    """Error raised when stop loss price is not positive for an approved signal."""

    def __init__(self, sl_price: Decimal) -> None:
        """Initialize NonPositiveApprovedStopLossPriceError."""
        super().__init__(f"Stop loss price must be positive: {sl_price}")


class NonPositiveApprovedTakeProfitPriceError(ValidationError):
    """Error raised when take profit price is not positive for an approved signal."""

    def __init__(self, tp_price: Decimal) -> None:
        """Initialize NonPositiveApprovedTakeProfitPriceError."""
        super().__init__(f"Take profit price must be positive: {tp_price}")


class BuyStopLossNotBelowApprovedLimitError(ValidationError):
    """Error for BUY orders when stop loss is not below the approved limit price."""

    def __init__(self, sl_price: Decimal, limit_price: Decimal) -> None:
        """Initialize BuyStopLossNotBelowApprovedLimitError."""
        message = (
            f"For BUY orders, stop loss price ({sl_price}) must be below "
            f"entry price ({limit_price})"
        )
        super().__init__(message)


class BuyTakeProfitNotAboveApprovedLimitError(ValidationError):
    """Error for BUY orders when take profit is not above the approved limit price."""

    def __init__(self, tp_price: Decimal, limit_price: Decimal) -> None:
        """Initialize BuyTakeProfitNotAboveApprovedLimitError."""
        message = (
            f"For BUY orders, take profit price ({tp_price}) must be above "
            f"entry price ({limit_price})"
        )
        super().__init__(message)


class EventType(Enum):
    """Enumeration of possible event types within the system."""

    # Data Flow Events
    MARKET_DATA_L2 = auto()  # L2 Order Book Update (bids/asks)
    MARKET_DATA_OHLCV = auto()  # OHLCV Bar Update
    MARKET_DATA_TRADE = auto()  # Individual Trade Update (NEW)
    FEATURES_CALCULATED = auto()  # New features calculated by FeatureEngine
    PREDICTION_GENERATED = auto()  # New prediction from PredictionService
    TRADE_SIGNAL_PROPOSED = auto()  # Proposed trade signal from StrategyArbitrator
    TRADE_SIGNAL_APPROVED = auto()  # Approved trade signal from RiskManager
    TRADE_SIGNAL_REJECTED = auto()  # Rejected trade signal from RiskManager
    EXECUTION_REPORT = auto()  # Report from ExecutionHandler (fill, error, etc.)

    # System & Operational Events
    SYSTEM_STATE_CHANGE = auto()  # Change in global system state (HALTED, RUNNING)
    LOG_ENTRY = auto()  # Log message event (for potential event-based logging)
    POTENTIAL_HALT_TRIGGER = auto()  # Potential Halt signal
    PREDICTION_CONFIG_UPDATED = auto()  # Prediction service specific configuration updated
    SYSTEM_ERROR = auto()  # System error event (e.g. API errors)

    # Portfolio and Risk Events
    PORTFOLIO_UPDATE = auto()  # Portfolio state update
    PORTFOLIO_RECONCILIATION = auto()  # Portfolio reconciliation request/result
    PORTFOLIO_DISCREPANCY = auto()  # Identified discrepancy in portfolio state
    RISK_LIMIT_ALERT = auto()  # Risk limit breach alert

    # Additional Events
    MARKET_DATA_RAW = auto()  # Raw market data before processing
    FEATURE_CALCULATED = auto()  # Alias for FEATURES_CALCULATED (for backward compatibility)
    # Event carrying ticker/quote data with best bid/ask and 24h statistics (E501)
    MARKET_DATA_TICKER = auto()


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
    timestamp: dt.datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        from dataclasses import asdict

        # Convert dataclass to dict
        data = asdict(self)

        # Convert special types to strings
        if "event_id" in data:
            data["event_id"] = str(data["event_id"])
        if "timestamp" in data:
            data["timestamp"] = data["timestamp"].isoformat()
        if "event_type" in data:
            data["event_type"] = data["event_type"].value

        # Convert UUID fields
        for key, value in data.items():
            if isinstance(value, uuid.UUID):
                data[key] = str(value)
            elif isinstance(value, dt.datetime): # F821
                data[key] = value.isoformat()
            elif isinstance(value, Decimal):
                data[key] = str(value)
            elif isinstance(value, Enum):
                data[key] = value.value

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        # Convert string UUIDs back to UUID objects
        if "event_id" in data:
            data["event_id"] = uuid.UUID(data["event_id"])

        # Convert ISO strings back to datetime
        if "timestamp" in data:
            data["timestamp"] = dt.datetime.fromisoformat(data["timestamp"])

        # Handle specific field conversions based on event type
        for key, value in data.items():
            if key.endswith("_id") and isinstance(value, str) and value:
                with contextlib.suppress(ValueError): # SIM105
                    data[key] = uuid.UUID(value)
            elif "timestamp" in key and isinstance(value, str):
                with contextlib.suppress(ValueError): # SIM105
                    data[key] = dt.datetime.fromisoformat(value)
            # SIM102: Combined nested if
            elif key in ["price", "volume", "quantity", "sl_price", "tp_price",
                         "limit_price", "bid", "ask", "bid_size", "ask_size",
                         "last_price", "last_size", "volume_24h", "vwap_24h",
                         "high_24h", "low_24h", "quantity_ordered", "quantity_filled",
                         "average_fill_price", "stop_price", "commission",
                         "proposed_sl_price", "proposed_tp_price", "proposed_entry_price"] \
                 and value is not None and value != "None":
                data[key] = Decimal(str(value))

        # Remove event_type if present (it's set by the class)
        data.pop("event_type", None)

        return cls(**data)


# --- Specific Event Definitions ---


@dataclass(frozen=True)
class SystemStateEvent(Event):
    """Event representing a change in the global system state."""

    new_state: str
    reason: str
    halt_action: str | None = None  # e.g., "LIQUIDATE_POSITIONS"
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
    bids: Sequence[tuple[str, str]]  # [[price_str, volume_str], ...]
    asks: Sequence[tuple[str, str]]  # [[price_str, volume_str], ...]
    is_snapshot: bool
    timestamp_exchange: dt.datetime | None = None
    event_type: EventType = field(default=EventType.MARKET_DATA_L2, init=False)


@dataclass(frozen=True)
class MarketDataOHLCVEvent(Event):
    """Event carrying OHLCV bar updates."""

    trading_pair: str
    exchange: str
    interval: str  # e.g., "1m", "5m"
    timestamp_bar_start: dt.datetime
    open: str  # Using string representation from inter_module_comm doc
    high: str
    low: str
    close: str
    volume: str
    event_type: EventType = field(default=EventType.MARKET_DATA_OHLCV, init=False)


@dataclass(frozen=True)
class MarketDataTradeEvent(Event):
    """Event carrying individual executed trade data."""

    trading_pair: str
    exchange: str
    timestamp_exchange: dt.datetime  # Timestamp from the exchange for the trade
    price: Decimal
    volume: Decimal
    side: str  # Aggressor side: "buy" or "sell"
    trade_id: str | None = None  # Optional: Exchange-specific trade ID
    event_type: EventType = field(default=EventType.MARKET_DATA_TRADE, init=False)

    # Error messages
    _INVALID_SIDE_MSG = "Invalid trade side: '{side}'. Must be 'buy' or 'sell'."
    _NON_POSITIVE_PRICE_MSG = "Trade price must be positive: {price}"
    _NON_POSITIVE_VOLUME_MSG = "Trade volume must be positive: {volume}"

    def __post_init__(self) -> None:
        """Validate trade side after initialization."""
        if self.side.lower() not in ["buy", "sell"]:
            raise ValueError(self._INVALID_SIDE_MSG.format(side=self.side))
        if self.price <= Decimal("0"):
            raise ValueError(self._NON_POSITIVE_PRICE_MSG.format(price=self.price))
        if self.volume <= Decimal("0"):
            raise ValueError(self._NON_POSITIVE_VOLUME_MSG.format(volume=self.volume))


@dataclass(frozen=True)
class MarketDataTickerEvent(Event):
    """Event carrying ticker/quote data with best bid/ask and 24h statistics."""

    trading_pair: str
    exchange: str
    timestamp_exchange: dt.datetime  # Timestamp from the exchange
    bid: Decimal  # Best bid price
    bid_size: Decimal  # Best bid size
    ask: Decimal  # Best ask price
    ask_size: Decimal  # Best ask size
    last_price: Decimal  # Last traded price
    last_size: Decimal  # Last traded size
    volume_24h: Decimal  # 24-hour volume
    vwap_24h: Decimal  # 24-hour VWAP
    high_24h: Decimal  # 24-hour high
    low_24h: Decimal  # 24-hour low
    trades_24h: int  # Number of trades in 24h
    event_type: EventType = field(default=EventType.MARKET_DATA_TICKER, init=False)

    # Error messages
    _NON_POSITIVE_PRICE_MSG = "Price must be positive: {field}={price}"
    _NON_POSITIVE_SIZE_MSG = "Size must be positive: {field}={size}"
    _NEGATIVE_COUNT_MSG = "Trade count cannot be negative: {count}"

    def __post_init__(self) -> None:
        """Validate ticker data after initialization."""
        # Validate prices
        price_fields = [
            ("bid", self.bid),
            ("ask", self.ask),
            ("last_price", self.last_price),
            ("vwap_24h", self.vwap_24h),
            ("high_24h", self.high_24h),
            ("low_24h", self.low_24h),
        ]

        for field_name, price in price_fields:
            if price <= Decimal("0"):
                # E501
                error_msg = self._NON_POSITIVE_PRICE_MSG.format(field=field_name, price=price)
                raise ValueError(error_msg)

        # Validate sizes
        size_fields = [
            ("bid_size", self.bid_size),
            ("ask_size", self.ask_size),
            ("last_size", self.last_size),
            ("volume_24h", self.volume_24h),
        ]

        for field_name, size in size_fields:
            if size < Decimal("0"):
                raise ValueError(self._NON_POSITIVE_SIZE_MSG.format(field=field_name, size=size))

        # Validate trade count
        if self.trades_24h < 0:
            raise ValueError(self._NEGATIVE_COUNT_MSG.format(count=self.trades_24h))

        # Validate bid/ask spread
        if self.bid >= self.ask:
            error_msg = f"Invalid bid/ask spread: bid={self.bid} >= ask={self.ask}" # EM102, TRY003
            raise ValueError(error_msg)


@dataclass(frozen=True)
class FeatureEvent(Event):
    """Event carrying calculated features."""

    trading_pair: str
    exchange: str
    timestamp_features_for: dt.datetime
    # Feature values (consider specific types if known)
    features: dict
    event_type: EventType = field(default=EventType.FEATURES_CALCULATED, init=False)


@dataclass(frozen=True)
class PredictionEvent(Event):
    """Event carrying model predictions."""

    trading_pair: str
    exchange: str
    timestamp_prediction_for: dt.datetime
    model_id: str
    prediction_target: str  # e.g., "prob_price_up_0.1pct_5min"
    # Use float or Decimal for probability/value
    prediction_value: float
    confidence: float | None = None
    associated_features: dict | None = None
    event_type: EventType = field(default=EventType.PREDICTION_GENERATED, init=False)


@dataclass
class TradeSignalProposedParams:
    """Parameters for creating a TradeSignalProposedEvent."""

    source_module: str
    trading_pair: str
    exchange: str
    side: str
    entry_type: str
    proposed_sl_price: Decimal
    proposed_tp_price: Decimal
    strategy_id: str
    proposed_entry_price: Decimal | None = None
    triggering_prediction_event_id: uuid.UUID | None = None
    triggering_prediction: dict | None = None


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
    proposed_entry_price: Decimal | None = None
    triggering_prediction_event_id: uuid.UUID | None = None
    triggering_prediction: dict | None = None  # Added full prediction data
    event_type: EventType = field(default=EventType.TRADE_SIGNAL_PROPOSED, init=False)

    @classmethod
    def _validate_input(
        cls,
        side: str,
        entry_type: str,
        proposed_sl_price: Decimal,
        proposed_tp_price: Decimal,
        proposed_entry_price: Decimal | None = None,
    ) -> None:
        """Validate inputs for trade signal proposal."""
        # Validate basic parameters
        cls._validate_basic_params(side, entry_type, proposed_entry_price)

        # Validate price values
        cls._validate_price_values(proposed_sl_price, proposed_tp_price, proposed_entry_price)

        # Validate SL/TP positions if entry price is provided
        if proposed_entry_price is not None:
            cls._validate_sl_tp_positions(
                side,
                proposed_sl_price,
                proposed_tp_price,
                proposed_entry_price,
            )

    @staticmethod
    def _validate_basic_params(
        side: str,
        entry_type: str,
        proposed_entry_price: Decimal | None,
    ) -> None:
        """Validate basic parameters for trade signal."""
        if side not in ["BUY", "SELL"]:
            raise InvalidTradeSignalSideError(side)

        if entry_type not in ["LIMIT", "MARKET"]:
            raise InvalidTradeSignalEntryTypeError(entry_type)

        if entry_type == "LIMIT" and proposed_entry_price is None:
            raise MissingProposedEntryPriceError

    @staticmethod
    def _validate_price_values(
        sl_price: Decimal,
        tp_price: Decimal,
        entry_price: Decimal | None,
    ) -> None:
        """Validate that price values are positive."""
        if sl_price <= Decimal(0):
            raise NonPositiveStopLossPriceError(sl_price)

        if tp_price <= Decimal(0):
            raise NonPositiveTakeProfitPriceError(tp_price)

        if entry_price is not None and entry_price <= Decimal(0):
            raise NonPositiveProposedEntryPriceError(entry_price)

    @staticmethod
    def _validate_sl_tp_positions(
        side: str,
        sl_price: Decimal,
        tp_price: Decimal,
        entry_price: Decimal,
    ) -> None:
        """Validate stop loss and take profit positions relative to entry price and side."""
        if side == "BUY":
            if sl_price >= entry_price:
                raise BuyStopLossNotBelowEntryError(sl_price, entry_price)

            if tp_price <= entry_price:
                raise BuyTakeProfitNotAboveEntryError(tp_price, entry_price)
        else:  # side == "SELL"
            if sl_price <= entry_price:
                raise SellStopLossNotAboveProposedEntryError(sl_price, entry_price)

            if tp_price >= entry_price:
                raise SellTakeProfitNotBelowProposedEntryError(tp_price, entry_price)

    @classmethod
    def create(cls, params: TradeSignalProposedParams) -> "TradeSignalProposedEvent":
        """Create a validated TradeSignalProposedEvent instance."""
        # Validate inputs
        cls._validate_input(
            side=params.side,
            entry_type=params.entry_type,
            proposed_sl_price=params.proposed_sl_price,
            proposed_tp_price=params.proposed_tp_price,
            proposed_entry_price=params.proposed_entry_price,
        )

        # Create and return instance
        return cls(
            source_module=params.source_module,
            event_id=uuid.uuid4(),
            timestamp=dt.datetime.now(dt.UTC), # DTZ003
            signal_id=uuid.uuid4(),  # Generate a new UUID for this signal
            trading_pair=params.trading_pair,
            exchange=params.exchange,
            side=params.side,
            entry_type=params.entry_type,
            proposed_sl_price=params.proposed_sl_price,
            proposed_tp_price=params.proposed_tp_price,
            strategy_id=params.strategy_id,
            proposed_entry_price=params.proposed_entry_price,
            triggering_prediction_event_id=params.triggering_prediction_event_id,
            triggering_prediction=params.triggering_prediction,
        )


@dataclass
class TradeSignalApprovedParams:
    """Parameters for creating a TradeSignalApprovedEvent."""

    source_module: str
    signal_id: uuid.UUID
    trading_pair: str
    exchange: str
    side: str
    order_type: str
    quantity: Decimal
    sl_price: Decimal
    tp_price: Decimal
    risk_parameters: dict
    limit_price: Decimal | None = None


@dataclass
class TradeSignalApprovedValidationParams:
    """Parameters for validating an approved trade signal."""

    side: str
    order_type: str
    quantity: Decimal
    sl_price: Decimal
    tp_price: Decimal
    limit_price: Decimal | None = None


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
    risk_parameters: dict  # Parameters used by RiskManager for approval
    limit_price: Decimal | None = None
    event_type: EventType = field(default=EventType.TRADE_SIGNAL_APPROVED, init=False)

    @classmethod
    def _validate_input(cls, params: TradeSignalApprovedValidationParams) -> None:
        """Validate inputs for trade signal approval."""
        # Validate basic parameters
        cls._validate_basic_params(params.side, params.order_type, params.limit_price)

        # Validate prices and quantity
        cls._validate_prices_and_quantity(
            params.quantity,
            params.sl_price,
            params.tp_price,
            params.limit_price,
        )

        # Validate SL/TP positions if limit price is provided
        if params.limit_price is not None:
            cls._validate_sl_tp_positions(
                params.side,
                params.sl_price,
                params.tp_price,
                params.limit_price,
            )

    @staticmethod
    def _validate_basic_params(side: str, order_type: str, limit_price: Decimal | None) -> None:
        """Validate basic parameters for trade signal."""
        if side not in ["BUY", "SELL"]:
            raise InvalidSideError(side)

        if order_type not in ["LIMIT", "MARKET"]:
            raise InvalidOrderTypeError(order_type, ["LIMIT", "MARKET"])

        if order_type == "LIMIT" and limit_price is None:
            raise MissingLimitPriceError

    @staticmethod
    def _validate_prices_and_quantity(
        quantity: Decimal,
        sl_price: Decimal,
        tp_price: Decimal,
        limit_price: Decimal | None,
    ) -> None:
        """Validate that prices and quantity are positive."""
        if quantity <= Decimal(0):
            raise NonPositiveApprovedQuantityError(quantity)

        if sl_price <= Decimal(0):
            raise NonPositiveApprovedStopLossPriceError(sl_price)

        if tp_price <= Decimal(0):
            raise NonPositiveApprovedTakeProfitPriceError(tp_price)

        if limit_price is not None and limit_price <= Decimal(0):
            raise NonPositiveLimitPriceError(limit_price)

    @staticmethod
    def _validate_sl_tp_positions(
        side: str,
        sl_price: Decimal,
        tp_price: Decimal,
        limit_price: Decimal,
    ) -> None:
        """Validate stop loss and take profit positions relative to entry price and side."""
        if side == "BUY":
            if sl_price >= limit_price:
                raise BuyStopLossNotBelowApprovedLimitError(sl_price, limit_price)

            if tp_price <= limit_price:
                raise BuyTakeProfitNotAboveApprovedLimitError(tp_price, limit_price)
        else:  # side == "SELL"
            if sl_price <= limit_price:
                raise SellStopLossNotAboveEntryError(sl_price, limit_price)

            if tp_price >= limit_price:
                raise SellTakeProfitNotBelowEntryError(tp_price, limit_price)

    @classmethod
    def create(cls, params: TradeSignalApprovedParams) -> "TradeSignalApprovedEvent":
        """Create a validated TradeSignalApprovedEvent instance."""
        # Validate inputs
        validation_params = TradeSignalApprovedValidationParams(
            side=params.side,
            order_type=params.order_type,
            quantity=params.quantity,
            sl_price=params.sl_price,
            tp_price=params.tp_price,
            limit_price=params.limit_price,
        )
        cls._validate_input(validation_params)

        # Create and return instance
        return cls(
            source_module=params.source_module,
            event_id=uuid.uuid4(),
            timestamp=dt.datetime.now(dt.UTC), # DTZ003
            signal_id=params.signal_id,
            trading_pair=params.trading_pair,
            exchange=params.exchange,
            side=params.side,
            order_type=params.order_type,
            quantity=params.quantity,
            sl_price=params.sl_price,
            tp_price=params.tp_price,
            risk_parameters=params.risk_parameters,
            limit_price=params.limit_price,
        )


@dataclass
class TradeSignalRejectedParams:
    """Parameters for creating a TradeSignalRejectedEvent."""

    source_module: str
    signal_id: uuid.UUID
    trading_pair: str
    exchange: str
    side: str
    reason: str


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
    def create(cls, params: TradeSignalRejectedParams) -> "TradeSignalRejectedEvent":
        """Create a TradeSignalRejectedEvent instance.

        Args:
        ----
            params: Dataclass with all parameters for creating a TradeSignalRejectedEvent

        Returns:
        -------
            A TradeSignalRejectedEvent instance
        """
        return cls(
            source_module=params.source_module,
            event_id=uuid.uuid4(),
            timestamp=dt.datetime.now(dt.UTC), # F821, DTZ003
            signal_id=params.signal_id,
            trading_pair=params.trading_pair,
            exchange=params.exchange,
            side=params.side,
            reason=params.reason,
        )


@dataclass
class ExecutionReportParams:
    """Parameters for creating an ExecutionReportEvent."""

    source_module: str
    exchange_order_id: str
    trading_pair: str
    exchange: str
    order_status: str
    order_type: str
    side: str
    quantity_ordered: Decimal
    signal_id: uuid.UUID | None = None
    client_order_id: str | None = None
    quantity_filled: Decimal = Decimal(0)
    average_fill_price: Decimal | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    commission: Decimal | None = None
    commission_asset: str | None = None
    timestamp_exchange: dt.datetime | None = None
    error_message: str | None = None


@dataclass
class ExecutionReportValidationParams:
    """Parameters for validating an ExecutionReportEvent."""

    order_status: str
    order_type: str
    side: str
    quantity_ordered: Decimal
    quantity_filled: Decimal
    average_fill_price: Decimal | None
    limit_price: Decimal | None
    commission: Decimal | None
    commission_asset: str | None


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
    signal_id: uuid.UUID | None = None  # Originating signal ID
    client_order_id: str | None = None  # Internal ID if used
    quantity_filled: Decimal = Decimal(0)
    average_fill_price: Decimal | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    commission: Decimal | None = None
    commission_asset: str | None = None
    timestamp_exchange: dt.datetime | None = None
    error_message: str | None = None
    event_type: EventType = field(default=EventType.EXECUTION_REPORT, init=False)

    @classmethod
    def _validate_input(cls, validation_params: ExecutionReportValidationParams) -> None:
        """Validate execution report inputs."""
        # Validate order properties
        cls._validate_order_properties(
            validation_params.order_status,
            validation_params.order_type,
            validation_params.side,
        )

        # Validate quantities
        cls._validate_quantities(
            validation_params.quantity_ordered,
            validation_params.quantity_filled,
        )

        # Validate prices
        cls._validate_prices(
            validation_params.order_type,
            validation_params.order_status,
            validation_params.limit_price,
            validation_params.average_fill_price,
        )

        # Validate commission
        cls._validate_commission(validation_params.commission, validation_params.commission_asset)

    @staticmethod
    def _validate_order_properties(order_status: str, order_type: str, side: str) -> None:
        """Validate order status, type, and side."""
        valid_statuses = ["NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "REJECTED", "EXPIRED"]
        if order_status not in valid_statuses:
            raise InvalidOrderStatusError(order_status, valid_statuses)

        valid_order_types = ["LIMIT", "MARKET", "STOP", "STOP_LIMIT"]
        if order_type not in valid_order_types:
            raise InvalidOrderTypeError(order_type, valid_order_types)

        if side not in ["BUY", "SELL"]:
            raise InvalidSideError(side)

    @staticmethod
    def _validate_quantities(quantity_ordered: Decimal, quantity_filled: Decimal) -> None:
        """Validate order quantities."""
        if quantity_ordered <= Decimal(0):
            raise NonPositiveQuantityOrderedError(quantity_ordered)

        if quantity_filled < Decimal(0):
            raise NegativeQuantityFilledError(quantity_filled)

        if quantity_filled > quantity_ordered:
            raise QuantityFilledExceedsOrderedError(quantity_filled, quantity_ordered)

    @staticmethod
    def _validate_prices(
        order_type: str,
        order_status: str,
        limit_price: Decimal | None,
        average_fill_price: Decimal | None,
    ) -> None:
        """Validate price values."""
        if order_type == "LIMIT" and limit_price is None:
            raise MissingLimitPriceError

        if limit_price is not None and limit_price <= Decimal(0):
            raise NonPositiveLimitPriceError(limit_price)

        if average_fill_price is not None and average_fill_price <= Decimal(0):
            raise NonPositiveAverageFillPriceError(average_fill_price)

        filled_statuses = ["FILLED", "PARTIALLY_FILLED"]
        if order_status in filled_statuses and average_fill_price is None:
            raise MissingAverageFillPriceError

    @staticmethod
    def _validate_commission(
        commission: Decimal | None,
        commission_asset: str | None,
    ) -> None:
        """Validate commission details."""
        if commission is not None:
            if commission < Decimal(0):
                raise NegativeCommissionError(commission)
            if commission_asset is None:
                raise CommissionAssetMissingError

    @classmethod
    def create(cls, params: ExecutionReportParams) -> "ExecutionReportEvent":
        """Create a validated ExecutionReportEvent instance.

        Args:
        ----
            params: A dataclass containing all the parameters for creating an ExecutionReportEvent

        Returns:
        -------
            A validated ExecutionReportEvent instance

        Raises:
        ------
            ValueError: If any validation check fails
        """
        # Validate inputs
        cls._validate_input(
            ExecutionReportValidationParams(
                order_status=params.order_status,
                order_type=params.order_type,
                side=params.side,
                quantity_ordered=params.quantity_ordered,
                quantity_filled=params.quantity_filled,
                average_fill_price=params.average_fill_price,
                limit_price=params.limit_price,
                commission=params.commission,
                commission_asset=params.commission_asset,
            ),
        )

        # Create and return instance
        return cls(
            source_module=params.source_module,
            event_id=uuid.uuid4(),
            timestamp=dt.datetime.now(dt.UTC), # DTZ003
            exchange_order_id=params.exchange_order_id,
            trading_pair=params.trading_pair,
            exchange=params.exchange,
            order_status=params.order_status,
            order_type=params.order_type,
            side=params.side,
            quantity_ordered=params.quantity_ordered,
            signal_id=params.signal_id,
            client_order_id=params.client_order_id,
            quantity_filled=params.quantity_filled,
            average_fill_price=params.average_fill_price,
            limit_price=params.limit_price,
            stop_price=params.stop_price,
            commission=params.commission,
            commission_asset=params.commission_asset,
            timestamp_exchange=params.timestamp_exchange,
            error_message=params.error_message,
        )


@dataclass(frozen=True)
class LogEvent(Event):
    """Event carrying structured log information."""

    level: str  # e.g., "INFO", "ERROR"
    message: str
    context: dict | None = None
    event_type: EventType = field(default=EventType.LOG_ENTRY, init=False)


@dataclass(frozen=True)
class APIErrorEvent(Event):
    """Event indicating an API error occurred with the trading platform."""

    error_message: str
    http_status: int | None = None
    endpoint: str | None = None
    request_data: dict | None = None
    retry_attempted: bool = False
    event_type: EventType = field(default=EventType.SYSTEM_ERROR, init=False)

    @classmethod
    def create(cls, source_module: str, error_message: str, **details: Any) -> "APIErrorEvent":  # noqa: ANN401
        """Create a new APIErrorEvent with a generated UUID and current timestamp.

        Args:
        ----
            source_module: The module that created this event
            error_message: Description of the error that occurred
            **details: Additional fields like http_status, endpoint, etc.

        Returns:
        -------
            A new APIErrorEvent instance
        """
        return cls(
            source_module=source_module,
            event_id=uuid.uuid4(),
            timestamp=dt.datetime.now(dt.UTC), # DTZ003
            error_message=error_message,
            **details,
        )


@dataclass(frozen=True)
class ClosePositionCommand(Event):
    """Command to close an open position, typically triggered during a system HALT."""

    trading_pair: str
    quantity: Decimal
    side: str  # "BUY" or "SELL" (opposite of the position's side)
    order_type: str = "MARKET"  # Default to market order for immediate execution
    event_type: EventType = field(default=EventType.TRADE_SIGNAL_APPROVED, init=False)

    @classmethod
    def create(
        cls,
        source_module: str,
        trading_pair: str,
        quantity: Decimal,
        side: str,
    ) -> "ClosePositionCommand":
        """Create a new ClosePositionCommand with a generated UUID and current timestamp.

        Args:
        ----
            source_module: The module that created this command
            trading_pair: The trading pair to close (e.g., "XBT/USD")
            quantity: The absolute quantity to close
            side: "BUY" or "SELL" (opposite of the position's side)

        Returns:
        -------
            A new ClosePositionCommand instance
        """
        return cls(
            source_module=source_module,
            event_id=uuid.uuid4(),
            timestamp=dt.datetime.now(dt.UTC), # DTZ003
            trading_pair=trading_pair,
            quantity=quantity,
            side=side,
        )


@dataclass(frozen=True)
class PredictionConfigUpdatedEvent(Event):
    """Signals that the prediction service configuration has been updated."""

    # The payload could carry the entire new prediction_service config section,
    # or specific details about what changed.
    # For this implementation, let's assume it carries the new prediction_service config dict.
    new_prediction_service_config: dict[str, Any]
    event_type: EventType = field(default=EventType.PREDICTION_CONFIG_UPDATED, init=False)

    @classmethod
    def create(
        cls,
        source_module: str,
        new_config: dict[str, Any],
    ) -> "PredictionConfigUpdatedEvent":
        """Create a new PredictionConfigUpdatedEvent instance.

        Args:
            source_module: The module that created this event
            new_config: The new prediction service configuration

        Returns:
        -------
            A new PredictionConfigUpdatedEvent instance
        """
        return cls(
            source_module=source_module,
            event_id=uuid.uuid4(),
            timestamp=dt.datetime.now(dt.UTC), # DTZ003
            new_prediction_service_config=new_config,
        )


# Cleanup placeholder comment
# Removed the TODO section as definitions are now added.
# Removed the example creation comment.
