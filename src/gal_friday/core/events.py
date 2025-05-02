import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


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
    halt_action: Optional[str] = None # e.g., "LIQUIDATE_POSITIONS"
    event_type: EventType = field(
        default=EventType.SYSTEM_STATE_CHANGE, init=False
    )


@dataclass(frozen=True)
class PotentialHaltTriggerEvent(Event):
    """Event indicating a condition that might warrant a system HALT."""
    reason: str
    event_type: EventType = field(
        default=EventType.POTENTIAL_HALT_TRIGGER, init=False
    )


@dataclass(frozen=True)
class MarketDataL2Event(Event):
    """Event carrying L2 order book updates."""
    trading_pair: str
    exchange: str
    # Use Decimal for price/volume internally if possible, converting from str early.
    # Keep as str if exact representation from exchange is critical for checksums etc.
    bids: List[Tuple[str, str]] # [[price_str, volume_str], ...]
    asks: List[Tuple[str, str]] # [[price_str, volume_str], ...]
    is_snapshot: bool
    timestamp_exchange: Optional[datetime] = None
    event_type: EventType = field(
        default=EventType.MARKET_DATA_L2, init=False
    )


@dataclass(frozen=True)
class MarketDataOHLCVEvent(Event):
    """Event carrying OHLCV bar updates."""
    trading_pair: str
    exchange: str
    interval: str # e.g., "1m", "5m"
    timestamp_bar_start: datetime
    open: str # Using string representation from inter_module_comm doc
    high: str
    low: str
    close: str
    volume: str
    event_type: EventType = field(
        default=EventType.MARKET_DATA_OHLCV, init=False
    )


@dataclass(frozen=True)
class FeatureEvent(Event):
    """Event carrying calculated features."""
    trading_pair: str
    exchange: str
    timestamp_features_for: datetime
    features: Dict[str, Any] # Feature values (consider specific types if known)
    event_type: EventType = field(
        default=EventType.FEATURES_CALCULATED, init=False
    )


@dataclass(frozen=True)
class PredictionEvent(Event):
    """Event carrying model predictions."""
    trading_pair: str
    exchange: str
    timestamp_prediction_for: datetime
    model_id: str
    prediction_target: str # e.g., "prob_price_up_0.1pct_5min"
    # Use float or Decimal for probability/value
    prediction_value: float 
    confidence: Optional[float] = None
    associated_features: Optional[Dict[str, Any]] = None
    event_type: EventType = field(
        default=EventType.PREDICTION_GENERATED, init=False
    )


@dataclass(frozen=True)
class TradeSignalProposedEvent(Event):
    """Event carrying a proposed trade signal from the strategy."""
    signal_id: uuid.UUID # Unique ID for this proposal
    trading_pair: str
    exchange: str
    side: str # "BUY" or "SELL"
    entry_type: str # "LIMIT" or "MARKET"
    # Use Decimal for prices
    proposed_sl_price: Decimal
    proposed_tp_price: Decimal
    strategy_id: str
    proposed_entry_price: Optional[Decimal] = None 
    triggering_prediction_event_id: Optional[uuid.UUID] = None
    event_type: EventType = field(
        default=EventType.TRADE_SIGNAL_PROPOSED, init=False
    )


@dataclass(frozen=True)
class TradeSignalApprovedEvent(Event):
    """Event carrying an approved trade signal from the risk manager."""
    signal_id: uuid.UUID # Corresponds to proposed event
    trading_pair: str
    exchange: str
    side: str
    order_type: str # "LIMIT" or "MARKET"
    # Use Decimal for quantity/prices
    quantity: Decimal 
    sl_price: Decimal
    tp_price: Decimal
    limit_price: Optional[Decimal] = None
    event_type: EventType = field(
        default=EventType.TRADE_SIGNAL_APPROVED, init=False
    )


@dataclass(frozen=True)
class TradeSignalRejectedEvent(Event):
    """Event carrying a rejected trade signal from the risk manager."""
    signal_id: uuid.UUID # Corresponds to proposed event
    trading_pair: str
    exchange: str
    side: str
    reason: str
    event_type: EventType = field(
        default=EventType.TRADE_SIGNAL_REJECTED, init=False
    )


@dataclass(frozen=True)
class ExecutionReportEvent(Event):
    """Event carrying updates on order execution from the exchange."""
    exchange_order_id: str 
    trading_pair: str
    exchange: str
    order_status: str # e.g., "NEW", "FILLED", "CANCELED", "REJECTED"
    order_type: str # e.g., "LIMIT", "MARKET"
    side: str
    # Use Decimal for quantity/prices/commission
    quantity_ordered: Decimal
    signal_id: Optional[uuid.UUID] = None # Originating signal ID
    client_order_id: Optional[str] = None # Internal ID if used
    quantity_filled: Decimal = Decimal(0)
    average_fill_price: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    commission_asset: Optional[str] = None
    timestamp_exchange: Optional[datetime] = None
    error_message: Optional[str] = None
    event_type: EventType = field(
        default=EventType.EXECUTION_REPORT, init=False
    )


@dataclass(frozen=True)
class LogEvent(Event):
    """Event carrying structured log information."""
    level: str # e.g., "INFO", "ERROR"
    message: str
    context: Optional[Dict[str, Any]] = None
    event_type: EventType = field(default=EventType.LOG_ENTRY, init=False)


# Cleanup placeholder comment
# Removed the TODO section as definitions are now added.
# Removed the example creation comment.
