# Risk Manager Module
"""Risk management module for trading operations.

This module provides risk management functionality for trading operations,
including position sizing, drawdown limits, and trade validation.
"""

import asyncio
import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_UP, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json

import numpy as np

# Direct imports of actual service implementations
from .core.events import (
    EventType,
    PotentialHaltTriggerEvent,
    TradeSignalProposedEvent,
    ExecutionReportEvent,
)

# First-party (Gal-Friday) core component imports
from .core.feature_registry_client import FeatureRegistryClient
from .core.pubsub import PubSubManager
from .exchange_info_service import ExchangeInfoService
from .logger_service import LoggerService
from .market_price_service import MarketPriceService
from .portfolio_manager import PortfolioManager


# Enhanced Enums for Risk Management
class RejectionReason(str, Enum):
    """Categorized reasons for signal rejection."""
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    RISK_THRESHOLD_BREACH = "risk_threshold_breach"
    MARKET_CONDITION_INVALID = "market_condition_invalid"
    SIGNAL_QUALITY_POOR = "signal_quality_poor"
    CORRELATION_TOO_HIGH = "correlation_too_high"
    VOLATILITY_TOO_HIGH = "volatility_too_high"
    LIQUIDITY_INSUFFICIENT = "liquidity_insufficient"
    BLACKOUT_PERIOD = "blackout_period"
    TECHNICAL_ERROR = "technical_error"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    FAT_FINGER_DETECTED = "fat_finger_detected"
    INVALID_STOP_LOSS = "invalid_stop_loss"
    DRAWDOWN_LIMIT_BREACH = "drawdown_limit_breach"


class RejectionSeverity(str, Enum):
    """Severity levels for rejection reasons."""
    LOW = "low"       # Signal quality issue
    MEDIUM = "medium" # Risk threshold breach
    HIGH = "high"     # Position limit exceeded
    CRITICAL = "critical" # System safety issue


class ApprovalStatus(str, Enum):
    """Approval status for trade signals."""
    APPROVED = "approved"
    CONDITIONALLY_APPROVED = "conditionally_approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"


class OrderStatus(str, Enum):
    """Order status types."""
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    NEW = "NEW"
    PENDING = "PENDING"


class OrderSide(str, Enum):
    """Order side types."""
    BUY = "BUY"
    SELL = "SELL"


class MarketCondition(str, Enum):
    """Market condition types for validation."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    LOW_LIQUIDITY = "low_liquidity"
    TRENDING = "trending"
    GAPPING = "gapping"


class PositionSizingMethod(str, Enum):
    """Position sizing methods."""
    FIXED_AMOUNT = "fixed_amount"
    PERCENTAGE_OF_PORTFOLIO = "percentage_of_portfolio"
    RISK_BASED = "risk_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"
    EQUAL_WEIGHT = "equal_weight"


class OrderSizeType(str, Enum):
    """Order size representation types."""
    UNITS = "units"              # Raw units/shares
    NOTIONAL = "notional"        # Dollar/currency amount
    PERCENTAGE = "percentage"    # Percentage of portfolio
    LOTS = "lots"               # Exchange-specific lots
    CONTRACTS = "contracts"      # Futures/options contracts


# Enhanced Dataclasses
@dataclass
class RiskMetrics:
    """Current risk state metrics."""
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    current_drawdown: Decimal = Decimal("0")
    current_drawdown_pct: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    total_realized_pnl: Decimal = Decimal("0")
    active_positions_count: int = 0
    total_exposure: Decimal = Decimal("0")
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    var_95: Decimal = Decimal("0")  # Value at Risk
    expected_shortfall: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")


@dataclass
class SignalRejectionEvent:
    """Event published when a signal is rejected."""
    signal_id: uuid.UUID
    trading_pair: str
    strategy_id: str
    rejection_reason: RejectionReason
    severity: RejectionSeverity
    rejection_timestamp: datetime
    signal_data: Dict[str, Any]
    risk_metrics: Dict[str, float]
    rejection_details: str
    auto_retry_eligible: bool = False


@dataclass
class SignalApprovalEvent:
    """Event published when a signal is approved."""
    signal_id: uuid.UUID
    trading_pair: str
    strategy_id: str
    approval_status: ApprovalStatus
    approved_position_size: Decimal
    original_position_size: Decimal
    approval_timestamp: datetime
    approval_conditions: List[str]
    portfolio_impact: Dict[str, float]
    risk_adjustments: Dict[str, Any]
    execution_priority: int  # 1-10, higher is more urgent
    valid_until: datetime


@dataclass
class ExecutionReport:
    """Standardized execution report model."""
    order_id: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    realized_pnl: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    signal_id: Optional[str] = None
    strategy_id: Optional[str] = None


@dataclass
class ExchangePrecisionInfo:
    """Exchange precision specifications for a symbol."""
    symbol: str
    price_precision: int  # decimal places
    quantity_precision: int  # decimal places
    min_price: Decimal
    max_price: Decimal
    tick_size: Decimal  # minimum price increment
    min_quantity: Decimal
    max_quantity: Decimal
    step_size: Decimal  # minimum quantity increment
    min_notional: Decimal  # minimum order value


@dataclass
class MarketPriceContext:
    """Current market price context for validation."""
    symbol: str
    current_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    spread_pct: Decimal
    volatility_1h: Decimal
    volatility_24h: Decimal
    volume_24h: Decimal
    price_change_24h_pct: Decimal
    market_condition: MarketCondition
    last_trade_time: datetime


@dataclass
class ValidationResult:
    """Result of price/quantity validation."""
    is_valid: bool
    original_value: Decimal
    validated_value: Decimal
    adjustments_made: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class LotSizeResult:
    """Result of lot size calculation."""
    original_size: Decimal
    original_type: OrderSizeType
    calculated_lots: Decimal
    calculated_units: Decimal
    calculated_notional: Decimal
    exchange_valid: bool
    adjustments_made: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# Custom exceptions
class RiskManagerError(Exception):
    """Custom exception for risk management errors.

    Used to indicate errors in risk management operations, such as
    invalid configurations or trade validation failures.
    """


class SignalValidationStageError(RiskManagerError):
    """Custom exception for failures during specific trade signal validation stages."""

    def __init__(self, reason: str, stage_name: str) -> None:
        """Initialize the SignalValidationStageError with reason and stage name.

        Args:
            reason: The reason for the validation failure
            stage_name: The name of the validation stage that failed
        """
        super().__init__(f"Validation failed at {stage_name}: {reason}")
        self.reason = reason
        self.stage_name = stage_name


# The TYPE_CHECKING block is now only used for genuine type-hint-only
# forward references if absolutely necessary to break typing cycles


# Using default Decimal precision

MIN_SYMBOL_PARTS = 2
MIN_SYMBOL_LENGTH_FOR_FALLBACK_SPLIT = 3
CACHE_EXPIRY_SECONDS = 300
MIN_SAMPLES_FOR_STDEV_FUNCTION = 2  # Minimum samples required by statistics.stdev


# --- Event Payloads ---
@dataclass
class TradeSignalProposedPayload:
    """Payload for trade signal proposals."""

    signal_id: uuid.UUID
    trading_pair: str
    exchange: str
    side: str
    entry_type: str
    proposed_entry_price: str | None = None
    proposed_sl_price: str | None = None
    proposed_tp_price: str | None = None
    strategy_id: str = "default"
    triggering_prediction_event_id: uuid.UUID | None = None


@dataclass
class Stage1Context:
    """Context for Stage 1: Initial Validation & Price Rounding."""

    event: TradeSignalProposedEvent
    proposed_entry_price_decimal: Decimal | None
    proposed_sl_price_decimal: Decimal | None
    proposed_tp_price_decimal: Decimal | None


@dataclass
class Stage2Context:
    """Context for Stage 2: Fat Finger & Stop-Loss Distance."""

    event: TradeSignalProposedEvent
    rounded_entry_price: Decimal | None  # From Stage 1
    rounded_sl_price: Decimal  # From Stage 1, guaranteed non-None
    current_market_price_for_validation: Decimal | None


@dataclass
class Stage3Context:
    """Context for Stage 3: Position Sizing & Portfolio Checks."""

    event: TradeSignalProposedEvent
    current_equity_decimal: Decimal
    ref_entry_for_calculation: Decimal  # From Stage 2
    rounded_sl_price: Decimal  # From Stage 1/2
    portfolio_state: dict[str, Any]


@dataclass
class FinalValidationDataContext:
    """Holds data for the final stage of pre-trade checks."""

    event: TradeSignalProposedEvent
    signal_id: uuid.UUID
    trading_pair: str
    side: str
    entry_type: str
    exchange: str
    strategy_id: str
    current_equity: Decimal
    portfolio_state: dict[str, Any]
    state_values: dict[str, Decimal]
    initial_rounded_calculated_qty: Decimal
    rounded_entry_price: Decimal | None
    rounded_sl_price: Decimal
    rounded_tp_price: Decimal | None
    effective_entry_price: Decimal  # Guaranteed non-None if this stage is reached
    ref_entry_for_calculation: Decimal  # Guaranteed non-None if this stage is reached


@dataclass
class PriceValidationContext:
    """Holds data for price validation steps (fat finger, SL)."""

    event: TradeSignalProposedEvent
    entry_type: str
    side: str
    rounded_entry_price: Decimal | None
    rounded_sl_price: Decimal  # Must be non-None if this stage is reached
    effective_entry_price_for_non_limit: Decimal | None
    current_market_price: Decimal | None


@dataclass
class PositionScalingContext:
    """Holds data for the position scaling check."""

    signal_id: uuid.UUID
    trading_pair: str
    side: str
    ref_entry_price: Decimal  # Guaranteed non-None
    portfolio_state: dict[str, Any]
    initial_calculated_qty: Decimal  # Before scaling


@dataclass
class PriceRoundingContext:
    """Holds data for the price rounding and initial validation step."""

    entry_type: str
    side: str
    trading_pair: str
    effective_entry_price: Decimal | None
    sl_price: Decimal | None  # Initial SL before rounding, can be None if not proposed
    tp_price: Decimal | None


@dataclass
class SystemHaltPayload:
    """Payload for system halt events."""

    reason: str
    details: dict[str, Any]


@dataclass
class SizingResult:
    """Result of position sizing calculation."""

    is_valid: bool
    quantity: Decimal | None = None
    rejection_reason: str | None = None
    risk_amount: Decimal | None = None
    position_value: Decimal | None = None


# --- RiskManager Class ---
class RiskManager:
    """Assess trade signals against risk parameters and portfolio state.

    Assesses trade signals against risk parameters and current portfolio state.

    Key responsibilities include:
    - Validating proposed trade signals based on configured risk limits (e.g., max drawdown,
      exposure per asset, total exposure).
    - Calculating appropriate position sizes based on risk per trade percentages.
    - Performing pre-trade checks like fat-finger validation and ensuring sufficient balance.
    - Optionally, dynamically adjusting `risk_per_trade_pct` based on market volatility.
      This dynamic adjustment can be driven by a configured volatility feature (e.g., "atr_14_default")
      received from `FeatureEngine` via `FeatureEvent`s (payload: `dict[str, float]`).
      A baseline "normal" volatility for selected pairs is calibrated on startup using
      historical OHLCV data from the `MarketPriceService`. See `_calibrate_normal_volatility`
      and `_update_risk_parameters_based_on_volatility`.
    - Monitoring for excessive consecutive losses and triggering system alerts or halts.
    - Publishing `TradeSignalApprovedEvent` or `TradeSignalRejectedEvent`.
    - Publishing `PotentialHaltTriggerEvent` if critical risk limits are breached.

    It instantiates an internal `FeatureRegistryClient`. If dynamic risk adjustment
    is enabled and a `dynamic_risk_volatility_feature_key` is specified in the
    configuration, this key is validated against the Feature Registry during startup.
    """

    def __init__(  # - Multiple dependencies are required
        self,
        config: dict[str, Any],
        pubsub_manager: PubSubManager,
        portfolio_manager: PortfolioManager,
        logger_service: LoggerService,
        market_price_service: MarketPriceService,
        exchange_info_service: ExchangeInfoService,
    ) -> None:
        """Initialize the RiskManager with configuration and dependencies.

        Args:
            config: Overall application configuration dictionary. `RiskManager` uses the
                "risk_manager" section for its settings, including static limits,
                position sizing rules, and parameters for dynamic risk adjustment
                (e.g., `enable_dynamic_risk_adjustment`,
                `dynamic_risk_volatility_feature_key`, `dynamic_risk_target_pairs`).
            pubsub_manager: The application PubSubManager instance for event communication.
            portfolio_manager: The PortfolioManager instance to access portfolio state.
            logger_service: Shared logger instance for consistent logging.
            market_price_service: MarketPriceService instance for price and volatility data.
            exchange_info_service: ExchangeInfoService instance for exchange-specific details.

        Initializes an internal `FeatureRegistryClient`. Loads specific risk configurations
        via `_load_risk_config`, which includes validating the
        `dynamic_risk_volatility_feature_key` against the registry if dynamic risk
        adjustment is enabled and the key is configured. Sets up handlers for
        `TradeSignalProposedEvent`, `ExecutionReportEvent`, and `FeatureEvent`
        (if dynamic risk adjustment based on features is enabled).
        """
        self._config = config.get("risk_manager", {}) # Service specific config
        self.pubsub = pubsub_manager
        self._portfolio_manager = portfolio_manager
        self._market_price_service = market_price_service
        self._exchange_info_service = exchange_info_service
        self.logger = logger_service
        self.feature_registry_client = FeatureRegistryClient()

        self._is_running = False
        self._main_task: asyncio.Task | None = None
        self._periodic_check_task: asyncio.Task | None = None
        self._dynamic_risk_adjustment_task: asyncio.Task | None = None
        self._risk_metrics_task: asyncio.Task | None = None
        self._source_module = self.__class__.__name__

        # Event Handlers
        self._signal_proposal_handler = self._handle_trade_signal_proposed
        self._exec_report_handler = self._handle_execution_report_for_losses
        self._feature_event_handler = self._handle_feature_event # For dynamic risk

        # State variables
        self._consecutive_loss_count: int = 0
        self._consecutive_win_count: int = 0
        self._recent_trades: list[dict[str, Any]] = []
        self._cached_conversion_rates: dict[str, Decimal] = {}
        self._cached_conversion_timestamps: dict[str, datetime] = {}
        self._normal_volatility: dict[str, Decimal] = {} # Type hint Dict
        self._normal_volatility_logged_missing: dict[str, bool] = {}

        # Enhanced risk metrics tracking
        self._risk_metrics = RiskMetrics()
        self._metrics_history: List[RiskMetrics] = []
        self._max_metrics_history = 1000  # Keep last 1000 snapshots
        
        # Exchange precision cache
        self._precision_cache: Dict[str, ExchangePrecisionInfo] = {}
        self._precision_cache_expiry_seconds = 300  # 5 minutes
        
        # Market condition tracking
        self._market_conditions: Dict[str, MarketCondition] = {}
        self._last_market_update: Dict[str, datetime] = {}
        
        # Enhanced rejection/approval tracking
        self._rejection_stats: Dict[str, int] = {
            'total_rejections': 0,
            'rejections_by_reason': {},
            'rejections_by_symbol': {},
        }
        self._approval_stats: Dict[str, int] = {
            'total_approvals': 0,
            'conditional_approvals': 0,
            'approvals_by_symbol': {},
        }
        
        # Execution report processing
        self._execution_buffer: List[ExecutionReport] = []
        self._max_execution_buffer_size = 500
        
        # Risk threshold tracking
        self._threshold_breach_count: Dict[str, int] = {}
        self._last_threshold_reset = datetime.now(UTC)

        # For dynamic risk based on FeatureEngine features
        self._dynamic_risk_volatility_feature_key: str | None = None
        self._dynamic_risk_target_pairs: list[str] = [] # Type hint List
        self._latest_volatility_features: dict[str, float] = {} # Type hint Dict

        # Initialize risk metrics repository if available
        try:
            from gal_friday.dal.repositories.risk_metrics_repository import RiskMetricsRepository
            self._risk_metrics_repository = RiskMetricsRepository()
        except ImportError:
            self.logger.warning(
                "RiskMetricsRepository not available - risk metrics will not be persisted to database",
                source_module=self._source_module,
            )
            self._risk_metrics_repository = None

        # Load configuration (including new dynamic risk params)
        self._load_risk_config()

    def _load_risk_config(self) -> None:
        """Load risk parameters from the 'risk_manager' section of the application configuration.

        Extracts and initializes static risk limits (e.g., drawdown, exposure),
        position sizing parameters (e.g., `risk_per_trade_pct`), and settings for
        dynamic risk adjustment. This includes `enable_dynamic_risk_adjustment`,
        the `dynamic_risk_volatility_feature_key` (which specifies the feature from
        FeatureEngine to monitor, e.g., "atr_14_default"), and
        `dynamic_risk_target_pairs` for applying these adjustments.
        If dynamic risk adjustment is enabled and `dynamic_risk_volatility_feature_key`
        is set, this method also validates the feature key against the Feature Registry.
        Logs warnings if dynamic adjustment is enabled but necessary keys are missing
        or if the specified feature key is not found in the registry.
        """
        limits = self._config.get("limits", {})
        self._max_total_drawdown_pct = Decimal(str(limits.get("max_total_drawdown_pct", 15.0)))
        self._max_daily_drawdown_pct = Decimal(str(limits.get("max_daily_drawdown_pct", 2.0)))
        self._max_weekly_drawdown_pct = Decimal(str(limits.get("max_weekly_drawdown_pct", 5.0)))
        self._max_consecutive_losses = int(limits.get("max_consecutive_losses", 5))
        self._max_exposure_per_asset_pct = Decimal(
            str(limits.get("max_exposure_per_asset_pct", 10.0)),
        )
        self._max_total_exposure_pct = Decimal(str(limits.get("max_total_exposure_pct", 25.0)))
        self._max_order_size_usd = Decimal(str(limits.get("max_order_size_usd", 10000)))
        self._risk_per_trade_pct = Decimal(
            str(self._config.get("sizing", {}).get("risk_per_trade_pct", "0.5")),
        )
        # Added default_tp_rr_ratio from config or defaults to 2.0
        self._default_tp_rr_ratio = Decimal(
            str(self._config.get("sizing", {}).get("default_tp_rr_ratio", "2.0")),
        )
        self._check_interval_s = self._config.get("check_interval_s", 60)
        self._min_sl_distance_pct = Decimal(str(self._config.get("min_sl_distance_pct", "0.01")))
        self._max_single_position_pct = Decimal(
            str(self._config.get("max_single_position_pct", "100.0")),
        )
        # New config values for pre-trade checks
        self._fat_finger_max_deviation_pct = Decimal(
            str(self._config.get("fat_finger_max_deviation_pct", "5.0")),
        )
        self._taker_fee_pct = Decimal(
            str(self._config.get("exchange", {}).get("taker_fee_pct", "0.26")),
        ) / Decimal("100")  # Convert percentage to decimal
        # Portfolio valuation currency (e.g., "USD")
        self._valuation_currency = str(self._config.get("portfolio_valuation_currency", "USD"))

        # Add new configuration for dynamic risk adjustment
        self._enable_dynamic_risk_adjustment = bool(
            self._config.get("enable_dynamic_risk_adjustment", False),
        )
        self._risk_adjustment_interval_s = int(
            self._config.get("risk_adjustment_interval_s", 900),
        )
        self._volatility_window_size = int(
            self._config.get("volatility_window_size", 24),
        )
        self._risk_metrics_interval_s = int(
            self._config.get("risk_metrics_interval_s", 60),
        )

        # New: Config for dynamic risk adjustment via FeatureEngine features
        self._dynamic_risk_volatility_feature_key = self._config.get("dynamic_risk_volatility_feature_key")
        self._dynamic_risk_target_pairs = self._config.get("dynamic_risk_target_pairs", [])

        # Initialize monitored pairs - use dynamic risk target pairs or fall back to configured list
        self._monitored_pairs = self._dynamic_risk_target_pairs or self._config.get("monitored_pairs", ["XRP/USD", "DOGE/USD"])

        if self._enable_dynamic_risk_adjustment and not self._dynamic_risk_volatility_feature_key:
            self.logger.warning(
                "Dynamic risk adjustment is enabled, but no 'dynamic_risk_volatility_feature_key' "
                "is configured. Dynamic adjustments based on FeatureEngine features will not occur.",
            )
        if self._enable_dynamic_risk_adjustment and not self._dynamic_risk_target_pairs:
            self.logger.warning(
                "Dynamic risk adjustment is enabled, but 'dynamic_risk_target_pairs' is empty. "
                "Adjustments will not apply to any specific pair via FeatureEngine features.",
            )

        # Validate dynamic_risk_volatility_feature_key against Feature Registry
        if self._enable_dynamic_risk_adjustment and self._dynamic_risk_volatility_feature_key:
            if not self.feature_registry_client or not self.feature_registry_client.is_loaded():
                self.logger.warning(
                    "FeatureRegistryClient not available or not loaded. "
                    "Cannot validate 'dynamic_risk_volatility_feature_key': '%s' against the registry.",
                    self._dynamic_risk_volatility_feature_key,
                    source_module=self._source_module,
                )
            else:
                definition = self.feature_registry_client.get_feature_definition(self._dynamic_risk_volatility_feature_key)
                if definition is None:
                    self.logger.warning(
                        "The configured 'dynamic_risk_volatility_feature_key': '%s' was not found in the Feature Registry. "
                        "Dynamic risk adjustment based on this feature may not work as expected.",
                        self._dynamic_risk_volatility_feature_key,
                        source_module=self._source_module,
                    )
                else:
                    self.logger.debug(
                        "Dynamic risk volatility feature '%s' validated against Feature Registry.",
                        self._dynamic_risk_volatility_feature_key,
                        source_module=self._source_module,
                    )

        self.logger.info("RiskManager configured.", source_module=self._source_module)
        self._validate_config()

    def get_available_risk_budget(self) -> Decimal:
        """Return available risk budget based on current exposure limits."""

        try:
            portfolio_state = self._portfolio_manager.get_current_state()
        except Exception as exc:  # pragma: no cover - defensive catch
            self.logger.error(
                f"Failed to retrieve portfolio state for risk budget: {exc}",
                source_module=self._source_module,
            )
            return Decimal("0")

        try:
            equity_raw = portfolio_state.get("total_equity") or portfolio_state.get(
                "total_equity_usd",
            )
            equity = Decimal(str(equity_raw)) if equity_raw is not None else Decimal("0")
        except (InvalidOperation, ValueError):  # pragma: no cover - logging handles
            self.logger.warning(
                "Invalid total_equity value from PortfolioManager.",
                source_module=self._source_module,
            )
            equity = Decimal("0")

        try:
            exposure_pct_raw = portfolio_state.get("total_exposure_pct")
            exposure_pct = (
                Decimal(str(exposure_pct_raw)) if exposure_pct_raw is not None else Decimal("0")
            )
        except (InvalidOperation, ValueError):  # pragma: no cover - logging handles
            self.logger.warning(
                "Invalid total_exposure_pct value from PortfolioManager.",
                source_module=self._source_module,
            )
            exposure_pct = Decimal("0")

        available_pct = self._max_total_exposure_pct - exposure_pct
        if available_pct < 0:
            available_pct = Decimal("0")

        return equity * (available_pct / Decimal("100"))

    def _validate_config(self) -> None:
        """Validate risk parameters from configuration.

        Checks that all required parameters are present and have valid values.
        Raises RiskManagerError if validation fails.
        """
        config_errors = []

        # Helper to check percentage values
        def check_percentage(
            value: Decimal,
            name: str,
            lower_bound: Decimal = Decimal("0"),
            upper_bound: Decimal = Decimal("100"),
        ) -> None:
            if not (lower_bound <= value <= upper_bound):
                config_errors.append(
                    f"{name} ({value}%) is outside the valid range [{lower_bound}-{upper_bound}].",
                )

        check_percentage(self._max_total_drawdown_pct, "max_total_drawdown_pct")
        check_percentage(self._max_daily_drawdown_pct, "max_daily_drawdown_pct")
        check_percentage(self._max_weekly_drawdown_pct, "max_weekly_drawdown_pct")
        check_percentage(self._max_exposure_per_asset_pct, "max_exposure_per_asset_pct")
        check_percentage(self._max_total_exposure_pct, "max_total_exposure_pct")
        # risk_per_trade_pct typically small, but can be up to 100
        check_percentage(
            self._risk_per_trade_pct,
            "risk_per_trade_pct",
            upper_bound=Decimal("100"),
        )
        # min_sl_distance_pct must be > 0
        check_percentage(
            self._min_sl_distance_pct,
            "min_sl_distance_pct",
            lower_bound=Decimal("0.001"),
        )
        check_percentage(self._max_single_position_pct, "max_single_position_pct")

        if self._default_tp_rr_ratio <= 0:
            config_errors.append(
                f"default_tp_rr_ratio ({self._default_tp_rr_ratio}) must be positive.",
            )

        if self._max_consecutive_losses < 1:
            config_errors.append(
                f"max_consecutive_losses ({self._max_consecutive_losses}) must be at least 1.",
            )

        if config_errors:
            for error_msg in config_errors:
                self.logger.debug(
                    f"Configuration Error: {error_msg}",
                    source_module=self._source_module,
                )
            # Log critical message and continue as per whiteboard,
            # rather than raising an exception.
            error_msg_str = "; ".join(config_errors)
            self.logger.critical(
                f"RiskManager has configuration errors. Review settings: {error_msg_str}",
                source_module=self._source_module,
            )

    async def start(self) -> None:
        """Starts the RiskManager service.

        Subscribes to:
        - `EventType.TRADE_SIGNAL_PROPOSED` for assessing new trade signals.
        - `EventType.EXECUTION_REPORT` for tracking trade outcomes (e.g., losses).
        - `EventType.FEATURES_CALCULATED` via `_feature_event_handler` if dynamic risk
          adjustment based on FeatureEngine features is enabled. This allows the
          RiskManager to receive live volatility metrics.

        If dynamic risk adjustment is enabled:
        - Calls `_calibrate_normal_volatility` to establish baseline volatility.
        - If a specific `dynamic_risk_volatility_feature_key` is NOT configured, it may
          start a fallback `_dynamic_risk_adjustment_loop_fallback` that uses
          `MarketPriceService.get_volatility()`. (Note: This fallback's direct update
          mechanism is currently simplified and may need further refinement if used).

        Starts periodic tasks for overall risk checks (`_periodic_risk_check_loop`)
        and risk metrics calculation (`_risk_metrics_loop`).
        """
        self.logger.info("Starting RiskManager...", source_module=self._source_module)

        # Subscribe to events
        self.pubsub.subscribe(EventType.TRADE_SIGNAL_PROPOSED, self._signal_proposal_handler)
        self.pubsub.subscribe(EventType.EXECUTION_REPORT, self._exec_report_handler)
        self.pubsub.subscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)

        # Calibrate normal volatility for dynamic risk adjustment (baseline from market data)
        if self._enable_dynamic_risk_adjustment:
            await self._calibrate_normal_volatility()

            # If a specific feature key for volatility is NOT defined for dynamic adjustment,
            # start the fallback loop that uses MarketPriceService.get_volatility.
            # Otherwise, dynamic adjustments are primarily event-driven by _handle_feature_event.
            if not self._dynamic_risk_volatility_feature_key:
                self.logger.info("No dynamic_risk_volatility_feature_key configured. Starting fallback dynamic risk adjustment loop.")
                self._dynamic_risk_adjustment_task = asyncio.create_task(
                    self._dynamic_risk_adjustment_loop_fallback(),
                )
            else:
                self.logger.info(f"Dynamic risk adjustment configured with feature key: {self._dynamic_risk_volatility_feature_key}. Adjustments will be event-driven.")

        # Start periodic risk check task
        self._periodic_check_task = asyncio.create_task(self._periodic_risk_check_loop())

        # Start risk metrics calculation task
        self._risk_metrics_task = asyncio.create_task(self._risk_metrics_loop()) # Explicitly re-set

        self._is_running = True
        self.logger.info("RiskManager started.", source_module=self._source_module)

    async def stop(self) -> None:
        """Stops the RiskManager service.

        Unsubscribes from `TRADE_SIGNAL_PROPOSED`, `EXECUTION_REPORT`, and
        `FEATURES_CALCULATED` events.
        Cancels any running asynchronous tasks like periodic risk checks and
        dynamic risk adjustment loops.
        """
        self.logger.info("Stopping RiskManager...", source_module=self._source_module)

        self._is_running = False

        # Unsubscribe from events
        try:
            self.pubsub.unsubscribe(EventType.TRADE_SIGNAL_PROPOSED, self._signal_proposal_handler)
            self.pubsub.unsubscribe(EventType.EXECUTION_REPORT, self._exec_report_handler)
            self.pubsub.unsubscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)
            self.logger.info(
                "Unsubscribed from TRADE_SIGNAL_PROPOSED, EXECUTION_REPORT, and FEATURES_CALCULATED.",
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception(
                "Error unsubscribing RiskManager handlers",
                source_module=self._source_module,
            )

        # Stop periodic checks
        if self._periodic_check_task and not self._periodic_check_task.done():
            self._periodic_check_task.cancel()

        if self._dynamic_risk_adjustment_task and not self._dynamic_risk_adjustment_task.done():
            self._dynamic_risk_adjustment_task.cancel()

        if self._risk_metrics_task and not self._risk_metrics_task.done():
            self._risk_metrics_task.cancel()

        # Stop retry processing task
        if hasattr(self, '_retry_task') and self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.error(
                    f"Error stopping retry processing task: {e}",
                    source_module=self._source_module,
                )

        # Clean up retry queue and attempts tracking
        if hasattr(self, '_retry_queue'):
            retry_count = len(self._retry_queue)
            if retry_count > 0:
                self.logger.info(
                    f"Cleaning up {retry_count} pending signal retries",
                    source_module=self._source_module,
                )
            self._retry_queue.clear()
            
        if hasattr(self, '_retry_attempts'):
            self._retry_attempts.clear()

        # Clean up audit clients and resources
        await self._cleanup_audit_resources()

        self.logger.info(
            "Stopped periodic risk checks, retry processing, audit resources, and tasks.",
            source_module=self._source_module,
        )

    async def _periodic_risk_check_loop(self) -> None:
        """Periodically check portfolio risk metrics."""
        while self._is_running:
            try:
                await asyncio.sleep(self._check_interval_s)

                # Check drawdown limits
                portfolio_state = self._portfolio_manager.get_current_state()
                await self._check_drawdown_limits(portfolio_state, is_pre_trade_check=False)

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception(
                    "Error in periodic risk check",
                    source_module=self._source_module,
                )

    async def _handle_feature_event(self, event_dict: dict[str, Any]) -> None:
        """Handles incoming `FeatureEvent`s (payload expected as `dict[str, float]`)
        to update latest volatility data for dynamic risk adjustment.

        If dynamic risk adjustment is enabled, this method checks if the received
        feature event corresponds to one of the `_dynamic_risk_target_pairs` and
        if the event payload contains the `_dynamic_risk_volatility_feature_key`.
        If so, it extracts the float value of this feature, stores it in
        `_latest_volatility_features`, and then calls
        `_update_risk_parameters_based_on_volatility` for the specific trading pair.
        """
        if not self._is_running or not self._enable_dynamic_risk_adjustment or \
           not self._dynamic_risk_volatility_feature_key:
            return

        if event_dict.get("event_type") == EventType.FEATURES_CALCULATED.name:
            payload = event_dict.get("payload")
            if not payload or not isinstance(payload, dict):
                self.logger.warning("Received FEATURES_CALCULATED event with invalid payload.")
                return

            trading_pair = payload.get("trading_pair")
            # Features are expected as dict[str, float] from PublishedFeaturesV1.model_dump()
            features_data = payload.get("features")

            if not trading_pair or not features_data or not isinstance(features_data, dict):
                self.logger.warning("FEATURES_CALCULATED event missing trading_pair or valid features dict.")
                return

            if trading_pair in self._dynamic_risk_target_pairs:
                volatility_value = features_data.get(self._dynamic_risk_volatility_feature_key)

                if volatility_value is not None and isinstance(volatility_value, (float, int)): # Check type
                    if np.isnan(volatility_value):
                        self.logger.debug(
                            f"Volatility feature '{self._dynamic_risk_volatility_feature_key}' "
                            f"is NaN for {trading_pair}. Skipping update.",
                        )
                        return

                    self.logger.debug(
                        f"Received volatility feature '{self._dynamic_risk_volatility_feature_key}' "
                        f"value {volatility_value:.4f} for {trading_pair}.",
                    )
                    self._latest_volatility_features[trading_pair] = float(volatility_value)
                    await self._update_risk_parameters_based_on_volatility(trading_pair)
                else:
                    self.logger.debug(
                        f"Volatility feature '{self._dynamic_risk_volatility_feature_key}' "
                        f"not found or invalid type ({type(volatility_value)}) in event for {trading_pair}.",
                    )

    async def _dynamic_risk_adjustment_loop_fallback(self) -> None:
        """Fallback loop for dynamic risk adjustment if FeatureEngine-based feature
        is not configured. This uses MarketPriceService.get_volatility directly.
        This method might be deprecated or removed if feature-based adjustment is primary.
        """
        self.logger.info("Using fallback dynamic risk adjustment loop (MarketPriceService.get_volatility).")
        while self._is_running:
            try:
                await asyncio.sleep(self._risk_adjustment_interval_s)

                for trading_pair in self._dynamic_risk_target_pairs:
                    try:
                        volatility_raw = await self._market_price_service.get_volatility(
                            trading_pair,
                            self._volatility_window_size, # type: ignore
                        )
                        if volatility_raw is not None:
                            # This fallback loop's purpose is to populate the same
                            # self._latest_volatility_features that _handle_feature_event would,
                            # so that _update_risk_parameters_based_on_volatility can use it.
                            if np.isnan(volatility_raw):
                                self.logger.debug(f"Fallback: MarketPriceService returned NaN volatility for {trading_pair}. Skipping.")
                                continue

                            self._latest_volatility_features[trading_pair] = float(volatility_raw)
                            self.logger.debug(f"Fallback: Updated latest volatility for {trading_pair} to {volatility_raw:.4f} from MarketPriceService.")
                            await self._update_risk_parameters_based_on_volatility(trading_pair)
                        else:
                            self.logger.debug(f"Fallback: MarketPriceService returned None volatility for {trading_pair}.")

                    except AttributeError:
                        self.logger.warning("MarketPriceService does not implement get_volatility method (fallback loop).")
                        return # Stop this loop if service doesn't support it
                    except Exception:
                        self.logger.exception(f"Error fetching/processing volatility in fallback loop for {trading_pair}.")
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("Error in fallback dynamic risk adjustment loop")


    async def _handle_trade_signal_proposed(
        self,
        event: TradeSignalProposedEvent,
    ) -> None:
        """Handle incoming trade signal proposals.

        This is the main entry point for risk assessment of a trade.
        """
        try:
            # Convert proposed prices from string to Decimal
            try:
                proposed_entry_price_decimal = (
                    Decimal(event.proposed_entry_price) if event.proposed_entry_price else None
                )
                proposed_sl_price_decimal = (
                    Decimal(event.proposed_sl_price) if event.proposed_sl_price else None
                )
                proposed_tp_price_decimal = (
                    Decimal(event.proposed_tp_price) if event.proposed_tp_price else None
                )
            except InvalidOperation as e:
                await self._reject_signal(
                    event.signal_id,
                    event,
                    f"Invalid proposed price format: {e}",
                )
                return

            # --- Stage 1: Initial Validation & Price Rounding ---
            stage1_ctx = Stage1Context(
                event=event,
                proposed_entry_price_decimal=proposed_entry_price_decimal,
                proposed_sl_price_decimal=proposed_sl_price_decimal,
                proposed_tp_price_decimal=proposed_tp_price_decimal,
            )
            (
                rounded_entry_price,
                rounded_sl_price,
                rounded_tp_price,
            ) = await self._stage1_initial_validation_and_price_rounding(stage1_ctx)

            # --- Stage 2: Fat Finger & Stop-Loss Distance (Market Price Dependent) ---
            current_market_price_for_validation = await self._get_current_market_price(
                event.trading_pair,
            )
            stage2_ctx = Stage2Context(
                event=event,
                rounded_entry_price=rounded_entry_price,
                rounded_sl_price=rounded_sl_price,
                current_market_price_for_validation=current_market_price_for_validation,
            )
            (
                effective_entry_price_for_non_limit,
                ref_entry_for_calculation,
                final_rounded_entry_price,
            ) = await self._stage2_market_price_dependent_checks(stage2_ctx)

            self._validate_and_raise_if_error(
                error_condition=(
                    ref_entry_for_calculation is None or final_rounded_entry_price is None
                ),
                failure_reason="Internal error: critical price missing post Stage 2.",
                stage_name="Post-Stage2 Internal Check",
                log_message=(
                    "Internal error: ref_entry or final_rounded_entry is None after "
                    "Stage 2 for signal %(signal_id)s."
                ),
                log_context={"signal_id": str(event.signal_id)},
            )

            # Type assertions for mypy - after the validation above, these cannot be None
            assert ref_entry_for_calculation is not None
            assert final_rounded_entry_price is not None

            # --- Stage 3: Position Sizing & Portfolio Checks (Portfolio/Equity checks first) ---
            portfolio_state = self._portfolio_manager.get_current_state()
            stage_name_pre_s3 = "Pre-Stage3 Portfolio/Equity Check"

            self._validate_and_raise_if_error(
                error_condition=not portfolio_state or "total_equity_usd" not in portfolio_state,
                failure_reason="Portfolio state/equity unavailable.",
                stage_name=stage_name_pre_s3,
                log_message=(
                    "Portfolio state or total_equity_usd not available for signal "
                    "%(signal_id)s."
                ),
                log_context={"signal_id": str(event.signal_id)},
            )

            current_equity = portfolio_state.get("total_equity_usd")
            if (
                current_equity is None
            ):  # Nested check for 'equity' if 'total_equity_usd' is missing
                current_equity = portfolio_state.get("equity")
                self._validate_and_raise_if_error(
                    error_condition=current_equity is None,
                    failure_reason="Equity not found.",
                    stage_name=stage_name_pre_s3,
                    log_message=(
                        "Equity (neither total_equity_usd nor equity) not found for "
                        "signal %(signal_id)s."
                    ),
                    log_context={"signal_id": str(event.signal_id)},
                )

            current_equity_decimal: Decimal
            try:
                current_equity_decimal = Decimal(str(current_equity))
            except InvalidOperation as e:
                self.logger.exception(
                    "Invalid equity value '%(equity_val)s' in portfolio state for "
                    "signal %(signal_id)s.",
                    source_module=self._source_module,
                    context={
                        "signal_id": str(event.signal_id),
                        "equity_val": str(current_equity),
                    },
                )
                reason = "Invalid equity."
                raise SignalValidationStageError(reason, stage_name_pre_s3) from e

            stage3_ctx = Stage3Context(
                event=event,
                current_equity_decimal=current_equity_decimal,
                ref_entry_for_calculation=ref_entry_for_calculation,
                rounded_sl_price=rounded_sl_price,
                portfolio_state=portfolio_state,
            )
            initial_rounded_calculated_qty, state_values = await self._stage3_position_sizing_and_portfolio_checks(
                stage3_ctx,
            )

            # --- Stage 4: Position Scaling (if applicable) ---
            position_scaling_ctx = PositionScalingContext(
                signal_id=event.signal_id,
                trading_pair=event.trading_pair,
                side=event.side,
                ref_entry_price=ref_entry_for_calculation,
                portfolio_state=portfolio_state,
                initial_calculated_qty=initial_rounded_calculated_qty,
            )
            # _check_position_scaling & _perform_final_pre_trade_validations
            # return (bool, reason) and are not yet converted to raise
            # SignalValidationStageError directly. So they are handled by the
            # _validate_and_raise_if_error helper where the condition is their negation.

            (
                can_scale,
                scale_rejection_reason,
                final_trade_action,
                final_quantity,
            ) = self._check_position_scaling(position_scaling_ctx)

            self._validate_and_raise_if_error(
                error_condition=not can_scale,
                failure_reason=scale_rejection_reason or "Position scaling failed.",
                stage_name="Stage4: Position Scaling",
                log_message=(
                    "Position scaling check failed for signal %(signal_id)s. " "Reason: %(reason)s"
                ),
                log_context={"signal_id": str(event.signal_id), "reason": scale_rejection_reason},
            )

            current_qty_to_trade = (
                final_quantity if final_quantity is not None else initial_rounded_calculated_qty
            )

            self._validate_and_raise_if_error(
                error_condition=current_qty_to_trade.is_zero(),
                failure_reason="Quantity zero after scaling.",
                stage_name="Stage4: Position Scaling",
                log_message="Quantity became zero after scaling for signal %(signal_id)s.",
                log_context={"signal_id": str(event.signal_id)},
            )

            # --- Stage 5: Final Pre-Trade Validation (includes balance check) ---
            final_effective_entry_price = (
                final_rounded_entry_price
                if event.entry_type.upper() == "LIMIT"
                else effective_entry_price_for_non_limit
            )

            self._validate_and_raise_if_error(
                error_condition=final_effective_entry_price is None,
                failure_reason="Internal error: final effective price missing.",
                stage_name="Stage5: Pre-Final Validation",
                log_message="Final effective entry price is None for signal %(signal_id)s.",
                log_context={"signal_id": str(event.signal_id)},
            )
            if final_effective_entry_price is None:
                raise SignalValidationStageError(
                    "Final effective entry price is None",
                    "final_validation",
                )

            final_validation_ctx = FinalValidationDataContext(
                event=event,
                signal_id=event.signal_id,
                trading_pair=event.trading_pair,
                side=event.side,
                entry_type=event.entry_type,
                exchange=event.exchange,
                strategy_id=event.strategy_id,
                current_equity=current_equity_decimal,
                portfolio_state=portfolio_state,
                state_values=state_values,
                initial_rounded_calculated_qty=current_qty_to_trade,
                rounded_entry_price=final_rounded_entry_price,
                rounded_sl_price=rounded_sl_price,
                rounded_tp_price=rounded_tp_price,
                effective_entry_price=final_effective_entry_price,
                ref_entry_for_calculation=ref_entry_for_calculation,
            )

            (
                is_final_valid,
                final_rejection_reason,
            ) = await self._perform_final_pre_trade_validations(final_validation_ctx)

            self._validate_and_raise_if_error(
                error_condition=not is_final_valid,
                failure_reason=final_rejection_reason or "Final validation failed.",
                stage_name="Stage5: Final Pre-Trade Validation",
                log_message=(
                    "Final pre-trade validation failed for signal %(signal_id)s. "
                    "Reason: %(reason)s"
                ),
                log_context={"signal_id": str(event.signal_id), "reason": final_rejection_reason},
            )

            self.logger.info(
                "Trade signal %(signal_id)s passed all risk checks.",
                source_module=self._source_module,
                context={"signal_id": str(event.signal_id)},
            )

            # Approve the signal
            await self._approve_signal(
                event.signal_id,
                event,
                current_qty_to_trade,
                final_rounded_entry_price,
            )

        except SignalValidationStageError as e:
            await self._reject_signal(event.signal_id, event, e.reason)

    async def _perform_final_pre_trade_validations(
        self,
        ctx: FinalValidationDataContext,
    ) -> tuple[bool, str | None]:
        """Perform final overall portfolio and risk checks before approval."""
        # Check 1: Max Exposure per Asset
        # This requires knowing the value of existing position in this asset + new trade value.
        # Portfolio state should provide current position value for the asset.
        # New trade value = quantity * effective_entry_price

        # Convert new trade quantity to quote currency value
        # Assuming effective_entry_price is in quote currency
        value_of_new_trade_usd = ctx.initial_rounded_calculated_qty * ctx.effective_entry_price

        # Get current exposure for this specific asset from portfolio_state
        current_asset_exposure_usd = Decimal("0")
        if ctx.portfolio_state and "positions" in ctx.portfolio_state:
            asset_position_data = ctx.portfolio_state["positions"].get(ctx.trading_pair)
            if asset_position_data:
                # Assuming 'current_market_value' is in USD or valuation currency
                current_asset_exposure_usd = Decimal(
                    str(asset_position_data.get("current_market_value", "0")),
                )

        total_potential_asset_exposure_usd = current_asset_exposure_usd + value_of_new_trade_usd

        max_asset_exposure_allowed_usd = ctx.current_equity * (
            self._max_exposure_per_asset_pct / Decimal("100")
        )
        if total_potential_asset_exposure_usd > max_asset_exposure_allowed_usd:
            reason = (
                f"Exceeds max exposure per asset for {ctx.trading_pair} "
                f"({total_potential_asset_exposure_usd:.2f} > "
                f"{max_asset_exposure_allowed_usd:.2f} USD). "
                f"Limit: {self._max_exposure_per_asset_pct}%"
            )
            self.logger.info(
                reason,
                source_module=self._source_module,
                context={
                    "signal_id": str(ctx.signal_id),
                    "trading_pair": ctx.trading_pair,
                    "exposure": f"{total_potential_asset_exposure_usd:.2f}",
                    "limit": f"{max_asset_exposure_allowed_usd:.2f}",
                },
            )
            return False, reason

        # Check 2: Max Total Portfolio Exposure
        current_total_exposure_usd = Decimal("0")
        if ctx.portfolio_state and "positions" in ctx.portfolio_state:
            for pair_data in ctx.portfolio_state["positions"].values():
                current_total_exposure_usd += Decimal(
                    str(pair_data.get("current_market_value", "0")),
                )

        total_potential_portfolio_exposure_usd = (
            current_total_exposure_usd + value_of_new_trade_usd
        )

        max_portfolio_exposure_allowed_usd = ctx.current_equity * (
            self._max_total_exposure_pct / Decimal("100")
        )
        if total_potential_portfolio_exposure_usd > max_portfolio_exposure_allowed_usd:
            reason = (
                f"Exceeds max total portfolio exposure "
                f"({total_potential_portfolio_exposure_usd:.2f} > "
                f"{max_portfolio_exposure_allowed_usd:.2f} USD). "
                f"Limit: {self._max_total_exposure_pct}%"
            )
            self.logger.info(
                reason,
                source_module=self._source_module,
                context={
                    "signal_id": str(ctx.signal_id),
                    "exposure": f"{total_potential_portfolio_exposure_usd:.2f}",
                    "limit": f"{max_portfolio_exposure_allowed_usd:.2f}",
                },
            )
            return False, reason

        # Check 3: Sufficient Free Balance (SRS FR-506)
        available_balance_usd = ctx.state_values.get("available_balance_usd")
        if available_balance_usd is None:
            self.logger.warning(
                "Available balance not found in portfolio state for signal %(signal_id)s. "
                "Cannot verify funds.",
                source_module=self._source_module,
                context={"signal_id": str(ctx.signal_id)},
            )
            return False, "Available balance missing in portfolio state for fund check."

        estimated_order_cost_usd = ctx.initial_rounded_calculated_qty * ctx.effective_entry_price
        taker_fee_multiplier = self._taker_fee_pct
        estimated_fee_usd = estimated_order_cost_usd * taker_fee_multiplier
        total_estimated_cost_with_fee_usd = estimated_order_cost_usd + estimated_fee_usd

        if total_estimated_cost_with_fee_usd > available_balance_usd:
            reason = (
                f"Insufficient available balance for trade. "
                f"Estimated cost with fee {total_estimated_cost_with_fee_usd:.2f} "
                f"{self._valuation_currency} > "
                f"Available {available_balance_usd:.2f} {self._valuation_currency}."
            )
            self.logger.info(
                reason,
                source_module=self._source_module,
                context={
                    "signal_id": str(ctx.signal_id),
                    "cost": f"{total_estimated_cost_with_fee_usd:.2f}",
                    "balance": f"{available_balance_usd:.2f}",
                },
            )
            return False, reason

        # Check 4: Drawdown limits
        if not await self._check_drawdown_limits(ctx.portfolio_state, is_pre_trade_check=True):
            return False, "Portfolio drawdown limits would be breached or are currently breached."

        return True, None

    def _check_single_drawdown_limit(
        self,
        current_equity: Decimal,
        period_initial_equity: Decimal | None,
        max_period_drawdown_pct: Decimal,
        pm_provided_drawdown_pct_str: str | None,
        period_name: str,
    ) -> str | None:
        """Check a single drawdown limit (e.g., total, daily, weekly)."""
        breached_reason: str | None = None

        # Use PortfolioManager's drawdown if available and valid
        if pm_provided_drawdown_pct_str is not None:
            try:
                pm_provided_dd_val = Decimal(str(pm_provided_drawdown_pct_str))
                if pm_provided_dd_val > max_period_drawdown_pct:
                    breached_reason = (
                        f"{period_name} drawdown limit breached (from PM): "
                        f"{pm_provided_dd_val:.2f}% > {max_period_drawdown_pct:.2f}%"
                    )
            except InvalidOperation:
                self.logger.warning(
                    "Invalid %(period_name_lower)s_drawdown_pct '%(pm_val)s' from PM.",
                    source_module=self._source_module,
                    context={
                        "period_name_lower": period_name.lower(),
                        "pm_val": pm_provided_drawdown_pct_str,
                    },
                )
                # Fall through to recalculate if PM value is invalid

        # If PM didn't provide it, or it was invalid, or no breach yet, calculate it
        if (
            breached_reason is None
            and period_initial_equity is not None
            and period_initial_equity > 0
        ):
            calculated_dd = (
                (period_initial_equity - current_equity) / period_initial_equity
            ) * Decimal("100")
            if pm_provided_drawdown_pct_str is None:  # Log only if PM didn't provide it
                self.logger.info(
                    "PM did not provide valid %(period_name_lower)s_drawdown_pct. "
                    "Calculated: %(calc_dd).2f%%",
                    source_module=self._source_module,
                    context={"period_name_lower": period_name.lower(), "calc_dd": calculated_dd},
                )
            if calculated_dd > max_period_drawdown_pct:
                breached_reason = (
                    f"{period_name} drawdown limit breached (calculated): "
                    f"{calculated_dd:.2f}% > {max_period_drawdown_pct:.2f}%"
                )
        return breached_reason

    async def _check_drawdown_limits(
        self,
        portfolio_state: dict[str, Any],
        is_pre_trade_check: bool = False,
    ) -> bool:
        """Check portfolio against configured drawdown limits."""
        if not portfolio_state:
            self.logger.warning(
                "Cannot check drawdown limits: Portfolio state is empty.",
                source_module=self._source_module,
            )
            return True

        state_values = self._extract_relevant_portfolio_values(portfolio_state)
        current_equity = state_values.get("current_equity_usd")

        if current_equity is None:
            self.logger.warning(
                "Cannot check drawdown limits: Current equity not found in portfolio state.",
                source_module=self._source_module,
            )
            return True

        breached_limit_reason: str | None = None

        # Check Total Drawdown
        breached_limit_reason = self._check_single_drawdown_limit(
            current_equity,
            state_values.get("initial_equity_usd"),
            self._max_total_drawdown_pct,
            portfolio_state.get("total_drawdown_pct"),
            "Total",
        )

        # Check Daily Drawdown if no breach yet
        if breached_limit_reason is None:
            breached_limit_reason = self._check_single_drawdown_limit(
                current_equity,
                state_values.get("daily_initial_equity_usd"),
                self._max_daily_drawdown_pct,
                portfolio_state.get("daily_drawdown_pct"),
                "Daily",
            )

        # Check Weekly Drawdown if no breach yet
        if breached_limit_reason is None and self._max_weekly_drawdown_pct is not None:
            breached_limit_reason = self._check_single_drawdown_limit(
                current_equity,
                state_values.get("weekly_initial_equity_usd"),
                self._max_weekly_drawdown_pct,
                portfolio_state.get("weekly_drawdown_pct"),
                "Weekly",
            )

        if breached_limit_reason:
            self.logger.critical(
                breached_limit_reason,
                source_module=self._source_module,
                context={
                    "current_equity": current_equity,
                    "initial_equity_usd": state_values.get("initial_equity_usd"),
                    "daily_initial_equity_usd": state_values.get("daily_initial_equity_usd"),
                    "weekly_initial_equity_usd": state_values.get("weekly_initial_equity_usd"),
                },
            )
            if not is_pre_trade_check:
                details_dict = {
                    "check_type": "drawdown_limit",
                    "current_equity": str(current_equity),
                    "breach_details": breached_limit_reason,
                }
                reason_with_details = f"{breached_limit_reason} Details: {details_dict}"
                event_payload = PotentialHaltTriggerEvent(
                    source_module=self._source_module,
                    event_id=uuid.uuid4(),
                    timestamp=datetime.now(UTC),
                    reason=reason_with_details,
                )
                await self.pubsub.publish(event_payload)
            return False

        return True

    async def _update_risk_parameters_based_on_volatility(
        self,
        trading_pair: str,
        # current_volatility: Decimal, # No longer a direct arg
    ) -> None:
        """Dynamically adjusts `_risk_per_trade_pct` for a given `trading_pair`.

        This method uses the latest live volatility feature value for the `trading_pair`
        (cached in `self._latest_volatility_features` by `_handle_feature_event`) and
        compares it against a pre-calibrated "normal" volatility for that pair
        (from `self._normal_volatility`, typically based on historical daily log return
        standard deviation).

        - If current volatility is significantly higher than normal, `_risk_per_trade_pct` is reduced.
        - If current volatility is significantly lower than normal, `_risk_per_trade_pct` is
          increased (capped at the original statically configured maximum).
        - Otherwise (volatility is within a normal band), `_risk_per_trade_pct` is
          reverted to its static configured value.

        This adjustment is only performed if `self._enable_dynamic_risk_adjustment` is true
        and the necessary volatility data (both live and normal) is available.
        """
        if not self._enable_dynamic_risk_adjustment:
            return

        current_volatility_float = self._latest_volatility_features.get(trading_pair)
        if current_volatility_float is None:
            self.logger.debug(
                "No live volatility feature value available for %s to perform dynamic risk adjustment.",
                trading_pair,
            )
            return

        try:
            current_volatility = Decimal(str(current_volatility_float))
        except InvalidOperation:
            self.logger.warning(
                "Invalid volatility feature value '%s' for %s. Cannot perform dynamic risk adjustment.",
                current_volatility_float, trading_pair,
            )
            return

        normal_vol = self._normal_volatility.get(trading_pair) # This is Decimal
        current_risk_setting_before_adj = self._risk_per_trade_pct

        if normal_vol and normal_vol > Decimal(0):
            static_configured_risk_pct = Decimal(
                str(self._config.get("sizing", {}).get("risk_per_trade_pct", "0.5")),
            )
            new_risk_pct = current_risk_setting_before_adj

            if current_volatility > normal_vol * Decimal("1.5"):  # High volatility
                new_risk_pct = max(current_risk_setting_before_adj / Decimal("2"), Decimal("0.05"))
                if new_risk_pct != current_risk_setting_before_adj:
                    self.logger.info(
                        (
                            f"DYNAMIC RISK: Reducing risk per trade from "
                            f"{current_risk_setting_before_adj:.3f}% "
                            f"to {new_risk_pct:.3f}% "
                            f"for {trading_pair} due to high volatility "
                            f"({current_volatility:.4f} vs normal ~{normal_vol:.4f}). Volatility feature: {self._dynamic_risk_volatility_feature_key}"
                        ),
                        source_module=self._source_module,
                    )
                    self._risk_per_trade_pct = new_risk_pct

            elif current_volatility < normal_vol * Decimal("0.75"):  # Low volatility
                new_risk_pct = min(
                    current_risk_setting_before_adj * Decimal("1.5"),
                    static_configured_risk_pct, # Cap at the original config value
                )
                if new_risk_pct != current_risk_setting_before_adj:
                    self.logger.info(
                        (
                            f"DYNAMIC RISK: Increasing risk per trade from "
                            f"{current_risk_setting_before_adj:.3f}% "
                            f"to {new_risk_pct:.3f}% "
                            f"for {trading_pair} due to low volatility "
                            f"({current_volatility:.4f} vs normal ~{normal_vol:.4f}). "
                            f"Capped at {static_configured_risk_pct:.3f}%. Volatility feature: {self._dynamic_risk_volatility_feature_key}"
                        ),
                        source_module=self._source_module,
                    )
                    self._risk_per_trade_pct = new_risk_pct

            elif self._risk_per_trade_pct != static_configured_risk_pct: # Volatility is normal, ensure risk is at static config
                self.logger.info(
                    (
                        f"DYNAMIC RISK: Volatility normal for {trading_pair} "
                        f"({current_volatility:.4f} vs normal ~{normal_vol:.4f}). "
                        f"Reverting risk per trade from {self._risk_per_trade_pct:.3f}% "
                        f"to static configured {static_configured_risk_pct:.3f}%. Volatility feature: {self._dynamic_risk_volatility_feature_key}"
                    ),
                    source_module=self._source_module,
                )
                self._risk_per_trade_pct = static_configured_risk_pct

        else:
            if not hasattr(self, "_normal_volatility_logged_missing"):
                self._normal_volatility_logged_missing = {}

            if self._normal_volatility_logged_missing.get(trading_pair, False) is False:
                self.logger.warning(
                    (
                        f"Cannot dynamically adjust risk for {trading_pair}: "
                        f"Normal volatility not calibrated or is zero. "
                        "Using static risk per trade."
                    ),
                    source_module=self._source_module,
                )
                self._normal_volatility_logged_missing[trading_pair] = True

            static_configured_risk_pct = Decimal(
                str(self._config.get("sizing", {}).get("risk_per_trade_pct", "0.5")),
            )
            if self._risk_per_trade_pct != static_configured_risk_pct:
                self.logger.info(
                    (
                        f"DYNAMIC RISK: Reverting risk for {trading_pair} to static "
                        f"configured {static_configured_risk_pct:.3f}% "
                        "due to missing normal volatility calibration."
                    ),
                    source_module=self._source_module,
                    context={
                        "trading_pair": trading_pair,
                        "static_risk_pct": static_configured_risk_pct,
                    },
                )
                self._risk_per_trade_pct = static_configured_risk_pct

    async def _calibrate_normal_volatility(self) -> None:
        """Calibrate normal (baseline) volatility for each trading pair.
        
        This method now supports both daily and intraday volatility calculations
        based on configuration settings. Intraday volatility provides more
        responsive risk adjustments for high-frequency trading scenarios.
        """
        self.logger.info(
            "Starting normal volatility calibration process.",
            source_module=self._source_module,
        )

        # Check configuration for volatility calculation mode
        volatility_mode = self._config.get("risk.volatility_mode", "daily")  # "daily" or "intraday"
        intraday_timeframe = self._config.get("risk.intraday_timeframe", "15m")  # For intraday mode
        
        MIN_SAMPLES_FOR_STDEV_FUNCTION = 2  # Required by statistics.stdev

        # Calibration parameters based on mode
        if volatility_mode == "intraday":
            lookback_hours = self._config.get("risk.intraday_lookback_hours", 24)  # Last 24 hours
            min_data_points_for_stddev = self._config.get("risk.intraday_min_data_points", 50)
        else:  # daily mode
            lookback_days = self._config.get("risk.normal_volatility_lookback_days", 60)
            min_data_points_for_stddev = self._config.get("risk.min_data_points_for_stddev", 30)

        for trading_pair in self._monitored_pairs:
            # Skip if we've already logged a warning for this pair
            if self._normal_volatility_logged_missing.get(trading_pair, False):
                continue

            try:
                # Calculate time range based on mode
                current_time = datetime.now(UTC)
                if volatility_mode == "intraday":
                    since_datetime = current_time - timedelta(hours=lookback_hours)
                    timeframe = intraday_timeframe
                else:
                    since_datetime = current_time - timedelta(days=lookback_days)
                    timeframe = "1d"

                self.logger.debug(
                    f"Fetching {volatility_mode} volatility data for {trading_pair} "
                    f"from {since_datetime} with timeframe {timeframe}",
                    source_module=self._source_module,
                )

                # Fetch historical data with appropriate timeframe
                historical_data = await self._market_price_service.get_historical_ohlcv(
                    trading_pair=trading_pair,
                    timeframe=timeframe,
                    since=since_datetime,
                )

                if not historical_data or len(historical_data) < min_data_points_for_stddev:
                    self.logger.warning(
                        (
                            f"Could not calibrate {volatility_mode} volatility for {trading_pair}: "
                            f"Insufficient historical data fetched (got "
                            f"{len(historical_data) if historical_data else 0} points, "
                            f"need at least {min_data_points_for_stddev})."
                        ),
                        source_module=self._source_module,
                        context={
                            "trading_pair": trading_pair,
                            "fetched_points": len(historical_data) if historical_data else 0,
                            "min_points": min_data_points_for_stddev,
                            "volatility_mode": volatility_mode,
                        },
                    )
                    self._normal_volatility_logged_missing[trading_pair] = True
                    continue

                # Extract closing prices
                closing_prices = [
                    candle["close"]
                    for candle in historical_data
                    if "close" in candle and isinstance(candle["close"], Decimal)
                ]

                if len(closing_prices) < min_data_points_for_stddev:
                    self.logger.warning(
                        (
                            f"Could not calibrate {volatility_mode} volatility for {trading_pair}: "
                            f"Insufficient valid closing prices extracted "
                            f"({len(closing_prices)}, need at least {min_data_points_for_stddev})."
                        ),
                        source_module=self._source_module,
                        context={
                            "trading_pair": trading_pair,
                            "closing_prices_count": len(closing_prices),
                            "min_points": min_data_points_for_stddev,
                            "volatility_mode": volatility_mode,
                        },
                    )
                    self._normal_volatility_logged_missing[trading_pair] = True
                    continue

                # Calculate logarithmic returns: ln(Price_t / Price_{t-1})
                log_returns = []
                for i in range(1, len(closing_prices)):
                    # Avoid division by zero or issues with non-positive prices
                    if closing_prices[i - 1] > Decimal(0):
                        log_return = Decimal(math.log(closing_prices[i] / closing_prices[i - 1]))
                        log_returns.append(log_return)
                    else:
                        self.logger.debug(
                            (
                                f"Skipping log return calculation for {trading_pair} "
                                f"due to non-positive previous price: {closing_prices[i-1]}"
                            ),
                            source_module=self._source_module,
                            context={
                                "trading_pair": trading_pair,
                                "prev_price": closing_prices[i - 1],
                            },
                        )

                # Need at least 2 returns for stdev, realistically more
                if len(log_returns) < min_data_points_for_stddev - 1:
                    self.logger.warning(
                        (
                            f"Could not calibrate {volatility_mode} volatility for {trading_pair}: "
                            f"Insufficient log returns calculated ({len(log_returns)}, "
                            f"need at least {min_data_points_for_stddev -1} for meaningful stdev)."
                        ),
                        source_module=self._source_module,
                        context={
                            "trading_pair": trading_pair,
                            "log_returns_count": len(log_returns),
                            "min_needed_for_stdev": min_data_points_for_stddev - 1,
                            "volatility_mode": volatility_mode,
                        },
                    )
                    self._normal_volatility_logged_missing[trading_pair] = True
                    continue

                # Calculate standard deviation of log returns
                # statistics.stdev requires at least MIN_SAMPLES_FOR_STDEV_FUNCTION data points.
                if len(log_returns) >= MIN_SAMPLES_FOR_STDEV_FUNCTION:
                    # Convert Decimal log returns to float for statistics.stdev,
                    # then back to Decimal
                    float_log_returns = [float(lr) for lr in log_returns]
                    period_volatility = Decimal(str(statistics.stdev(float_log_returns)))

                    # Annualize volatility if needed
                    if volatility_mode == "intraday":
                        # Convert intraday volatility to annualized equivalent
                        # For crypto markets: 365 days * 24 hours * periods per hour
                        periods_per_day = self._get_periods_per_day(intraday_timeframe)
                        periods_per_year = periods_per_day * 365  # 365 days for crypto markets
                        annualized_volatility = period_volatility * Decimal(math.sqrt(periods_per_year))
                        
                        self._normal_volatility[trading_pair] = annualized_volatility
                        volatility_type = f"annualized intraday ({intraday_timeframe})"
                    else:
                        # For daily volatility, annualize using 365 days
                        # Daily volatility * sqrt(365) for annualized
                        annualized_daily_volatility = period_volatility * Decimal(math.sqrt(365))
                        self._normal_volatility[trading_pair] = annualized_daily_volatility
                        volatility_type = "annualized daily"

                    self.logger.info(
                        (
                            f"Successfully calibrated {volatility_type} volatility for {trading_pair}: "
                            f"{self._normal_volatility[trading_pair]:.8f}. "
                            f"(Based on {len(log_returns)} log returns from {len(closing_prices)} prices)"
                        ),
                        source_module=self._source_module,
                        context={
                            "trading_pair": trading_pair,
                            "volatility": self._normal_volatility[trading_pair],
                            "volatility_type": volatility_type,
                            "log_returns_count": len(log_returns),
                            "closing_prices_count": len(closing_prices),
                            "lookback_period": f"{lookback_hours}h" if volatility_mode == "intraday" else f"{lookback_days}d",
                        },
                    )
                    # Mark as calibrated
                    self._normal_volatility_logged_missing[trading_pair] = False
                else:
                    self.logger.warning(
                        (
                            f"Could not calculate stdev for {trading_pair}: "
                            f"Not enough log returns ({len(log_returns)})."
                        ),
                        source_module=self._source_module,
                        context={
                            "trading_pair": trading_pair,
                            "log_returns_count": len(log_returns),
                        },
                    )
                    self._normal_volatility_logged_missing[trading_pair] = True

            except Exception:
                self.logger.exception(
                    f"Error during {volatility_mode} volatility calibration for {trading_pair}",
                    source_module=self._source_module,
                    context={"trading_pair": trading_pair, "volatility_mode": volatility_mode},
                )
                self._normal_volatility_logged_missing[trading_pair] = True

        self.logger.info(
            f"{volatility_mode.capitalize()} volatility calibration process finished.",
            source_module=self._source_module,
        )

    def _get_periods_per_day(self, timeframe: str) -> int:
        """Convert timeframe string to number of periods per trading day."""
        # Map common timeframes to periods per day (24-hour trading for crypto)
        timeframe_map = {
            "1m": 1440,   # 24 hours * 60 minutes
            "5m": 288,    # 1440 / 5
            "15m": 96,    # 1440 / 15
            "30m": 48,    # 1440 / 30
            "1h": 24,     # 24 hours
            "4h": 6,      # 24 / 4
            "1d": 1,      # 1 day
        }
        return timeframe_map.get(timeframe, 96)  # Default to 15m if unknown

    def _calculate_and_validate_prices(
        self,
        ctx: PriceRoundingContext,
    ) -> tuple[
        bool,
        str | None,
        Decimal | None,
        Decimal | None,
        Decimal | None,
    ]:
        """Calculate and validate prices with proper rounding."""
        # Round entry price if provided
        rounded_entry_price = None
        if ctx.effective_entry_price is not None:
            rounded_entry_price = self._round_price_to_tick_size(
                ctx.effective_entry_price,
                ctx.trading_pair,
            )

        # Validate and calculate SL price
        if ctx.sl_price is None:
            # Calculate default SL based on minimum distance
            if rounded_entry_price is not None:
                if ctx.side == "BUY":
                    ctx.sl_price = rounded_entry_price * (Decimal("1") - self._min_sl_distance_pct / Decimal("100"))
                else:  # SELL
                    ctx.sl_price = rounded_entry_price * (Decimal("1") + self._min_sl_distance_pct / Decimal("100"))
            else:
                return False, "Cannot calculate SL without entry price", None, None, None

        rounded_sl_price = self._round_price_to_tick_size(ctx.sl_price, ctx.trading_pair)

        # Validate SL distance
        if rounded_entry_price is not None and rounded_sl_price is not None:
            sl_distance_pct = abs((rounded_sl_price - rounded_entry_price) / rounded_entry_price) * Decimal("100")
            if sl_distance_pct < self._min_sl_distance_pct:
                return False, f"SL distance {sl_distance_pct:.2f}% below minimum {self._min_sl_distance_pct}%", None, None, None

        # Calculate TP if not provided
        rounded_tp_price = None
        if ctx.tp_price is not None:
            rounded_tp_price = self._round_price_to_tick_size(ctx.tp_price, ctx.trading_pair)
        elif rounded_entry_price is not None and rounded_sl_price is not None:
            # Calculate TP based on risk/reward ratio
            risk = abs(rounded_entry_price - rounded_sl_price)
            if ctx.side == "BUY":
                rounded_tp_price = rounded_entry_price + (risk * self._default_tp_rr_ratio)
            else:  # SELL
                rounded_tp_price = rounded_entry_price - (risk * self._default_tp_rr_ratio)
            rounded_tp_price = self._round_price_to_tick_size(rounded_tp_price, ctx.trading_pair)

        return True, None, rounded_entry_price, rounded_sl_price, rounded_tp_price

    def _round_price_to_tick_size(
        self,
        price: Decimal | None,
        trading_pair: str,
    ) -> Decimal | None:
        """Round price to exchange tick size."""
        if price is None:
            return None

        try:
            tick_size = self._exchange_info_service.get_tick_size(trading_pair)
        except AttributeError:
            # Handle missing method in ExchangeInfoService
            self.logger.warning(
                "ExchangeInfoService does not implement get_tick_size method",
                source_module=self._source_module,
            )
            tick_size = None
        if not tick_size or tick_size <= 0:
            # Default tick size if not found
            tick_size = Decimal("0.00001")

        # Round to nearest tick
        try:
            if tick_size and tick_size > 0:
                result_price: Decimal = (price / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_size
                return result_price
            return price
        except Exception as e:
            self.logger.exception(
                f"Error rounding price to tick size for {trading_pair}",
                source_module=self._source_module,
                context={"trading_pair": trading_pair, "price": price, "tick_size": tick_size, "error": str(e)},
            )
            return None

    def _validate_prices_fat_finger_and_sl_distance(
        self,
        ctx: PriceValidationContext,
    ) -> tuple[bool, str | None]:
        """Validate prices for fat finger and stop loss distance."""
        # Fat finger check for limit orders
        if ctx.entry_type == "LIMIT" and ctx.current_market_price is not None and ctx.rounded_entry_price is not None:
            deviation_pct = abs((ctx.rounded_entry_price - ctx.current_market_price) / ctx.current_market_price) * Decimal("100")
            if deviation_pct > self._fat_finger_max_deviation_pct:
                return False, f"Entry price deviates {deviation_pct:.2f}% from market (max {self._fat_finger_max_deviation_pct}%)"

        # Validate SL distance
        effective_entry = ctx.rounded_entry_price or ctx.effective_entry_price_for_non_limit
        if effective_entry is not None:
            sl_distance_pct = abs((ctx.rounded_sl_price - effective_entry) / effective_entry) * Decimal("100")
            if sl_distance_pct < self._min_sl_distance_pct:
                return False, f"SL distance {sl_distance_pct:.2f}% below minimum {self._min_sl_distance_pct}%"

            # Validate SL is on correct side
            if ctx.side == "BUY" and ctx.rounded_sl_price >= effective_entry:
                return False, "BUY order SL must be below entry price"
            if ctx.side == "SELL" and ctx.rounded_sl_price <= effective_entry:
                return False, "SELL order SL must be above entry price"

        return True, None

    def _calculate_and_validate_position_size(
        self,
        event: TradeSignalProposedEvent,
        current_equity: Decimal,
        ref_entry_price: Decimal,
        rounded_sl_price: Decimal,
        portfolio_state: dict[str, Any],
    ) -> SizingResult:
        """Calculate position size based on risk parameters."""
        try:
            # Calculate risk amount
            risk_amount = current_equity * (self._risk_per_trade_pct / Decimal("100"))

            # Calculate risk per unit
            risk_per_unit = abs(ref_entry_price - rounded_sl_price)
            if risk_per_unit <= 0:
                return SizingResult(
                    is_valid=False,
                    rejection_reason="Invalid risk per unit (SL too close to entry)",
                )

            # Calculate raw position size
            raw_quantity = risk_amount / risk_per_unit

            # Apply lot size constraints
            rounded_quantity, _ = self._calculate_lot_size_with_fallback(
                raw_quantity,
                event.trading_pair,
            )

            if rounded_quantity <= 0:
                return SizingResult(
                    is_valid=False,
                    rejection_reason="Position size below minimum tradeable amount",
                )

            # Calculate position value
            position_value = rounded_quantity * ref_entry_price

            # Check against max order size
            if position_value > self._max_order_size_usd:
                # Scale down to max order size
                scaled_quantity = self._max_order_size_usd / ref_entry_price
                rounded_quantity, _ = self._calculate_lot_size_with_fallback(
                    scaled_quantity,
                    event.trading_pair,
                )

                if rounded_quantity <= 0:
                    return SizingResult(
                        is_valid=False,
                        rejection_reason=f"Position value exceeds max order size ${self._max_order_size_usd}",
                    )
                position_value = rounded_quantity * ref_entry_price

            # Check against max single position percentage
            position_pct = (position_value / current_equity) * Decimal("100")
            if position_pct > self._max_single_position_pct:
                return SizingResult(
                    is_valid=False,
                    rejection_reason=f"Position size {position_pct:.2f}% exceeds max {self._max_single_position_pct}%",
                )

            return SizingResult(
                is_valid=True,
                quantity=rounded_quantity,
                risk_amount=risk_amount,
                position_value=position_value,
            )

        except Exception as e:
            self.logger.exception(
                "Error calculating position size",
                source_module=self._source_module,
                context={"signal_id": str(event.signal_id)},
            )
            return SizingResult(
                is_valid=False,
                rejection_reason=f"Position sizing error: {e!s}",
            )

    def _check_position_scaling(
        self,
        ctx: PositionScalingContext,
    ) -> tuple[bool, str | None, str | None, Decimal | None]:
        """Check if position needs scaling based on existing positions."""
        # Check if we already have a position in this pair
        existing_position = None
        if "positions" in ctx.portfolio_state:
            existing_position = ctx.portfolio_state["positions"].get(ctx.trading_pair)

        if existing_position:
            existing_qty = Decimal(str(existing_position.get("quantity", "0")))
            existing_side = existing_position.get("side", "").upper()

            # Check if this is adding to position or reducing
            if existing_side == ctx.side:
                # Adding to position - check if total would exceed limits
                total_qty = existing_qty + ctx.initial_calculated_qty
                total_value = total_qty * ctx.ref_entry_price

                # Check against portfolio equity
                portfolio_equity = Decimal(str(ctx.portfolio_state.get("total_equity_usd", "0")))
                if portfolio_equity > 0:
                    position_pct = (total_value / portfolio_equity) * Decimal("100")
                    if position_pct > self._max_exposure_per_asset_pct:
                        # Scale down the new quantity
                        max_additional_value = (portfolio_equity * self._max_exposure_per_asset_pct / Decimal("100")) - (existing_qty * ctx.ref_entry_price)
                        if max_additional_value <= 0:
                            return False, "Position already at maximum exposure", None, None

                        scaled_qty = max_additional_value / ctx.ref_entry_price
                        scaled_qty, _ = self._calculate_lot_size_with_fallback(scaled_qty, ctx.trading_pair)

                        if scaled_qty <= 0:
                            return False, "Scaled position size below minimum", None, None

                        return True, None, "ADD_TO_POSITION", scaled_qty

                return True, None, "ADD_TO_POSITION", ctx.initial_calculated_qty
            # Opposite side - this would close/reduce position
            if existing_qty <= ctx.initial_calculated_qty:
                # This would close the position entirely
                return True, None, "CLOSE_POSITION", existing_qty
            # This would partially close
            return True, None, "REDUCE_POSITION", ctx.initial_calculated_qty

        # No existing position
        return True, None, "NEW_POSITION", ctx.initial_calculated_qty

    def _extract_relevant_portfolio_values(
        self,
        portfolio_state: dict[str, Any],
    ) -> dict[str, Decimal]:
        """Extract and convert relevant portfolio values to Decimal."""
        extracted = {}

        # Define fields to extract with defaults
        fields = {
            "available_balance_usd": "0",
            "current_equity_usd": "0",
            "total_equity_usd": "0",
            "initial_equity_usd": "0",
            "daily_initial_equity_usd": "0",
            "weekly_initial_equity_usd": "0",
            "total_exposure_usd": "0",
        }

        for field, default in fields.items():
            value = portfolio_state.get(field, default)
            try:
                extracted[field] = Decimal(str(value))
            except (InvalidOperation, ValueError):
                self.logger.warning(
                    f"Invalid value for {field}: {value}, using default {default}",
                    source_module=self._source_module,
                )
                extracted[field] = Decimal(default)

        # Handle special cases
        if "current_equity_usd" not in portfolio_state and "equity" in portfolio_state:
            try:
                extracted["current_equity_usd"] = Decimal(str(portfolio_state["equity"]))
            except (InvalidOperation, ValueError):
                pass

        # Calculate total exposure if not provided
        if extracted["total_exposure_usd"] == Decimal("0") and "positions" in portfolio_state:
            total_exposure = Decimal("0")
            for position in portfolio_state["positions"].values():
                try:
                    market_value = Decimal(str(position.get("current_market_value", "0")))
                    total_exposure += market_value
                except (InvalidOperation, ValueError):
                    pass

        return extracted

    async def _handle_execution_report_for_losses(self, event_dict: dict[str, Any]) -> None:
        """Handle execution reports to track losses for consecutive loss counting.
        
        Enhanced implementation with comprehensive execution report parsing,
        loss counter logic, event publishing and audit trail.
        """
        try:
            # Parse execution report based on format
            execution_report = await self._parse_execution_report(event_dict)
            if not execution_report:
                return
                
            # Only process filled or partially filled orders
            if execution_report.status not in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                return
            
            # Skip if no average fill price
            if not execution_report.average_price or execution_report.average_price <= 0:
                self.logger.warning(
                    f"Execution report {execution_report.order_id} has no valid average fill price",
                    source_module=self._source_module,
                )
                return
            
            # Add to execution buffer for tracking
            self._execution_buffer.append(execution_report)
            if len(self._execution_buffer) > self._max_execution_buffer_size:
                self._execution_buffer.pop(0)
            
            # Get portfolio state to determine P&L
            portfolio_state = await self._portfolio_manager.get_current_state()
            if not portfolio_state:
                self.logger.error(
                    "Could not get portfolio state for execution report processing",
                    source_module=self._source_module,
                )
                return
            
            # Calculate realized P&L from this execution
            realized_pnl = await self._calculate_realized_pnl_from_execution(execution_report, portfolio_state)
            
            # Update loss counters based on realized PnL
            await self._update_loss_counters(execution_report, realized_pnl)
            
            # Check for risk condition breaches
            risk_events = await self._evaluate_risk_conditions(execution_report, realized_pnl, portfolio_state)
            
            # Publish risk events if any
            if risk_events:
                await self._publish_risk_events(risk_events, execution_report)
            
            # Update audit trail
            await self._update_execution_audit_trail(execution_report, realized_pnl, risk_events)
            
            # Update comprehensive risk metrics
            await self._update_risk_metrics_from_execution(execution_report, realized_pnl, portfolio_state)
            
            # Check consecutive loss thresholds
            await self._check_consecutive_loss_thresholds()
            
            # Publish risk metrics update event
            await self._publish_risk_metrics_update(execution_report, realized_pnl)
            
            self.logger.info(
                f"Processed execution report {execution_report.order_id}: "
                f"PnL={realized_pnl:.4f} if realized_pnl else 'N/A', "
                f"Consecutive losses={self._consecutive_loss_count}",
                source_module=self._source_module,
            )
            
        except Exception as e:
            self.logger.error(
                f"Error processing execution report for loss tracking: {e}",
                source_module=self._source_module,
                context={"event_dict": event_dict},
            )
            # Publish error event for monitoring
            await self._publish_execution_error_event(event_dict, str(e))

    async def _calculate_realized_pnl_from_execution(
        self, 
        execution_report: ExecutionReport, 
        portfolio_state: dict[str, Any]
    ) -> Decimal | None:
        """Calculate realized P&L from an execution report."""
        try:
            # For closing trades, we need to compare with the entry price
            positions = portfolio_state.get("positions", {})
            position = positions.get(execution_report.symbol)
            
            if not position:
                # No existing position, this might be an opening trade
                return None
            
            position_side = position.get("side", "").upper()
            position_quantity = Decimal(str(position.get("quantity", "0")))
            entry_price = Decimal(str(position.get("average_entry_price", "0")))
            
            if position_quantity <= 0 or entry_price <= 0:
                return None
            
            # Check if this is a closing trade (opposite side of position)
            if position_side == execution_report.side.value:
                # Same side as position, this is adding to position
                return None
            
            # Calculate P&L for closing trade
            fill_price = execution_report.average_price
            fill_quantity = execution_report.filled_quantity
            
            if position_side == "BUY":
                # Long position being closed by a sell
                realized_pnl = (fill_price - entry_price) * fill_quantity
            else:
                # Short position being closed by a buy
                realized_pnl = (entry_price - fill_price) * fill_quantity
            
            # Subtract commission
            if execution_report.commission:
                realized_pnl -= execution_report.commission
            
            # Update execution report with calculated PnL
            execution_report.realized_pnl = realized_pnl
            
            return realized_pnl
            
        except Exception as e:
            self.logger.error(
                f"Error calculating realized P&L: {e}",
                source_module=self._source_module,
            )
            return None

    async def _update_risk_metrics_from_execution(
        self,
        execution_report: ExecutionReport,
        realized_pnl: Decimal,
        portfolio_state: dict[str, Any]
    ) -> None:
        """Update internal risk metrics based on execution."""
        try:
            # Update drawdown tracking
            current_equity = Decimal(str(portfolio_state.get("total_equity_usd", "0")))
            
            # Track peak equity for drawdown calculation
            if not hasattr(self, "_peak_equity"):
                self._peak_equity = current_equity
            else:
                self._peak_equity = max(self._peak_equity, current_equity)
            
            # Calculate current drawdown
            if self._peak_equity > 0:
                current_drawdown = self._peak_equity - current_equity
                current_drawdown_pct = (current_drawdown / self._peak_equity) * Decimal("100")
                
                # Update risk metrics object
                self._risk_metrics.current_drawdown = current_drawdown
                self._risk_metrics.current_drawdown_pct = current_drawdown_pct
                
                # Update max drawdown if necessary
                if not hasattr(self, "_max_drawdown"):
                    self._max_drawdown = current_drawdown
                    self._max_drawdown_pct = current_drawdown_pct
                    self._risk_metrics.max_drawdown = current_drawdown
                    self._risk_metrics.max_drawdown_pct = current_drawdown_pct
                else:
                    if current_drawdown > self._max_drawdown:
                        self._max_drawdown = current_drawdown
                        self._max_drawdown_pct = current_drawdown_pct
                        self._risk_metrics.max_drawdown = current_drawdown
                        self._risk_metrics.max_drawdown_pct = current_drawdown_pct
                
                # Log significant drawdowns
                if current_drawdown_pct > Decimal("5"):
                    self.logger.warning(
                        f"Significant drawdown detected: {current_drawdown_pct:.2f}%",
                        source_module=self._source_module,
                        context={
                            "current_equity": float(current_equity),
                            "peak_equity": float(self._peak_equity),
                            "drawdown": float(current_drawdown),
                            "drawdown_pct": float(current_drawdown_pct),
                        }
                    )
            
            # Update daily PnL if available
            if realized_pnl is not None:
                self._risk_metrics.daily_pnl += realized_pnl
                self._risk_metrics.total_realized_pnl += realized_pnl
            
            # Update last updated timestamp
            self._risk_metrics.last_updated = datetime.now(UTC)
            
        except Exception as e:
            self.logger.error(
                f"Error updating risk metrics: {e}",
                source_module=self._source_module,
            )

    async def _check_consecutive_loss_thresholds(self) -> None:
        """Check if consecutive loss thresholds are breached and take action."""
        try:
            # Check against configured maximum consecutive losses
            if self._consecutive_loss_count >= self._max_consecutive_losses:
                self.logger.error(
                    f"CRITICAL: Maximum consecutive losses reached: {self._consecutive_loss_count}",
                    source_module=self._source_module,
                )
                
                # Publish halt trigger event
                halt_event = {
                    "type": "PotentialHaltTrigger",
                    "source_module": self._source_module,
                    "reason": f"Maximum consecutive losses exceeded: {self._consecutive_loss_count}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "context": {
                        "consecutive_losses": self._consecutive_loss_count,
                        "max_allowed": self._max_consecutive_losses,
                        "recent_trades": self._recent_trades[-5:]  # Last 5 trades
                    }
                }
                
                await self.pubsub.publish(EventType.POTENTIAL_HALT_TRIGGER, halt_event)
                
                # Reduce risk parameters
                self._risk_per_trade_pct = max(
                    self._risk_per_trade_pct / Decimal("2"),
                    Decimal("0.1")  # Minimum 0.1%
                )
                
                self.logger.warning(
                    f"Risk per trade reduced to {self._risk_per_trade_pct:.2f}% due to consecutive losses",
                    source_module=self._source_module,
                )
                
        except Exception as e:
            self.logger.error(
                f"Error checking consecutive loss thresholds: {e}",
                source_module=self._source_module,
            )

    async def _publish_risk_metrics_update(
        self,
        execution_report: ExecutionReport,
        realized_pnl: Decimal
    ) -> None:
        """Publish risk metrics update event."""
        try:
            risk_metrics_event = {
                "type": "RiskMetricsUpdated",
                "source_module": self._source_module,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "consecutive_losses": self._consecutive_loss_count,
                    "consecutive_wins": self._consecutive_win_count,
                    "current_drawdown": float(self._risk_metrics.current_drawdown),
                    "current_drawdown_pct": float(self._risk_metrics.current_drawdown_pct),
                    "max_drawdown": float(self._risk_metrics.max_drawdown),
                    "max_drawdown_pct": float(self._risk_metrics.max_drawdown_pct),
                    "risk_per_trade_pct": float(self._risk_per_trade_pct),
                    "recent_trades_count": len(self._recent_trades),
                    "daily_pnl": float(self._risk_metrics.daily_pnl),
                    "total_realized_pnl": float(self._risk_metrics.total_realized_pnl),
                },
                "trigger_report": {
                    "order_id": execution_report.order_id,
                    "symbol": execution_report.symbol,
                    "realized_pnl": float(realized_pnl) if realized_pnl else None,
                }
            }
            
            # Publish to a generic event type or create a specific one
            await self.pubsub.publish("risk.metrics.updated", risk_metrics_event)
            
            # Also add to metrics history
            current_snapshot = RiskMetrics(
                consecutive_losses=self._consecutive_loss_count,
                consecutive_wins=self._consecutive_win_count,
                current_drawdown=self._risk_metrics.current_drawdown,
                current_drawdown_pct=self._risk_metrics.current_drawdown_pct,
                max_drawdown=self._risk_metrics.max_drawdown,
                max_drawdown_pct=self._risk_metrics.max_drawdown_pct,
                daily_pnl=self._risk_metrics.daily_pnl,
                total_realized_pnl=self._risk_metrics.total_realized_pnl,
                active_positions_count=self._risk_metrics.active_positions_count,
                total_exposure=self._risk_metrics.total_exposure,
                var_95=self._risk_metrics.var_95,
                expected_shortfall=self._risk_metrics.expected_shortfall,
                sharpe_ratio=self._risk_metrics.sharpe_ratio,
                win_rate=self._risk_metrics.win_rate,
            )
            self._metrics_history.append(current_snapshot)
            if len(self._metrics_history) > self._max_metrics_history:
                self._metrics_history.pop(0)
            
        except Exception as e:
            self.logger.error(
                f"Error publishing risk metrics update: {e}",
                source_module=self._source_module,
            )

    async def _risk_metrics_loop(self) -> None:
        """Periodically calculate and update comprehensive risk metrics.
        
        Enhanced implementation with VaR, Expected Shortfall, Sharpe ratio,
        and comprehensive drawdown/exposure monitoring.
        """
        while self._is_running:
            try:
                await asyncio.sleep(self._risk_metrics_interval_s)
                
                self.logger.debug(
                    "Starting periodic risk metrics calculation",
                    source_module=self._source_module
                )
                
                # Get current portfolio state
                portfolio_state = await self._portfolio_manager.get_current_state()
                if not portfolio_state:
                    self.logger.error(
                        "Could not get portfolio state for risk metrics calculation",
                        source_module=self._source_module,
                    )
                    continue
                
                # Extract key values
                current_equity = Decimal(str(portfolio_state.get("total_equity_usd", "0")))
                available_balance = Decimal(str(portfolio_state.get("available_balance_usd", "0")))
                initial_equity = Decimal(str(portfolio_state.get("initial_equity_usd", "0")))
                
                # Skip if no equity data
                if current_equity <= 0 or initial_equity <= 0:
                    self.logger.warning(
                        "Skipping risk metrics calculation - no valid equity data",
                        source_module=self._source_module,
                        context={
                            "current_equity": float(current_equity),
                            "initial_equity": float(initial_equity),
                        }
                    )
                    continue
                
                # Calculate portfolio metrics
                total_pnl = current_equity - initial_equity
                total_pnl_pct = (total_pnl / initial_equity) * Decimal("100")
                
                # Calculate exposure metrics with enhanced tracking
                positions = portfolio_state.get("positions", {})
                total_exposure = Decimal("0")
                open_positions_count = 0
                long_exposure = Decimal("0")
                short_exposure = Decimal("0")
                exposure_by_symbol = {}
                
                for symbol, position in positions.items():
                    try:
                        market_value = abs(Decimal(str(position.get("current_market_value", "0"))))
                        total_exposure += market_value
                        open_positions_count += 1
                        
                        side = position.get("side", "").upper()
                        if side == "BUY":
                            long_exposure += market_value
                        elif side == "SELL":
                            short_exposure += market_value
                            
                        # Track exposure by symbol for concentration analysis
                        exposure_by_symbol[symbol] = market_value
                    except (InvalidOperation, ValueError) as e:
                        self.logger.warning(
                            f"Invalid position value: {e}",
                            source_module=self._source_module,
                        )
                
                # Calculate exposure percentages
                total_exposure_pct = (total_exposure / current_equity) * Decimal("100") if current_equity > 0 else Decimal("0")
                long_exposure_pct = (long_exposure / current_equity) * Decimal("100") if current_equity > 0 else Decimal("0")
                short_exposure_pct = (short_exposure / current_equity) * Decimal("100") if current_equity > 0 else Decimal("0")
                
                # Calculate concentration risk (Herfindahl index)
                concentration_risk = Decimal("0")
                if total_exposure > 0:
                    for exposure in exposure_by_symbol.values():
                        weight = exposure / total_exposure
                        concentration_risk += weight ** 2
                
                # Calculate drawdown metrics
                if not hasattr(self, "_peak_equity"):
                    self._peak_equity = current_equity
                else:
                    self._peak_equity = max(self._peak_equity, current_equity)
                
                current_drawdown = self._peak_equity - current_equity if self._peak_equity > current_equity else Decimal("0")
                current_drawdown_pct = (current_drawdown / self._peak_equity) * Decimal("100") if self._peak_equity > 0 else Decimal("0")
                
                # Update max drawdown tracking
                if not hasattr(self, "_max_drawdown"):
                    self._max_drawdown = current_drawdown
                    self._max_drawdown_pct = current_drawdown_pct
                else:
                    if current_drawdown > self._max_drawdown:
                        self._max_drawdown = current_drawdown
                        self._max_drawdown_pct = current_drawdown_pct
                
                # Calculate win rate and profit/loss statistics
                win_rate = Decimal("0")
                avg_win = Decimal("0")
                avg_loss = Decimal("0")
                profit_factor = Decimal("0")
                returns_for_risk_metrics = []
                
                if hasattr(self, "_recent_trades") and self._recent_trades:
                    winning_trades = [t for t in self._recent_trades if t["realized_pnl"] > 0]
                    losing_trades = [t for t in self._recent_trades if t["realized_pnl"] < 0]
                    
                    total_trades = len(self._recent_trades)
                    if total_trades > 0:
                        win_rate = (Decimal(len(winning_trades)) / Decimal(total_trades)) * Decimal("100")
                    
                    if winning_trades:
                        total_wins = sum(Decimal(str(t["realized_pnl"])) for t in winning_trades)
                        avg_win = total_wins / len(winning_trades)
                    
                    if losing_trades:
                        total_losses = abs(sum(Decimal(str(t["realized_pnl"])) for t in losing_trades))
                        avg_loss = total_losses / len(losing_trades)
                        
                        # Calculate profit factor
                        if total_losses > 0 and winning_trades:
                            profit_factor = total_wins / total_losses
                    
                    # Extract returns for VaR and Sharpe calculations
                    if initial_equity > 0:
                        for trade in self._recent_trades:
                            pnl = Decimal(str(trade["realized_pnl"]))
                            return_pct = (pnl / initial_equity) * Decimal("100")
                            returns_for_risk_metrics.append(float(return_pct))
                
                # Calculate VaR and Expected Shortfall
                var_95 = Decimal("0")
                expected_shortfall = Decimal("0")
                
                if len(returns_for_risk_metrics) >= 30:  # Need sufficient data
                    returns_array = np.array(returns_for_risk_metrics)
                    
                    # 95% Value at Risk (5th percentile of returns)
                    var_95_value = np.percentile(returns_array, 5)
                    var_95 = abs(Decimal(str(var_95_value)))
                    
                    # Expected Shortfall (average of returns below VaR)
                    tail_returns = returns_array[returns_array <= var_95_value]
                    if len(tail_returns) > 0:
                        expected_shortfall = abs(Decimal(str(np.mean(tail_returns))))
                    else:
                        expected_shortfall = var_95
                
                # Calculate Sharpe ratio
                sharpe_ratio = Decimal("0")
                if len(returns_for_risk_metrics) >= 30:
                    returns_array = np.array(returns_for_risk_metrics)
                    mean_return = np.mean(returns_array)
                    std_return = np.std(returns_array)
                    
                    if std_return > 0:
                        # Annualize Sharpe ratio (assuming daily returns)
                        sharpe_ratio = Decimal(str((mean_return * 252) / (std_return * np.sqrt(252))))
                
                # Calculate risk score based on multiple factors
                risk_score = self._calculate_composite_risk_score(
                    current_drawdown_pct=current_drawdown_pct,
                    total_exposure_pct=total_exposure_pct,
                    consecutive_losses=self._consecutive_loss_count,
                    win_rate=win_rate,
                )
                
                # Update risk metrics object with all calculated values
                self._risk_metrics.consecutive_losses = self._consecutive_loss_count
                self._risk_metrics.consecutive_wins = self._consecutive_win_count
                self._risk_metrics.current_drawdown = current_drawdown
                self._risk_metrics.current_drawdown_pct = current_drawdown_pct
                self._risk_metrics.max_drawdown = self._max_drawdown
                self._risk_metrics.max_drawdown_pct = self._max_drawdown_pct
                self._risk_metrics.daily_pnl = Decimal("0")  # Reset daily PnL if new day
                self._risk_metrics.total_realized_pnl = total_pnl
                self._risk_metrics.active_positions_count = open_positions_count
                self._risk_metrics.total_exposure = total_exposure
                self._risk_metrics.var_95 = var_95
                self._risk_metrics.expected_shortfall = expected_shortfall
                self._risk_metrics.sharpe_ratio = sharpe_ratio
                self._risk_metrics.win_rate = win_rate
                self._risk_metrics.last_updated = datetime.now(UTC)
                
                # Add current metrics to history
                metrics_snapshot = RiskMetrics(
                    consecutive_losses=self._consecutive_loss_count,
                    consecutive_wins=self._consecutive_win_count,
                    current_drawdown=current_drawdown,
                    current_drawdown_pct=current_drawdown_pct,
                    max_drawdown=self._max_drawdown,
                    max_drawdown_pct=self._max_drawdown_pct,
                    daily_pnl=self._risk_metrics.daily_pnl,
                    total_realized_pnl=total_pnl,
                    active_positions_count=open_positions_count,
                    total_exposure=total_exposure,
                    var_95=var_95,
                    expected_shortfall=expected_shortfall,
                    sharpe_ratio=sharpe_ratio,
                    win_rate=win_rate
                )
                self._metrics_history.append(metrics_snapshot)
                if len(self._metrics_history) > self._max_metrics_history:
                    self._metrics_history.pop(0)
                
                # Persist metrics to database
                if hasattr(self, "_risk_metrics_repository"):
                    try:
                        await self._risk_metrics_repository.update_metrics(
                            consecutive_losses=self._consecutive_loss_count,
                            consecutive_wins=self._consecutive_win_count,
                            current_drawdown=current_drawdown,
                            current_drawdown_pct=current_drawdown_pct,
                            max_drawdown=self._max_drawdown,
                            max_drawdown_pct=self._max_drawdown_pct,
                            total_pnl=total_pnl,
                            total_pnl_pct=total_pnl_pct,
                            win_rate=win_rate,
                            avg_win=avg_win,
                            avg_loss=avg_loss,
                            profit_factor=profit_factor,
                            total_exposure=total_exposure,
                            total_exposure_pct=total_exposure_pct,
                            long_exposure=long_exposure,
                            long_exposure_pct=long_exposure_pct,
                            short_exposure=short_exposure,
                            short_exposure_pct=short_exposure_pct,
                            concentration_risk=concentration_risk,
                            open_positions_count=open_positions_count,
                            risk_score=risk_score,
                            current_equity=current_equity,
                            peak_equity=self._peak_equity,
                            var_95=var_95,
                            expected_shortfall=expected_shortfall,
                            sharpe_ratio=sharpe_ratio,
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to persist risk metrics: {e}",
                            source_module=self._source_module,
                        )
                
                # Publish comprehensive risk metrics event
                risk_metrics_event = {
                    "type": "RiskMetricsCalculated",
                    "source_module": self._source_module,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": {
                        # Loss/win tracking
                        "consecutive_losses": self._consecutive_loss_count,
                        "consecutive_wins": self._consecutive_win_count,
                        "win_rate": float(win_rate),
                        "avg_win": float(avg_win),
                        "avg_loss": float(avg_loss),
                        "profit_factor": float(profit_factor),
                        
                        # Drawdown metrics
                        "current_drawdown": float(current_drawdown),
                        "current_drawdown_pct": float(current_drawdown_pct),
                        "max_drawdown": float(self._max_drawdown),
                        "max_drawdown_pct": float(self._max_drawdown_pct),
                        
                        # P&L metrics
                        "total_pnl": float(total_pnl),
                        "total_pnl_pct": float(total_pnl_pct),
                        "daily_pnl": float(self._risk_metrics.daily_pnl),
                        "total_realized_pnl": float(self._risk_metrics.total_realized_pnl),
                        
                        # Exposure metrics
                        "total_exposure": float(total_exposure),
                        "total_exposure_pct": float(total_exposure_pct),
                        "long_exposure": float(long_exposure),
                        "long_exposure_pct": float(long_exposure_pct),
                        "short_exposure": float(short_exposure),
                        "short_exposure_pct": float(short_exposure_pct),
                        "concentration_risk": float(concentration_risk),
                        "exposure_by_symbol": {k: float(v) for k, v in exposure_by_symbol.items()},
                        
                        # Risk metrics
                        "var_95": float(var_95),
                        "expected_shortfall": float(expected_shortfall),
                        "sharpe_ratio": float(sharpe_ratio),
                        "risk_score": float(risk_score),
                        "risk_per_trade_pct": float(self._risk_per_trade_pct),
                        
                        # Portfolio state
                        "open_positions_count": open_positions_count,
                        "current_equity": float(current_equity),
                        "peak_equity": float(self._peak_equity),
                        "available_balance": float(available_balance),
                    }
                }
                
                await self.pubsub.publish("risk.metrics.calculated", risk_metrics_event)
                
                # Check for risk threshold breaches
                await self._check_risk_thresholds(
                    current_drawdown_pct=current_drawdown_pct,
                    total_exposure_pct=total_exposure_pct,
                    risk_score=risk_score,
                )
                
                self.logger.debug(
                    "Risk metrics calculated and published",
                    source_module=self._source_module,
                    context={
                        "drawdown_pct": float(current_drawdown_pct),
                        "exposure_pct": float(total_exposure_pct),
                        "risk_score": float(risk_score),
                        "consecutive_losses": self._consecutive_loss_count,
                    }
                )
                
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception(
                    "Error in risk metrics loop",
                    source_module=self._source_module,
                )

    async def _reject_signal(
        self,
        signal_id: uuid.UUID,
        event: TradeSignalProposedEvent,
        reason: str,
    ) -> None:
        """Enhanced signal rejection workflow with categorization and comprehensive logging."""
        try:
            # Categorize rejection reason and determine severity
            rejection_reason, severity = self._categorize_rejection_reason(reason)
            
            # Create comprehensive signal rejection event
            signal_rejection = SignalRejectionEvent(
                signal_id=signal_id,
                trading_pair=event.trading_pair,
                strategy_id=event.strategy_id,
                rejection_reason=rejection_reason,
                severity=severity,
                rejection_timestamp=datetime.now(UTC),
                signal_data={
                    "exchange": event.exchange,
                    "side": event.side,
                    "entry_type": event.entry_type,
                    "proposed_entry_price": event.proposed_entry_price,
                    "proposed_sl_price": event.proposed_sl_price,
                    "proposed_tp_price": event.proposed_tp_price,
                    "triggering_prediction_event_id": str(event.triggering_prediction_event_id) if event.triggering_prediction_event_id else None,
                },
                risk_metrics={
                    "consecutive_losses": float(self._consecutive_loss_count),
                    "current_drawdown_pct": float(self._risk_metrics.current_drawdown_pct),
                    "max_drawdown_pct": float(self._risk_metrics.max_drawdown_pct),
                    "risk_per_trade_pct": float(self._risk_per_trade_pct),
                    "risk_score": float(self._calculate_composite_risk_score(
                        self._risk_metrics.current_drawdown_pct,
                        self._risk_metrics.total_exposure / self._portfolio_manager.get_current_state().get("total_equity_usd", 1) * 100 if self._risk_metrics.total_exposure > 0 else Decimal("0"),
                        self._consecutive_loss_count,
                        self._risk_metrics.win_rate
                    )),
                },
                rejection_details=reason,
                auto_retry_eligible=self._is_auto_retry_eligible(rejection_reason)
            )
            
            # Log rejection decision with appropriate severity
            await self._log_rejection_decision(signal_rejection)
            
            # Publish rejection event
            await self._publish_rejection_event(signal_rejection)
            
            # Update rejection statistics
            self._update_rejection_statistics(signal_rejection)
            
            # Check if auto-retry should be scheduled
            if signal_rejection.auto_retry_eligible:
                await self._schedule_signal_retry(event, signal_rejection)
                
        except Exception as e:
            self.logger.error(
                f"Error in signal rejection workflow: {e}",
                source_module=self._source_module,
                context={
                    "signal_id": str(signal_id),
                    "reason": reason,
                }
            )
    
    def _categorize_rejection_reason(self, reason: str) -> Tuple[RejectionReason, RejectionSeverity]:
        """Categorize rejection reason and determine severity."""
        reason_lower = reason.lower()
        
        # Map reasons to categories
        if "insufficient" in reason_lower and "balance" in reason_lower:
            return RejectionReason.INSUFFICIENT_BALANCE, RejectionSeverity.HIGH
        elif "drawdown" in reason_lower:
            return RejectionReason.DRAWDOWN_LIMIT_BREACH, RejectionSeverity.CRITICAL
        elif "fat" in reason_lower and "finger" in reason_lower:
            return RejectionReason.FAT_FINGER_DETECTED, RejectionSeverity.MEDIUM
        elif "stop" in reason_lower and "loss" in reason_lower:
            return RejectionReason.INVALID_STOP_LOSS, RejectionSeverity.MEDIUM
        elif "position" in reason_lower and ("limit" in reason_lower or "exceed" in reason_lower):
            return RejectionReason.POSITION_LIMIT_EXCEEDED, RejectionSeverity.HIGH
        elif "exposure" in reason_lower:
            return RejectionReason.RISK_THRESHOLD_BREACH, RejectionSeverity.HIGH
        elif "confidence" in reason_lower:
            return RejectionReason.INSUFFICIENT_CONFIDENCE, RejectionSeverity.LOW
        elif "volatility" in reason_lower:
            return RejectionReason.VOLATILITY_TOO_HIGH, RejectionSeverity.MEDIUM
        elif "market" in reason_lower and "condition" in reason_lower:
            return RejectionReason.MARKET_CONDITION_INVALID, RejectionSeverity.MEDIUM
        else:
            return RejectionReason.TECHNICAL_ERROR, RejectionSeverity.LOW
    
    def _is_auto_retry_eligible(self, rejection_reason: RejectionReason) -> bool:
        """Determine if rejection reason is eligible for auto-retry."""
        # These reasons may be temporary and worth retrying
        retry_eligible_reasons = {
            RejectionReason.INSUFFICIENT_BALANCE,  # Balance might be replenished
            RejectionReason.MARKET_CONDITION_INVALID,  # Market conditions change
            RejectionReason.VOLATILITY_TOO_HIGH,  # Volatility may decrease
            RejectionReason.LIQUIDITY_INSUFFICIENT,  # Liquidity may improve
        }
        return rejection_reason in retry_eligible_reasons
    
    async def _log_rejection_decision(self, rejection_event: SignalRejectionEvent) -> None:
        """Comprehensive logging of rejection decision."""
        log_data = {
            'event_type': 'signal_rejection',
            'signal_id': str(rejection_event.signal_id),
            'trading_pair': rejection_event.trading_pair,
            'strategy_id': rejection_event.strategy_id,
            'rejection_reason': rejection_event.rejection_reason.value,
            'severity': rejection_event.severity.value,
            'timestamp': rejection_event.rejection_timestamp.isoformat(),
            'auto_retry_eligible': rejection_event.auto_retry_eligible,
            'risk_metrics': rejection_event.risk_metrics,
        }
        
        # Log at appropriate level based on severity
        if rejection_event.severity == RejectionSeverity.CRITICAL:
            self.logger.error(f"CRITICAL signal rejection: {log_data}")
        elif rejection_event.severity == RejectionSeverity.HIGH:
            self.logger.warning(f"HIGH severity signal rejection: {log_data}")
        else:
            self.logger.info(f"Signal rejection: {log_data}")
    
    async def _publish_rejection_event(self, rejection_event: SignalRejectionEvent) -> None:
        """Publish rejection event to interested subscribers."""
        event_data = {
            'type': 'TradeSignalRejected',
            'source_module': self._source_module,
            'signal_id': str(rejection_event.signal_id),
            'trading_pair': rejection_event.trading_pair,
            'strategy_id': rejection_event.strategy_id,
            'rejection_reason': rejection_event.rejection_reason.value,
            'severity': rejection_event.severity.value,
            'timestamp': rejection_event.rejection_timestamp.isoformat(),
            'rejection_details': rejection_event.rejection_details,
            'auto_retry_eligible': rejection_event.auto_retry_eligible,
            'signal_data': rejection_event.signal_data,
            'risk_metrics': rejection_event.risk_metrics,
        }
        
        # Publish to general rejection topic
        await self.pubsub.publish(EventType.TRADE_SIGNAL_REJECTED, event_data)
        
        # Publish to severity-specific topic for urgent attention
        if rejection_event.severity in [RejectionSeverity.HIGH, RejectionSeverity.CRITICAL]:
            await self.pubsub.publish(f'signals.rejected.{rejection_event.severity.value}', event_data)
        
        # Publish to strategy-specific topic for strategy optimization
        await self.pubsub.publish(f'signals.rejected.{rejection_event.strategy_id}', event_data)
    
    def _update_rejection_statistics(self, rejection_event: SignalRejectionEvent) -> None:
        """Update rejection statistics for monitoring."""
        self._rejection_stats['total_rejections'] += 1
        
        # Update by reason
        reason = rejection_event.rejection_reason.value
        if reason not in self._rejection_stats['rejections_by_reason']:
            self._rejection_stats['rejections_by_reason'][reason] = 0
        self._rejection_stats['rejections_by_reason'][reason] += 1
        
        # Update by symbol
        symbol = rejection_event.trading_pair
        if symbol not in self._rejection_stats['rejections_by_symbol']:
            self._rejection_stats['rejections_by_symbol'][symbol] = 0
        self._rejection_stats['rejections_by_symbol'][symbol] += 1
    
    async def _schedule_signal_retry(self, event: TradeSignalProposedEvent, rejection: SignalRejectionEvent) -> None:
        """Schedule signal for retry if eligible with intelligent backoff and condition checking."""
        try:
            # Get retry configuration
            retry_config = self._config.get("retry", {})
            max_retries = retry_config.get("max_attempts", 3)
            base_delay_seconds = retry_config.get("base_delay_seconds", 300)  # 5 minutes
            max_delay_seconds = retry_config.get("max_delay_seconds", 3600)   # 1 hour
            backoff_multiplier = retry_config.get("backoff_multiplier", 2.0)
            
            # Check if we have a retry tracking system
            if not hasattr(self, '_retry_queue'):
                self._retry_queue = {}
                self._retry_attempts = {}
                self._retry_task = asyncio.create_task(self._retry_processing_loop())
            
            signal_key = str(rejection.signal_id)
            
            # Check current retry count
            current_attempts = self._retry_attempts.get(signal_key, 0)
            if current_attempts >= max_retries:
                self.logger.warning(
                    f"Signal {rejection.signal_id} has exhausted retry attempts ({current_attempts}/{max_retries})",
                    source_module=self._source_module,
                    context={
                        "signal_id": str(rejection.signal_id),
                        "rejection_reason": rejection.rejection_reason.value,
                        "attempts": current_attempts,
                        "max_retries": max_retries,
                    }
                )
                await self._publish_retry_exhausted_event(event, rejection, current_attempts)
                return
            
            # Calculate retry delay with exponential backoff
            retry_delay = min(
                base_delay_seconds * (backoff_multiplier ** current_attempts),
                max_delay_seconds
            )
            
            # Add jitter to prevent thundering herd (±20% randomization)
            import random
            jitter = random.uniform(0.8, 1.2)
            retry_delay = int(retry_delay * jitter)
            
            # Schedule retry
            retry_time = datetime.now(UTC) + timedelta(seconds=retry_delay)
            
            retry_entry = {
                "signal_id": rejection.signal_id,
                "original_event": event,
                "rejection_event": rejection,
                "retry_time": retry_time,
                "attempt_number": current_attempts + 1,
                "created_at": datetime.now(UTC),
                "retry_reason": rejection.rejection_reason.value,
                "conditions_to_check": self._get_retry_conditions(rejection.rejection_reason),
            }
            
            self._retry_queue[signal_key] = retry_entry
            self._retry_attempts[signal_key] = current_attempts + 1
            
            # Publish retry scheduled event
            await self._publish_retry_scheduled_event(retry_entry)
            
            self.logger.info(
                f"Signal {rejection.signal_id} scheduled for retry #{retry_entry['attempt_number']} "
                f"in {retry_delay} seconds (at {retry_time.isoformat()}) due to {rejection.rejection_reason.value}",
                source_module=self._source_module,
                context={
                    "signal_id": str(rejection.signal_id),
                    "rejection_reason": rejection.rejection_reason.value,
                    "retry_delay_seconds": retry_delay,
                    "retry_time": retry_time.isoformat(),
                    "attempt_number": retry_entry['attempt_number'],
                    "max_retries": max_retries,
                    "conditions_to_check": retry_entry['conditions_to_check'],
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Error scheduling signal retry: {e}",
                source_module=self._source_module,
                context={
                    "signal_id": str(rejection.signal_id) if rejection else "unknown",
                    "rejection_reason": rejection.rejection_reason.value if rejection else "unknown",
                                 }
             )

    def _get_retry_conditions(self, rejection_reason: RejectionReason) -> List[str]:
        """Get list of conditions to check before retrying based on rejection reason."""
        condition_map = {
            RejectionReason.INSUFFICIENT_BALANCE: [
                "check_available_balance",
                "check_portfolio_equity",
            ],
            RejectionReason.MARKET_CONDITION_INVALID: [
                "check_market_volatility",
                "check_market_liquidity",
                "check_spread_conditions",
            ],
            RejectionReason.VOLATILITY_TOO_HIGH: [
                "check_market_volatility",
                "check_price_stability",
            ],
            RejectionReason.LIQUIDITY_INSUFFICIENT: [
                "check_market_liquidity",
                "check_orderbook_depth",
            ],
            RejectionReason.DRAWDOWN_LIMIT_BREACH: [
                "check_portfolio_recovery",
                "check_drawdown_levels",
            ],
            RejectionReason.POSITION_LIMIT_EXCEEDED: [
                "check_position_exposure",
                "check_portfolio_rebalancing",
            ],
            RejectionReason.RISK_THRESHOLD_BREACH: [
                "check_risk_metrics",
                "check_exposure_levels",
            ],
        }
        
        return condition_map.get(rejection_reason, ["check_general_conditions"])

    async def _retry_processing_loop(self) -> None:
        """Main loop to process scheduled retries."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not hasattr(self, '_retry_queue') or not self._retry_queue:
                    continue
                
                current_time = datetime.now(UTC)
                signals_to_retry = []
                
                # Find signals ready for retry
                for signal_key, retry_entry in list(self._retry_queue.items()):
                    if current_time >= retry_entry["retry_time"]:
                        signals_to_retry.append((signal_key, retry_entry))
                
                # Process ready retries
                for signal_key, retry_entry in signals_to_retry:
                    try:
                        await self._process_signal_retry(signal_key, retry_entry)
                    except Exception as e:
                        self.logger.error(
                            f"Error processing retry for signal {retry_entry['signal_id']}: {e}",
                            source_module=self._source_module,
                            exc_info=True
                        )
                        # Remove failed retry from queue
                        self._retry_queue.pop(signal_key, None)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error in retry processing loop: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )

    async def _process_signal_retry(self, signal_key: str, retry_entry: Dict[str, Any]) -> None:
        """Process a scheduled signal retry."""
        try:
            signal_id = retry_entry["signal_id"]
            original_event = retry_entry["original_event"]
            rejection_event = retry_entry["rejection_event"]
            attempt_number = retry_entry["attempt_number"]
            conditions_to_check = retry_entry["conditions_to_check"]
            
            self.logger.info(
                f"Processing retry #{attempt_number} for signal {signal_id}",
                source_module=self._source_module,
                context={
                    "signal_id": str(signal_id),
                    "attempt_number": attempt_number,
                    "original_rejection": rejection_event.rejection_reason.value,
                    "conditions_to_check": conditions_to_check,
                }
            )
            
            # Check if conditions have improved
            conditions_met = await self._check_retry_conditions(
                original_event, 
                rejection_event.rejection_reason, 
                conditions_to_check
            )
            
            if not conditions_met:
                self.logger.info(
                    f"Retry conditions not yet met for signal {signal_id}, will retry later",
                    source_module=self._source_module,
                    context={
                        "signal_id": str(signal_id),
                        "attempt_number": attempt_number,
                        "conditions_checked": conditions_to_check,
                    }
                )
                
                # Reschedule for later (double the delay)
                retry_config = self._config.get("retry", {})
                base_delay = retry_config.get("base_delay_seconds", 300)
                new_delay = base_delay * (2 ** (attempt_number - 1))
                new_retry_time = datetime.now(UTC) + timedelta(seconds=new_delay)
                
                retry_entry["retry_time"] = new_retry_time
                retry_entry["attempt_number"] = attempt_number  # Don't increment on condition check failure
                
                return  # Keep in queue for later retry
            
            # Remove from retry queue
            self._retry_queue.pop(signal_key, None)
            
            # Publish retry attempt event
            await self._publish_retry_attempt_event(retry_entry)
            
            # Reprocess the original signal
            self.logger.info(
                f"Conditions met, reprocessing signal {signal_id} (retry #{attempt_number})",
                source_module=self._source_module,
                context={
                    "signal_id": str(signal_id),
                    "attempt_number": attempt_number,
                    "original_rejection": rejection_event.rejection_reason.value,
                }
            )
            
            # Create a new signal event with retry metadata
            retry_event = TradeSignalProposedEvent(
                source_module=original_event.source_module,
                event_id=uuid.uuid4(),  # New event ID for the retry
                timestamp=datetime.now(UTC),
                signal_id=signal_id,  # Keep original signal ID
                trading_pair=original_event.trading_pair,
                exchange=original_event.exchange,
                side=original_event.side,
                entry_type=original_event.entry_type,
                proposed_entry_price=original_event.proposed_entry_price,
                proposed_sl_price=original_event.proposed_sl_price,
                proposed_tp_price=original_event.proposed_tp_price,
                strategy_id=original_event.strategy_id,
                triggering_prediction_event_id=original_event.triggering_prediction_event_id,
            )
            
            # Add retry metadata
            retry_event.metadata = getattr(original_event, 'metadata', {})
            retry_event.metadata.update({
                "is_retry": True,
                "retry_attempt": attempt_number,
                "original_rejection_reason": rejection_event.rejection_reason.value,
                "retry_conditions_checked": conditions_to_check,
            })
            
            # Resubmit for processing
            await self._handle_trade_signal_proposed(retry_event)
            
        except Exception as e:
            self.logger.error(
                f"Error in signal retry processing: {e}",
                source_module=self._source_module,
                context={
                    "signal_key": signal_key,
                    "retry_entry": retry_entry,
                },
                exc_info=True
            )
            # Remove failed retry from queue
            self._retry_queue.pop(signal_key, None)

    async def _check_retry_conditions(
        self, 
        original_event: TradeSignalProposedEvent, 
        rejection_reason: RejectionReason, 
        conditions: List[str]
    ) -> bool:
        """Check if conditions have improved enough to warrant a retry."""
        try:
            conditions_met = 0
            total_conditions = len(conditions)
            
            for condition in conditions:
                if await self._evaluate_retry_condition(condition, original_event, rejection_reason):
                    conditions_met += 1
            
            # Require at least 80% of conditions to be met
            success_threshold = 0.8
            success_rate = conditions_met / total_conditions if total_conditions > 0 else 0
            
            self.logger.debug(
                f"Retry condition check for signal {original_event.signal_id}: "
                f"{conditions_met}/{total_conditions} conditions met ({success_rate:.1%})",
                source_module=self._source_module,
                context={
                    "signal_id": str(original_event.signal_id),
                    "rejection_reason": rejection_reason.value,
                    "conditions_checked": conditions,
                    "conditions_met": conditions_met,
                    "total_conditions": total_conditions,
                    "success_rate": success_rate,
                    "threshold": success_threshold,
                }
            )
            
            return success_rate >= success_threshold
            
        except Exception as e:
            self.logger.error(
                f"Error checking retry conditions: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False

    async def _evaluate_retry_condition(
        self, 
        condition: str, 
        original_event: TradeSignalProposedEvent, 
        rejection_reason: RejectionReason
    ) -> bool:
        """Evaluate a specific retry condition."""
        try:
            if condition == "check_available_balance":
                portfolio_state = self._portfolio_manager.get_current_state()
                available_balance = Decimal(str(portfolio_state.get("available_balance_usd", "0")))
                required_balance = Decimal("1000")  # Minimum required balance
                return available_balance >= required_balance
                
            elif condition == "check_portfolio_equity":
                portfolio_state = self._portfolio_manager.get_current_state()
                current_equity = Decimal(str(portfolio_state.get("total_equity_usd", "0")))
                return current_equity > Decimal("0")
                
            elif condition == "check_market_volatility":
                # Check if volatility has decreased
                if self._dynamic_risk_volatility_feature_key:
                    current_vol = self._latest_volatility_features.get(original_event.trading_pair)
                    normal_vol = self._normal_volatility.get(original_event.trading_pair)
                    if current_vol is not None and normal_vol:
                        return current_vol <= float(normal_vol * Decimal("1.2"))  # 20% above normal
                return True  # Default to true if no volatility data
                
            elif condition == "check_market_liquidity":
                # Simplified liquidity check - could be enhanced with orderbook analysis
                return True  # Placeholder implementation
                
            elif condition == "check_spread_conditions":
                # Check if bid-ask spread is reasonable
                try:
                    current_price = await self._get_current_market_price(original_event.trading_pair)
                    if current_price:
                        # Assume spread is acceptable if we can get price
                        return True
                except Exception:
                    pass
                return False
                
            elif condition == "check_price_stability":
                # Check if price has stabilized
                return True  # Placeholder implementation
                
            elif condition == "check_orderbook_depth":
                # Check orderbook depth for liquidity
                return True  # Placeholder implementation
                
            elif condition == "check_portfolio_recovery":
                # Check if portfolio has recovered from drawdown
                current_drawdown = self._risk_metrics.current_drawdown_pct
                return current_drawdown < self._max_total_drawdown_pct * Decimal("0.8")  # 80% of max
                
            elif condition == "check_drawdown_levels":
                # Check current drawdown levels
                current_drawdown = self._risk_metrics.current_drawdown_pct
                return current_drawdown < self._max_total_drawdown_pct * Decimal("0.9")  # 90% of max
                
            elif condition == "check_position_exposure":
                # Check if position exposure has decreased
                portfolio_state = self._portfolio_manager.get_current_state()
                total_exposure = Decimal("0")
                positions = portfolio_state.get("positions", {})
                for position in positions.values():
                    total_exposure += abs(Decimal(str(position.get("current_market_value", "0"))))
                
                current_equity = Decimal(str(portfolio_state.get("total_equity_usd", "1")))
                exposure_pct = (total_exposure / current_equity) * Decimal("100") if current_equity > 0 else Decimal("0")
                return exposure_pct < self._max_total_exposure_pct * Decimal("0.8")  # 80% of max
                
            elif condition == "check_portfolio_rebalancing":
                # Check if portfolio has been rebalanced
                return True  # Placeholder implementation
                
            elif condition == "check_risk_metrics":
                # Check overall risk metrics
                risk_score = self._calculate_composite_risk_score(
                    self._risk_metrics.current_drawdown_pct,
                    Decimal("50"),  # Placeholder exposure
                    self._consecutive_loss_count,
                    self._risk_metrics.win_rate
                )
                return risk_score < Decimal("60")  # Below warning threshold
                
            elif condition == "check_exposure_levels":
                # Check exposure levels
                return self._risk_metrics.total_exposure < Decimal("50000")  # Placeholder threshold
                
            elif condition == "check_general_conditions":
                # General condition check - system is running and healthy
                return self._is_running and self._consecutive_loss_count < self._max_consecutive_losses
                
            else:
                self.logger.warning(
                    f"Unknown retry condition: {condition}",
                    source_module=self._source_module
                )
                return False
                
        except Exception as e:
            self.logger.error(
                f"Error evaluating retry condition {condition}: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False

    async def _publish_retry_scheduled_event(self, retry_entry: Dict[str, Any]) -> None:
        """Publish event when a retry is scheduled."""
        try:
            event_data = {
                "type": "SignalRetryScheduled",
                "source_module": self._source_module,
                "timestamp": datetime.now(UTC).isoformat(),
                "signal_id": str(retry_entry["signal_id"]),
                "trading_pair": retry_entry["original_event"].trading_pair,
                "strategy_id": retry_entry["original_event"].strategy_id,
                "retry_time": retry_entry["retry_time"].isoformat(),
                "attempt_number": retry_entry["attempt_number"],
                "retry_reason": retry_entry["retry_reason"],
                "conditions_to_check": retry_entry["conditions_to_check"],
                "created_at": retry_entry["created_at"].isoformat(),
            }
            
            await self.pubsub.publish("signals.retry.scheduled", event_data)
            
        except Exception as e:
            self.logger.error(
                f"Error publishing retry scheduled event: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _publish_retry_attempt_event(self, retry_entry: Dict[str, Any]) -> None:
        """Publish event when a retry attempt is made."""
        try:
            event_data = {
                "type": "SignalRetryAttempt",
                "source_module": self._source_module,
                "timestamp": datetime.now(UTC).isoformat(),
                "signal_id": str(retry_entry["signal_id"]),
                "trading_pair": retry_entry["original_event"].trading_pair,
                "strategy_id": retry_entry["original_event"].strategy_id,
                "attempt_number": retry_entry["attempt_number"],
                "retry_reason": retry_entry["retry_reason"],
                "conditions_checked": retry_entry["conditions_to_check"],
            }
            
            await self.pubsub.publish("signals.retry.attempt", event_data)
            
        except Exception as e:
            self.logger.error(
                f"Error publishing retry attempt event: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _publish_retry_exhausted_event(
        self, 
        original_event: TradeSignalProposedEvent, 
        rejection: SignalRejectionEvent, 
        attempts: int
    ) -> None:
        """Publish event when retry attempts are exhausted."""
        try:
            event_data = {
                "type": "SignalRetryExhausted",
                "source_module": self._source_module,
                "timestamp": datetime.now(UTC).isoformat(),
                "signal_id": str(rejection.signal_id),
                "trading_pair": original_event.trading_pair,
                "strategy_id": original_event.strategy_id,
                "total_attempts": attempts,
                "final_rejection_reason": rejection.rejection_reason.value,
                "exhausted_at": datetime.now(UTC).isoformat(),
            }
            
            await self.pubsub.publish("signals.retry.exhausted", event_data)
            
            # Clean up tracking for this signal
            signal_key = str(rejection.signal_id)
            self._retry_attempts.pop(signal_key, None)
            
        except Exception as e:
            self.logger.error(
                f"Error publishing retry exhausted event: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _approve_signal(
        self,
        signal_id: uuid.UUID,
        event: TradeSignalProposedEvent,
        approved_quantity: Decimal,
        approved_entry_price: Decimal | None,
    ) -> None:
        """Enhanced signal approval handling with comprehensive portfolio checks and event publication."""
        try:
            # Get current portfolio state for impact analysis
            portfolio_state = self._portfolio_manager.get_current_state()
            
            # Calculate risk amount and portfolio impact
            risk_amount = None
            if event.proposed_sl_price and approved_entry_price:
                sl_price = Decimal(str(event.proposed_sl_price))
                price_diff = abs(approved_entry_price - sl_price)
                risk_amount = approved_quantity * price_diff
            
            # Calculate portfolio impact
            portfolio_impact = await self._calculate_portfolio_impact(
                event.trading_pair,
                approved_quantity,
                approved_entry_price,
                portfolio_state
            )
            
            # Determine approval conditions and priority
            approval_conditions = await self._determine_approval_conditions(
                event,
                approved_quantity,
                portfolio_state
            )
            
            # Calculate execution priority
            execution_priority = self._calculate_execution_priority(
                event,
                self._risk_metrics.current_drawdown_pct,
                portfolio_impact
            )
            
            # Determine approval status
            approval_status = ApprovalStatus.APPROVED
            if approval_conditions:
                approval_status = ApprovalStatus.CONDITIONALLY_APPROVED
            
            # Create comprehensive approval event
            signal_approval = SignalApprovalEvent(
                signal_id=signal_id,
                trading_pair=event.trading_pair,
                strategy_id=event.strategy_id,
                approval_status=approval_status,
                approved_position_size=approved_quantity,
                original_position_size=Decimal(str(event.proposed_entry_price)) if event.proposed_entry_price else approved_quantity,
                approval_timestamp=datetime.now(UTC),
                approval_conditions=approval_conditions,
                portfolio_impact=portfolio_impact,
                risk_adjustments={
                    "risk_amount": float(risk_amount) if risk_amount else None,
                    "risk_per_trade_pct": float(self._risk_per_trade_pct),
                    "stop_loss_price": event.proposed_sl_price,
                    "take_profit_price": event.proposed_tp_price,
                },
                execution_priority=execution_priority,
                valid_until=datetime.now(UTC) + timedelta(hours=1)  # 1 hour validity
            )
            
            # Log approval decision
            await self._log_approval_decision(signal_approval, event)
            
            # Publish approval event
            await self._publish_approval_event(signal_approval, event)
            
            # Update approval statistics
            self._update_approval_statistics(signal_approval)
            
            # Handle post-approval actions
            await self._handle_post_approval_actions(signal_approval, event)
            
        except Exception as e:
            self.logger.error(
                f"Error in signal approval workflow: {e}",
                source_module=self._source_module,
                context={
                    "signal_id": str(signal_id),
                    "approved_quantity": float(approved_quantity),
                    "approved_entry_price": float(approved_entry_price) if approved_entry_price else None,
                }
            )
    
    async def _calculate_portfolio_impact(
        self,
        trading_pair: str,
        quantity: Decimal,
        price: Optional[Decimal],
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate the impact of the trade on the portfolio."""
        current_equity = Decimal(str(portfolio_state.get("total_equity_usd", "0")))
        
        # Calculate position value
        position_value = quantity * price if price else Decimal("0")
        
        # Calculate impact metrics
        portfolio_impact = {
            "position_value": float(position_value),
            "position_pct_of_portfolio": float((position_value / current_equity * 100) if current_equity > 0 else 0),
            "new_exposure_increase": float(position_value),
            "margin_impact": float(position_value * Decimal("0.1")),  # Assuming 10% margin requirement
        }
        
        # Check if this adds to existing position
        positions = portfolio_state.get("positions", {})
        if trading_pair in positions:
            existing_position = positions[trading_pair]
            existing_value = Decimal(str(existing_position.get("current_market_value", "0")))
            portfolio_impact["total_position_value"] = float(existing_value + position_value)
            portfolio_impact["position_increase_pct"] = float((position_value / existing_value * 100) if existing_value > 0 else 100)
        
        return portfolio_impact
    
    async def _determine_approval_conditions(
        self,
        event: TradeSignalProposedEvent,
        quantity: Decimal,
        portfolio_state: Dict[str, Any]
    ) -> List[str]:
        """Determine any conditions for the approval."""
        conditions = []
        
        # Check if position size was reduced
        if event.proposed_entry_price:
            original_qty = Decimal("1")  # Placeholder - should get from signal
            if quantity < original_qty:
                conditions.append("position_size_reduced_due_to_risk_limits")
        
        # Check if close to exposure limits
        current_exposure = Decimal(str(portfolio_state.get("total_exposure_usd", "0")))
        current_equity = Decimal(str(portfolio_state.get("total_equity_usd", "0")))
        if current_equity > 0:
            exposure_pct = (current_exposure / current_equity) * Decimal("100")
            if exposure_pct > self._max_total_exposure_pct * Decimal("0.8"):
                conditions.append("approaching_exposure_limit")
        
        # Check if in drawdown
        if self._risk_metrics.current_drawdown_pct > Decimal("5"):
            conditions.append("trading_during_drawdown")
        
        # Check consecutive losses
        if self._consecutive_loss_count > 2:
            conditions.append("consecutive_losses_detected")
        
        return conditions
    
    def _calculate_execution_priority(
        self,
        event: TradeSignalProposedEvent,
        current_drawdown_pct: Decimal,
        portfolio_impact: Dict[str, float]
    ) -> int:
        """Calculate execution priority (1-10, higher is more urgent)."""
        priority = 5  # Default medium priority
        
        # Increase priority for smaller positions (easier to fill)
        if portfolio_impact["position_pct_of_portfolio"] < 2:
            priority += 1
        
        # Decrease priority during drawdown
        if current_drawdown_pct > Decimal("10"):
            priority -= 2
        elif current_drawdown_pct > Decimal("5"):
            priority -= 1
        
        # Increase priority for limit orders (better pricing)
        if event.entry_type.upper() == "LIMIT":
            priority += 1
        
        # Ensure priority is within bounds
        priority = max(1, min(10, priority))
        
        return priority
    
    async def _log_approval_decision(self, approval_event: SignalApprovalEvent, original_event: TradeSignalProposedEvent) -> None:
        """Log approval decision with comprehensive details."""
        log_data = {
            'event_type': 'signal_approval',
            'signal_id': str(approval_event.signal_id),
            'trading_pair': approval_event.trading_pair,
            'strategy_id': approval_event.strategy_id,
            'approval_status': approval_event.approval_status.value,
            'approved_size': float(approval_event.approved_position_size),
            'execution_priority': approval_event.execution_priority,
            'conditions': approval_event.approval_conditions,
            'portfolio_impact': approval_event.portfolio_impact,
            'risk_adjustments': approval_event.risk_adjustments,
            'valid_until': approval_event.valid_until.isoformat(),
        }
        
        if approval_event.approval_status == ApprovalStatus.CONDITIONALLY_APPROVED:
            self.logger.warning(f"Conditional approval granted: {log_data}")
        else:
            self.logger.info(f"Signal approved: {log_data}")
    
    async def _publish_approval_event(self, approval_event: SignalApprovalEvent, original_event: TradeSignalProposedEvent) -> None:
        """Publish approval event to interested subscribers."""
        event_data = {
            'type': 'TradeSignalApproved',
            'source_module': self._source_module,
            'signal_id': str(approval_event.signal_id),
            'trading_pair': approval_event.trading_pair,
            'exchange': original_event.exchange,
            'side': original_event.side,
            'entry_type': original_event.entry_type,
            'strategy_id': approval_event.strategy_id,
            'approval_status': approval_event.approval_status.value,
            'approved_position_size': float(approval_event.approved_position_size),
            'original_position_size': float(approval_event.original_position_size),
            'approval_conditions': approval_event.approval_conditions,
            'portfolio_impact': approval_event.portfolio_impact,
            'risk_adjustments': approval_event.risk_adjustments,
            'execution_priority': approval_event.execution_priority,
            'valid_until': approval_event.valid_until.isoformat(),
            'timestamp': approval_event.approval_timestamp.isoformat(),
        }
        
        # Publish to general approval topic
        await self.pubsub.publish(EventType.TRADE_SIGNAL_APPROVED, event_data)
        
        # Publish to execution service
        await self.pubsub.publish('execution.signals.approved', event_data)
        
        # Publish to portfolio manager for position tracking
        await self.pubsub.publish('portfolio.signals.approved', event_data)
        
        # Publish to strategy-specific topic
        await self.pubsub.publish(f'signals.approved.{approval_event.strategy_id}', event_data)
    
    def _update_approval_statistics(self, approval_event: SignalApprovalEvent) -> None:
        """Update approval statistics for monitoring."""
        self._approval_stats['total_approvals'] += 1
        
        if approval_event.approval_status == ApprovalStatus.CONDITIONALLY_APPROVED:
            self._approval_stats['conditional_approvals'] += 1
        
        # Update by symbol
        symbol = approval_event.trading_pair
        if symbol not in self._approval_stats['approvals_by_symbol']:
            self._approval_stats['approvals_by_symbol'][symbol] = 0
        self._approval_stats['approvals_by_symbol'][symbol] += 1
    
    async def _handle_post_approval_actions(self, approval_event: SignalApprovalEvent, original_event: TradeSignalProposedEvent) -> None:
        """Handle any post-approval actions."""
        # Log approval metrics
        self.logger.info(
            f"Trade signal approved with priority {approval_event.execution_priority}",
            source_module=self._source_module,
            context={
                "signal_id": str(approval_event.signal_id),
                "trading_pair": approval_event.trading_pair,
                "approved_quantity": float(approval_event.approved_position_size),
                "conditions": len(approval_event.approval_conditions),
                "portfolio_impact_pct": approval_event.portfolio_impact.get("position_pct_of_portfolio", 0),
            }
        )

    async def _stage1_initial_validation_and_price_rounding(
        self,
        ctx: Stage1Context,
    ) -> tuple[Decimal | None, Decimal, Decimal | None]:
        """Stage 1: Initial validation and price rounding."""
        try:
            # Create PriceRoundingContext for the helper method
            price_rounding_ctx = PriceRoundingContext(
                entry_type=ctx.event.entry_type,
                side=ctx.event.side,
                trading_pair=ctx.event.trading_pair,
                effective_entry_price=ctx.proposed_entry_price_decimal,
                sl_price=ctx.proposed_sl_price_decimal,
                tp_price=ctx.proposed_tp_price_decimal,
            )
            
            # Use existing price calculation and validation method
            (
                is_valid,
                error_msg,
                rounded_entry_price,
                rounded_sl_price,
                rounded_tp_price,
            ) = self._calculate_and_validate_prices(price_rounding_ctx)
            
            if not is_valid:
                self._validate_and_raise_if_error(
                    error_condition=True,
                    failure_reason=error_msg or "Price validation failed",
                    stage_name="Stage1_InitialValidation",
                    log_message=f"Stage 1 validation failed: {error_msg}",
                    log_context={
                        "signal_id": str(ctx.event.signal_id),
                        "trading_pair": ctx.event.trading_pair,
                        "entry_type": ctx.event.entry_type,
                        "proposed_entry": float(ctx.proposed_entry_price_decimal) if ctx.proposed_entry_price_decimal else None,
                        "proposed_sl": float(ctx.proposed_sl_price_decimal) if ctx.proposed_sl_price_decimal else None,
                        "proposed_tp": float(ctx.proposed_tp_price_decimal) if ctx.proposed_tp_price_decimal else None,
                    }
                )
            
            # Ensure we have a valid stop loss
            if rounded_sl_price is None:
                self._validate_and_raise_if_error(
                    error_condition=True,
                    failure_reason="Stop loss price is required",
                    stage_name="Stage1_InitialValidation",
                    log_message="Stage 1 validation failed: No stop loss price",
                    log_context={
                        "signal_id": str(ctx.event.signal_id),
                        "trading_pair": ctx.event.trading_pair,
                    }
                )
            
            self.logger.debug(
                "Stage 1 validation completed",
                source_module=self._source_module,
                context={
                    "signal_id": str(ctx.event.signal_id),
                    "rounded_entry": float(rounded_entry_price) if rounded_entry_price else None,
                    "rounded_sl": float(rounded_sl_price) if rounded_sl_price else None,
                    "rounded_tp": float(rounded_tp_price) if rounded_tp_price else None,
                }
            )
            
            return rounded_entry_price, rounded_sl_price, rounded_tp_price
            
        except SignalValidationStageError:
            # Re-raise validation errors
            raise
        except Exception as e:
            self.logger.exception(
                "Unexpected error in Stage 1 validation",
                source_module=self._source_module,
                context={"signal_id": str(ctx.event.signal_id)},
            )
            self._validate_and_raise_if_error(
                error_condition=True,
                failure_reason=f"Stage 1 error: {str(e)}",
                stage_name="Stage1_InitialValidation",
                log_message=f"Stage 1 unexpected error: {str(e)}",
                log_context={"signal_id": str(ctx.event.signal_id)},
            )

    async def _get_current_market_price(self, trading_pair: str) -> Decimal | None:
        """Get current market price for a trading pair."""
        return await self._market_price_service.get_latest_price(trading_pair)

    async def _stage2_market_price_dependent_checks(
        self,
        ctx: Stage2Context,
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Stage 2: Market price dependent checks (fat finger, SL distance)."""
        try:
            # Determine effective entry price for non-limit orders
            effective_entry_price_for_non_limit = None
            if ctx.event.entry_type != "LIMIT":
                # For market orders, use current market price
                if ctx.current_market_price_for_validation is not None:
                    effective_entry_price_for_non_limit = ctx.current_market_price_for_validation
                else:
                    self._validate_and_raise_if_error(
                        error_condition=True,
                        failure_reason="Market price unavailable for non-limit order",
                        stage_name="Stage2_MarketPriceChecks",
                        log_message="Stage 2 validation failed: No market price for non-limit order",
                        log_context={
                            "signal_id": str(ctx.event.signal_id),
                            "trading_pair": ctx.event.trading_pair,
                            "entry_type": ctx.event.entry_type,
                        }
                    )
            
            # Create validation context
            price_validation_ctx = PriceValidationContext(
                event=ctx.event,
                entry_type=ctx.event.entry_type,
                side=ctx.event.side,
                rounded_entry_price=ctx.rounded_entry_price,
                rounded_sl_price=ctx.rounded_sl_price,
                effective_entry_price_for_non_limit=effective_entry_price_for_non_limit,
                current_market_price=ctx.current_market_price_for_validation,
            )
            
            # Perform fat finger and SL distance validation
            is_valid, error_msg = self._validate_prices_fat_finger_and_sl_distance(price_validation_ctx)
            
            if not is_valid:
                self._validate_and_raise_if_error(
                    error_condition=True,
                    failure_reason=error_msg or "Price validation failed",
                    stage_name="Stage2_MarketPriceChecks",
                    log_message=f"Stage 2 validation failed: {error_msg}",
                    log_context={
                        "signal_id": str(ctx.event.signal_id),
                        "trading_pair": ctx.event.trading_pair,
                        "entry_type": ctx.event.entry_type,
                        "rounded_entry_price": float(ctx.rounded_entry_price) if ctx.rounded_entry_price else None,
                        "rounded_sl_price": float(ctx.rounded_sl_price),
                        "current_market_price": float(ctx.current_market_price_for_validation) if ctx.current_market_price_for_validation else None,
                    }
                )
            
            # Determine reference entry price for calculations
            ref_entry_for_calculation = ctx.rounded_entry_price
            if ref_entry_for_calculation is None and effective_entry_price_for_non_limit is not None:
                ref_entry_for_calculation = effective_entry_price_for_non_limit
            
            # Final check that we have a valid reference price
            if ref_entry_for_calculation is None:
                self._validate_and_raise_if_error(
                    error_condition=True,
                    failure_reason="No valid entry price for calculations",
                    stage_name="Stage2_MarketPriceChecks",
                    log_message="Stage 2 validation failed: No valid entry price",
                    log_context={
                        "signal_id": str(ctx.event.signal_id),
                        "trading_pair": ctx.event.trading_pair,
                        "entry_type": ctx.event.entry_type,
                    }
                )
            
            self.logger.debug(
                "Stage 2 validation completed",
                source_module=self._source_module,
                context={
                    "signal_id": str(ctx.event.signal_id),
                    "effective_entry_price": float(effective_entry_price_for_non_limit) if effective_entry_price_for_non_limit else None,
                    "ref_entry_for_calculation": float(ref_entry_for_calculation) if ref_entry_for_calculation else None,
                }
            )
            
            return effective_entry_price_for_non_limit, ref_entry_for_calculation, ctx.rounded_entry_price
            
        except SignalValidationStageError:
            # Re-raise validation errors
            raise
        except Exception as e:
            self.logger.exception(
                "Unexpected error in Stage 2 validation",
                source_module=self._source_module,
                context={"signal_id": str(ctx.event.signal_id)},
            )
            self._validate_and_raise_if_error(
                error_condition=True,
                failure_reason=f"Stage 2 error: {str(e)}",
                stage_name="Stage2_MarketPriceChecks",
                log_message=f"Stage 2 unexpected error: {str(e)}",
                log_context={"signal_id": str(ctx.event.signal_id)},
            )

    def _validate_and_raise_if_error(
        self,
        error_condition: bool,
        failure_reason: str,
        stage_name: str,
        log_message: str,
        log_context: dict[str, Any] | None = None,
    ) -> None:
        """Validate condition and raise SignalValidationStageError if error."""
        if error_condition:
            self.logger.error(
                log_message,
                source_module=self._source_module,
                context=log_context,
            )
            raise SignalValidationStageError(failure_reason, stage_name)

    async def _stage3_position_sizing_and_portfolio_checks(
        self,
        ctx: Stage3Context,
    ) -> tuple[Decimal, dict[str, Decimal]]:
        """Stage 3: Position sizing and portfolio checks."""
        try:
            # Calculate position size
            sizing_result = self._calculate_and_validate_position_size(
                event=ctx.event,
                current_equity=ctx.current_equity_decimal,
                ref_entry_price=ctx.ref_entry_for_calculation,
                rounded_sl_price=ctx.rounded_sl_price,
                portfolio_state=ctx.portfolio_state,
            )
            
            if not sizing_result.is_valid:
                self._validate_and_raise_if_error(
                    error_condition=True,
                    failure_reason=sizing_result.rejection_reason or "Position sizing failed",
                    stage_name="Stage3_PositionSizing",
                    log_message=f"Stage 3 validation failed: {sizing_result.rejection_reason}",
                    log_context={
                        "signal_id": str(ctx.event.signal_id),
                        "trading_pair": ctx.event.trading_pair,
                        "current_equity": float(ctx.current_equity_decimal),
                        "ref_entry_price": float(ctx.ref_entry_for_calculation),
                        "rounded_sl_price": float(ctx.rounded_sl_price),
                    }
                )
            
            initial_calculated_qty = sizing_result.quantity
            
            # Check position scaling if needed
            position_scaling_ctx = PositionScalingContext(
                signal_id=ctx.event.signal_id,
                trading_pair=ctx.event.trading_pair,
                side=ctx.event.side,
                ref_entry_price=ctx.ref_entry_for_calculation,
                portfolio_state=ctx.portfolio_state,
                initial_calculated_qty=initial_calculated_qty,
            )
            
            is_valid, error_msg, position_action, final_quantity = self._check_position_scaling(position_scaling_ctx)
            
            if not is_valid:
                self._validate_and_raise_if_error(
                    error_condition=True,
                    failure_reason=error_msg or "Position scaling check failed",
                    stage_name="Stage3_PositionScaling",
                    log_message=f"Stage 3 scaling check failed: {error_msg}",
                    log_context={
                        "signal_id": str(ctx.event.signal_id),
                        "trading_pair": ctx.event.trading_pair,
                        "initial_qty": float(initial_calculated_qty),
                        "position_action": position_action,
                    }
                )
            
            # Extract relevant portfolio values for state tracking
            state_values = self._extract_relevant_portfolio_values(ctx.portfolio_state)
            
            # Add additional metrics to state values
            state_values["calculated_risk_amount"] = sizing_result.risk_amount or Decimal("0")
            state_values["calculated_position_value"] = sizing_result.position_value or Decimal("0")
            state_values["position_action"] = Decimal("1") if position_action == "NEW_POSITION" else Decimal("0")
            
            self.logger.debug(
                "Stage 3 validation completed",
                source_module=self._source_module,
                context={
                    "signal_id": str(ctx.event.signal_id),
                    "initial_calculated_qty": float(initial_calculated_qty),
                    "final_quantity": float(final_quantity) if final_quantity else None,
                    "position_action": position_action,
                    "risk_amount": float(sizing_result.risk_amount) if sizing_result.risk_amount else None,
                    "position_value": float(sizing_result.position_value) if sizing_result.position_value else None,
                }
            )
            
            return final_quantity or initial_calculated_qty, state_values
            
        except SignalValidationStageError:
            # Re-raise validation errors
            raise
        except Exception as e:
            self.logger.exception(
                "Unexpected error in Stage 3 validation",
                source_module=self._source_module,
                context={"signal_id": str(ctx.event.signal_id)},
            )
            self._validate_and_raise_if_error(
                error_condition=True,
                failure_reason=f"Stage 3 error: {str(e)}",
                stage_name="Stage3_PositionSizing",
                log_message=f"Stage 3 unexpected error: {str(e)}",
                log_context={"signal_id": str(ctx.event.signal_id)},
            )

    def _calculate_lot_size_with_fallback(
        self,
        raw_quantity: Decimal,
        trading_pair: str,
    ) -> tuple[Decimal, bool]:
        """Calculate lot size with fallback logic.
        
        Args:
            raw_quantity: Raw calculated quantity
            trading_pair: Trading pair symbol
            
        Returns:
            Tuple of (rounded_quantity, success_flag)
        """
        try:
            # Try to get step size from exchange info service
            step_size = None
            min_quantity = None
            max_quantity = None
            
            try:
                step_size = self._exchange_info_service.get_step_size(trading_pair)
                # Try to get min/max quantity if available
                if hasattr(self._exchange_info_service, "get_min_quantity"):
                    min_quantity = self._exchange_info_service.get_min_quantity(trading_pair)
                if hasattr(self._exchange_info_service, "get_max_quantity"):
                    max_quantity = self._exchange_info_service.get_max_quantity(trading_pair)
            except AttributeError:
                self.logger.warning(
                    "ExchangeInfoService missing expected methods",
                    source_module=self._source_module,
                    context={"trading_pair": trading_pair}
                )
            except Exception as e:
                self.logger.warning(
                    f"Error getting exchange info for {trading_pair}: {e}",
                    source_module=self._source_module,
                )
            
            # Use default step size if not available
            if not step_size or step_size <= 0:
                # Common default step sizes for crypto
                if "BTC" in trading_pair:
                    step_size = Decimal("0.00001")  # 5 decimal places for BTC pairs
                elif "USD" in trading_pair or "USDT" in trading_pair:
                    step_size = Decimal("0.01")  # 2 decimal places for USD pairs
                else:
                    step_size = Decimal("0.001")  # 3 decimal places as default
                
                self.logger.debug(
                    f"Using default step size {step_size} for {trading_pair}",
                    source_module=self._source_module,
                )
            
            # Round down to step size
            if step_size > 0:
                rounded_qty = (raw_quantity // step_size) * step_size
            else:
                rounded_qty = raw_quantity
            
            # Apply min/max constraints if available
            if min_quantity and rounded_qty < min_quantity:
                self.logger.warning(
                    f"Quantity {rounded_qty} below minimum {min_quantity} for {trading_pair}",
                    source_module=self._source_module,
                )
                return Decimal("0"), False
            
            if max_quantity and rounded_qty > max_quantity:
                self.logger.warning(
                    f"Quantity {rounded_qty} above maximum {max_quantity} for {trading_pair}, capping",
                    source_module=self._source_module,
                )
                rounded_qty = (max_quantity // step_size) * step_size
            
            # Final validation
            if rounded_qty <= 0:
                self.logger.warning(
                    f"Rounded quantity is zero or negative for {trading_pair}",
                    source_module=self._source_module,
                    context={
                        "raw_quantity": float(raw_quantity),
                        "step_size": float(step_size),
                        "rounded_qty": float(rounded_qty),
                    }
                )
                return Decimal("0"), False
            
            self.logger.debug(
                f"Lot size calculated for {trading_pair}",
                source_module=self._source_module,
                context={
                    "raw_quantity": float(raw_quantity),
                    "rounded_quantity": float(rounded_qty),
                    "step_size": float(step_size),
                }
            )
            
            return rounded_qty, True
            
        except Exception as e:
            self.logger.error(
                f"Error calculating lot size: {e}",
                source_module=self._source_module,
                context={
                    "trading_pair": trading_pair,
                    "raw_quantity": float(raw_quantity),
                }
            )
            # Return raw quantity as fallback
            return raw_quantity, False

    def _calculate_composite_risk_score(
        self,
        current_drawdown_pct: Decimal,
        total_exposure_pct: Decimal,
        consecutive_losses: int,
        win_rate: Decimal,
    ) -> Decimal:
        """Calculate a composite risk score from 0-100 based on multiple factors.
        
        Higher scores indicate higher risk levels.
        
        Args:
            current_drawdown_pct: Current drawdown percentage
            total_exposure_pct: Total portfolio exposure percentage
            consecutive_losses: Number of consecutive losing trades
            win_rate: Win rate percentage over recent trades
            
        Returns:
            Risk score from 0 to 100
        """
        try:
            # Weight factors (sum to 1.0)
            drawdown_weight = Decimal("0.4")
            exposure_weight = Decimal("0.3")
            loss_streak_weight = Decimal("0.2")
            win_rate_weight = Decimal("0.1")
            
            # Normalize drawdown (0-100 scale)
            # Max drawdown threshold for normalization
            max_drawdown_threshold = self._max_total_drawdown_pct
            drawdown_score = min(
                (current_drawdown_pct / max_drawdown_threshold) * Decimal("100"),
                Decimal("100")
            )
            
            # Normalize exposure (0-100 scale)
            # Max exposure threshold for normalization
            max_exposure_threshold = self._max_total_exposure_pct
            exposure_score = min(
                (total_exposure_pct / max_exposure_threshold) * Decimal("100"),
                Decimal("100")
            )
            
            # Normalize consecutive losses (0-100 scale)
            max_loss_threshold = Decimal(str(self._max_consecutive_losses))
            loss_streak_score = min(
                (Decimal(str(consecutive_losses)) / max_loss_threshold) * Decimal("100"),
                Decimal("100")
            )
            
            # Normalize win rate (inverted - lower win rate = higher risk)
            # 0% win rate = 100 risk score, 100% win rate = 0 risk score
            win_rate_score = Decimal("100") - win_rate
            
            # Calculate weighted composite score
            composite_score = (
                drawdown_score * drawdown_weight +
                exposure_score * exposure_weight +
                loss_streak_score * loss_streak_weight +
                win_rate_score * win_rate_weight
            )
            
            # Apply non-linear scaling for extreme cases
            if composite_score > Decimal("80"):
                # Amplify high risk scores
                composite_score = Decimal("80") + (composite_score - Decimal("80")) * Decimal("1.5")
            elif composite_score < Decimal("20"):
                # Dampen low risk scores
                composite_score = composite_score * Decimal("0.8")
            
            # Ensure score is within bounds
            composite_score = max(Decimal("0"), min(Decimal("100"), composite_score))
            
            self.logger.debug(
                "Calculated composite risk score",
                source_module=self._source_module,
                context={
                    "risk_score": float(composite_score),
                    "drawdown_score": float(drawdown_score),
                    "exposure_score": float(exposure_score),
                    "loss_streak_score": float(loss_streak_score),
                    "win_rate_score": float(win_rate_score),
                }
            )
            
            return composite_score
            
        except Exception as e:
            self.logger.error(
                f"Error calculating composite risk score: {e}",
                source_module=self._source_module,
            )
            # Return high risk score on error
            return Decimal("75")

    async def _check_risk_thresholds(
        self,
        current_drawdown_pct: Decimal,
        total_exposure_pct: Decimal,
        risk_score: Decimal,
    ) -> None:
        """Check if risk thresholds are breached and publish alerts.
        
        Args:
            current_drawdown_pct: Current drawdown percentage
            total_exposure_pct: Total portfolio exposure percentage
            risk_score: Composite risk score (0-100)
        """
        try:
            alerts = []
            
            # Check critical risk score threshold (>80)
            if risk_score > Decimal("80"):
                alerts.append({
                    "level": "CRITICAL",
                    "type": "HIGH_RISK_SCORE",
                    "message": f"Critical risk score: {risk_score:.2f}",
                    "threshold": 80,
                    "value": float(risk_score),
                })
                
                # Auto-reduce risk parameters
                self._risk_per_trade_pct = max(
                    self._risk_per_trade_pct * Decimal("0.5"),
                    Decimal("0.1")
                )
                self.logger.warning(
                    f"Auto-reduced risk per trade to {self._risk_per_trade_pct:.2f}% due to critical risk score",
                    source_module=self._source_module,
                )
            
            # Check warning risk score threshold (>60)
            elif risk_score > Decimal("60"):
                alerts.append({
                    "level": "WARNING",
                    "type": "ELEVATED_RISK_SCORE",
                    "message": f"Elevated risk score: {risk_score:.2f}",
                    "threshold": 60,
                    "value": float(risk_score),
                })
            
            # Check drawdown approaching max
            drawdown_warning_threshold = self._max_total_drawdown_pct * Decimal("0.8")  # 80% of max
            if current_drawdown_pct > drawdown_warning_threshold:
                alerts.append({
                    "level": "WARNING",
                    "type": "HIGH_DRAWDOWN",
                    "message": f"Drawdown {current_drawdown_pct:.2f}% approaching max {self._max_total_drawdown_pct:.2f}%",
                    "threshold": float(drawdown_warning_threshold),
                    "value": float(current_drawdown_pct),
                })
            
            # Check exposure approaching max
            exposure_warning_threshold = self._max_total_exposure_pct * Decimal("0.9")  # 90% of max
            if total_exposure_pct > exposure_warning_threshold:
                alerts.append({
                    "level": "WARNING",
                    "type": "HIGH_EXPOSURE",
                    "message": f"Exposure {total_exposure_pct:.2f}% approaching max {self._max_total_exposure_pct:.2f}%",
                    "threshold": float(exposure_warning_threshold),
                    "value": float(total_exposure_pct),
                })
            
            # Publish alerts if any
            if alerts:
                alert_event = {
                    "type": "RiskThresholdAlerts",
                    "source_module": self._source_module,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alerts": alerts,
                    "metrics": {
                        "risk_score": float(risk_score),
                        "drawdown_pct": float(current_drawdown_pct),
                        "exposure_pct": float(total_exposure_pct),
                        "consecutive_losses": self._consecutive_loss_count,
                    }
                }
                
                await self.pubsub.publish("risk.threshold.alerts", alert_event)
                
                # Log critical alerts
                for alert in alerts:
                    if alert["level"] == "CRITICAL":
                        self.logger.error(
                            f"CRITICAL RISK ALERT: {alert['message']}",
                            source_module=self._source_module,
                            context=alert,
                        )
                    else:
                        self.logger.warning(
                            f"Risk alert: {alert['message']}",
                            source_module=self._source_module,
                            context=alert,
                        )
                        
        except Exception as e:
            self.logger.error(
                f"Error checking risk thresholds: {e}",
                source_module=self._source_module,
            )

    async def _parse_execution_report(self, event_dict: Dict[str, Any]) -> Optional[ExecutionReport]:
        """Parse execution report from various formats into standardized model."""
        try:
            # Handle different report formats
            if "event_type" in event_dict and event_dict["event_type"] == EventType.EXECUTION_REPORT.name:
                # Standard ExecutionReportEvent format
                event = ExecutionReportEvent.from_dict(event_dict)
                return ExecutionReport(
                    order_id=event.exchange_order_id,
                    symbol=event.trading_pair,
                    side=OrderSide(event.side.upper()),
                    status=OrderStatus(event.order_status.upper()),
                    filled_quantity=Decimal(str(event.quantity_filled)),
                    average_price=Decimal(str(event.average_fill_price)),
                    commission=Decimal(str(event.commission)) if event.commission else Decimal("0"),
                    realized_pnl=None,  # Will be calculated separately
                    timestamp=event.timestamp,
                    signal_id=str(event.signal_id) if hasattr(event, 'signal_id') else None,
                    strategy_id=event.strategy_id if hasattr(event, 'strategy_id') else None,
                )
            elif "txid" in event_dict:
                # Kraken format
                return self._parse_kraken_execution_report(event_dict)
            elif "simulation_id" in event_dict:
                # Simulated format
                return self._parse_simulated_execution_report(event_dict)
            else:
                # Try to parse as generic format
                return self._parse_generic_execution_report(event_dict)
                
        except Exception as e:
            self.logger.warning(
                f"Failed to parse execution report: {e}",
                source_module=self._source_module,
                context={"event_dict": event_dict}
            )
            return None

    def _parse_kraken_execution_report(self, raw_report: Dict[str, Any]) -> Optional[ExecutionReport]:
        """Parse Kraken-specific execution report format."""
        try:
            return ExecutionReport(
                order_id=raw_report.get("txid", ""),
                symbol=raw_report.get("pair", ""),
                side=OrderSide.BUY if raw_report.get("type", "").lower() == "buy" else OrderSide.SELL,
                status=OrderStatus.FILLED if raw_report.get("status", "") == "closed" else OrderStatus.PARTIALLY_FILLED,
                filled_quantity=Decimal(str(raw_report.get("vol_exec", "0"))),
                average_price=Decimal(str(raw_report.get("price", "0"))),
                commission=Decimal(str(raw_report.get("fee", "0"))),
                timestamp=datetime.fromtimestamp(raw_report.get("closetm", 0), tz=UTC),
            )
        except Exception as e:
            self.logger.error(f"Error parsing Kraken execution report: {e}")
            return None

    def _parse_simulated_execution_report(self, raw_report: Dict[str, Any]) -> Optional[ExecutionReport]:
        """Parse simulated execution report format."""
        try:
            return ExecutionReport(
                order_id=raw_report.get("simulation_id", ""),
                symbol=raw_report.get("symbol", ""),
                side=OrderSide(raw_report.get("side", "BUY").upper()),
                status=OrderStatus(raw_report.get("status", "FILLED").upper()),
                filled_quantity=Decimal(str(raw_report.get("quantity", "0"))),
                average_price=Decimal(str(raw_report.get("fill_price", "0"))),
                commission=Decimal(str(raw_report.get("commission", "0"))),
                timestamp=datetime.fromisoformat(raw_report.get("timestamp", datetime.now(UTC).isoformat())),
                realized_pnl=Decimal(str(raw_report.get("realized_pnl", "0"))) if "realized_pnl" in raw_report else None,
            )
        except Exception as e:
            self.logger.error(f"Error parsing simulated execution report: {e}")
            return None

    def _parse_generic_execution_report(self, raw_report: Dict[str, Any]) -> Optional[ExecutionReport]:
        """Parse generic execution report format."""
        try:
            return ExecutionReport(
                order_id=raw_report.get("order_id", ""),
                symbol=raw_report.get("symbol", ""),
                side=OrderSide(raw_report.get("side", "BUY").upper()),
                status=OrderStatus(raw_report.get("status", "FILLED").upper()),
                filled_quantity=Decimal(str(raw_report.get("filled_quantity", "0"))),
                average_price=Decimal(str(raw_report.get("average_price", "0"))),
                commission=Decimal(str(raw_report.get("commission", "0"))),
                timestamp=datetime.now(UTC),
            )
        except Exception as e:
            self.logger.error(f"Error parsing generic execution report: {e}")
            return None

    async def _update_loss_counters(self, report: ExecutionReport, realized_pnl: Optional[Decimal]) -> None:
        """Update consecutive loss/win counters with configurable reset logic."""
        if realized_pnl is None:
            return
            
        if realized_pnl < 0:
            # Loss - increment loss counter
            self._consecutive_loss_count += 1
            self._consecutive_win_count = 0
            self._risk_metrics.consecutive_losses = self._consecutive_loss_count
            self._risk_metrics.consecutive_wins = 0
            
            self.logger.warning(
                f"Trade loss detected for {report.symbol}: "
                f"consecutive losses now {self._consecutive_loss_count} (PnL: {realized_pnl:.4f})",
                source_module=self._source_module,
                context={
                    "order_id": report.order_id,
                    "symbol": report.symbol,
                    "realized_pnl": float(realized_pnl),
                    "consecutive_losses": self._consecutive_loss_count,
                }
            )
        else:
            # Win - increment win counter
            self._consecutive_win_count += 1
            if self._consecutive_loss_count > 0:
                self.logger.info(
                    f"Profit recorded for {report.symbol}: reset consecutive losses "
                    f"from {self._consecutive_loss_count} to 0 (PnL: {realized_pnl:.4f})",
                    source_module=self._source_module,
                    context={
                        "order_id": report.order_id,
                        "symbol": report.symbol,
                        "realized_pnl": float(realized_pnl),
                        "previous_losses": self._consecutive_loss_count,
                    }
                )
            self._consecutive_loss_count = 0
            self._risk_metrics.consecutive_losses = 0
            self._risk_metrics.consecutive_wins = self._consecutive_win_count
        
        # Update recent trades
        self._recent_trades.append({
            "timestamp": report.timestamp,
            "order_id": report.order_id,
            "symbol": report.symbol,
            "side": report.side.value,
            "quantity": float(report.filled_quantity),
            "price": float(report.average_price),
            "realized_pnl": float(realized_pnl),
        })
        
        # Keep only recent trades (last 100)
        if len(self._recent_trades) > 100:
            self._recent_trades = self._recent_trades[-100:]

    async def _evaluate_risk_conditions(self, report: ExecutionReport, realized_pnl: Optional[Decimal], 
                                       portfolio_state: Dict[str, Any]) -> List[str]:
        """Evaluate risk conditions and return list of triggered events."""
        risk_events = []
        
        # Check consecutive loss limits
        max_losses = self._max_consecutive_losses
        if self._consecutive_loss_count >= max_losses:
            risk_events.append('consecutive_loss_limit_reached')
        
        # Check daily loss limits
        daily_loss = await self._calculate_daily_loss(report.symbol)
        max_daily_loss = self._config.get("limits", {}).get("max_daily_loss_usd", 1000.0)
        if daily_loss >= max_daily_loss:
            risk_events.append('daily_loss_limit_reached')
        
        # Check drawdown limits
        current_equity = Decimal(str(portfolio_state.get("total_equity_usd", "0")))
        if hasattr(self, "_peak_equity") and self._peak_equity > 0:
            current_drawdown_pct = ((self._peak_equity - current_equity) / self._peak_equity) * Decimal("100")
            if current_drawdown_pct > self._max_total_drawdown_pct:
                risk_events.append('total_drawdown_limit_reached')
        
        return risk_events

    async def _publish_risk_events(self, risk_events: List[str], report: ExecutionReport) -> None:
        """Publish risk events to interested subscribers."""
        for event_type in risk_events:
            event_data = {
                'type': event_type,
                'symbol': report.symbol,
                'order_id': report.order_id,
                'consecutive_losses': self._consecutive_loss_count,
                'realized_pnl': float(report.realized_pnl) if report.realized_pnl else None,
                'timestamp': datetime.now(UTC).isoformat(),
                'action_required': True,
                'risk_metrics': {
                    'current_drawdown_pct': float(self._risk_metrics.current_drawdown_pct),
                    'max_drawdown_pct': float(self._risk_metrics.max_drawdown_pct),
                    'consecutive_losses': self._consecutive_loss_count,
                    'total_exposure': float(self._risk_metrics.total_exposure),
                }
            }
            
            await self.pubsub.publish(f'risk.{event_type}', event_data)
            
            self.logger.warning(
                f"Risk event published: {event_type} for {report.symbol}",
                source_module=self._source_module,
                context=event_data
            )

    async def _update_execution_audit_trail(self, report: ExecutionReport, realized_pnl: Optional[Decimal],
                                          risk_events: List[str]) -> None:
        """Update audit trail for execution report processing."""
        audit_entry = {
            'timestamp': datetime.now(UTC).isoformat(),
            'order_id': report.order_id,
            'symbol': report.symbol,
            'side': report.side.value,
            'status': report.status.value,
            'filled_quantity': float(report.filled_quantity),
            'average_price': float(report.average_price),
            'commission': float(report.commission),
            'realized_pnl': float(realized_pnl) if realized_pnl else None,
            'consecutive_losses': self._consecutive_loss_count,
            'risk_events': risk_events,
            'risk_metrics_snapshot': {
                'current_drawdown_pct': float(self._risk_metrics.current_drawdown_pct),
                'total_exposure': float(self._risk_metrics.total_exposure),
                'win_rate': float(self._risk_metrics.win_rate),
            }
        }
        
        # Log audit entry
        self.logger.info(
            "Execution audit trail entry",
            source_module=self._source_module,
            context=audit_entry
        )
        
        # Persist to audit database if available
        await self._persist_audit_entry_to_database(audit_entry)

    async def _persist_audit_entry_to_database(self, audit_entry: Dict[str, Any]) -> None:
        """Persist audit entry to database with comprehensive error handling and retry logic."""
        try:
            # Check if audit persistence is enabled in configuration
            audit_config = self._config.get("audit", {})
            if not audit_config.get("enabled", True):
                self.logger.debug(
                    "Audit persistence disabled in configuration",
                    source_module=self._source_module
                )
                return
            
            # Get storage backends from configuration
            storage_backends = audit_config.get("storage_backends", ["database"])
            
            # Attempt to persist to each configured backend
            for backend in storage_backends:
                try:
                    if backend == "database":
                        await self._persist_to_audit_database(audit_entry, audit_config)
                    elif backend == "file":
                        await self._persist_to_audit_file(audit_entry, audit_config)
                    elif backend == "elasticsearch":
                        await self._persist_to_elasticsearch(audit_entry, audit_config)
                    elif backend == "s3":
                        await self._persist_to_s3(audit_entry, audit_config)
                    else:
                        self.logger.warning(
                            f"Unknown audit storage backend: {backend}",
                            source_module=self._source_module
                        )
                        
                except Exception as backend_error:
                    self.logger.error(
                        f"Failed to persist audit entry to {backend}: {backend_error}",
                        source_module=self._source_module,
                        context={
                            "backend": backend,
                            "order_id": audit_entry.get("order_id"),
                            "error": str(backend_error),
                        }
                    )
                    # Continue to next backend if current one fails
                    continue
            
            # Update audit statistics
            if not hasattr(self, '_audit_stats'):
                self._audit_stats = {
                    'total_entries': 0,
                    'successful_persists': 0,
                    'failed_persists': 0,
                    'last_persist_time': None,
                }
            
            self._audit_stats['total_entries'] += 1
            self._audit_stats['successful_persists'] += 1
            self._audit_stats['last_persist_time'] = datetime.now(UTC)
            
        except Exception as e:
            if hasattr(self, '_audit_stats'):
                self._audit_stats['failed_persists'] += 1
            
            self.logger.error(
                f"Error in audit entry persistence: {e}",
                source_module=self._source_module,
                context={
                    "order_id": audit_entry.get("order_id"),
                    "symbol": audit_entry.get("symbol"),
                },
                exc_info=True
            )

    async def _persist_to_audit_database(self, audit_entry: Dict[str, Any], audit_config: Dict[str, Any]) -> None:
        """Persist audit entry to PostgreSQL/database with retry logic."""
        try:
            # Initialize audit repository if not already done
            if not hasattr(self, '_audit_repository'):
                await self._initialize_audit_repository(audit_config)
            
            if not self._audit_repository:
                self.logger.warning(
                    "Audit repository not available for database persistence",
                    source_module=self._source_module
                )
                return
            
            # Enhanced audit entry with additional metadata
            enhanced_entry = {
                **audit_entry,
                'id': str(uuid.uuid4()),
                'created_at': datetime.now(UTC).isoformat(),
                'risk_manager_version': getattr(self, '_version', '1.0.0'),
                'audit_schema_version': '1.0',
                'environment': audit_config.get('environment', 'production'),
                'instance_id': audit_config.get('instance_id', 'unknown'),
            }
            
            # Attempt database persistence with retry logic
            max_retries = audit_config.get('database_max_retries', 3)
            retry_delay = audit_config.get('database_retry_delay_seconds', 1)
            
            for attempt in range(max_retries):
                try:
                    await self._audit_repository.create_audit_entry(enhanced_entry)
                    
                    self.logger.debug(
                        f"Successfully persisted audit entry to database (attempt {attempt + 1})",
                        source_module=self._source_module,
                        context={
                            "audit_id": enhanced_entry['id'],
                            "order_id": audit_entry.get("order_id"),
                        }
                    )
                    return  # Success, exit retry loop
                    
                except Exception as db_error:
                    if attempt < max_retries - 1:
                        self.logger.warning(
                            f"Database persist attempt {attempt + 1} failed, retrying in {retry_delay}s: {db_error}",
                            source_module=self._source_module
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise db_error  # Re-raise on final attempt
            
        except Exception as e:
            self.logger.error(
                f"Failed to persist audit entry to database after retries: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise

    async def _persist_to_audit_file(self, audit_entry: Dict[str, Any], audit_config: Dict[str, Any]) -> None:
        """Persist audit entry to file system with rotation support."""
        try:
            import json
            import os
            from pathlib import Path
            
            # Get file configuration
            file_config = audit_config.get('file', {})
            audit_dir = Path(file_config.get('directory', './audit_logs'))
            max_file_size = file_config.get('max_file_size_mb', 100) * 1024 * 1024  # Convert to bytes
            max_files = file_config.get('max_files', 10)
            
            # Ensure audit directory exists
            audit_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            current_date = datetime.now(UTC).strftime('%Y-%m-%d')
            audit_file = audit_dir / f"execution_audit_{current_date}.jsonl"
            
            # Check if file rotation is needed
            if audit_file.exists() and audit_file.stat().st_size > max_file_size:
                await self._rotate_audit_files(audit_dir, max_files)
                # Create new file with sequence number
                sequence = 1
                while True:
                    audit_file = audit_dir / f"execution_audit_{current_date}_{sequence:03d}.jsonl"
                    if not audit_file.exists() or audit_file.stat().st_size < max_file_size:
                        break
                    sequence += 1
            
            # Append audit entry to file
            with open(audit_file, 'a', encoding='utf-8') as f:
                json.dump(audit_entry, f, default=str, ensure_ascii=False)
                f.write('\n')
            
            self.logger.debug(
                f"Successfully persisted audit entry to file: {audit_file}",
                source_module=self._source_module,
                context={"file_path": str(audit_file)}
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to persist audit entry to file: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise

    async def _persist_to_elasticsearch(self, audit_entry: Dict[str, Any], audit_config: Dict[str, Any]) -> None:
        """Persist audit entry to Elasticsearch for searchable audit logs."""
        try:
            # Check if Elasticsearch client is available
            if not hasattr(self, '_elasticsearch_client'):
                await self._initialize_elasticsearch_client(audit_config)
            
            if not self._elasticsearch_client:
                self.logger.warning(
                    "Elasticsearch client not available",
                    source_module=self._source_module
                )
                return
            
            # Get Elasticsearch configuration
            es_config = audit_config.get('elasticsearch', {})
            index_name = es_config.get('index_name', 'risk-audit')
            doc_type = es_config.get('doc_type', '_doc')
            
            # Add Elasticsearch-specific metadata
            es_entry = {
                **audit_entry,
                '@timestamp': datetime.now(UTC).isoformat(),
                'audit_type': 'execution_report',
                'service': 'risk_manager',
                'environment': es_config.get('environment', 'production'),
            }
            
            # Index document in Elasticsearch
            await self._elasticsearch_client.index(
                index=index_name,
                doc_type=doc_type,
                body=es_entry
            )
            
            self.logger.debug(
                f"Successfully persisted audit entry to Elasticsearch index: {index_name}",
                source_module=self._source_module,
                context={
                    "index": index_name,
                    "order_id": audit_entry.get("order_id"),
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to persist audit entry to Elasticsearch: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise

    async def _persist_to_s3(self, audit_entry: Dict[str, Any], audit_config: Dict[str, Any]) -> None:
        """Persist audit entry to S3 for long-term storage and compliance."""
        try:
            # Check if S3 client is available
            if not hasattr(self, '_s3_client'):
                await self._initialize_s3_client(audit_config)
            
            if not self._s3_client:
                self.logger.warning(
                    "S3 client not available",
                    source_module=self._source_module
                )
                return
            
            # Get S3 configuration
            s3_config = audit_config.get('s3', {})
            bucket_name = s3_config.get('bucket_name')
            if not bucket_name:
                raise ValueError("S3 bucket_name not configured")
            
            # Generate S3 key with partitioning
            timestamp = datetime.now(UTC)
            s3_key = f"audit-logs/year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}/{uuid.uuid4()}.json"
            
            # Prepare audit entry for S3
            s3_entry = {
                **audit_entry,
                's3_key': s3_key,
                'uploaded_at': timestamp.isoformat(),
                'bucket': bucket_name,
            }
            
            # Upload to S3
            import json
            await self._s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=json.dumps(s3_entry, default=str, ensure_ascii=False),
                ContentType='application/json',
                Metadata={
                    'order_id': str(audit_entry.get("order_id", "")),
                    'symbol': str(audit_entry.get("symbol", "")),
                    'audit_type': 'execution_report',
                }
            )
            
            self.logger.debug(
                f"Successfully persisted audit entry to S3: s3://{bucket_name}/{s3_key}",
                source_module=self._source_module,
                context={
                    "bucket": bucket_name,
                    "key": s3_key,
                    "order_id": audit_entry.get("order_id"),
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to persist audit entry to S3: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise

    async def _rotate_audit_files(self, audit_dir: Path, max_files: int) -> None:
        """Rotate audit files to manage disk space."""
        try:
            # Get all audit files sorted by modification time
            audit_files = list(audit_dir.glob("execution_audit_*.jsonl"))
            audit_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove excess files
            for old_file in audit_files[max_files:]:
                old_file.unlink()
                self.logger.info(
                    f"Rotated old audit file: {old_file}",
                    source_module=self._source_module
                )
                
        except Exception as e:
            self.logger.error(
                f"Error rotating audit files: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _initialize_audit_repository(self, audit_config: Dict[str, Any]) -> None:
        """Initialize audit repository for database persistence."""
        try:
            # Try to import and initialize audit repository
            try:
                from gal_friday.dal.repositories.audit_repository import AuditRepository
                self._audit_repository = AuditRepository()
                
                # Test connection
                await self._audit_repository.health_check()
                
                self.logger.info(
                    "Audit repository initialized successfully",
                    source_module=self._source_module
                )
                
            except ImportError:
                self.logger.warning(
                    "AuditRepository not available - audit data will not be persisted to database",
                    source_module=self._source_module
                )
                self._audit_repository = None
                
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize audit repository: {e}",
                    source_module=self._source_module
                )
                self._audit_repository = None
                
        except Exception as e:
            self.logger.error(
                f"Error in audit repository initialization: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            self._audit_repository = None

    async def _initialize_elasticsearch_client(self, audit_config: Dict[str, Any]) -> None:
        """Initialize Elasticsearch client for audit persistence."""
        try:
            es_config = audit_config.get('elasticsearch', {})
            if not es_config.get('enabled', False):
                self._elasticsearch_client = None
                return
            
            try:
                from elasticsearch import AsyncElasticsearch
                
                self._elasticsearch_client = AsyncElasticsearch(
                    hosts=es_config.get('hosts', ['http://localhost:9200']),
                    timeout=es_config.get('timeout', 30),
                    max_retries=es_config.get('max_retries', 3),
                )
                
                # Test connection
                await self._elasticsearch_client.ping()
                
                self.logger.info(
                    "Elasticsearch client initialized successfully",
                    source_module=self._source_module
                )
                
            except ImportError:
                self.logger.warning(
                    "Elasticsearch library not available - install with: pip install elasticsearch",
                    source_module=self._source_module
                )
                self._elasticsearch_client = None
                
        except Exception as e:
            self.logger.error(
                f"Failed to initialize Elasticsearch client: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            self._elasticsearch_client = None

    async def _initialize_s3_client(self, audit_config: Dict[str, Any]) -> None:
        """Initialize S3 client for audit persistence."""
        try:
            s3_config = audit_config.get('s3', {})
            if not s3_config.get('enabled', False):
                self._s3_client = None
                return
            
            try:
                import aioboto3
                
                session = aioboto3.Session()
                self._s3_client = session.client(
                    's3',
                    region_name=s3_config.get('region', 'us-east-1'),
                    aws_access_key_id=s3_config.get('access_key_id'),
                    aws_secret_access_key=s3_config.get('secret_access_key'),
                )
                
                self.logger.info(
                    "S3 client initialized successfully",
                    source_module=self._source_module
                )
                
            except ImportError:
                self.logger.warning(
                    "AWS S3 library not available - install with: pip install aioboto3",
                    source_module=self._source_module
                )
                self._s3_client = None
                
        except Exception as e:
            self.logger.error(
                f"Failed to initialize S3 client: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            self._s3_client = None

    async def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics for monitoring."""
        try:
            if not hasattr(self, '_audit_stats'):
                return {"error": "Audit statistics not available"}
            
            stats = {
                **self._audit_stats,
                'success_rate': (
                    self._audit_stats['successful_persists'] / 
                    max(1, self._audit_stats['total_entries'])
                ) * 100,
                'failure_rate': (
                    self._audit_stats['failed_persists'] / 
                    max(1, self._audit_stats['total_entries'])
                ) * 100,
                'last_persist_time_iso': (
                    self._audit_stats['last_persist_time'].isoformat() 
                    if self._audit_stats['last_persist_time'] else None
                ),
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(
                f"Error getting audit statistics: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return {"error": str(e)}

    async def _cleanup_audit_resources(self) -> None:
        """Clean up audit-related resources during shutdown."""
        try:
            # Clean up Elasticsearch client
            if hasattr(self, '_elasticsearch_client') and self._elasticsearch_client:
                try:
                    await self._elasticsearch_client.close()
                    self.logger.debug(
                        "Elasticsearch client closed successfully",
                        source_module=self._source_module
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error closing Elasticsearch client: {e}",
                        source_module=self._source_module
                    )
                finally:
                    self._elasticsearch_client = None
            
            # Clean up S3 client
            if hasattr(self, '_s3_client') and self._s3_client:
                try:
                    await self._s3_client.close()
                    self.logger.debug(
                        "S3 client closed successfully",
                        source_module=self._source_module
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error closing S3 client: {e}",
                        source_module=self._source_module
                    )
                finally:
                    self._s3_client = None
            
            # Clean up audit repository
            if hasattr(self, '_audit_repository') and self._audit_repository:
                try:
                    if hasattr(self._audit_repository, 'close'):
                        await self._audit_repository.close()
                    self.logger.debug(
                        "Audit repository closed successfully",
                        source_module=self._source_module
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error closing audit repository: {e}",
                        source_module=self._source_module
                    )
                finally:
                    self._audit_repository = None
            
            # Log final audit statistics
            if hasattr(self, '_audit_stats'):
                final_stats = await self.get_audit_statistics()
                self.logger.info(
                    f"Final audit statistics: {final_stats}",
                    source_module=self._source_module
                )
            
        except Exception as e:
            self.logger.error(
                f"Error during audit resource cleanup: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _publish_execution_error_event(self, event_dict: Dict[str, Any], error: str) -> None:
        """Publish error event for execution report processing failures."""
        error_event = {
            'type': 'ExecutionReportProcessingError',
            'source_module': self._source_module,
            'timestamp': datetime.now(UTC).isoformat(),
            'error': error,
            'event_dict': event_dict,
        }
        
        await self.pubsub.publish('risk.execution_report.error', error_event)

    async def _calculate_daily_loss(self, symbol: str) -> Decimal:
        """Calculate total daily loss for a symbol."""
        daily_loss = Decimal("0")
        current_date = datetime.now(UTC).date()
        
        for trade in self._recent_trades:
            trade_date = trade['timestamp'].date() if isinstance(trade['timestamp'], datetime) else None
            if trade_date == current_date and trade['symbol'] == symbol and trade['realized_pnl'] < 0:
                daily_loss += abs(Decimal(str(trade['realized_pnl'])))
        
        return daily_loss
