# Risk Manager Module
"""Risk management module for trading operations.

This module provides risk management functionality for trading operations,
including position sizing, drawdown limits, and trade validation.
"""

import asyncio
import math
import statistics
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import TYPE_CHECKING, Any

# Event Definitions
from .core.events import (
    ExecutionReportEvent,
    PotentialHaltTriggerEvent,
    TradeSignalProposedEvent,
    TradeSignalApprovedEvent,
    TradeSignalRejectedEvent,
    EventType, # Added for subscription
    FeatureEvent # Added for type hint
)
from gal_friday.core.feature_models import PublishedFeaturesV1 # For type hint reference
from gal_friday.core.feature_registry_client import FeatureRegistryClient # Added

# Import PubSubManager
from .core.pubsub import PubSubManager

# Import logger service
from .logger_service import LoggerService


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


# Type hint for PortfolioManager without circular import
if TYPE_CHECKING:
    from .exchange_info_service import ExchangeInfoService
    from .market_price_service import MarketPriceService
    from .portfolio_manager import PortfolioManager
else:
    # Define placeholder or attempt runtime import carefully
    try:
        from .portfolio_manager import PortfolioManager
    except ImportError:
        # Define a minimal placeholder if import fails at runtime
        # This allows basic script execution but will fail if methods are called
        class PortfolioManager:  # type: ignore
            """Placeholder for PortfolioManager when not available at runtime.

            Provides minimal implementations of methods used by RiskManager.
            """

            def get_current_state(self) -> dict[str, Any]:
                """Return empty portfolio state dictionary.

                Returns:
                -------
                    dict: Empty dictionary representing portfolio state
                """
                return {}

    try:
        from .market_price_service import MarketPriceService
    except ImportError:

        class MarketPriceService:  # type: ignore
            """Placeholder for MarketPriceService when not available at runtime.

            This allows the RiskManager to be imported and used without the actual service.
            """

            async def get_latest_price(self, trading_pair: str) -> Decimal | None:
                """Return None as placeholder for latest price."""
                _ = trading_pair  # Unused parameter
                return None

            async def get_volatility(
                self,
                trading_pair: str,
                lookback_hours: int = 24,
            ) -> float | None:
                """Return None as placeholder for volatility calculation.

                Args:
                    trading_pair: The trading pair to calculate volatility for
                    lookback_hours: Number of hours to look back for calculation

                Returns:
                -------
                    None as placeholder
                """
                _ = (trading_pair, lookback_hours)  # Unused parameters
                return None

            async def convert_amount(
                self,
                from_amount: Decimal,
                from_currency: str,
                to_currency: str,
            ) -> Decimal | None:
                """Return None as placeholder for currency conversion.

                Args:
                    from_amount: Amount to convert
                    from_currency: Source currency
                    to_currency: Target currency

                Returns:
                -------
                    None as placeholder
                """
                _ = (from_amount, from_currency, to_currency)  # Unused parameters
                return None

            async def get_historical_ohlcv(
                self,
                trading_pair: str,
                timeframe: str,
                since: datetime,
                limit: int | None = None,
            ) -> list[dict[str, Any]] | None:
                """Define a placeholder docstring for get_historical_ohlcv."""
                _ = (trading_pair, timeframe, since, limit)
                return None

    try:  # Added for ExchangeInfoService
        from .exchange_info_service import ExchangeInfoService
    except ImportError:

        class ExchangeInfoService:  # type: ignore
            """Placeholder for ExchangeInfoService."""

            def get_symbol_info(self, trading_pair: str) -> dict[str, Any] | None:
                """Get information for a specific symbol.

                Args:
                    trading_pair: The trading symbol to get info for

                Returns:
                -------
                    Dictionary with symbol information or None if not found
                """
                _ = trading_pair  # Unused
                return None

            def get_tick_size(self, trading_pair: str) -> Decimal | None:
                """Get the minimum price movement for a trading pair.

                Args:
                    trading_pair: The trading pair to get tick size for

                Returns:
                -------
                    The minimum price movement or None if not available
                """
                _ = trading_pair  # Unused parameter
                return None

            def get_step_size(self, trading_pair: str) -> Decimal | None:
                """Get the minimum trade size for a trading pair.

                Args:
                    trading_pair: The trading pair to get step size for

                Returns:
                -------
                    The minimum trade size or None if not available
                """
                _ = trading_pair  # Unused parameter
                return None


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
        portfolio_manager: "PortfolioManager",
        logger_service: LoggerService,
        market_price_service: "MarketPriceService",
        exchange_info_service: "ExchangeInfoService",
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
        self._recent_trades: list[dict[str, Any]] = []
        self._cached_conversion_rates: dict[str, Decimal] = {}
        self._cached_conversion_timestamps: dict[str, datetime] = {}
        self._normal_volatility: Dict[str, Decimal] = {} # Type hint Dict
        self._normal_volatility_logged_missing: dict[str, bool] = {}

        # For dynamic risk based on FeatureEngine features
        self._dynamic_risk_volatility_feature_key: Optional[str] = None
        self._dynamic_risk_target_pairs: List[str] = [] # Type hint List
        self._latest_volatility_features: Dict[str, float] = {} # Type hint Dict


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

        if self._enable_dynamic_risk_adjustment and not self._dynamic_risk_volatility_feature_key:
            self.logger.warning(
                "Dynamic risk adjustment is enabled, but no 'dynamic_risk_volatility_feature_key' "
                "is configured. Dynamic adjustments based on FeatureEngine features will not occur."
            )
        if self._enable_dynamic_risk_adjustment and not self._dynamic_risk_target_pairs:
            self.logger.warning(
                "Dynamic risk adjustment is enabled, but 'dynamic_risk_target_pairs' is empty. "
                "Adjustments will not apply to any specific pair via FeatureEngine features."
            )

        # Validate dynamic_risk_volatility_feature_key against Feature Registry
        if self._enable_dynamic_risk_adjustment and self._dynamic_risk_volatility_feature_key:
            if not self.feature_registry_client or not self.feature_registry_client.is_loaded():
                self.logger.warning(
                    "FeatureRegistryClient not available or not loaded. "
                    "Cannot validate 'dynamic_risk_volatility_feature_key': '%s' against the registry.",
                    self._dynamic_risk_volatility_feature_key,
                    source_module=self._source_module
                )
            else:
                definition = self.feature_registry_client.get_feature_definition(self._dynamic_risk_volatility_feature_key)
                if definition is None:
                    self.logger.warning(
                        "The configured 'dynamic_risk_volatility_feature_key': '%s' was not found in the Feature Registry. "
                        "Dynamic risk adjustment based on this feature may not work as expected.",
                        self._dynamic_risk_volatility_feature_key,
                        source_module=self._source_module
                    )
                else:
                    self.logger.debug(
                        "Dynamic risk volatility feature '%s' validated against Feature Registry.",
                        self._dynamic_risk_volatility_feature_key,
                        source_module=self._source_module
                    )

        self.logger.info("RiskManager configured.", source_module=self._source_module)
        self._validate_config()

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
        """
        Starts the RiskManager service.

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
                    self._dynamic_risk_adjustment_loop_fallback()
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
        """
        Stops the RiskManager service.

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

        self.logger.info(
            "Stopped periodic risk checks and tasks.",
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

    async def _handle_feature_event(self, event_dict: Dict[str, Any]) -> None:
        """
        Handles incoming `FeatureEvent`s (payload expected as `dict[str, float]`)
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

        if event_dict.get('event_type') == EventType.FEATURES_CALCULATED.name:
            payload = event_dict.get('payload')
            if not payload or not isinstance(payload, dict):
                self.logger.warning("Received FEATURES_CALCULATED event with invalid payload.")
                return

            trading_pair = payload.get('trading_pair')
            # Features are expected as dict[str, float] from PublishedFeaturesV1.model_dump()
            features_data = payload.get('features')

            if not trading_pair or not features_data or not isinstance(features_data, dict):
                self.logger.warning("FEATURES_CALCULATED event missing trading_pair or valid features dict.")
                return

            if trading_pair in self._dynamic_risk_target_pairs:
                volatility_value = features_data.get(self._dynamic_risk_volatility_feature_key)

                if volatility_value is not None and isinstance(volatility_value, (float, int)): # Check type
                    if np.isnan(volatility_value):
                        self.logger.debug(
                            f"Volatility feature '{self._dynamic_risk_volatility_feature_key}' "
                            f"is NaN for {trading_pair}. Skipping update."
                        )
                        return

                    self.logger.debug(
                        f"Received volatility feature '{self._dynamic_risk_volatility_feature_key}' "
                        f"value {volatility_value:.4f} for {trading_pair}."
                    )
                    self._latest_volatility_features[trading_pair] = float(volatility_value)
                    await self._update_risk_parameters_based_on_volatility(trading_pair)
                else:
                    self.logger.debug(
                        f"Volatility feature '{self._dynamic_risk_volatility_feature_key}' "
                        f"not found or invalid type ({type(volatility_value)}) in event for {trading_pair}."
                    )

    async def _dynamic_risk_adjustment_loop_fallback(self) -> None:
        """
        Fallback loop for dynamic risk adjustment if FeatureEngine-based feature
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
                            self._volatility_window_size # type: ignore
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
            initial_rounded_calculated_qty = self._stage3_position_sizing_and_portfolio_checks(
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
                state_values=self._extract_relevant_portfolio_values(portfolio_state),
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
                event,
                current_qty_to_trade,
                final_rounded_entry_price,
                rounded_sl_price,
                rounded_tp_price,
                final_trade_action or "MARKET"  # Default to MARKET if order_type is not available on event
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
        """
        Dynamically adjusts `_risk_per_trade_pct` for a given `trading_pair`.

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
                trading_pair
            )
            return

        try:
            current_volatility = Decimal(str(current_volatility_float))
        except InvalidOperation:
            self.logger.warning(
                "Invalid volatility feature value '%s' for %s. Cannot perform dynamic risk adjustment.",
                current_volatility_float, trading_pair
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
        """
        Calibrates and stores a baseline "normal" historical volatility for configured trading pairs.

        This method is typically called during startup if dynamic risk adjustment is enabled.
        It fetches historical daily OHLCV data (e.g., for the last 60 days, as defined
        by `self._volatility_window_size`) via the `MarketPriceService`. It then
        calculates the standard deviation of daily logarithmic returns for these pairs.
        This calculated standard deviation is stored in `self._normal_volatility` for
        each pair and serves as a baseline.

        The `_update_risk_parameters_based_on_volatility` method later compares a live
        volatility metric (e.g., ATR, sourced from `FeatureEngine` via
        `self._dynamic_risk_volatility_feature_key`) against this calibrated baseline
        to make risk adjustments.

        **Note on Comparability:**
        The user/configurator must be mindful that the baseline volatility calculated here
        (std dev of log returns) might differ in scale and nature from the live
        volatility feature provided by `FeatureEngine` (e.g., an ATR value).
        The thresholds for dynamic risk adjustment (e.g., "current > normal * 1.5")
        need to be set with this potential difference in mind to ensure sensible behavior.
        Future enhancements could involve calibrating normal levels for the *specific*
        `FeatureEngine` feature if historical values of that feature were available.
        """
        if not hasattr(self, "_normal_volatility_logged_missing"): # Ensure attribute exists
            self._normal_volatility_logged_missing = {}

        # Define pairs and parameters for calibration
        pairs_to_calibrate = ["XRP/USD", "DOGE/USD"]  # As per SRS FR-102
        lookback_days = 60  # Number of days of historical data to use
        min_data_points_for_stddev = 10  # Need at least a few points to calculate stdev reliably

        self.logger.info(
            (
                f"Starting normal volatility calibration for pairs: "
                f"{pairs_to_calibrate} using {lookback_days}-day lookback."
            ),
            source_module=self._source_module,
            context={"pairs": pairs_to_calibrate, "lookback": lookback_days},
        )

        for trading_pair in pairs_to_calibrate:
            try:
                # Calculate the 'since' date for fetching historical data
                since_datetime = datetime.now(UTC) - timedelta(days=lookback_days)

                self.logger.debug(
                    (
                        f"Fetching historical OHLCV for {trading_pair} "
                        f"since {since_datetime} for volatility calibration."
                    ),
                    source_module=self._source_module,
                    context={"pair": trading_pair, "date": since_datetime},
                )

                # Fetch historical daily data
                # Assuming MarketPriceService is correctly instantiated and available
                historical_data = await self._market_price_service.get_historical_ohlcv(
                    trading_pair=trading_pair,
                    timeframe="1d",  # Daily timeframe
                    since=since_datetime,
                    # limit can be omitted if we want all data since 'since_datetime'
                    # up to Kraken's max (720 points)
                    # For 60 days, we expect about 60 points, which is well within the limit.
                )

                if not historical_data or len(historical_data) < min_data_points_for_stddev:
                    self.logger.warning(
                        (
                            f"Could not calibrate normal volatility for {trading_pair}: "
                            f"Insufficient historical data fetched (got "
                            f"{len(historical_data) if historical_data else 0} points, "
                            f"need at least {min_data_points_for_stddev})."
                        ),
                        source_module=self._source_module,
                        context={
                            "trading_pair": trading_pair,
                            "fetched_points": len(historical_data) if historical_data else 0,
                            "min_points": min_data_points_for_stddev,
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
                            f"Could not calibrate normal volatility for {trading_pair}: "
                            f"Insufficient valid closing prices extracted "
                            f"({len(closing_prices)}, need at least {min_data_points_for_stddev})."
                        ),
                        source_module=self._source_module,
                        context={
                            "trading_pair": trading_pair,
                            "closing_prices_count": len(closing_prices),
                            "min_points": min_data_points_for_stddev,
                        },
                    )
                    self._normal_volatility_logged_missing[trading_pair] = True
                    continue

                # Calculate daily logarithmic returns: ln(Price_t / Price_{t-1})
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
                            f"Could not calibrate normal volatility for {trading_pair}: "
                            f"Insufficient log returns calculated ({len(log_returns)}, "
                            f"need at least {min_data_points_for_stddev -1} for meaningful stdev)."
                        ),
                        source_module=self._source_module,
                        context={
                            "trading_pair": trading_pair,
                            "log_returns_count": len(log_returns),
                            "min_needed_for_stdev": min_data_points_for_stddev - 1,
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
                    daily_volatility = Decimal(str(statistics.stdev(float_log_returns)))

                    # Optional: Annualize volatility (daily might be fine for dynamic adjustment)
                    # For now, we store the daily volatility.
                    self._normal_volatility[trading_pair] = daily_volatility
                    self.logger.info(
                        (
                            f"Successfully calibrated normal daily volatility for {trading_pair}: "
                        f"{daily_volatility:.8f}. (Based on {len(log_returns)} log returns "
                        f"from {len(closing_prices)} prices over approx. {lookback_days} days)."
                        ),
                        source_module=self._source_module,
                        context={
                            "trading_pair": trading_pair,
                            "volatility": daily_volatility,
                            "log_returns_count": len(log_returns),
                            "closing_prices_count": len(closing_prices),
                            "lookback_days": lookback_days,
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
                    f"Error during normal volatility calibration for {trading_pair}",
                    source_module=self._source_module,
                    context={"trading_pair": trading_pair},
                )
                self._normal_volatility_logged_missing[trading_pair] = True

        self.logger.info(
            "Normal volatility calibration process finished.",
            source_module=self._source_module,
        )

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
                ctx.trading_pair
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
                source_module=self._source_module
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
            else:
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
            elif ctx.side == "SELL" and ctx.rounded_sl_price <= effective_entry:
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
                    rejection_reason="Invalid risk per unit (SL too close to entry)"
                )
            
            # Calculate raw position size
            raw_quantity = risk_amount / risk_per_unit
            
            # Apply lot size constraints
            rounded_quantity = self._calculate_lot_size_with_fallback(
                event.trading_pair,
                raw_quantity
            )
            
            if rounded_quantity is None or rounded_quantity <= 0:
                return SizingResult(
                    is_valid=False,
                    rejection_reason="Position size below minimum tradeable amount"
                )
            
            # Calculate position value
            position_value = rounded_quantity * ref_entry_price
            
            # Check against max order size
            if position_value > self._max_order_size_usd:
                # Scale down to max order size
                rounded_quantity = self._max_order_size_usd / ref_entry_price
                rounded_quantity = self._calculate_lot_size_with_fallback(
                    event.trading_pair,
                    rounded_quantity
                )
                
                if rounded_quantity is None or rounded_quantity <= 0:
                    return SizingResult(
                        is_valid=False,
                        rejection_reason=f"Position value exceeds max order size ${self._max_order_size_usd}"
                    )
                position_value = rounded_quantity * ref_entry_price
            
            # Check against max single position percentage
            position_pct = (position_value / current_equity) * Decimal("100")
            if position_pct > self._max_single_position_pct:
                return SizingResult(
                    is_valid=False,
                    rejection_reason=f"Position size {position_pct:.2f}% exceeds max {self._max_single_position_pct}%"
                )
            
            return SizingResult(
                is_valid=True,
                quantity=rounded_quantity,
                risk_amount=risk_amount,
                position_value=position_value
            )
            
        except Exception as e:
            self.logger.exception(
                "Error calculating position size",
                source_module=self._source_module,
                context={"signal_id": str(event.signal_id)}
            )
            return SizingResult(
                is_valid=False,
                rejection_reason=f"Position sizing error: {str(e)}"
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
                        scaled_qty = self._calculate_lot_size_with_fallback(ctx.trading_pair, scaled_qty)
                        
                        if scaled_qty is None or scaled_qty <= 0:
                            return False, "Scaled position size below minimum", None, None
                            
                        return True, None, "ADD_TO_POSITION", scaled_qty
                        
                return True, None, "ADD_TO_POSITION", ctx.initial_calculated_qty
            else:
                # Opposite side - this would close/reduce position
                if existing_qty <= ctx.initial_calculated_qty:
                    # This would close the position entirely
                    return True, None, "CLOSE_POSITION", existing_qty
                else:
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
                    source_module=self._source_module
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
