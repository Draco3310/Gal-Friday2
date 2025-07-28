"""Chain of Responsibility pattern implementation for retry condition evaluation."""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from gal_friday.core.events import TradeSignalProposedEvent
    from gal_friday.risk_manager import RejectionReason


class RetryConditionEvaluator(ABC):
    """Abstract base class for retry condition evaluators."""

    def __init__(self, risk_manager: Any) -> None:
        """Initialize with risk manager reference."""
        self.risk_manager = risk_manager
        self.next_evaluator: RetryConditionEvaluator | None = None
        # Cache commonly accessed private members to avoid repeated access
        self._source_module = getattr(risk_manager, "_source_module", "RiskManager")

    @property
    def portfolio_manager(self) -> Any:
        """Access portfolio manager."""
        return getattr(self.risk_manager, "_portfolio_manager", None)

    @property
    def risk_metrics(self) -> dict[str, Any]:
        """Access risk metrics."""
        return getattr(self.risk_manager, "_risk_metrics", {})

    @property
    def dynamic_risk_volatility_feature_key(self) -> str:
        """Access dynamic risk volatility feature key."""
        return getattr(self.risk_manager, "_dynamic_risk_volatility_feature_key", "default_volatility_key")

    @property
    def latest_volatility_features(self) -> dict[str, Any]:
        """Access latest volatility features."""
        return getattr(self.risk_manager, "_latest_volatility_features", {})

    @property
    def normal_volatility(self) -> dict[str, float]:
        """Access normal volatility."""
        return getattr(self.risk_manager, "_normal_volatility", {})

    async def get_current_market_price(self, trading_pair: str) -> Decimal | None:
        """Access get_current_market_price method."""
        method = getattr(self.risk_manager, "_get_current_market_price", None)
        if method:
            return cast("Decimal | None", await method(trading_pair))
        return None

    @property
    def max_total_drawdown_pct(self) -> float:
        """Access max total drawdown percentage."""
        return getattr(self.risk_manager, "_max_total_drawdown_pct", 0.15)

    @property
    def max_total_exposure_pct(self) -> float:
        """Access max total exposure percentage."""
        return getattr(self.risk_manager, "_max_total_exposure_pct", 0.95)

    @property
    def consecutive_loss_count(self) -> int:
        """Access consecutive loss count."""
        return getattr(self.risk_manager, "_consecutive_loss_count", 0)

    @property
    def max_consecutive_losses(self) -> int:
        """Access max consecutive losses."""
        return getattr(self.risk_manager, "_max_consecutive_losses", 5)

    @property
    def is_running(self) -> bool:
        """Access is_running flag."""
        return getattr(self.risk_manager, "_is_running", False)

    def calculate_composite_risk_score(self, risk_metrics: dict[str, Any]) -> Decimal:
        """Access calculate_composite_risk_score method."""
        method = getattr(self.risk_manager, "_calculate_composite_risk_score", None)
        if method:
            return cast("Decimal", method(risk_metrics))
        return Decimal(0)

    def set_next(self, evaluator: "RetryConditionEvaluator") -> "RetryConditionEvaluator":
        """Set the next evaluator in the chain."""
        self.next_evaluator = evaluator
        return evaluator

    async def evaluate(
        self,
        condition: str,
        original_event: "TradeSignalProposedEvent",
        rejection_reason: "RejectionReason",
    ) -> bool:
        """Evaluate the condition or pass to next evaluator."""
        if self.can_handle(condition):
            return await self.handle(condition, original_event, rejection_reason)
        if self.next_evaluator:
            return await self.next_evaluator.evaluate(condition, original_event, rejection_reason)
        # Unknown condition
        self.risk_manager.logger.warning(
            f"Unknown retry condition: {condition}",
            source_module=self._source_module,
        )
        return False

    @abstractmethod
    def can_handle(self, condition: str) -> bool:
        """Check if this evaluator can handle the condition."""

    @abstractmethod
    async def handle(
        self,
        condition: str,
        original_event: "TradeSignalProposedEvent",
        rejection_reason: "RejectionReason",
    ) -> bool:
        """Handle the condition evaluation."""


class BalanceEvaluator(RetryConditionEvaluator):
    """Evaluates balance-related retry conditions."""

    def can_handle(self, condition: str) -> bool:
        """Check if this is a balance-related condition."""
        return condition in ["check_available_balance", "check_portfolio_equity"]

    async def handle(
        self,
        condition: str,
        original_event: "TradeSignalProposedEvent",
        rejection_reason: "RejectionReason",
    ) -> bool:
        """Handle balance-related conditions."""
        portfolio_state = self.portfolio_manager.get_current_state()

        if condition == "check_available_balance":
            available_balance = Decimal(str(portfolio_state.get("available_balance_usd", "0")))
            required_balance = Decimal(1000)  # Minimum required balance
            return available_balance >= required_balance

        if condition == "check_portfolio_equity":
            current_equity = Decimal(str(portfolio_state.get("total_equity_usd", "0")))
            return current_equity > Decimal(0)

        return False


class VolatilityEvaluator(RetryConditionEvaluator):
    """Evaluates volatility-related retry conditions."""

    def can_handle(self, condition: str) -> bool:
        """Check if this is a volatility-related condition."""
        return condition == "check_market_volatility"

    async def handle(
        self,
        condition: str,
        original_event: "TradeSignalProposedEvent",
        rejection_reason: "RejectionReason",
    ) -> bool:
        """Handle volatility conditions."""
        # Check if volatility has decreased
        if self.dynamic_risk_volatility_feature_key:
            current_vol = self.latest_volatility_features.get(original_event.trading_pair)
            normal_vol = self.normal_volatility.get(original_event.trading_pair)
            if current_vol is not None and normal_vol:
                return cast("bool", current_vol <= float(Decimal(str(normal_vol)) * Decimal("1.2")))  # 20% above normal
        return True  # Default to true if no volatility data


class MarketConditionEvaluator(RetryConditionEvaluator):
    """Evaluates market condition-related retry conditions."""

    def can_handle(self, condition: str) -> bool:
        """Check if this is a market condition."""
        return condition in [
            "check_market_liquidity",
            "check_spread_conditions",
            "check_price_stability",
            "check_orderbook_depth",
        ]

    async def handle(
        self,
        condition: str,
        original_event: "TradeSignalProposedEvent",
        rejection_reason: "RejectionReason",
    ) -> bool:
        """Handle market condition evaluations."""
        if condition == "check_spread_conditions":
            # Check if bid-ask spread is reasonable
            try:
                current_price = await self.get_current_market_price(original_event.trading_pair)
                if current_price:
                    # Assume spread is acceptable if we can get price
                    return True
            except Exception:
                self.risk_manager.logger.debug(
                    f"Failed to get current market price for {original_event.trading_pair}",
                    source_module=self._source_module,
                )
            return False

        # Placeholder implementations for other conditions
        return True


class DrawdownEvaluator(RetryConditionEvaluator):
    """Evaluates drawdown-related retry conditions."""

    def can_handle(self, condition: str) -> bool:
        """Check if this is a drawdown-related condition."""
        return condition in ["check_portfolio_recovery", "check_drawdown_levels"]

    async def handle(
        self,
        condition: str,
        original_event: "TradeSignalProposedEvent",
        rejection_reason: "RejectionReason",
    ) -> bool:
        """Handle drawdown conditions."""
        current_drawdown = self.risk_metrics.get("current_drawdown_pct", Decimal(0))

        if condition == "check_portfolio_recovery":
            # Check if portfolio has recovered from drawdown
            recovery_threshold = Decimal(str(self.max_total_drawdown_pct)) * Decimal("0.8")  # 80% of max
            return cast("bool", current_drawdown < recovery_threshold)

        if condition == "check_drawdown_levels":
            # Check current drawdown levels
            levels_threshold = Decimal(str(self.max_total_drawdown_pct)) * Decimal("0.9")  # 90% of max
            return cast("bool", current_drawdown < levels_threshold)

        return False


class ExposureEvaluator(RetryConditionEvaluator):
    """Evaluates exposure-related retry conditions."""

    def can_handle(self, condition: str) -> bool:
        """Check if this is an exposure-related condition."""
        return condition in ["check_position_exposure", "check_exposure_levels"]

    async def handle(
        self,
        condition: str,
        original_event: "TradeSignalProposedEvent",
        rejection_reason: "RejectionReason",
    ) -> bool:
        """Handle exposure conditions."""
        if condition == "check_position_exposure":
            # Check if position exposure has decreased
            portfolio_state = self.portfolio_manager.get_current_state()
            total_exposure = Decimal(0)
            positions = portfolio_state.get("positions", {})
            for position in positions.values():
                total_exposure += abs(Decimal(str(position.get("current_market_value", "0"))))

            current_equity = Decimal(str(portfolio_state.get("total_equity_usd", "1")))
            exposure_pct = (total_exposure / current_equity) * Decimal(100) if current_equity > 0 else Decimal(0)
            return exposure_pct < Decimal(str(self.max_total_exposure_pct)) * Decimal("0.8")  # 80% of max

        if condition == "check_exposure_levels":
            # Check exposure levels
            return cast(
                "bool",
                self.risk_metrics.get("total_exposure", Decimal(0)) < Decimal(50000),
            )  # Placeholder threshold

        return False


class RiskMetricsEvaluator(RetryConditionEvaluator):
    """Evaluates risk metrics-related retry conditions."""

    def can_handle(self, condition: str) -> bool:
        """Check if this is a risk metrics condition."""
        return condition in ["check_risk_metrics", "check_general_conditions"]

    async def handle(
        self,
        condition: str,
        original_event: "TradeSignalProposedEvent",
        rejection_reason: "RejectionReason",
    ) -> bool:
        """Handle risk metrics conditions."""
        if condition == "check_risk_metrics":
            # Check overall risk metrics
            # Use the risk manager's method directly for complex calculation
            calc_method = getattr(self.risk_manager, "_calculate_composite_risk_score", None)
            if calc_method:
                risk_score = calc_method(
                    self.risk_metrics.get("current_drawdown_pct", Decimal(0)),
                    Decimal(50),  # Placeholder exposure
                    self.consecutive_loss_count,
                    self.risk_metrics.get("win_rate", Decimal(0)),
                )
            else:
                risk_score = Decimal(0)
            return cast("bool", risk_score < Decimal(60))  # Below warning threshold

        if condition == "check_general_conditions":
            # General condition check - system is running and healthy
            return self.is_running and self.consecutive_loss_count < self.max_consecutive_losses

        return False


class PlaceholderEvaluator(RetryConditionEvaluator):
    """Handles placeholder conditions that always return True."""

    def can_handle(self, condition: str) -> bool:
        """Check if this is a placeholder condition."""
        return condition == "check_portfolio_rebalancing"

    async def handle(
        self,
        condition: str,
        original_event: "TradeSignalProposedEvent",
        rejection_reason: "RejectionReason",
    ) -> bool:
        """Handle placeholder conditions."""
        # Placeholder implementation
        return True


def create_retry_condition_chain(risk_manager: Any) -> RetryConditionEvaluator:
    """Create and configure the chain of retry condition evaluators."""
    # Create evaluators
    balance_eval = BalanceEvaluator(risk_manager)
    volatility_eval = VolatilityEvaluator(risk_manager)
    market_eval = MarketConditionEvaluator(risk_manager)
    drawdown_eval = DrawdownEvaluator(risk_manager)
    exposure_eval = ExposureEvaluator(risk_manager)
    risk_metrics_eval = RiskMetricsEvaluator(risk_manager)
    placeholder_eval = PlaceholderEvaluator(risk_manager)

    # Build the chain
    balance_eval.set_next(volatility_eval).set_next(market_eval).set_next(
        drawdown_eval,
    ).set_next(exposure_eval).set_next(risk_metrics_eval).set_next(placeholder_eval)

    return balance_eval
