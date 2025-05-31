"""Enhanced strategy interface supporting traditional and MARL-based strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Protocol

import numpy as np

from ..core.asset_registry import AssetSpecification
from ..interfaces.feature_engine_interface import FeatureVector


class StrategyType(Enum):
    """Types of trading strategies."""
    RULES_BASED = auto()      # Traditional rule-based strategies
    ML_SINGLE_AGENT = auto()  # Single ML model/agent
    MARL_MULTI_AGENT = auto() # Multi-agent reinforcement learning
    ENSEMBLE = auto()         # Combination of multiple strategies
    HYBRID = auto()           # Mix of rules and ML


class ActionType(Enum):
    """Types of actions a strategy can recommend."""
    HOLD = auto()
    BUY = auto()
    SELL = auto()
    CLOSE_LONG = auto()
    CLOSE_SHORT = auto()
    SCALE_IN = auto()       # Increase position size
    SCALE_OUT = auto()      # Decrease position size
    HEDGE = auto()          # Hedge existing position


@dataclass(frozen=True)
class StrategyAction:
    """Action recommended by a strategy."""
    action_type: ActionType
    symbol: str
    exchange_id: str
    confidence: float  # 0.0 to 1.0

    # Order details
    suggested_quantity: Decimal | None = None
    suggested_price: Decimal | None = None
    order_type: str = "LIMIT"  # "LIMIT", "MARKET", "STOP"

    # Risk management
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    max_risk_amount: Decimal | None = None

    # Strategy context
    strategy_id: str = ""
    strategy_version: str = "1.0"
    reasoning: str = ""

    # MARL-specific fields
    agent_id: str | None = None
    episode_step: int | None = None
    q_values: dict[str, float] | None = None

    # Timing
    valid_until: datetime | None = None
    urgency: float = 0.5  # 0.0 = not urgent, 1.0 = very urgent

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyState:
    """Current state of a strategy for persistence and monitoring."""
    strategy_id: str
    strategy_type: StrategyType
    is_active: bool

    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")

    # MARL-specific state
    episode_number: int | None = None
    exploration_rate: float | None = None
    learning_rate: float | None = None

    # Resource usage
    last_execution_time_ms: float | None = None
    memory_usage_mb: float | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated_at: datetime = field(default_factory=datetime.utcnow)

    # Configuration
    parameters: dict[str, Any] = field(default_factory=dict)


class StrategyInterface(ABC):
    """Enhanced interface for trading strategies supporting traditional and MARL approaches."""

    def __init__(self, strategy_id: str, strategy_type: StrategyType,
                 asset_specifications: list[AssetSpecification], **kwargs: dict[str, Any]) -> None:
        """Initialize strategy with configuration."""
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.asset_specifications = {spec.symbol: spec for spec in asset_specifications}
        self.is_active = False
        self._state = StrategyState(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            is_active=False,
        )

    @property
    def state(self) -> StrategyState:
        """Get current strategy state."""
        return self._state

    # Core strategy methods
    @abstractmethod
    async def analyze_market(self, feature_vector: FeatureVector) -> StrategyAction:
        """Analyze market conditions and recommend action.

        Args:
            feature_vector: Current market features and state
        Returns:
            Recommended trading action
        """

    @abstractmethod
    async def update_state(self, market_event: dict[str, Any]) -> None:
        """Update internal strategy state based on market events.

        Args:
            market_event: Market data update or execution result
        """

    @abstractmethod
    def get_required_features(self) -> list[str]:
        """Get list of required feature names for this strategy.

        Returns:
            List of feature names needed for analysis
        """

    # Lifecycle management
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize strategy (load models, set up state, etc.)."""

    @abstractmethod
    async def start(self) -> None:
        """Start strategy execution."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop strategy execution and save state."""

    @abstractmethod
    async def reset(self) -> None:
        """Reset strategy to initial state."""

    # Configuration and parameters
    @abstractmethod
    def update_parameters(self, parameters: dict[str, Any]) -> None:
        """Update strategy parameters.

        Args:
            parameters: New parameter values
        """

    def get_parameters(self) -> dict[str, Any]:
        """Get current strategy parameters.

        Returns:
            Current parameter values
        """
        return self.state.parameters.copy()

    # Performance tracking
    def get_performance_metrics(self) -> dict[str, Any]:
        """Get strategy performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        win_rate = (
            self.state.winning_trades / self.state.total_trades
            if self.state.total_trades > 0 else 0.0
        )

        return {
            "total_trades": self.state.total_trades,
            "win_rate": win_rate,
            "total_pnl": float(self.state.total_pnl),
            "max_drawdown": float(self.state.max_drawdown),
            "last_execution_time_ms": self.state.last_execution_time_ms,
        }

    def record_trade_result(self, pnl: Decimal, was_winner: bool) -> None:
        """Record the result of a completed trade.

        Args:
            pnl: Profit/loss from the trade
            was_winner: Whether the trade was successful
        """
        # Create new state with updated values since StrategyState is frozen
        new_total_trades = self._state.total_trades + 1
        new_winning_trades = self._state.winning_trades + (1 if was_winner else 0)
        new_total_pnl = self._state.total_pnl + pnl
        new_max_drawdown = self._state.max_drawdown

        # Update max drawdown if necessary
        if pnl < 0 and abs(pnl) > new_max_drawdown:
            new_max_drawdown = abs(pnl)

        # Replace the state with updated version
        self._state = StrategyState(
            strategy_id=self._state.strategy_id,
            strategy_type=self._state.strategy_type,
            is_active=self._state.is_active,
            total_trades=new_total_trades,
            winning_trades=new_winning_trades,
            total_pnl=new_total_pnl,
            max_drawdown=new_max_drawdown,
            episode_number=self._state.episode_number,
            exploration_rate=self._state.exploration_rate,
            learning_rate=self._state.learning_rate,
            last_execution_time_ms=self._state.last_execution_time_ms,
            memory_usage_mb=self._state.memory_usage_mb,
            created_at=self._state.created_at,
            last_updated_at=datetime.utcnow(),
            parameters=self._state.parameters,
        )

    def validate_action(self, action: StrategyAction) -> list[str]:
        """Validate if an action is allowed in current market state.

        Args:
            action: Action to validate
        Returns:
            List of validation errors, empty if valid
        """
        errors = []

        # Basic validation
        if action.confidence < 0 or action.confidence > 1:
            errors.append("Confidence must be between 0 and 1")

        if action.suggested_quantity and action.suggested_quantity <= 0:
            errors.append("Suggested quantity must be positive")

        if action.suggested_price and action.suggested_price <= 0:
            errors.append("Suggested price must be positive")

        # Portfolio-specific validation
        if action.action_type in [ActionType.SELL, ActionType.CLOSE_LONG]:
            asset_spec_or_default_dict = self.asset_specifications.get(action.symbol, {})
            if isinstance(asset_spec_or_default_dict, dict):
                current_position = asset_spec_or_default_dict.get("position", 0)
            else:
                # asset_spec_or_default_dict is an AssetSpecification object.
                # AssetSpecification does not have a 'position' attribute.
                # Defaulting to 0 as per the original code's fallback.
                current_position = 0
            if current_position <= 0:
                errors.append(f"Cannot sell {action.symbol}: no long position")

        return errors

    # Asset support
    def supports_asset(self, symbol: str) -> bool:
        """Check if strategy supports trading a specific asset.

        Args:
            symbol: Asset symbol to check
        Returns:
            True if supported, False otherwise
        """
        return symbol in self.asset_specifications

    def get_supported_assets(self) -> list[str]:
        """Get list of supported asset symbols.

        Returns:
            List of asset symbols this strategy can trade
        """
        return list(self.asset_specifications.keys())


class MARLStrategyInterface(StrategyInterface):
    """Enhanced interface specifically for MARL-based strategies."""

    def __init__(self, strategy_id: str, agent_configs: list[dict[str, Any]],
                 asset_specifications: list[AssetSpecification], **kwargs: dict[str, Any]) -> None:
        """Initialize MARL strategy with multiple agents."""
        super().__init__(
            strategy_id,
            StrategyType.MARL_MULTI_AGENT,
            asset_specifications,
            **kwargs,
        )
        self.agent_configs = agent_configs
        self.agents: dict[str, Any] = {}  # Will hold actual agent instances
        self.current_episode = 0

    # MARL-specific methods
    @abstractmethod
    async def get_joint_action(
        self,
        feature_vector: FeatureVector,
        agent_observations: dict[str, np.ndarray],
    ) -> dict[str, StrategyAction]:
        """Get joint actions from all agents.

        Args:
            feature_vector: Market features
            agent_observations: Agent-specific observations
        Returns:
            Dictionary mapping agent IDs to their recommended actions
        """

    @abstractmethod
    async def update_agent_rewards(self, agent_rewards: dict[str, float]) -> None:
        """Update rewards for all agents based on market outcomes.

        Args:
            agent_rewards: Dictionary mapping agent IDs to their rewards
        """

    @abstractmethod
    async def step_episode(self) -> None:
        """Advance to the next episode step."""

    @abstractmethod
    def get_agent_states(self) -> dict[str, dict[str, Any]]:
        """Get current state of all agents.

        Returns:
            Dictionary mapping agent IDs to their states
        """

    # Ensemble decision making
    async def ensemble_actions(
        self,
        agent_actions: dict[str, StrategyAction],
    ) -> StrategyAction:
        """Combine multiple agent actions into a single strategy action.

        Args:
            agent_actions: Actions from individual agents
        Returns:
            Combined strategy action
        """
        # Default implementation: simple voting/averaging
        # Can be overridden for more sophisticated ensemble methods

        if not agent_actions:
            return StrategyAction(
                action_type=ActionType.HOLD,
                symbol="",
                exchange_id="",
                confidence=0.0,
                strategy_id=self.strategy_id,
                reasoning="No agent actions available",
            )

        # Simple voting for action type
        action_votes: dict[ActionType, int] = {}
        total_confidence = 0.0

        for action in agent_actions.values():
            action_votes[action.action_type] = action_votes.get(action.action_type, 0) + 1
            total_confidence += action.confidence

        # Most voted action
        winning_action = max(action_votes.keys(), key=lambda x: action_votes[x])
        avg_confidence = total_confidence / len(agent_actions)

        # Use the action details from the most confident agent with the winning action
        representative_action = max(
            (a for a in agent_actions.values() if a.action_type == winning_action),
            key=lambda x: x.confidence,
            default=next(iter(agent_actions.values())),
        )

        return StrategyAction(
            action_type=winning_action,
            symbol=representative_action.symbol,
            exchange_id=representative_action.exchange_id,
            confidence=avg_confidence,
            suggested_quantity=representative_action.suggested_quantity,
            suggested_price=representative_action.suggested_price,
            strategy_id=self.strategy_id,
            reasoning=f"Ensemble of {len(agent_actions)} agents",
            metadata={
                "agent_votes": action_votes,
                "agent_actions": {
                    aid: str(action.action_type)
                    for aid, action in agent_actions.items()
                },
            },
        )


class EnsembleStrategyInterface(StrategyInterface):
    """Interface for ensemble strategies combining multiple sub-strategies."""

    def __init__(
        self,
        strategy_id: str,
        sub_strategies: list[StrategyInterface],
        ensemble_method: str = "weighted_voting",
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize ensemble strategy.

        Args:
            strategy_id: Unique identifier for this ensemble
            sub_strategies: List of constituent strategies
            ensemble_method: Method for combining strategies
                ("weighted_voting", "meta_learning", etc.)
            **kwargs: Additional keyword arguments to pass to parent class
        """
        # Combine asset specifications from all sub-strategies
        all_assets = {}
        for strategy in sub_strategies:
            all_assets.update(strategy.asset_specifications)

        super().__init__(strategy_id, StrategyType.ENSEMBLE, list(all_assets.values()), **kwargs)
        self.sub_strategies = sub_strategies
        self.ensemble_method = ensemble_method
        self.strategy_weights = {s.strategy_id: 1.0 / len(sub_strategies) for s in sub_strategies}

    @abstractmethod
    async def combine_actions(
        self,
        sub_actions: dict[str, StrategyAction],
    ) -> StrategyAction:
        """Combine actions from sub-strategies into final action.

        Args:
            sub_actions: Actions from constituent strategies
        Returns:
            Combined ensemble action
        """

    def update_strategy_weights(
        self,
        performance_metrics: dict[str, dict[str, float]],
    ) -> None:
        """Update weights of constituent strategies based on performance.

        Args:
            performance_metrics: Performance data for each strategy
        """
        # Simple performance-based reweighting
        total_performance = sum(
            metrics.get("total_pnl", 0) for metrics in performance_metrics.values()
        )

        if total_performance > 0:
            for strategy_id, metrics in performance_metrics.items():
                strategy_pnl = metrics.get("total_pnl", 0)
                self.strategy_weights[strategy_id] = max(0.1, strategy_pnl / total_performance)

        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy_id in self.strategy_weights:
                self.strategy_weights[strategy_id] /= total_weight


# Protocol for strategy factory
class StrategyFactory(Protocol):
    """Protocol for creating different types of strategies."""

    def create_strategy(
        self,
        strategy_type: StrategyType,
        config: dict[str, Any],
        asset_specs: list[AssetSpecification],
    ) -> StrategyInterface:
        """Create a strategy instance.

        Args:
            strategy_type: Type of strategy to create
            config: Strategy configuration
            asset_specs: Supported asset specifications
        Returns:
            Strategy instance
        """


# Utility functions for strategy management
def validate_strategy_action(
    action: StrategyAction,
    current_portfolio: dict[str, Any],
) -> list[str]:
    """Validate a strategy action against current portfolio state.

    Args:
        action: Strategy action to validate
        current_portfolio: Current portfolio state
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Basic validation
    if action.confidence < 0 or action.confidence > 1:
        errors.append("Confidence must be between 0 and 1")

    if action.suggested_quantity and action.suggested_quantity <= 0:
        errors.append("Suggested quantity must be positive")

    if action.suggested_price and action.suggested_price <= 0:
        errors.append("Suggested price must be positive")

    # Portfolio-specific validation
    if action.action_type in [ActionType.SELL, ActionType.CLOSE_LONG]:
        current_position = current_portfolio.get("positions", {}).get(action.symbol, 0)
        if current_position <= 0:
            errors.append(f"Cannot sell {action.symbol}: no long position")

    return errors
