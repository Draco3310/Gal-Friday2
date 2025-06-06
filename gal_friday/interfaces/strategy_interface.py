"""Enhanced strategy interface supporting traditional and MARL-based strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Protocol, Optional
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

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


class EnsembleMethod(str, Enum):
    """Methods for combining multiple strategy predictions."""
    SIMPLE_MAJORITY = "simple_majority"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    SHARPE_WEIGHTED = "sharpe_weighted"
    VOLATILITY_INVERSE = "volatility_inverse"
    META_LEARNING = "meta_learning"


class ReweightingMethod(str, Enum):
    """Methods for performance-based strategy reweighting."""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"


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


@dataclass
class PerformanceMetrics:
    """Comprehensive statistical performance metrics for strategy evaluation."""
    strategy_id: str
    returns: np.ndarray
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    value_at_risk_95: float
    conditional_var_95: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    
    # Statistical significance
    t_statistic: float
    p_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    
    # Performance periods
    total_return: float
    annualized_return: float
    win_rate: float
    profit_factor: float
    
    # Metadata
    observation_count: int
    calculation_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble strategy decision making."""
    method: EnsembleMethod = EnsembleMethod.CONFIDENCE_WEIGHTED
    
    # Weighting parameters
    performance_window: int = 252  # 1 year lookback
    min_observations: int = 30
    confidence_threshold: float = 0.6
    volatility_adjustment: bool = True
    
    # Dynamic weighting
    adaptation_rate: float = 0.1
    recency_bias: float = 0.8
    
    # Risk management
    max_individual_weight: float = 0.4
    min_individual_weight: float = 0.05
    diversification_penalty: float = 0.1
    
    # Statistical parameters
    confidence_level: float = 0.95
    significance_threshold: float = 0.05


@dataclass  
class ReweightingConfig:
    """Configuration for statistical performance-based reweighting."""
    method: ReweightingMethod = ReweightingMethod.SHARPE_RATIO
    
    # Lookback parameters
    lookback_window: int = 252  # 1 year
    min_observations: int = 30
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly, quarterly
    
    # Statistical parameters
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    outlier_threshold: float = 3.0  # Standard deviations
    
    # Risk parameters
    target_volatility: Optional[float] = None
    max_weight: float = 0.5
    min_weight: float = 0.1
    
    # Regularization
    l1_penalty: float = 0.0
    l2_penalty: float = 0.01
    transaction_cost: float = 0.001


class StatisticalAnalyzer:
    """Statistical analysis utilities for strategy performance evaluation."""
    
    @staticmethod
    def calculate_performance_metrics(
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for a strategy."""
        if len(returns) < 2:
            raise ValueError("Insufficient data for performance analysis")
        
        returns_array = returns.values
        
        # Basic statistics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information ratio (vs benchmark if provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = active_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # Distribution metrics
        skewness = stats.skew(returns_array)
        kurtosis = stats.kurtosis(returns_array)
        
        # Statistical significance test
        t_stat, p_value = stats.ttest_1samp(returns_array, 0)
        
        # Confidence interval for mean return
        confidence_interval = stats.t.interval(
            0.95, len(returns) - 1, 
            loc=returns.mean(), 
            scale=stats.sem(returns)
        )
        
        # Win rate and profit factor
        winning_trades = (returns > 0).sum()
        win_rate = winning_trades / len(returns)
        
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else float('inf')
        
        return PerformanceMetrics(
            strategy_id="",  # Will be set by caller
            returns=returns_array,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            volatility=volatility,
            max_drawdown=max_drawdown,
            value_at_risk_95=var_95,
            conditional_var_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            t_statistic=t_stat,
            p_value=p_value,
            confidence_interval_lower=confidence_interval[0],
            confidence_interval_upper=confidence_interval[1],
            total_return=total_return,
            annualized_return=annualized_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            observation_count=len(returns)
        )

    @staticmethod
    def test_statistical_significance(
        strategy_metrics: list[PerformanceMetrics],
        significance_level: float = 0.05
    ) -> dict[str, dict[str, float]]:
        """Perform statistical significance tests between strategies."""
        if len(strategy_metrics) < 2:
            return {}
        
        results = {}
        
        for i, metrics_a in enumerate(strategy_metrics):
            for j, metrics_b in enumerate(strategy_metrics[i+1:], i+1):
                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(metrics_a.returns, metrics_b.returns)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_p_value = stats.mannwhitneyu(
                    metrics_a.returns, metrics_b.returns, alternative='two-sided'
                )
                
                pair_key = f"{metrics_a.strategy_id}_vs_{metrics_b.strategy_id}"
                results[pair_key] = {
                    "t_statistic": t_stat,
                    "t_p_value": p_value,
                    "t_significant": p_value < significance_level,
                    "u_statistic": u_stat,
                    "u_p_value": u_p_value,
                    "u_significant": u_p_value < significance_level,
                    "effect_size": (metrics_a.returns.mean() - metrics_b.returns.mean()) / 
                                  np.sqrt((metrics_a.returns.var() + metrics_b.returns.var()) / 2)
                }
        
        return results


class ConfigurableEnsemble:
    """Advanced configurable ensemble strategy combiner."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.performance_history: dict[str, list[float]] = {}
        self.weights_history: dict[str, list[float]] = {}
        self.logger = logging.getLogger(__name__)
    
    def combine_actions(
        self,
        actions: dict[str, StrategyAction],
        performance_metrics: Optional[dict[str, PerformanceMetrics]] = None
    ) -> StrategyAction:
        """Combine multiple strategy actions using configured ensemble method."""
        if not actions:
            return StrategyAction(
                action_type=ActionType.HOLD,
                symbol="",
                exchange_id="",
                confidence=0.0,
                reasoning="No actions to combine"
            )
        
        try:
            if self.config.method == EnsembleMethod.SIMPLE_MAJORITY:
                return self._simple_majority_vote(actions)
            elif self.config.method == EnsembleMethod.WEIGHTED_AVERAGE:
                return self._weighted_average(actions)
            elif self.config.method == EnsembleMethod.CONFIDENCE_WEIGHTED:
                return self._confidence_weighted(actions)
            elif self.config.method == EnsembleMethod.PERFORMANCE_WEIGHTED:
                return self._performance_weighted(actions, performance_metrics)
            elif self.config.method == EnsembleMethod.SHARPE_WEIGHTED:
                return self._sharpe_weighted(actions, performance_metrics)
            elif self.config.method == EnsembleMethod.VOLATILITY_INVERSE:
                return self._volatility_inverse_weighted(actions, performance_metrics)
            else:
                self.logger.warning(f"Unknown ensemble method: {self.config.method}")
                return self._confidence_weighted(actions)
        except Exception as e:
            self.logger.error(f"Error in ensemble combination: {e}")
            return self._simple_majority_vote(actions)
    
    def _weighted_average(self, actions: dict[str, StrategyAction]) -> StrategyAction:
        """Simple weighted average of all actions (equal weights)."""
        if not actions:
            return StrategyAction(
                action_type=ActionType.HOLD,
                symbol="",
                exchange_id="",
                confidence=0.0,
                reasoning="No actions for weighted average"
            )
        
        # Use equal weights for each action
        weight = 1.0 / len(actions)
        equal_weights = {action.strategy_id: weight for action in actions.values()}
        
        return self._apply_weights_to_actions(actions, equal_weights, "weighted_average")
    
    def _simple_majority_vote(self, actions: dict[str, StrategyAction]) -> StrategyAction:
        """Simple majority voting for action types."""
        action_votes: dict[ActionType, list[StrategyAction]] = {}
        
        for action in actions.values():
            if action.action_type not in action_votes:
                action_votes[action.action_type] = []
            action_votes[action.action_type].append(action)
        
        # Find the action type with most votes
        winning_action_type = max(action_votes.keys(), key=lambda x: len(action_votes[x]))
        winning_actions = action_votes[winning_action_type]
        
        # Use the most confident action of the winning type
        representative_action = max(winning_actions, key=lambda x: x.confidence)
        
        return StrategyAction(
            action_type=winning_action_type,
            symbol=representative_action.symbol,
            exchange_id=representative_action.exchange_id,
            confidence=np.mean([a.confidence for a in winning_actions]),
            suggested_quantity=representative_action.suggested_quantity,
            suggested_price=representative_action.suggested_price,
            reasoning=f"Majority vote: {len(winning_actions)}/{len(actions)} strategies",
            metadata={"ensemble_method": "simple_majority", "vote_count": len(winning_actions)}
        )
    
    def _confidence_weighted(self, actions: dict[str, StrategyAction]) -> StrategyAction:
        """Weight actions by their confidence scores."""
        total_confidence = sum(action.confidence for action in actions.values())
        if total_confidence == 0:
            return self._simple_majority_vote(actions)
        
        # Weight votes by confidence
        action_weights: dict[ActionType, float] = {}
        action_details: dict[ActionType, list[tuple[StrategyAction, float]]] = {}
        
        for action in actions.values():
            weight = action.confidence / total_confidence
            
            if action.action_type not in action_weights:
                action_weights[action.action_type] = 0
                action_details[action.action_type] = []
            
            action_weights[action.action_type] += weight
            action_details[action.action_type].append((action, weight))
        
        # Find the action type with highest weighted vote
        winning_action_type = max(action_weights.keys(), key=lambda x: action_weights[x])
        winning_actions = action_details[winning_action_type]
        
        # Calculate weighted average of parameters
        total_weight = sum(weight for _, weight in winning_actions)
        weighted_confidence = sum(action.confidence * weight for action, weight in winning_actions) / total_weight
        
        # Use the highest weighted action as representative
        representative_action = max(winning_actions, key=lambda x: x[1])[0]
        
        return StrategyAction(
            action_type=winning_action_type,
            symbol=representative_action.symbol,
            exchange_id=representative_action.exchange_id,
            confidence=min(weighted_confidence, 1.0),
            suggested_quantity=representative_action.suggested_quantity,
            suggested_price=representative_action.suggested_price,
            reasoning=f"Confidence-weighted ensemble (weight: {action_weights[winning_action_type]:.3f})",
            metadata={
                "ensemble_method": "confidence_weighted",
                "winning_weight": action_weights[winning_action_type],
                "total_confidence": total_confidence
            }
        )
    
    def _performance_weighted(
        self, 
        actions: dict[str, StrategyAction],
        performance_metrics: Optional[dict[str, PerformanceMetrics]]
    ) -> StrategyAction:
        """Weight actions by historical performance metrics."""
        if not performance_metrics:
            return self._confidence_weighted(actions)
        
        # Calculate performance-based weights
        performance_weights = {}
        total_performance = 0
        
        for strategy_id, metrics in performance_metrics.items():
            if strategy_id in [action.strategy_id for action in actions.values()]:
                # Use annualized return adjusted by Sharpe ratio
                performance_score = metrics.annualized_return * max(0.1, metrics.sharpe_ratio)
                performance_weights[strategy_id] = max(0.01, performance_score)
                total_performance += performance_weights[strategy_id]
        
        if total_performance <= 0:
            return self._confidence_weighted(actions)
        
        # Normalize weights
        for strategy_id in performance_weights:
            performance_weights[strategy_id] /= total_performance
        
        # Apply weights to actions
        weighted_actions: dict[ActionType, list[tuple[StrategyAction, float]]] = {}
        
        for action in actions.values():
            weight = performance_weights.get(action.strategy_id, 0)
            if weight > 0:
                if action.action_type not in weighted_actions:
                    weighted_actions[action.action_type] = []
                weighted_actions[action.action_type].append((action, weight))
        
        if not weighted_actions:
            return self._confidence_weighted(actions)
        
        # Calculate total weight for each action type
        action_total_weights = {
            action_type: sum(weight for _, weight in action_list)
            for action_type, action_list in weighted_actions.items()
        }
        
        # Choose action type with highest total weight
        winning_action_type = max(action_total_weights.keys(), key=lambda x: action_total_weights[x])
        winning_actions = weighted_actions[winning_action_type]
        
        # Use the highest weighted action as representative
        representative_action = max(winning_actions, key=lambda x: x[1])[0]
        
        return StrategyAction(
            action_type=winning_action_type,
            symbol=representative_action.symbol,
            exchange_id=representative_action.exchange_id,
            confidence=min(representative_action.confidence * action_total_weights[winning_action_type], 1.0),
            suggested_quantity=representative_action.suggested_quantity,
            suggested_price=representative_action.suggested_price,
            reasoning=f"Performance-weighted ensemble (weight: {action_total_weights[winning_action_type]:.3f})",
            metadata={
                "ensemble_method": "performance_weighted",
                "winning_weight": action_total_weights[winning_action_type],
                "performance_weights": performance_weights
            }
        )
    
    def _sharpe_weighted(
        self,
        actions: dict[str, StrategyAction],
        performance_metrics: Optional[dict[str, PerformanceMetrics]]
    ) -> StrategyAction:
        """Weight actions by Sharpe ratio."""
        if not performance_metrics:
            return self._confidence_weighted(actions)
        
        # Calculate Sharpe-based weights
        sharpe_weights = {}
        min_sharpe = float('inf')
        
        for strategy_id, metrics in performance_metrics.items():
            if strategy_id in [action.strategy_id for action in actions.values()]:
                min_sharpe = min(min_sharpe, metrics.sharpe_ratio)
        
        # Shift Sharpe ratios to be positive
        sharpe_offset = max(0, -min_sharpe + 0.1)
        total_sharpe = 0
        
        for strategy_id, metrics in performance_metrics.items():
            if strategy_id in [action.strategy_id for action in actions.values()]:
                adjusted_sharpe = metrics.sharpe_ratio + sharpe_offset
                sharpe_weights[strategy_id] = adjusted_sharpe
                total_sharpe += adjusted_sharpe
        
        if total_sharpe <= 0:
            return self._confidence_weighted(actions)
        
        # Normalize weights
        for strategy_id in sharpe_weights:
            sharpe_weights[strategy_id] /= total_sharpe
        
        return self._apply_weights_to_actions(actions, sharpe_weights, "sharpe_weighted")
    
    def _volatility_inverse_weighted(
        self,
        actions: dict[str, StrategyAction],
        performance_metrics: Optional[dict[str, PerformanceMetrics]]
    ) -> StrategyAction:
        """Weight actions inversely to their volatility."""
        if not performance_metrics:
            return self._confidence_weighted(actions)
        
        # Calculate inverse volatility weights
        inv_vol_weights = {}
        total_inv_vol = 0
        
        for strategy_id, metrics in performance_metrics.items():
            if strategy_id in [action.strategy_id for action in actions.values()]:
                if metrics.volatility > 0:
                    inv_weight = 1.0 / metrics.volatility
                    inv_vol_weights[strategy_id] = inv_weight
                    total_inv_vol += inv_weight
                else:
                    inv_vol_weights[strategy_id] = 1.0
                    total_inv_vol += 1.0
        
        if total_inv_vol <= 0:
            return self._confidence_weighted(actions)
        
        # Normalize weights
        for strategy_id in inv_vol_weights:
            inv_vol_weights[strategy_id] /= total_inv_vol
        
        return self._apply_weights_to_actions(actions, inv_vol_weights, "volatility_inverse")
    
    def _apply_weights_to_actions(
        self, 
        actions: dict[str, StrategyAction], 
        weights: dict[str, float],
        method_name: str
    ) -> StrategyAction:
        """Apply calculated weights to actions and return combined result."""
        weighted_actions: dict[ActionType, list[tuple[StrategyAction, float]]] = {}
        
        for action in actions.values():
            weight = weights.get(action.strategy_id, 0)
            if weight > 0:
                if action.action_type not in weighted_actions:
                    weighted_actions[action.action_type] = []
                weighted_actions[action.action_type].append((action, weight))
        
        if not weighted_actions:
            return self._confidence_weighted(actions)
        
        # Calculate total weight for each action type
        action_total_weights = {
            action_type: sum(weight for _, weight in action_list)
            for action_type, action_list in weighted_actions.items()
        }
        
        # Choose action type with highest total weight
        winning_action_type = max(action_total_weights.keys(), key=lambda x: action_total_weights[x])
        winning_actions = weighted_actions[winning_action_type]
        
        # Use the highest weighted action as representative
        representative_action = max(winning_actions, key=lambda x: x[1])[0]
        
        return StrategyAction(
            action_type=winning_action_type,
            symbol=representative_action.symbol,
            exchange_id=representative_action.exchange_id,
            confidence=min(representative_action.confidence * action_total_weights[winning_action_type], 1.0),
            suggested_quantity=representative_action.suggested_quantity,
            suggested_price=representative_action.suggested_price,
            reasoning=f"{method_name} ensemble (weight: {action_total_weights[winning_action_type]:.3f})",
            metadata={
                "ensemble_method": method_name,
                "winning_weight": action_total_weights[winning_action_type],
                "strategy_weights": weights
            }
        )


class StatisticalReweighting:
    """Advanced statistical reweighting system for strategy portfolios."""
    
    def __init__(self, config: ReweightingConfig):
        self.config = config
        self.performance_cache: dict[str, PerformanceMetrics] = {}
        self.weight_history: list[dict[str, float]] = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_optimal_weights(
        self,
        strategy_performances: dict[str, pd.Series],
        current_weights: Optional[dict[str, float]] = None
    ) -> dict[str, float]:
        """Calculate optimal strategy weights using statistical analysis."""
        if len(strategy_performances) < 2:
            return {list(strategy_performances.keys())[0]: 1.0} if strategy_performances else {}
        
        try:
            # Calculate performance metrics for each strategy
            metrics = {}
            for strategy_id, returns in strategy_performances.items():
                if len(returns) >= self.config.min_observations:
                    metrics[strategy_id] = StatisticalAnalyzer.calculate_performance_metrics(returns)
                    metrics[strategy_id].strategy_id = strategy_id
            
            if not metrics:
                return self._equal_weights(list(strategy_performances.keys()))
            
            # Apply chosen reweighting method
            if self.config.method == ReweightingMethod.SHARPE_RATIO:
                weights = self._sharpe_ratio_weighting(metrics)
            elif self.config.method == ReweightingMethod.SORTINO_RATIO:
                weights = self._sortino_ratio_weighting(metrics)
            elif self.config.method == ReweightingMethod.RISK_PARITY:
                weights = self._risk_parity_weighting(metrics)
            elif self.config.method == ReweightingMethod.MINIMUM_VARIANCE:
                weights = self._minimum_variance_weighting(strategy_performances)
            elif self.config.method == ReweightingMethod.MAXIMUM_DIVERSIFICATION:
                weights = self._maximum_diversification_weighting(strategy_performances)
            else:
                self.logger.warning(f"Unknown reweighting method: {self.config.method}")
                weights = self._sharpe_ratio_weighting(metrics)
            
            # Apply constraints and regularization
            weights = self._apply_constraints(weights, current_weights)
            
            # Cache performance metrics
            self.performance_cache.update(metrics)
            self.weight_history.append(weights.copy())
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error in weight calculation: {e}")
            return self._equal_weights(list(strategy_performances.keys()))
    
    def _sharpe_ratio_weighting(self, metrics: dict[str, PerformanceMetrics]) -> dict[str, float]:
        """Calculate weights based on Sharpe ratio."""
        sharpe_scores = {}
        min_sharpe = min(m.sharpe_ratio for m in metrics.values())
        
        # Shift Sharpe ratios to be positive
        sharpe_offset = max(0, -min_sharpe + 0.1)
        
        total_adjusted_sharpe = 0
        for strategy_id, m in metrics.items():
            adjusted_sharpe = m.sharpe_ratio + sharpe_offset
            # Filter out strategies with poor statistical significance
            if m.p_value > self.config.significance_threshold:
                adjusted_sharpe *= 0.5  # Penalty for insignificant performance
            
            sharpe_scores[strategy_id] = adjusted_sharpe
            total_adjusted_sharpe += adjusted_sharpe
        
        if total_adjusted_sharpe <= 0:
            return self._equal_weights(list(metrics.keys()))
        
        return {sid: score / total_adjusted_sharpe for sid, score in sharpe_scores.items()}
    
    def _sortino_ratio_weighting(self, metrics: dict[str, PerformanceMetrics]) -> dict[str, float]:
        """Calculate weights based on Sortino ratio."""
        sortino_scores = {}
        min_sortino = min(m.sortino_ratio for m in metrics.values())
        
        # Shift Sortino ratios to be positive
        sortino_offset = max(0, -min_sortino + 0.1)
        
        total_adjusted_sortino = 0
        for strategy_id, m in metrics.items():
            adjusted_sortino = m.sortino_ratio + sortino_offset
            # Apply statistical significance penalty
            if m.p_value > self.config.significance_threshold:
                adjusted_sortino *= 0.5
            
            sortino_scores[strategy_id] = adjusted_sortino
            total_adjusted_sortino += adjusted_sortino
        
        if total_adjusted_sortino <= 0:
            return self._equal_weights(list(metrics.keys()))
        
        return {sid: score / total_adjusted_sortino for sid, score in sortino_scores.items()}
    
    def _risk_parity_weighting(self, metrics: dict[str, PerformanceMetrics]) -> dict[str, float]:
        """Calculate risk parity weights (inverse volatility)."""
        inv_vol_scores = {}
        total_inv_vol = 0
        
        for strategy_id, m in metrics.items():
            if m.volatility > 0:
                inv_vol = 1.0 / m.volatility
                inv_vol_scores[strategy_id] = inv_vol
                total_inv_vol += inv_vol
            else:
                inv_vol_scores[strategy_id] = 1.0
                total_inv_vol += 1.0
        
        if total_inv_vol <= 0:
            return self._equal_weights(list(metrics.keys()))
        
        return {sid: score / total_inv_vol for sid, score in inv_vol_scores.items()}
    
    def _minimum_variance_weighting(self, performances: dict[str, pd.Series]) -> dict[str, float]:
        """Calculate minimum variance portfolio weights."""
        try:
            # Create returns matrix
            strategy_ids = list(performances.keys())
            returns_matrix = pd.DataFrame({sid: performances[sid] for sid in strategy_ids})
            returns_matrix = returns_matrix.dropna()
            
            if len(returns_matrix) < self.config.min_observations:
                return self._equal_weights(strategy_ids)
            
            # Calculate covariance matrix
            cov_matrix = returns_matrix.cov().values
            
            # Regularize covariance matrix
            if self.config.l2_penalty > 0:
                identity = np.eye(len(cov_matrix))
                cov_matrix = cov_matrix + self.config.l2_penalty * identity
            
            # Minimum variance optimization: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
            inv_cov = np.linalg.pinv(cov_matrix)
            ones = np.ones((len(strategy_ids), 1))
            weights = inv_cov @ ones
            weights = weights / (ones.T @ inv_cov @ ones)
            weights = weights.flatten()
            
            # Ensure non-negative weights
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
            
            return dict(zip(strategy_ids, weights))
            
        except Exception as e:
            self.logger.error(f"Error in minimum variance calculation: {e}")
            return self._equal_weights(list(performances.keys()))
    
    def _maximum_diversification_weighting(self, performances: dict[str, pd.Series]) -> dict[str, float]:
        """Calculate maximum diversification portfolio weights."""
        try:
            strategy_ids = list(performances.keys())
            returns_matrix = pd.DataFrame({sid: performances[sid] for sid in strategy_ids})
            returns_matrix = returns_matrix.dropna()
            
            if len(returns_matrix) < self.config.min_observations:
                return self._equal_weights(strategy_ids)
            
            # Calculate volatilities and correlation matrix
            volatilities = returns_matrix.std().values
            correlation_matrix = returns_matrix.corr().values
            
            # Maximum diversification: maximize (w^T * σ) / sqrt(w^T * Σ * w)
            # This is equivalent to minimum variance with volatility scaling
            inv_corr = np.linalg.pinv(correlation_matrix)
            vol_scaled_weights = inv_corr @ (1 / volatilities)
            vol_scaled_weights = vol_scaled_weights / vol_scaled_weights.sum()
            
            # Ensure non-negative weights
            vol_scaled_weights = np.maximum(vol_scaled_weights, 0)
            vol_scaled_weights = vol_scaled_weights / vol_scaled_weights.sum()
            
            return dict(zip(strategy_ids, vol_scaled_weights))
            
        except Exception as e:
            self.logger.error(f"Error in maximum diversification calculation: {e}")
            return self._equal_weights(list(performances.keys()))
    
    def _apply_constraints(
        self,
        weights: dict[str, float],
        current_weights: Optional[dict[str, float]] = None
    ) -> dict[str, float]:
        """Apply weight constraints and regularization."""
        if not weights:
            return weights
        
        # Apply min/max weight constraints
        for strategy_id in weights:
            weights[strategy_id] = max(self.config.min_weight, 
                                     min(self.config.max_weight, weights[strategy_id]))
        
        # Renormalize after constraints
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {sid: w / total_weight for sid, w in weights.items()}
        
        # Apply transaction cost penalty if current weights provided
        if current_weights and self.config.transaction_cost > 0:
            weights = self._apply_transaction_cost_penalty(weights, current_weights)
        
        return weights
    
    def _apply_transaction_cost_penalty(
        self,
        new_weights: dict[str, float],
        current_weights: dict[str, float]
    ) -> dict[str, float]:
        """Apply transaction cost penalty to reduce turnover."""
        penalty_factor = self.config.transaction_cost
        
        # Calculate turnover
        total_turnover = 0
        for strategy_id in new_weights:
            current_weight = current_weights.get(strategy_id, 0)
            turnover = abs(new_weights[strategy_id] - current_weight)
            total_turnover += turnover
        
        # If turnover is below threshold, return new weights
        if total_turnover < 0.1:  # 10% turnover threshold
            return new_weights
        
        # Otherwise, blend with current weights
        blend_factor = min(0.8, 1.0 - penalty_factor * total_turnover)
        
        blended_weights = {}
        for strategy_id in new_weights:
            current_weight = current_weights.get(strategy_id, 0)
            blended_weight = (blend_factor * new_weights[strategy_id] + 
                            (1 - blend_factor) * current_weight)
            blended_weights[strategy_id] = blended_weight
        
        # Renormalize
        total_weight = sum(blended_weights.values())
        if total_weight > 0:
            blended_weights = {sid: w / total_weight for sid, w in blended_weights.items()}
        
        return blended_weights
    
    def _equal_weights(self, strategy_ids: list[str]) -> dict[str, float]:
        """Return equal weights for all strategies."""
        if not strategy_ids:
            return {}
        weight = 1.0 / len(strategy_ids)
        return {sid: weight for sid in strategy_ids}
    
    def get_performance_attribution(self) -> dict[str, Any]:
        """Get performance attribution analysis."""
        if not self.performance_cache or not self.weight_history:
            return {}
        
        latest_weights = self.weight_history[-1] if self.weight_history else {}
        
        attribution = {
            "individual_contributions": {},
            "total_portfolio_metrics": {},
            "diversification_ratio": 0.0,
            "concentration_index": 0.0
        }
        
        # Calculate individual contributions
        total_return = 0
        total_volatility = 0
        
        for strategy_id, metrics in self.performance_cache.items():
            weight = latest_weights.get(strategy_id, 0)
            contribution = weight * metrics.annualized_return
            risk_contribution = weight * metrics.volatility
            
            attribution["individual_contributions"][strategy_id] = {
                "weight": weight,
                "return_contribution": contribution,
                "risk_contribution": risk_contribution,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown
            }
            
            total_return += contribution
            total_volatility += risk_contribution  # Simplified calculation
        
        # Portfolio-level metrics
        attribution["total_portfolio_metrics"] = {
            "expected_return": total_return,
            "expected_volatility": total_volatility,
            "expected_sharpe": total_return / total_volatility if total_volatility > 0 else 0
        }
        
        # Concentration metrics
        weights_array = np.array(list(latest_weights.values()))
        attribution["concentration_index"] = np.sum(weights_array ** 2)  # Herfindahl index
        attribution["effective_strategies"] = 1.0 / attribution["concentration_index"] if attribution["concentration_index"] > 0 else 0
        
        return attribution


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
            asset_spec_or_default_dict: AssetSpecification | dict[str, Any] = self.asset_specifications.get(action.symbol, {})
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
                 asset_specifications: list[AssetSpecification], 
                 ensemble_config: Optional[EnsembleConfig] = None,
                 **kwargs: dict[str, Any]) -> None:
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
        
        # Enhanced ensemble configuration
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.ensemble_combiner = ConfigurableEnsemble(self.ensemble_config)
        self.agent_performance_history: dict[str, list[float]] = {}
        self.agent_metrics_cache: dict[str, PerformanceMetrics] = {}

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

    # Enhanced ensemble decision making
    async def ensemble_actions(
        self,
        agent_actions: dict[str, StrategyAction],
    ) -> StrategyAction:
        """Combine multiple agent actions using advanced ensemble methods.

        Args:
            agent_actions: Actions from individual agents
        Returns:
            Combined strategy action using configured ensemble method
        """
        if not agent_actions:
            return StrategyAction(
                action_type=ActionType.HOLD,
                symbol="",
                exchange_id="",
                confidence=0.0,
                strategy_id=self.strategy_id,
                reasoning="No agent actions available",
            )

        try:
            # Update agent performance metrics if available
            await self._update_agent_performance_metrics()
            
            # Use the configurable ensemble combiner
            combined_action = self.ensemble_combiner.combine_actions(
                actions=agent_actions,
                performance_metrics=self.agent_metrics_cache
            )
            
            # Ensure the strategy_id is set correctly
            combined_action = StrategyAction(
                action_type=combined_action.action_type,
                symbol=combined_action.symbol,
                exchange_id=combined_action.exchange_id,
                confidence=combined_action.confidence,
                suggested_quantity=combined_action.suggested_quantity,
                suggested_price=combined_action.suggested_price,
                order_type=combined_action.order_type,
                stop_loss_price=combined_action.stop_loss_price,
                take_profit_price=combined_action.take_profit_price,
                max_risk_amount=combined_action.max_risk_amount,
                strategy_id=self.strategy_id,
                strategy_version=combined_action.strategy_version,
                reasoning=f"MARL Ensemble ({self.ensemble_config.method.value}): {combined_action.reasoning}",
                agent_id=None,  # Ensemble result, not from specific agent
                episode_step=self.current_episode,
                q_values=combined_action.q_values,
                valid_until=combined_action.valid_until,
                urgency=combined_action.urgency,
                metadata={
                    **combined_action.metadata,
                    "ensemble_config": {
                        "method": self.ensemble_config.method.value,
                        "performance_window": self.ensemble_config.performance_window,
                        "confidence_threshold": self.ensemble_config.confidence_threshold
                    },
                    "agent_count": len(agent_actions),
                    "episode": self.current_episode
                }
            )
            
            return combined_action
            
        except Exception as e:
            # Fallback to simple majority voting on error
            warnings.warn(f"Error in ensemble combination, falling back to simple voting: {e}")
            return await self._fallback_simple_voting(agent_actions)
    
    async def _update_agent_performance_metrics(self) -> None:
        """Update performance metrics for each agent based on historical performance."""
        try:
            for agent_id, performance_history in self.agent_performance_history.items():
                if len(performance_history) >= self.ensemble_config.min_observations:
                    returns_series = pd.Series(performance_history)
                    
                    # Calculate comprehensive metrics for this agent
                    metrics = StatisticalAnalyzer.calculate_performance_metrics(returns_series)
                    metrics.strategy_id = agent_id
                    
                    self.agent_metrics_cache[agent_id] = metrics
                    
        except Exception as e:
            warnings.warn(f"Error updating agent performance metrics: {e}")
    
    def record_agent_performance(self, agent_id: str, performance_value: float) -> None:
        """Record performance for an agent to update ensemble weights.
        
        Args:
            agent_id: ID of the agent
            performance_value: Performance metric (e.g., return, PnL)
        """
        if agent_id not in self.agent_performance_history:
            self.agent_performance_history[agent_id] = []
        
        self.agent_performance_history[agent_id].append(performance_value)
        
        # Keep only recent history within performance window
        max_history = self.ensemble_config.performance_window
        if len(self.agent_performance_history[agent_id]) > max_history:
            self.agent_performance_history[agent_id] = (
                self.agent_performance_history[agent_id][-max_history:]
            )
    
    async def _fallback_simple_voting(self, agent_actions: dict[str, StrategyAction]) -> StrategyAction:
        """Fallback to simple voting if ensemble method fails."""
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
            reasoning=f"Fallback ensemble of {len(agent_actions)} agents",
            metadata={
                "ensemble_method": "fallback_simple_voting",
                "agent_votes": action_votes,
                "agent_actions": {
                    aid: str(action.action_type)
                    for aid, action in agent_actions.items()
                },
            },
        )


class EnsembleStrategyInterface(StrategyInterface):
    """Interface for ensemble strategies combining multiple sub-strategies with advanced reweighting."""

    def __init__(
        self,
        strategy_id: str,
        sub_strategies: list[StrategyInterface],
        ensemble_config: Optional[EnsembleConfig] = None,
        reweighting_config: Optional[ReweightingConfig] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize ensemble strategy with advanced statistical reweighting.

        Args:
            strategy_id: Unique identifier for this ensemble
            sub_strategies: List of constituent strategies
            ensemble_config: Configuration for ensemble combination methods
            reweighting_config: Configuration for statistical reweighting
            **kwargs: Additional keyword arguments to pass to parent class
        """
        # Combine asset specifications from all sub-strategies
        all_assets = {}
        for strategy in sub_strategies:
            all_assets.update(strategy.asset_specifications)

        super().__init__(strategy_id, StrategyType.ENSEMBLE, list(all_assets.values()), **kwargs)
        self.sub_strategies = sub_strategies
        
        # Enhanced configuration systems
        self.ensemble_config = ensemble_config or EnsembleConfig()
        self.reweighting_config = reweighting_config or ReweightingConfig()
        
        # Advanced components
        self.ensemble_combiner = ConfigurableEnsemble(self.ensemble_config)
        self.reweighting_system = StatisticalReweighting(self.reweighting_config)
        
        # Performance tracking
        self.strategy_performance_history: dict[str, list[float]] = {}
        self.strategy_returns_series: dict[str, pd.Series] = {}
        self.strategy_weights = {s.strategy_id: 1.0 / len(sub_strategies) for s in sub_strategies}
        self.last_rebalance_time: Optional[datetime] = None

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
        performance_metrics: Optional[dict[str, dict[str, float]]] = None,
        force_rebalance: bool = False
    ) -> None:
        """Update weights using advanced statistical reweighting methods.

        Args:
            performance_metrics: Legacy performance data (deprecated, use record_strategy_performance)
            force_rebalance: Force rebalancing regardless of frequency settings
        """
        try:
            current_time = datetime.utcnow()
            
            # Check if rebalancing is needed based on frequency
            if not force_rebalance and self.last_rebalance_time:
                if not self._should_rebalance(current_time):
                    return
            
            # Use statistical reweighting if sufficient data available
            if self.strategy_returns_series and len(self.strategy_returns_series) > 1:
                # Filter strategies with sufficient observations
                valid_series = {
                    sid: series for sid, series in self.strategy_returns_series.items()
                    if len(series) >= self.reweighting_config.min_observations
                }
                
                if valid_series:
                    # Calculate optimal weights using statistical methods
                    new_weights = self.reweighting_system.calculate_optimal_weights(
                        strategy_performances=valid_series,
                        current_weights=self.strategy_weights
                    )
                    
                    # Update weights for strategies with sufficient data
                    for strategy_id, weight in new_weights.items():
                        if strategy_id in self.strategy_weights:
                            self.strategy_weights[strategy_id] = weight
                    
                    # Normalize all weights (including strategies without sufficient data)
                    self._normalize_weights()
                    
                    self.last_rebalance_time = current_time
                    return
            
            # Fallback to legacy performance-based reweighting if provided
            if performance_metrics:
                self._legacy_performance_reweighting(performance_metrics)
                self.last_rebalance_time = current_time
                
        except Exception as e:
            warnings.warn(f"Error in statistical reweighting, using fallback: {e}")
            if performance_metrics:
                self._legacy_performance_reweighting(performance_metrics)
    
    def _should_rebalance(self, current_time: datetime) -> bool:
        """Check if rebalancing is needed based on frequency configuration."""
        if not self.last_rebalance_time:
            return True
        
        time_diff = current_time - self.last_rebalance_time
        
        if self.reweighting_config.rebalance_frequency == "daily":
            return time_diff.days >= 1
        elif self.reweighting_config.rebalance_frequency == "weekly":
            return time_diff.days >= 7
        elif self.reweighting_config.rebalance_frequency == "monthly":
            return time_diff.days >= 30
        elif self.reweighting_config.rebalance_frequency == "quarterly":
            return time_diff.days >= 90
        else:
            return True  # Default to always rebalance
    
    def _legacy_performance_reweighting(self, performance_metrics: dict[str, dict[str, float]]) -> None:
        """Legacy simple performance-based reweighting for backward compatibility."""
        total_performance = sum(
            metrics.get("total_pnl", 0) for metrics in performance_metrics.values()
        )

        if total_performance > 0:
            for strategy_id, metrics in performance_metrics.items():
                if strategy_id in self.strategy_weights:
                    strategy_pnl = metrics.get("total_pnl", 0)
                    self.strategy_weights[strategy_id] = max(
                        self.reweighting_config.min_weight, 
                        min(self.reweighting_config.max_weight, strategy_pnl / total_performance)
                    )

        self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """Normalize strategy weights to sum to 1.0."""
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy_id in self.strategy_weights:
                self.strategy_weights[strategy_id] /= total_weight
    
    def record_strategy_performance(self, strategy_id: str, return_value: float) -> None:
        """Record performance for a strategy to update reweighting calculations.
        
        Args:
            strategy_id: ID of the strategy
            return_value: Return value for this period (e.g., daily return)
        """
        if strategy_id not in self.strategy_performance_history:
            self.strategy_performance_history[strategy_id] = []
        
        self.strategy_performance_history[strategy_id].append(return_value)
        
        # Keep only recent history within lookback window
        max_history = self.reweighting_config.lookback_window
        if len(self.strategy_performance_history[strategy_id]) > max_history:
            self.strategy_performance_history[strategy_id] = (
                self.strategy_performance_history[strategy_id][-max_history:]
            )
        
        # Update pandas series for statistical analysis
        self.strategy_returns_series[strategy_id] = pd.Series(
            self.strategy_performance_history[strategy_id]
        )
    
    async def combine_actions_with_weights(
        self,
        sub_actions: dict[str, StrategyAction]
    ) -> StrategyAction:
        """Combine sub-strategy actions using current weights and ensemble configuration."""
        if not sub_actions:
            return StrategyAction(
                action_type=ActionType.HOLD,
                symbol="",
                exchange_id="",
                confidence=0.0,
                strategy_id=self.strategy_id,
                reasoning="No sub-strategy actions available"
            )
        
        try:
            # Apply current strategy weights to actions
            weighted_actions = {}
            total_weight = 0
            
            for strategy_id, action in sub_actions.items():
                weight = self.strategy_weights.get(strategy_id, 0)
                if weight > 0:
                    # Create weighted action with strategy_id for ensemble processing
                    weighted_action = StrategyAction(
                        action_type=action.action_type,
                        symbol=action.symbol,
                        exchange_id=action.exchange_id,
                        confidence=action.confidence * weight,
                        suggested_quantity=action.suggested_quantity,
                        suggested_price=action.suggested_price,
                        strategy_id=strategy_id,  # Keep original strategy ID
                        reasoning=action.reasoning,
                        metadata={**action.metadata, "ensemble_weight": weight}
                    )
                    weighted_actions[strategy_id] = weighted_action
                    total_weight += weight
            
            if not weighted_actions:
                return await self.combine_actions(sub_actions)
            
            # Get performance metrics for ensemble combination
            performance_metrics = {}
            for strategy_id in weighted_actions.keys():
                if strategy_id in self.reweighting_system.performance_cache:
                    performance_metrics[strategy_id] = self.reweighting_system.performance_cache[strategy_id]
            
            # Use ensemble combiner with performance metrics
            combined_action = self.ensemble_combiner.combine_actions(
                actions=weighted_actions,
                performance_metrics=performance_metrics
            )
            
            # Update metadata with ensemble information
            combined_action = StrategyAction(
                action_type=combined_action.action_type,
                symbol=combined_action.symbol,
                exchange_id=combined_action.exchange_id,
                confidence=combined_action.confidence,
                suggested_quantity=combined_action.suggested_quantity,
                suggested_price=combined_action.suggested_price,
                strategy_id=self.strategy_id,
                reasoning=f"Weighted ensemble: {combined_action.reasoning}",
                metadata={
                    **combined_action.metadata,
                    "ensemble_weights": self.strategy_weights,
                    "reweighting_method": self.reweighting_config.method.value,
                    "total_ensemble_weight": total_weight,
                    "sub_strategy_count": len(weighted_actions)
                }
            )
            
            return combined_action
            
        except Exception as e:
            warnings.warn(f"Error in weighted ensemble combination: {e}")
            return await self.combine_actions(sub_actions)
    
    def get_ensemble_analytics(self) -> dict[str, Any]:
        """Get comprehensive analytics about the ensemble performance and weights."""
        analytics = {
            "current_weights": self.strategy_weights.copy(),
            "reweighting_config": {
                "method": self.reweighting_config.method.value,
                "lookback_window": self.reweighting_config.lookback_window,
                "rebalance_frequency": self.reweighting_config.rebalance_frequency,
                "last_rebalance": self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
            },
            "ensemble_config": {
                "method": self.ensemble_config.method.value,
                "performance_window": self.ensemble_config.performance_window,
                "confidence_threshold": self.ensemble_config.confidence_threshold
            }
        }
        
        # Add performance attribution if available
        if hasattr(self.reweighting_system, 'get_performance_attribution'):
            analytics["performance_attribution"] = self.reweighting_system.get_performance_attribution()
        
        # Add strategy-level analytics
        strategy_analytics = {}
        for strategy_id in self.strategy_weights.keys():
            strategy_analytics[strategy_id] = {
                "current_weight": self.strategy_weights.get(strategy_id, 0),
                "observation_count": len(self.strategy_performance_history.get(strategy_id, [])),
                "has_sufficient_data": len(self.strategy_performance_history.get(strategy_id, [])) >= self.reweighting_config.min_observations
            }
        
        analytics["strategy_analytics"] = strategy_analytics
        
        return analytics


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
