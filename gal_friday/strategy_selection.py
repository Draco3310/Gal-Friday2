"""Enterprise-grade strategy selection system for Gal-Friday.

This module implements a sophisticated multi-criteria decision analysis framework
for selecting optimal trading strategies based on performance metrics, market
conditions, risk alignment, and operational efficiency.
"""

import asyncio
import uuid
import time
import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Deque, Set

import numpy as np
from scipy import stats

# Import from existing Gal-Friday modules
from .logger_service import LoggerService
from .core.events import EventType


# === Data Models ===

class MarketRegime(str, Enum):
    """Market regime classifications."""
    LOW_VOLATILITY = "low_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    HIGH_VOLATILITY = "high_volatility"
    
    
class TrendState(str, Enum):
    """Market trend classifications."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class LiquidityCondition(str, Enum):
    """Market liquidity conditions."""
    NORMAL = "normal"
    REDUCED = "reduced"
    POOR = "poor"


class TransitionPhase(str, Enum):
    """Strategy transition phases."""
    SHADOW_MODE = "shadow_mode"
    PHASE_25_PERCENT = "phase_25_percent"
    PHASE_50_PERCENT = "phase_50_percent"
    PHASE_75_PERCENT = "phase_75_percent"
    FULL_DEPLOYMENT = "full_deployment"


@dataclass
class StrategyPerformanceMetrics:
    """Comprehensive performance metrics for a strategy."""
    strategy_id: str
    evaluation_timestamp: datetime
    
    # Financial metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    average_win: Decimal
    average_loss: Decimal
    largest_win: Decimal
    largest_loss: Decimal
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration_days: float
    current_drawdown: float
    volatility: float
    downside_deviation: float
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional Value at Risk
    
    # Execution metrics
    total_trades: int
    average_slippage_bps: float  # basis points
    fill_rate: float
    average_latency_ms: float
    api_error_rate: float
    
    # Operational metrics
    cpu_usage_avg: float
    memory_usage_avg_mb: float
    signal_generation_rate: float  # signals per hour
    halt_frequency: float  # halts per week
    
    # Return metrics
    total_return: float
    annualized_return: float
    monthly_returns: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    
    # Additional context
    trading_pairs: List[str] = field(default_factory=list)
    market_conditions_exposure: Dict[str, float] = field(default_factory=dict)
    correlation_with_market: float = 0.0
    
    
@dataclass
class MarketConditionSnapshot:
    """Current market condition assessment."""
    timestamp: datetime
    
    # Volatility metrics
    volatility_regime: MarketRegime
    realized_volatility_24h: float
    implied_volatility: Optional[float]
    volatility_percentile: float  # 0-100
    
    # Trend metrics
    trend_state: TrendState
    trend_strength: float  # 0-1
    momentum_score: float  # -1 to 1
    
    # Liquidity metrics
    liquidity_condition: LiquidityCondition
    average_spread_bps: float
    order_book_depth_score: float  # 0-1
    
    # Volume metrics
    volume_24h_usd: Decimal
    volume_percentile: float  # 0-100
    
    # Market session
    active_sessions: List[str]  # ["asia", "europe", "us"]
    
    # Risk factors
    correlation_matrix: Dict[str, Dict[str, float]]
    systemic_risk_score: float  # 0-1
    

@dataclass
class StrategySelectionContext:
    """Context for strategy selection decision."""
    timestamp: datetime
    current_strategy_id: str
    available_strategies: List[str]
    portfolio_state: Dict[str, Any]
    market_conditions: MarketConditionSnapshot
    risk_budget_available: Decimal
    recent_performance_window: timedelta = timedelta(days=30)
    

@dataclass
class StrategyEvaluationResult:
    """Result of strategy evaluation."""
    strategy_id: str
    composite_score: float
    component_scores: Dict[str, float]
    meets_minimum_thresholds: bool
    improvement_over_current: float
    risk_assessment: Dict[str, Any]
    recommendation: str  # "deploy", "monitor", "reject"
    confidence_level: float  # 0-1
    reasons: List[str]
    

@dataclass
class StrategyTransitionPlan:
    """Execution plan for strategy transition."""
    transition_id: str
    from_strategy: str
    to_strategy: str
    phases: List[TransitionPhase]
    current_phase: TransitionPhase
    phase_durations: Dict[TransitionPhase, timedelta]
    rollback_triggers: Dict[str, float]
    validation_checkpoints: List[datetime]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    

# === Core Components ===

class StrategyPerformanceAnalyzer:
    """Analyzes and tracks strategy performance metrics."""

    def __init__(
        self,
        logger: LoggerService,
        config: Dict[str, Any],
        trade_repository: Any,
        portfolio_snapshot_repository: Any,
        execution_repository: Any,
    ) -> None:
        self.logger = logger
        self.config = config
        self.trade_repository = trade_repository
        self.portfolio_snapshot_repository = portfolio_snapshot_repository
        self.execution_repository = execution_repository
        self._source_module = self.__class__.__name__
        
        # Performance history storage
        self._performance_history: Dict[str, Deque[StrategyPerformanceMetrics]] = defaultdict(
            lambda: deque(maxlen=config.get("history_retention_days", 90))
        )
        
        # Cache for computed metrics
        self._metrics_cache: Dict[str, Tuple[datetime, StrategyPerformanceMetrics]] = {}
        self._cache_ttl = timedelta(minutes=config.get("cache_ttl_minutes", 15))
        
        # Benchmark metrics for comparison
        self._benchmark_metrics = config.get("benchmark_metrics", {
            "minimum_sharpe": 1.0,
            "minimum_win_rate": 0.50,
            "maximum_drawdown": 0.10,
            "minimum_profit_factor": 1.5
        })
        
    async def analyze_strategy_performance(
        self, 
        strategy_id: str, 
        lookback_period: timedelta
    ) -> StrategyPerformanceMetrics:
        """Analyze comprehensive performance metrics for a strategy."""
        
        # Check cache first
        if strategy_id in self._metrics_cache:
            cache_time, cached_metrics = self._metrics_cache[strategy_id]
            if datetime.utcnow() - cache_time < self._cache_ttl:
                return cached_metrics
        
        try:
            # Fetch trade data
            trades = await self._fetch_strategy_trades(strategy_id, lookback_period)
            portfolio_snapshots = await self._fetch_portfolio_snapshots(lookback_period)
            execution_metrics = await self._fetch_execution_metrics(strategy_id, lookback_period)
            
            # Calculate all metrics
            metrics = StrategyPerformanceMetrics(
                strategy_id=strategy_id,
                evaluation_timestamp=datetime.utcnow(),
                **self._calculate_financial_metrics(trades),
                **self._calculate_risk_metrics(trades, portfolio_snapshots),
                **self._calculate_execution_metrics(execution_metrics),
                **self._calculate_operational_metrics(strategy_id, lookback_period)
            )
            
            # Update cache and history
            self._metrics_cache[strategy_id] = (datetime.utcnow(), metrics)
            self._performance_history[strategy_id].append(metrics)
            
            # Persist performance metrics if repository is available
            if hasattr(self, 'performance_metrics_repository') and self.performance_metrics_repository:
                try:
                    from .repositories.strategy_repositories import StrategyRepository
                    from .models.strategy_models import StrategyPerformanceSnapshot
                    from decimal import Decimal
                    
                    # First ensure strategy exists in DB
                    strategy_repo = StrategyRepository(self.performance_metrics_repository.session)
                    strategy_config = await strategy_repo.get_by_strategy_id(strategy_id)
                    
                    if strategy_config:
                        # Create performance snapshot
                        snapshot = StrategyPerformanceSnapshot(
                            strategy_config_id=strategy_config.id,
                            snapshot_date=metrics.evaluation_timestamp,
                            evaluation_period_start=metrics.evaluation_timestamp - lookback_period,
                            evaluation_period_end=metrics.evaluation_timestamp,
                            total_return=Decimal(str(metrics.total_return)),
                            annualized_return=Decimal(str(metrics.annualized_return)),
                            sharpe_ratio=Decimal(str(metrics.sharpe_ratio)) if metrics.sharpe_ratio else None,
                            sortino_ratio=Decimal(str(metrics.sortino_ratio)) if metrics.sortino_ratio else None,
                            calmar_ratio=Decimal(str(metrics.calmar_ratio)) if metrics.calmar_ratio else None,
                            max_drawdown=Decimal(str(metrics.max_drawdown)),
                            current_drawdown=Decimal(str(metrics.current_drawdown)),
                            total_trades=metrics.total_trades,
                            win_rate=Decimal(str(metrics.win_rate)),
                            profit_factor=Decimal(str(metrics.profit_factor)) if metrics.profit_factor else None,
                            average_win=metrics.average_win,
                            average_loss=metrics.average_loss,
                            largest_win=metrics.largest_win,
                            largest_loss=metrics.largest_loss,
                            volatility=Decimal(str(metrics.volatility)),
                            downside_deviation=Decimal(str(metrics.downside_deviation)) if metrics.downside_deviation else None,
                            var_95=Decimal(str(metrics.var_95)) if metrics.var_95 else None,
                            cvar_95=Decimal(str(metrics.cvar_95)) if metrics.cvar_95 else None,
                            max_drawdown_duration_days=Decimal(str(metrics.max_drawdown_duration_days)) if metrics.max_drawdown_duration_days else None,
                            average_slippage_bps=Decimal(str(metrics.average_slippage_bps)) if metrics.average_slippage_bps else None,
                            fill_rate=Decimal(str(metrics.fill_rate)),
                            average_latency_ms=Decimal(str(metrics.average_latency_ms)) if metrics.average_latency_ms else None,
                            api_error_rate=Decimal(str(metrics.api_error_rate)),
                            cpu_usage_avg=Decimal(str(metrics.cpu_usage_avg)) if metrics.cpu_usage_avg else None,
                            memory_usage_avg_mb=Decimal(str(metrics.memory_usage_avg_mb)) if metrics.memory_usage_avg_mb else None,
                            signal_generation_rate=Decimal(str(metrics.signal_generation_rate)) if metrics.signal_generation_rate else None,
                            halt_frequency=Decimal(str(metrics.halt_frequency))
                        )
                        
                        await self.performance_metrics_repository.create(snapshot)
                        self.logger.info(
                            f"Persisted performance snapshot for strategy {strategy_id}",
                            source_module=self._source_module
                        )
                except Exception as e:
                    self.logger.error(
                        f"Failed to persist performance metrics: {e}",
                        source_module=self._source_module
                    )
            
            # Log performance summary
            self.logger.info(
                f"Strategy {strategy_id} performance: Sharpe={metrics.sharpe_ratio:.2f}, "
                f"WinRate={metrics.win_rate:.2%}, MaxDD={metrics.max_drawdown:.2%}",
                source_module=self._source_module
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(
                f"Error analyzing performance for strategy {strategy_id}: {e}",
                source_module=self._source_module
            )
            raise
    
    def _calculate_financial_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate financial performance metrics from trades."""
        if not trades:
            return self._empty_financial_metrics()
            
        # Extract P&L data
        pnls = [Decimal(str(t["realized_pnl"])) for t in trades]
        returns = [float(t["realized_pnl_pct"]) for t in trades]
        
        # Basic metrics
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_rate = len(wins) / len(trades) if trades else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else Decimal("0")
        gross_loss = abs(sum(losses)) if losses else Decimal("1")  # Avoid division by zero
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Risk-adjusted returns
        if len(returns) > 1:
            returns_array = np.array(returns)
            
            # Sharpe Ratio (annualized, assuming 252 trading days)
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array, ddof=1)
            sharpe_ratio = np.sqrt(252) * avg_return / std_return if std_return > 0 else 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else std_return
            sortino_ratio = np.sqrt(252) * avg_return / downside_std if downside_std > 0 else 0
            
            # Calmar Ratio (return / max drawdown)
            cumulative_returns = np.cumprod(1 + returns_array / 100) - 1
            max_drawdown = self._calculate_max_drawdown_from_returns(cumulative_returns)
            annualized_return = (1 + np.sum(returns_array / 100)) ** (252 / len(returns_array)) - 1
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
            annualized_return = 0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": sum(wins) / len(wins) if wins else Decimal("0"),
            "average_loss": sum(losses) / len(losses) if losses else Decimal("0"),
            "largest_win": max(wins) if wins else Decimal("0"),
            "largest_loss": min(losses) if losses else Decimal("0"),
            "total_return": float(sum(pnls)),
            "annualized_return": annualized_return,
            "total_trades": len(trades)
        }
    
    def _calculate_risk_metrics(
        self, 
        trades: List[Dict[str, Any]], 
        portfolio_snapshots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate risk metrics from trades and portfolio data."""
        
        # Drawdown analysis from portfolio equity curve
        if portfolio_snapshots:
            equity_curve = [float(s["total_equity"]) for s in portfolio_snapshots]
            drawdowns = self._calculate_drawdown_series(equity_curve)
            
            max_drawdown = abs(min(drawdowns)) if drawdowns else 0
            current_drawdown = abs(drawdowns[-1]) if drawdowns else 0
            
            # Drawdown duration
            dd_durations = self._calculate_drawdown_durations(drawdowns, portfolio_snapshots)
            max_dd_duration = max(dd_durations) if dd_durations else 0
        else:
            max_drawdown = current_drawdown = max_dd_duration = 0
        
        # Volatility and VaR from returns
        if trades and len(trades) > 1:
            returns = np.array([float(t["realized_pnl_pct"]) for t in trades])
            
            volatility = np.std(returns, ddof=1)
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else volatility
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5) if len(returns) > 20 else 0
            
            # Conditional Value at Risk
            returns_below_var = returns[returns <= var_95]
            cvar_95 = np.mean(returns_below_var) if len(returns_below_var) > 0 else var_95
        else:
            volatility = downside_deviation = var_95 = cvar_95 = 0
            
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration_days": max_dd_duration,
            "current_drawdown": current_drawdown,
            "volatility": volatility,
            "downside_deviation": downside_deviation,
            "var_95": var_95,
            "cvar_95": cvar_95
        }
        
    def _calculate_execution_metrics(self, execution_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate execution quality metrics."""
        if not execution_data:
            return {
                "average_slippage_bps": 0,
                "fill_rate": 1.0,
                "average_latency_ms": 0,
                "api_error_rate": 0
            }
            
        # Slippage analysis
        slippages = []
        for exec in execution_data:
            if exec.get("expected_price") and exec.get("actual_price"):
                expected = float(exec["expected_price"])
                actual = float(exec["actual_price"])
                slippage_bps = abs(actual - expected) / expected * 10000
                slippages.append(slippage_bps)
        
        # Fill rate
        total_orders = len(execution_data)
        filled_orders = sum(1 for e in execution_data if e.get("status") == "FILLED")
        fill_rate = filled_orders / total_orders if total_orders > 0 else 1.0
        
        # Latency
        latencies = [float(e["latency_ms"]) for e in execution_data if "latency_ms" in e]
        
        # Error rate
        error_count = sum(1 for e in execution_data if e.get("status") in ["ERROR", "REJECTED"])
        error_rate = error_count / total_orders if total_orders > 0 else 0
        
        return {
            "average_slippage_bps": np.mean(slippages) if slippages else 0,
            "fill_rate": fill_rate,
            "average_latency_ms": np.mean(latencies) if latencies else 0,
            "api_error_rate": error_rate
        }
        
    def _calculate_operational_metrics(
        self, 
        strategy_id: str, 
        lookback_period: timedelta
    ) -> Dict[str, Any]:
        """Calculate operational efficiency metrics."""
        # This would fetch from monitoring service in production
        # For now, return realistic placeholder values
        return {
            "cpu_usage_avg": 45.0,  # percentage
            "memory_usage_avg_mb": 512.0,
            "signal_generation_rate": 12.5,  # per hour
            "halt_frequency": 0.2  # per week
        }
        
    def _calculate_max_drawdown_from_returns(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns series."""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        return float(np.min(drawdown))
        
    def _calculate_drawdown_series(self, equity_curve: List[float]) -> List[float]:
        """Calculate drawdown series from equity curve."""
        if not equity_curve:
            return []
            
        peak = equity_curve[0]
        drawdowns = []
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak if peak > 0 else 0
            drawdowns.append(drawdown)
            
        return drawdowns
        
    def _calculate_drawdown_durations(
        self, 
        drawdowns: List[float], 
        snapshots: List[Dict[str, Any]]
    ) -> List[float]:
        """Calculate duration of each drawdown period in days."""
        if not drawdowns or not snapshots:
            return []
            
        durations = []
        current_duration = 0
        in_drawdown = False
        
        for i, dd in enumerate(drawdowns):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_time = snapshots[i]["snapshot_timestamp"]
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                end_time = snapshots[i]["snapshot_timestamp"]
                duration = (end_time - start_time).total_seconds() / 86400  # days
                durations.append(duration)
                
        return durations
        
    def _empty_financial_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure when no data available."""
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_win": Decimal("0"),
            "average_loss": Decimal("0"),
            "largest_win": Decimal("0"),
            "largest_loss": Decimal("0"),
            "total_return": 0.0,
            "annualized_return": 0.0,
            "total_trades": 0
        }
        
    async def _fetch_strategy_trades(
        self,
        strategy_id: str,
        lookback_period: timedelta
    ) -> List[Dict[str, Any]]:
        """Fetch trades for a strategy within lookback period."""
        start_date = datetime.utcnow() - lookback_period
        end_date = datetime.utcnow()
        try:
            trades = await self.trade_repository.get_trades_by_strategy(
                strategy_id, start_date, end_date
            )
        except Exception as exc:  # pragma: no cover - DB errors
            self.logger.error(
                f"Failed to fetch trades for {strategy_id}: {exc}",
                source_module=self._source_module,
            )
            return []

        return [t.to_dict() if hasattr(t, "to_dict") else t for t in trades]
        
    async def _fetch_portfolio_snapshots(
        self,
        lookback_period: timedelta
    ) -> List[Dict[str, Any]]:
        """Fetch portfolio snapshots within lookback period."""
        start_date = datetime.utcnow() - lookback_period
        end_date = datetime.utcnow()
        try:
            snapshots = await self.portfolio_snapshot_repository.get_snapshots_in_period(
                start_date, end_date
            )
        except Exception as exc:  # pragma: no cover - DB errors
            self.logger.error(
                f"Failed to fetch portfolio snapshots: {exc}",
                source_module=self._source_module,
            )
            return []

        return [s.to_dict() if hasattr(s, "to_dict") else s for s in snapshots]
        
    async def _fetch_execution_metrics(
        self,
        strategy_id: str,
        lookback_period: timedelta
    ) -> List[Dict[str, Any]]:
        """Fetch execution metrics for a strategy."""
        start_date = datetime.utcnow() - lookback_period
        end_date = datetime.utcnow()
        try:
            metrics = await self.execution_repository.get_metrics_by_strategy(
                strategy_id, start_date, end_date
            )
        except Exception as exc:  # pragma: no cover - DB errors
            self.logger.error(
                f"Failed to fetch execution metrics for {strategy_id}: {exc}",
                source_module=self._source_module,
            )
            return []

        return [m.to_dict() if hasattr(m, "to_dict") else m for m in metrics]
        
    def get_performance_trend(
        self, 
        strategy_id: str, 
        metric: str, 
        periods: int = 10
    ) -> Optional[float]:
        """Calculate trend of a specific metric over recent periods."""
        history = self._performance_history.get(strategy_id, [])
        
        if len(history) < 2:
            return None
            
        # Get recent values for the metric
        recent_values = []
        for metrics in list(history)[-periods:]:
            value = getattr(metrics, metric, None)
            if value is not None:
                recent_values.append(float(value))
                
        if len(recent_values) < 2:
            return None
            
        # Calculate linear regression slope
        x = np.arange(len(recent_values))
        slope, _ = np.polyfit(x, recent_values, 1)
        
        return slope


class MarketConditionMonitor:
    """Monitors and classifies current market conditions."""
    
    def __init__(self, logger: LoggerService, config: Dict[str, Any]):
        self.logger = logger
        self.config = config
        self._source_module = self.__class__.__name__
        
        # Thresholds for market regime classification
        self._volatility_thresholds = config.get("volatility_thresholds", {
            "low": 0.02,  # 2% daily
            "high": 0.05  # 5% daily
        })
        
        self._liquidity_thresholds = config.get("liquidity_thresholds", {
            "poor_spread_bps": 30,  # 0.3%
            "reduced_spread_bps": 10  # 0.1%
        })
        
        # Cache for market conditions
        self._condition_cache: Optional[Tuple[datetime, MarketConditionSnapshot]] = None
        self._cache_ttl = timedelta(minutes=config.get("cache_ttl_minutes", 5))
        
    async def assess_market_conditions(
        self, 
        trading_pairs: List[str]
    ) -> MarketConditionSnapshot:
        """Assess current market conditions across trading pairs."""
        
        # Check cache
        if self._condition_cache:
            cache_time, cached_snapshot = self._condition_cache
            if datetime.utcnow() - cache_time < self._cache_ttl:
                return cached_snapshot
                
        try:
            # Gather market data
            volatility_data = await self._calculate_volatility_metrics(trading_pairs)
            trend_data = await self._analyze_trend_state(trading_pairs)
            liquidity_data = await self._assess_liquidity_conditions(trading_pairs)
            volume_data = await self._analyze_volume_patterns(trading_pairs)
            
            # Classify market regime
            volatility_regime = self._classify_volatility_regime(
                volatility_data["realized_volatility_24h"]
            )
            
            trend_state = self._classify_trend_state(
                trend_data["trend_strength"],
                trend_data["momentum_score"]
            )
            
            liquidity_condition = self._classify_liquidity_condition(
                liquidity_data["average_spread_bps"]
            )
            
            # Determine active sessions
            active_sessions = self._determine_active_sessions()
            
            # Create snapshot
            snapshot = MarketConditionSnapshot(
                timestamp=datetime.utcnow(),
                volatility_regime=volatility_regime,
                realized_volatility_24h=volatility_data["realized_volatility_24h"],
                implied_volatility=volatility_data.get("implied_volatility"),
                volatility_percentile=volatility_data["volatility_percentile"],
                trend_state=trend_state,
                trend_strength=trend_data["trend_strength"],
                momentum_score=trend_data["momentum_score"],
                liquidity_condition=liquidity_condition,
                average_spread_bps=liquidity_data["average_spread_bps"],
                order_book_depth_score=liquidity_data["depth_score"],
                volume_24h_usd=Decimal(str(volume_data["volume_24h_usd"])),
                volume_percentile=volume_data["volume_percentile"],
                active_sessions=active_sessions,
                correlation_matrix=await self._calculate_correlation_matrix(trading_pairs),
                systemic_risk_score=await self._assess_systemic_risk()
            )
            
            # Update cache
            self._condition_cache = (datetime.utcnow(), snapshot)
            
            # Log market assessment
            self.logger.info(
                f"Market conditions: {volatility_regime.value} volatility, "
                f"{trend_state.value}, {liquidity_condition.value} liquidity",
                source_module=self._source_module
            )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(
                f"Error assessing market conditions: {e}",
                source_module=self._source_module
            )
            raise
            
    def _classify_volatility_regime(self, volatility: float) -> MarketRegime:
        """Classify volatility regime based on thresholds."""
        if volatility < self._volatility_thresholds["low"]:
            return MarketRegime.LOW_VOLATILITY
        elif volatility > self._volatility_thresholds["high"]:
            return MarketRegime.HIGH_VOLATILITY
        else:
            return MarketRegime.NORMAL_VOLATILITY
            
    def _classify_trend_state(self, trend_strength: float, momentum: float) -> TrendState:
        """Classify trend state based on strength and momentum."""
        if trend_strength > 0.7:
            return TrendState.STRONG_UPTREND if momentum > 0 else TrendState.STRONG_DOWNTREND
        elif trend_strength > 0.3:
            return TrendState.WEAK_UPTREND if momentum > 0 else TrendState.WEAK_DOWNTREND
        else:
            return TrendState.SIDEWAYS
            
    def _classify_liquidity_condition(self, spread_bps: float) -> LiquidityCondition:
        """Classify liquidity based on spread."""
        if spread_bps > self._liquidity_thresholds["poor_spread_bps"]:
            return LiquidityCondition.POOR
        elif spread_bps > self._liquidity_thresholds["reduced_spread_bps"]:
            return LiquidityCondition.REDUCED
        else:
            return LiquidityCondition.NORMAL
            
    def _determine_active_sessions(self) -> List[str]:
        """Determine which major trading sessions are active."""
        current_hour_utc = datetime.utcnow().hour
        
        active_sessions = []
        
        # Simplified session times (UTC)
        if 23 <= current_hour_utc or current_hour_utc < 9:
            active_sessions.append("asia")
        if 7 <= current_hour_utc < 16:
            active_sessions.append("europe")
        if 13 <= current_hour_utc < 22:
            active_sessions.append("us")
            
        return active_sessions
        
    async def _calculate_volatility_metrics(
        self, 
        trading_pairs: List[str]
    ) -> Dict[str, float]:
        """Calculate volatility metrics across trading pairs."""
        # This would fetch market data and calculate metrics in production
        # Placeholder implementation
        return {
            "realized_volatility_24h": 0.035,  # 3.5% daily volatility
            "implied_volatility": None,  # Would come from options if available
            "volatility_percentile": 65.0  # Current vol at 65th percentile historically
        }
        
    async def _analyze_trend_state(
        self, 
        trading_pairs: List[str]
    ) -> Dict[str, float]:
        """Analyze trend strength and momentum."""
        # This would analyze price action in production
        # Placeholder implementation
        return {
            "trend_strength": 0.45,  # 0-1 scale
            "momentum_score": 0.2  # -1 to 1 scale
        }
        
    async def _assess_liquidity_conditions(
        self, 
        trading_pairs: List[str]
    ) -> Dict[str, float]:
        """Assess market liquidity conditions."""
        # This would analyze order book data in production
        # Placeholder implementation
        return {
            "average_spread_bps": 8.5,  # basis points
            "depth_score": 0.75  # 0-1 scale
        }
        
    async def _analyze_volume_patterns(
        self, 
        trading_pairs: List[str]
    ) -> Dict[str, Any]:
        """Analyze volume patterns and percentiles."""
        # This would analyze volume data in production
        # Placeholder implementation
        return {
            "volume_24h_usd": 1500000.0,
            "volume_percentile": 72.0
        }
        
    async def _calculate_correlation_matrix(
        self, 
        trading_pairs: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between trading pairs."""
        # This would calculate actual correlations in production
        # Placeholder implementation
        correlations = {}
        for pair1 in trading_pairs:
            correlations[pair1] = {}
            for pair2 in trading_pairs:
                if pair1 == pair2:
                    correlations[pair1][pair2] = 1.0
                else:
                    # Placeholder correlation
                    correlations[pair1][pair2] = 0.65
                    
        return correlations
        
    async def _assess_systemic_risk(self) -> float:
        """Assess overall systemic risk in the market."""
        # This would analyze broader market indicators in production
        # Placeholder implementation
        return 0.3  # 0-1 scale


class RiskEvaluator:
    """Evaluates strategy risk alignment with portfolio constraints."""
    
    def __init__(
        self, 
        logger: LoggerService, 
        config: Dict[str, Any],
        risk_manager,  # Reference to RiskManager
        portfolio_manager  # Reference to PortfolioManager
    ):
        self.logger = logger
        self.config = config
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self._source_module = self.__class__.__name__
        
        # Risk thresholds
        self._risk_limits = config.get("risk_limits", {
            "max_strategy_drawdown": 0.10,  # 10%
            "max_correlation": 0.70,
            "min_risk_budget_buffer": 0.20,  # Keep 20% buffer
            "max_volatility_exposure": 0.15  # 15% portfolio volatility
        })
        
    async def evaluate_strategy_risk(
        self,
        strategy_id: str,
        strategy_metrics: StrategyPerformanceMetrics,
        selection_context: StrategySelectionContext
    ) -> Dict[str, Any]:
        """Comprehensive risk evaluation for strategy selection."""
        
        try:
            # Get current portfolio state
            portfolio_state = self.portfolio_manager.get_current_state()
            
            # Evaluate different risk dimensions
            drawdown_assessment = self._assess_drawdown_risk(
                strategy_metrics,
                portfolio_state
            )
            
            correlation_assessment = await self._assess_correlation_risk(
                strategy_id,
                selection_context.available_strategies
            )
            
            capacity_assessment = self._assess_capacity_constraints(
                strategy_metrics,
                portfolio_state,
                selection_context.risk_budget_available
            )
            
            volatility_assessment = self._assess_volatility_alignment(
                strategy_metrics,
                selection_context.market_conditions
            )
            
            # Aggregate risk score (0-1, lower is better)
            risk_score = self._calculate_aggregate_risk_score({
                "drawdown": drawdown_assessment,
                "correlation": correlation_assessment,
                "capacity": capacity_assessment,
                "volatility": volatility_assessment
            })
            
            # Determine if strategy passes risk criteria
            passes_risk_checks = all([
                drawdown_assessment["acceptable"],
                correlation_assessment["acceptable"],
                capacity_assessment["acceptable"],
                volatility_assessment["acceptable"]
            ])
            
            risk_evaluation = {
                "strategy_id": strategy_id,
                "risk_score": risk_score,
                "passes_checks": passes_risk_checks,
                "assessments": {
                    "drawdown": drawdown_assessment,
                    "correlation": correlation_assessment,
                    "capacity": capacity_assessment,
                    "volatility": volatility_assessment
                },
                "recommendations": self._generate_risk_recommendations(
                    strategy_metrics,
                    risk_score,
                    passes_risk_checks
                )
            }
            
            # Log risk assessment
            self.logger.info(
                f"Risk evaluation for {strategy_id}: score={risk_score:.3f}, "
                f"passes={passes_risk_checks}",
                source_module=self._source_module
            )
            
            return risk_evaluation
            
        except Exception as e:
            self.logger.error(
                f"Error evaluating risk for strategy {strategy_id}: {e}",
                source_module=self._source_module
            )
            raise
            
    def _assess_drawdown_risk(
        self,
        strategy_metrics: StrategyPerformanceMetrics,
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess drawdown risk of strategy."""
        
        # Current portfolio drawdown
        current_portfolio_dd = float(portfolio_state.get("total_drawdown_pct", 0))
        
        # Strategy's historical max drawdown
        strategy_max_dd = strategy_metrics.max_drawdown
        
        # Projected combined drawdown (conservative estimate)
        projected_dd = current_portfolio_dd + strategy_max_dd * 0.7  # Assume some diversification
        
        # Check against limits
        acceptable = strategy_max_dd <= self._risk_limits["max_strategy_drawdown"]
        
        return {
            "acceptable": acceptable,
            "strategy_max_drawdown": strategy_max_dd,
            "current_portfolio_drawdown": current_portfolio_dd,
            "projected_combined_drawdown": projected_dd,
            "limit": self._risk_limits["max_strategy_drawdown"],
            "severity": "low" if strategy_max_dd < 0.05 else "medium" if strategy_max_dd < 0.10 else "high"
        }
        
    async def _assess_correlation_risk(
        self,
        strategy_id: str,
        active_strategies: List[str]
    ) -> Dict[str, Any]:
        """Assess correlation risk with existing strategies."""
        
        # This would calculate actual correlations in production
        # Placeholder implementation
        max_correlation = 0.45  # Placeholder
        correlated_strategies = []
        
        acceptable = max_correlation <= self._risk_limits["max_correlation"]
        
        return {
            "acceptable": acceptable,
            "max_correlation": max_correlation,
            "correlated_strategies": correlated_strategies,
            "limit": self._risk_limits["max_correlation"],
            "diversification_score": 1 - max_correlation
        }
        
    def _assess_capacity_constraints(
        self,
        strategy_metrics: StrategyPerformanceMetrics,
        portfolio_state: Dict[str, Any],
        risk_budget_available: Decimal
    ) -> Dict[str, Any]:
        """Assess if portfolio has capacity for strategy."""
        
        # Estimate risk budget usage based on historical metrics
        avg_position_size = Decimal("5000")  # Placeholder - would calculate from trades
        typical_risk_per_trade = avg_position_size * Decimal(str(strategy_metrics.var_95 / 100))
        
        # Expected concurrent positions
        signals_per_day = strategy_metrics.signal_generation_rate * 24
        avg_holding_period_hours = 4  # Placeholder
        expected_concurrent_positions = signals_per_day * avg_holding_period_hours / 24
        
        # Total risk budget needed
        strategy_risk_budget_needed = typical_risk_per_trade * Decimal(str(expected_concurrent_positions))
        
        # Check if we have enough buffer
        risk_budget_after = risk_budget_available - strategy_risk_budget_needed
        buffer_ratio = float(risk_budget_after / risk_budget_available) if risk_budget_available > 0 else 0
        
        acceptable = buffer_ratio >= self._risk_limits["min_risk_budget_buffer"]
        
        return {
            "acceptable": acceptable,
            "risk_budget_available": float(risk_budget_available),
            "strategy_budget_needed": float(strategy_risk_budget_needed),
            "remaining_buffer_ratio": buffer_ratio,
            "min_buffer_required": self._risk_limits["min_risk_budget_buffer"]
        }
        
    def _assess_volatility_alignment(
        self,
        strategy_metrics: StrategyPerformanceMetrics,
        market_conditions: MarketConditionSnapshot
    ) -> Dict[str, Any]:
        """Assess if strategy volatility aligns with market conditions."""
        
        # Strategy's volatility profile
        strategy_vol = strategy_metrics.volatility
        
        # Market volatility regime
        market_vol = market_conditions.realized_volatility_24h
        
        # Check alignment
        vol_ratio = strategy_vol / market_vol if market_vol > 0 else 1
        
        # Strategies should not be too aggressive relative to market
        acceptable = strategy_vol <= self._risk_limits["max_volatility_exposure"]
        
        # Also check if strategy is appropriate for current regime
        regime_appropriate = True
        if market_conditions.volatility_regime == MarketRegime.HIGH_VOLATILITY:
            # In high vol, prefer lower vol strategies
            regime_appropriate = strategy_vol < market_vol * 1.2
        elif market_conditions.volatility_regime == MarketRegime.LOW_VOLATILITY:
            # In low vol, can accept higher vol strategies
            regime_appropriate = strategy_vol < market_vol * 2.0
            
        return {
            "acceptable": acceptable and regime_appropriate,
            "strategy_volatility": strategy_vol,
            "market_volatility": market_vol,
            "volatility_ratio": vol_ratio,
            "regime_appropriate": regime_appropriate,
            "market_regime": market_conditions.volatility_regime.value
        }
        
    def _calculate_aggregate_risk_score(
        self,
        assessments: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate aggregate risk score from individual assessments."""
        
        weights = {
            "drawdown": 0.35,
            "correlation": 0.25,
            "capacity": 0.25,
            "volatility": 0.15
        }
        
        risk_scores = {}
        
        # Convert each assessment to a risk score (0-1)
        # Drawdown risk
        dd_assessment = assessments["drawdown"]
        risk_scores["drawdown"] = min(
            dd_assessment["strategy_max_drawdown"] / dd_assessment["limit"],
            1.0
        )
        
        # Correlation risk
        corr_assessment = assessments["correlation"]
        risk_scores["correlation"] = min(
            corr_assessment["max_correlation"] / corr_assessment["limit"],
            1.0
        )
        
        # Capacity risk
        cap_assessment = assessments["capacity"]
        risk_scores["capacity"] = 1.0 - cap_assessment["remaining_buffer_ratio"]
        
        # Volatility risk
        vol_assessment = assessments["volatility"]
        risk_scores["volatility"] = min(vol_assessment["volatility_ratio"], 1.0)
        
        # Weighted average
        aggregate_score = sum(
            risk_scores[key] * weights[key]
            for key in weights
        )
        
        return aggregate_score
        
    def _generate_risk_recommendations(
        self,
        strategy_metrics: StrategyPerformanceMetrics,
        risk_score: float,
        passes_checks: bool
    ) -> List[str]:
        """Generate risk-based recommendations."""
        
        recommendations = []
        
        if not passes_checks:
            recommendations.append("Strategy does not meet risk criteria for deployment")
            
        if risk_score > 0.7:
            recommendations.append("High risk score - consider risk reduction measures")
        elif risk_score > 0.5:
            recommendations.append("Moderate risk - monitor closely during deployment")
        else:
            recommendations.append("Risk profile acceptable for deployment")
            
        if strategy_metrics.max_drawdown > 0.08:
            recommendations.append("Consider tighter stop-loss parameters to reduce drawdown")
            
        if strategy_metrics.volatility > 0.04:
            recommendations.append("High volatility - ensure position sizing accounts for this")
            
        return recommendations


class StrategySelector:
    """Core decision engine for strategy selection using multi-criteria analysis."""
    
    def __init__(
        self,
        logger: LoggerService,
        config: Dict[str, Any],
        performance_analyzer: StrategyPerformanceAnalyzer,
        market_monitor: MarketConditionMonitor,
        risk_evaluator: RiskEvaluator
    ):
        self.logger = logger
        self.config = config
        self.performance_analyzer = performance_analyzer
        self.market_monitor = market_monitor
        self.risk_evaluator = risk_evaluator
        self._source_module = self.__class__.__name__
        
        # Decision criteria weights
        self._criteria_weights = config.get("criteria_weights", {
            "performance_score": 0.40,
            "risk_alignment": 0.35,
            "market_fit": 0.20,
            "operational_efficiency": 0.05
        })
        
        # Selection thresholds
        self._selection_thresholds = config.get("selection_thresholds", {
            "minimum_composite_score": 0.60,
            "improvement_required": 0.10,  # 10% improvement
            "confidence_threshold": 0.70
        })
        
        # Historical selection decisions
        self._decision_history: Deque[StrategyEvaluationResult] = deque(maxlen=100)
        
    async def select_optimal_strategy(
        self,
        selection_context: StrategySelectionContext
    ) -> StrategyEvaluationResult:
        """Select the optimal strategy based on multi-criteria analysis."""
        
        try:
            self.logger.info(
                f"Starting strategy selection. Current: {selection_context.current_strategy_id}, "
                f"Candidates: {selection_context.available_strategies}",
                source_module=self._source_module
            )
            
            # Evaluate all candidate strategies
            evaluations = []
            for strategy_id in selection_context.available_strategies:
                evaluation = await self._evaluate_strategy(strategy_id, selection_context)
                evaluations.append(evaluation)
                
            # Sort by composite score
            evaluations.sort(key=lambda x: x.composite_score, reverse=True)
            
            # Select best strategy that meets criteria
            selected_evaluation = self._apply_selection_criteria(
                evaluations,
                selection_context.current_strategy_id
            )
            
            # Record decision
            self._decision_history.append(selected_evaluation)
            
            # Log decision
            self.logger.info(
                f"Strategy selection complete. Selected: {selected_evaluation.strategy_id}, "
                f"Score: {selected_evaluation.composite_score:.3f}, "
                f"Recommendation: {selected_evaluation.recommendation}",
                source_module=self._source_module
            )
            
            return selected_evaluation
            
        except Exception as e:
            self.logger.error(
                f"Error in strategy selection: {e}",
                source_module=self._source_module
            )
            # Return conservative decision on error
            return self._create_fallback_evaluation(selection_context.current_strategy_id)
            
    async def _evaluate_strategy(
        self,
        strategy_id: str,
        selection_context: StrategySelectionContext
    ) -> StrategyEvaluationResult:
        """Evaluate a single strategy across all criteria."""
        
        # Get performance metrics
        performance_metrics = await self.performance_analyzer.analyze_strategy_performance(
            strategy_id,
            selection_context.recent_performance_window
        )
        
        # Calculate component scores
        performance_score = self._calculate_performance_score(performance_metrics)
        
        risk_evaluation = await self.risk_evaluator.evaluate_strategy_risk(
            strategy_id,
            performance_metrics,
            selection_context
        )
        risk_score = 1.0 - risk_evaluation["risk_score"]  # Convert to positive score
        
        market_fit_score = self._calculate_market_fit_score(
            performance_metrics,
            selection_context.market_conditions
        )
        
        operational_score = self._calculate_operational_score(performance_metrics)
        
        # Calculate weighted composite score
        component_scores = {
            "performance_score": performance_score,
            "risk_alignment": risk_score,
            "market_fit": market_fit_score,
            "operational_efficiency": operational_score
        }
        
        composite_score = sum(
            score * self._criteria_weights[criterion]
            for criterion, score in component_scores.items()
        )
        
        # Check if meets minimum thresholds
        meets_thresholds = all([
            performance_metrics.sharpe_ratio >= 1.0,
            performance_metrics.win_rate >= 0.50,
            performance_metrics.max_drawdown <= 0.10,
            risk_evaluation["passes_checks"]
        ])
        
        # Calculate improvement over current
        improvement = 0.0
        if strategy_id != selection_context.current_strategy_id:
            current_metrics = await self.performance_analyzer.analyze_strategy_performance(
                selection_context.current_strategy_id,
                selection_context.recent_performance_window
            )
            current_score = self._calculate_performance_score(current_metrics)
            improvement = (performance_score - current_score) / current_score if current_score > 0 else 0
            
        # Generate recommendation
        recommendation = self._generate_recommendation(
            strategy_id,
            composite_score,
            meets_thresholds,
            improvement,
            risk_evaluation
        )
        
        # Calculate confidence level
        confidence = self._calculate_decision_confidence(
            performance_metrics,
            component_scores,
            selection_context
        )
        
        # Compile reasons
        reasons = self._compile_selection_reasons(
            strategy_id,
            performance_metrics,
            component_scores,
            risk_evaluation,
            selection_context
        )
        
        return StrategyEvaluationResult(
            strategy_id=strategy_id,
            composite_score=composite_score,
            component_scores=component_scores,
            meets_minimum_thresholds=meets_thresholds,
            improvement_over_current=improvement,
            risk_assessment=risk_evaluation,
            recommendation=recommendation,
            confidence_level=confidence,
            reasons=reasons
        )
        
    def _calculate_performance_score(
        self,
        metrics: StrategyPerformanceMetrics
    ) -> float:
        """Calculate normalized performance score (0-1)."""
        
        # Define ideal targets and weights
        performance_factors = {
            "sharpe_ratio": {
                "value": metrics.sharpe_ratio,
                "target": 2.0,
                "weight": 0.30
            },
            "win_rate": {
                "value": metrics.win_rate,
                "target": 0.65,
                "weight": 0.25
            },
            "profit_factor": {
                "value": metrics.profit_factor,
                "target": 2.0,
                "weight": 0.20
            },
            "max_drawdown": {
                "value": 1.0 - metrics.max_drawdown,  # Invert so higher is better
                "target": 0.95,  # Max 5% drawdown
                "weight": 0.25
            }
        }
        
        # Calculate weighted score
        score = 0.0
        for factor, data in performance_factors.items():
            # Normalize to target (capped at 1.0)
            normalized = min(data["value"] / data["target"], 1.0) if data["target"] > 0 else 0
            score += normalized * data["weight"]
            
        return score
        
    def _calculate_market_fit_score(
        self,
        metrics: StrategyPerformanceMetrics,
        market_conditions: MarketConditionSnapshot
    ) -> float:
        """Calculate how well strategy fits current market conditions."""
        
        score = 0.0
        
        # Volatility alignment
        if market_conditions.volatility_regime == MarketRegime.HIGH_VOLATILITY:
            # Prefer lower volatility strategies in high vol markets
            vol_score = 1.0 - min(metrics.volatility / 0.10, 1.0)
        elif market_conditions.volatility_regime == MarketRegime.LOW_VOLATILITY:
            # In low vol, moderate volatility is acceptable
            vol_score = 1.0 - abs(metrics.volatility - 0.03) / 0.03
        else:
            vol_score = 0.8  # Neutral
            
        score += vol_score * 0.4
        
        # Trend alignment
        if market_conditions.trend_state in [TrendState.STRONG_UPTREND, TrendState.STRONG_DOWNTREND]:
            # Strong trends favor trend-following
            if "trend" in metrics.strategy_id.lower():
                trend_score = 0.9
            else:
                trend_score = 0.6
        else:
            # Sideways markets favor mean reversion
            if "reversion" in metrics.strategy_id.lower() or "scalp" in metrics.strategy_id.lower():
                trend_score = 0.9
            else:
                trend_score = 0.7
                
        score += trend_score * 0.3
        
        # Liquidity conditions
        if market_conditions.liquidity_condition == LiquidityCondition.POOR:
            # Poor liquidity penalizes high-frequency strategies
            if metrics.signal_generation_rate > 20:  # >20 signals/hour
                liquidity_score = 0.4
            else:
                liquidity_score = 0.8
        else:
            liquidity_score = 0.9
            
        score += liquidity_score * 0.3
        
        return score
        
    def _calculate_operational_score(
        self,
        metrics: StrategyPerformanceMetrics
    ) -> float:
        """Calculate operational efficiency score."""
        
        factors = {
            "latency": {
                "value": 1.0 - min(metrics.average_latency_ms / 100, 1.0),
                "weight": 0.30
            },
            "fill_rate": {
                "value": metrics.fill_rate,
                "weight": 0.25
            },
            "resource_usage": {
                "value": 1.0 - (metrics.cpu_usage_avg / 100 + metrics.memory_usage_avg_mb / 4096) / 2,
                "weight": 0.25
            },
            "reliability": {
                "value": 1.0 - min(metrics.halt_frequency / 2, 1.0),  # <0.5 halts/week is good
                "weight": 0.20
            }
        }
        
        score = sum(
            data["value"] * data["weight"]
            for data in factors.values()
        )
        
        return score
        
    def _apply_selection_criteria(
        self,
        evaluations: List[StrategyEvaluationResult],
        current_strategy_id: str
    ) -> StrategyEvaluationResult:
        """Apply selection criteria to choose best strategy."""
        
        # Find current strategy evaluation
        current_eval = next(
            (e for e in evaluations if e.strategy_id == current_strategy_id),
            None
        )
        
        # Check each candidate
        for candidate in evaluations:
            # Skip if doesn't meet minimum thresholds
            if not candidate.meets_minimum_thresholds:
                continue
                
            # Skip if composite score too low
            if candidate.composite_score < self._selection_thresholds["minimum_composite_score"]:
                continue
                
            # If it's the current strategy, keep it unless something better
            if candidate.strategy_id == current_strategy_id:
                continue
                
            # Check if improvement is significant enough
            if current_eval and candidate.improvement_over_current >= self._selection_thresholds["improvement_required"]:
                return candidate
                
            # If no current eval (first run), take the best that meets criteria
            if not current_eval:
                return candidate
                
        # Default to keeping current strategy
        if current_eval:
            return current_eval
        else:
            # No suitable strategy found, return the best available
            return evaluations[0] if evaluations else self._create_fallback_evaluation("default")
            
    def _generate_recommendation(
        self,
        strategy_id: str,
        composite_score: float,
        meets_thresholds: bool,
        improvement: float,
        risk_evaluation: Dict[str, Any]
    ) -> str:
        """Generate recommendation for strategy."""
        
        if not meets_thresholds:
            return "reject"
        elif not risk_evaluation["passes_checks"]:
            return "monitor"
        elif composite_score >= 0.75 and improvement >= 0.15:
            return "deploy"
        elif composite_score >= 0.65:
            return "monitor"
        else:
            return "reject"
            
    def _calculate_decision_confidence(
        self,
        metrics: StrategyPerformanceMetrics,
        component_scores: Dict[str, float],
        context: StrategySelectionContext
    ) -> float:
        """Calculate confidence level in the decision."""
        
        confidence_factors = []
        
        # Data sufficiency
        data_confidence = min(metrics.total_trades / 100, 1.0)  # 100+ trades = full confidence
        confidence_factors.append(data_confidence * 0.3)
        
        # Score consistency
        score_variance = np.var(list(component_scores.values()))
        consistency_confidence = 1.0 - min(score_variance * 2, 1.0)
        confidence_factors.append(consistency_confidence * 0.2)
        
        # Market condition stability
        if context.market_conditions.volatility_regime == MarketRegime.NORMAL_VOLATILITY:
            market_confidence = 0.9
        else:
            market_confidence = 0.7
        confidence_factors.append(market_confidence * 0.2)
        
        # Historical performance stability
        performance_trend = self.performance_analyzer.get_performance_trend(
            metrics.strategy_id,
            "sharpe_ratio",
            periods=10
        )
        if performance_trend is not None:
            stability_confidence = 1.0 - min(abs(performance_trend) * 10, 1.0)
        else:
            stability_confidence = 0.5
        confidence_factors.append(stability_confidence * 0.3)
        
        return sum(confidence_factors)
        
    def _compile_selection_reasons(
        self,
        strategy_id: str,
        metrics: StrategyPerformanceMetrics,
        component_scores: Dict[str, float],
        risk_evaluation: Dict[str, Any],
        context: StrategySelectionContext
    ) -> List[str]:
        """Compile human-readable reasons for selection decision."""
        
        reasons = []
        
        # Performance reasons
        if metrics.sharpe_ratio >= 2.0:
            reasons.append(f"Excellent risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
        elif metrics.sharpe_ratio >= 1.5:
            reasons.append(f"Good risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
            
        if metrics.win_rate >= 0.65:
            reasons.append(f"High win rate: {metrics.win_rate:.1%}")
            
        if metrics.max_drawdown <= 0.05:
            reasons.append(f"Low maximum drawdown: {metrics.max_drawdown:.1%}")
            
        # Risk reasons
        if risk_evaluation["passes_checks"]:
            reasons.append("Passes all risk management criteria")
        else:
            reasons.append("Some risk concerns identified")
            
        # Market fit reasons
        market_fit = component_scores.get("market_fit", 0)
        if market_fit >= 0.8:
            reasons.append(f"Well-suited for current {context.market_conditions.volatility_regime.value} market")
            
        # Operational reasons
        if metrics.average_latency_ms < 30:
            reasons.append(f"Excellent execution latency: {metrics.average_latency_ms:.0f}ms")
            
        return reasons
        
    def _create_fallback_evaluation(self, strategy_id: str) -> StrategyEvaluationResult:
        """Create conservative fallback evaluation."""
        
        return StrategyEvaluationResult(
            strategy_id=strategy_id,
            composite_score=0.5,
            component_scores={
                "performance_score": 0.5,
                "risk_alignment": 0.5,
                "market_fit": 0.5,
                "operational_efficiency": 0.5
            },
            meets_minimum_thresholds=True,
            improvement_over_current=0.0,
            risk_assessment={"passes_checks": True, "risk_score": 0.5},
            recommendation="monitor",
            confidence_level=0.3,
            reasons=["Fallback evaluation due to error - maintaining current strategy"]
        )


class StrategyOrchestrator:
    """Manages safe strategy transitions and deployment."""
    
    def __init__(
        self,
        logger: LoggerService,
        config: Dict[str, Any],
        monitoring_service  # Reference to MonitoringService
    ):
        self.logger = logger
        self.config = config
        self.monitoring_service = monitoring_service
        self._source_module = self.__class__.__name__
        
        # Transition configuration
        self._transition_config = config.get("transition", {
            "shadow_mode_duration": timedelta(hours=2),
            "phase_durations": {
                TransitionPhase.PHASE_25_PERCENT: timedelta(hours=4),
                TransitionPhase.PHASE_50_PERCENT: timedelta(hours=8),
                TransitionPhase.PHASE_75_PERCENT: timedelta(hours=12)
            },
            "validation_frequency": timedelta(hours=1),
            "rollback_thresholds": {
                "sharpe_drop": 0.5,  # 50% drop
                "drawdown_limit": 0.05,  # 5% drawdown
                "win_rate_floor": 0.40  # 40% minimum
            }
        })
        
        # Active transition tracking
        self._active_transition: Optional[StrategyTransitionPlan] = None
        self._transition_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._phase_start_time: Optional[datetime] = None
        
    async def execute_strategy_transition(
        self,
        from_strategy: str,
        to_strategy: str,
        evaluation_result: StrategyEvaluationResult
    ) -> StrategyTransitionPlan:
        """Execute a safe, phased strategy transition."""
        
        try:
            self.logger.info(
                f"Starting strategy transition: {from_strategy} -> {to_strategy}",
                source_module=self._source_module
            )
            
            # Create transition plan
            transition_plan = self._create_transition_plan(from_strategy, to_strategy)
            self._active_transition = transition_plan
            
            # Start with shadow mode
            await self._enter_shadow_mode(to_strategy)
            
            # Begin transition phases
            for phase in transition_plan.phases:
                if phase == TransitionPhase.SHADOW_MODE:
                    continue  # Already in shadow mode
                    
                success = await self._execute_transition_phase(phase, to_strategy)
                
                if not success:
                    # Rollback on failure
                    await self._rollback_transition(transition_plan, f"Failed at {phase.value}")
                    return transition_plan
                    
            # Complete transition
            await self._complete_transition(transition_plan)
            
            return transition_plan
            
        except Exception as e:
            self.logger.error(
                f"Error during strategy transition: {e}",
                source_module=self._source_module
            )
            if self._active_transition:
                await self._rollback_transition(self._active_transition, str(e))
            raise
            
    def _create_transition_plan(
        self,
        from_strategy: str,
        to_strategy: str
    ) -> StrategyTransitionPlan:
        """Create detailed transition execution plan."""
        
        phases = [
            TransitionPhase.SHADOW_MODE,
            TransitionPhase.PHASE_25_PERCENT,
            TransitionPhase.PHASE_50_PERCENT,
            TransitionPhase.PHASE_75_PERCENT,
            TransitionPhase.FULL_DEPLOYMENT
        ]
        
        phase_durations = {
            TransitionPhase.SHADOW_MODE: self._transition_config["shadow_mode_duration"]
        }
        phase_durations.update(self._transition_config["phase_durations"])
        phase_durations[TransitionPhase.FULL_DEPLOYMENT] = timedelta(0)  # Immediate
        
        return StrategyTransitionPlan(
            transition_id=str(uuid.uuid4()),
            from_strategy=from_strategy,
            to_strategy=to_strategy,
            phases=phases,
            current_phase=TransitionPhase.SHADOW_MODE,
            phase_durations=phase_durations,
            rollback_triggers=self._transition_config["rollback_thresholds"],
            validation_checkpoints=[],
            started_at=datetime.utcnow()
        )
        
    async def _enter_shadow_mode(self, strategy_id: str) -> None:
        """Enter shadow mode for new strategy."""
        
        self.logger.info(
            f"Entering shadow mode for strategy {strategy_id}",
            source_module=self._source_module
        )
        
        # Configure strategy to run without actual trading
        # This would update the strategy configuration in production
        self._phase_start_time = datetime.utcnow()
        
        # Initialize metrics collection
        self._transition_metrics[strategy_id] = []
        
        # Notify monitoring service
        await self._notify_transition_status("shadow_mode_started", strategy_id)
        
    async def _execute_transition_phase(
        self,
        phase: TransitionPhase,
        strategy_id: str
    ) -> bool:
        """Execute a single transition phase."""
        
        self.logger.info(
            f"Starting transition phase: {phase.value} for {strategy_id}",
            source_module=self._source_module
        )
        
        self._phase_start_time = datetime.utcnow()
        phase_duration = self._active_transition.phase_durations[phase]
        
        # Update traffic allocation
        await self._update_traffic_allocation(phase, strategy_id)
        
        # Monitor phase execution
        phase_end = self._phase_start_time + phase_duration
        validation_interval = self._transition_config["validation_frequency"]
        
        while datetime.utcnow() < phase_end:
            # Validate performance
            validation_passed = await self._validate_transition_phase(phase, strategy_id)
            
            if not validation_passed:
                self.logger.warning(
                    f"Validation failed during {phase.value} for {strategy_id}",
                    source_module=self._source_module
                )
                return False
                
            # Wait for next validation
            await asyncio.sleep(validation_interval.total_seconds())
            
        # Phase completed successfully
        self.logger.info(
            f"Completed transition phase: {phase.value} for {strategy_id}",
            source_module=self._source_module
        )
        
        return True
        
    async def _validate_transition_phase(
        self,
        phase: TransitionPhase,
        strategy_id: str
    ) -> bool:
        """Validate strategy performance during transition phase."""
        
        # Collect current metrics
        current_metrics = await self._collect_transition_metrics(strategy_id)
        self._transition_metrics[strategy_id].append(current_metrics)
        
        # Check rollback triggers
        rollback_triggers = self._active_transition.rollback_triggers
        
        # Sharpe ratio check
        if current_metrics["sharpe_ratio"] < rollback_triggers["sharpe_drop"]:
            self.logger.error(
                f"Sharpe ratio dropped below threshold: {current_metrics['sharpe_ratio']}",
                source_module=self._source_module
            )
            return False
            
        # Drawdown check
        if current_metrics["drawdown"] > rollback_triggers["drawdown_limit"]:
            self.logger.error(
                f"Drawdown exceeded limit: {current_metrics['drawdown']}",
                source_module=self._source_module
            )
            return False
            
        # Win rate check
        if current_metrics["win_rate"] < rollback_triggers["win_rate_floor"]:
            self.logger.error(
                f"Win rate below floor: {current_metrics['win_rate']}",
                source_module=self._source_module
            )
            return False
            
        return True
        
    async def _update_traffic_allocation(
        self,
        phase: TransitionPhase,
        strategy_id: str
    ) -> None:
        """Update traffic allocation for transition phase."""
        
        allocations = {
            TransitionPhase.SHADOW_MODE: 0.0,
            TransitionPhase.PHASE_25_PERCENT: 0.25,
            TransitionPhase.PHASE_50_PERCENT: 0.50,
            TransitionPhase.PHASE_75_PERCENT: 0.75,
            TransitionPhase.FULL_DEPLOYMENT: 1.0
        }
        
        allocation = allocations.get(phase, 0.0)
        
        # This would update the actual traffic routing in production
        self.logger.info(
            f"Updated traffic allocation for {strategy_id}: {allocation:.0%}",
            source_module=self._source_module
        )
        
        await self._notify_transition_status(f"allocation_{int(allocation*100)}", strategy_id)
        
    async def _rollback_transition(
        self,
        transition_plan: StrategyTransitionPlan,
        reason: str
    ) -> None:
        """Rollback a failed transition."""
        
        self.logger.error(
            f"Rolling back transition {transition_plan.transition_id}: {reason}",
            source_module=self._source_module
        )
        
        # Restore original strategy to 100% traffic
        await self._update_traffic_allocation(
            TransitionPhase.FULL_DEPLOYMENT,
            transition_plan.from_strategy
        )
        
        # Clear transition state
        self._active_transition = None
        self._phase_start_time = None
        
        # Notify monitoring
        await self._notify_transition_status("rollback", transition_plan.from_strategy, reason)
        
    async def _complete_transition(
        self,
        transition_plan: StrategyTransitionPlan
    ) -> None:
        """Complete successful transition."""
        
        self.logger.info(
            f"Completing transition {transition_plan.transition_id}",
            source_module=self._source_module
        )
        
        # Set new strategy to 100%
        await self._update_traffic_allocation(
            TransitionPhase.FULL_DEPLOYMENT,
            transition_plan.to_strategy
        )
        
        # Update completion time
        transition_plan.completed_at = datetime.utcnow()
        
        # Clear transition state
        self._active_transition = None
        self._phase_start_time = None
        
        # Archive transition metrics
        self._archive_transition_metrics(transition_plan)
        
        # Notify completion
        await self._notify_transition_status("completed", transition_plan.to_strategy)
        
    async def _collect_transition_metrics(
        self,
        strategy_id: str
    ) -> Dict[str, float]:
        """Collect real-time metrics during transition."""
        
        # This would collect actual metrics in production
        # Placeholder implementation
        return {
            "timestamp": datetime.utcnow().timestamp(),
            "sharpe_ratio": 1.8,
            "win_rate": 0.62,
            "drawdown": 0.03,
            "latency_ms": 35,
            "error_rate": 0.001
        }
        
    def _archive_transition_metrics(
        self,
        transition_plan: StrategyTransitionPlan
    ) -> None:
        """Archive transition metrics for analysis."""
        
        # This would store metrics in database in production
        metrics_summary = {
            "transition_id": transition_plan.transition_id,
            "from_strategy": transition_plan.from_strategy,
            "to_strategy": transition_plan.to_strategy,
            "duration": (transition_plan.completed_at - transition_plan.started_at).total_seconds(),
            "metrics": self._transition_metrics.get(transition_plan.to_strategy, [])
        }
        
        self.logger.info(
            f"Archived transition metrics: {len(metrics_summary['metrics'])} data points",
            source_module=self._source_module
        )
        
    async def _notify_transition_status(
        self,
        status: str,
        strategy_id: str,
        details: Optional[str] = None
    ) -> None:
        """Notify monitoring service of transition status."""
        
        # This would integrate with monitoring service in production
        self.logger.info(
            f"Transition status: {status} for {strategy_id}. {details or ''}",
            source_module=self._source_module
        )
        
    def get_active_transition(self) -> Optional[StrategyTransitionPlan]:
        """Get current active transition if any."""
        return self._active_transition
        
    def is_in_transition(self) -> bool:
        """Check if system is currently in transition."""
        return self._active_transition is not None


# === Main Strategy Selection System ===

class StrategySelectionSystem:
    """Main orchestrator for the complete strategy selection system."""
    
    def __init__(
        self,
        logger: LoggerService,
        config: Dict[str, Any],
        risk_manager,
        portfolio_manager,
        monitoring_service,
        database_manager  # For data access
    ):
        self.logger = logger
        self.config = config
        self._source_module = self.__class__.__name__
        
        # Initialize components
        self.performance_analyzer = StrategyPerformanceAnalyzer(logger, config)
        self.market_monitor = MarketConditionMonitor(logger, config)
        self.risk_evaluator = RiskEvaluator(logger, config, risk_manager, portfolio_manager)
        
        self.strategy_selector = StrategySelector(
            logger,
            config,
            self.performance_analyzer,
            self.market_monitor,
            self.risk_evaluator
        )
        
        self.strategy_orchestrator = StrategyOrchestrator(logger, config, monitoring_service)
        
        # Selection state
        self._last_selection_time: Optional[datetime] = None
        self._selection_frequency = timedelta(hours=config.get("selection_frequency_hours", 4))
        self._is_running = False
        self._selection_task = None
        
    async def start(self) -> None:
        """Start the strategy selection system."""
        if self._is_running:
            self.logger.warning(
                "Strategy selection system already running",
                source_module=self._source_module
            )
            return
            
        self._is_running = True
        self._selection_task = asyncio.create_task(self._selection_loop())
        
        self.logger.info(
            "Strategy selection system started",
            source_module=self._source_module
        )
        
    async def stop(self) -> None:
        """Stop the strategy selection system."""
        self._is_running = False
        
        if self._selection_task:
            self._selection_task.cancel()
            try:
                await self._selection_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info(
            "Strategy selection system stopped",
            source_module=self._source_module
        )
        
    async def _selection_loop(self) -> None:
        """Main selection loop that runs periodically."""
        
        while self._is_running:
            try:
                # Check if it's time for selection
                if self._should_run_selection():
                    await self.run_strategy_selection()
                    
                # Sleep for a short interval before checking again
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(
                    f"Error in selection loop: {e}",
                    source_module=self._source_module
                )
                
    def _should_run_selection(self) -> bool:
        """Check if it's time to run strategy selection."""
        
        # Skip if in transition
        if self.strategy_orchestrator.is_in_transition():
            return False
            
        # Check frequency
        if self._last_selection_time:
            time_since_last = datetime.utcnow() - self._last_selection_time
            return time_since_last >= self._selection_frequency
        else:
            return True  # First run
            
    async def run_strategy_selection(self) -> Optional[StrategyEvaluationResult]:
        """Run a complete strategy selection cycle."""
        
        try:
            self.logger.info(
                "Starting strategy selection cycle",
                source_module=self._source_module
            )
            
            # Create selection context
            context = await self._create_selection_context()
            
            # Run selection
            evaluation_result = await self.strategy_selector.select_optimal_strategy(context)
            
            # Update timestamp
            self._last_selection_time = datetime.utcnow()
            
            # Check if we need to transition
            if (evaluation_result.recommendation == "deploy" and 
                evaluation_result.strategy_id != context.current_strategy_id):
                
                # Execute transition
                await self.strategy_orchestrator.execute_strategy_transition(
                    context.current_strategy_id,
                    evaluation_result.strategy_id,
                    evaluation_result
                )
                
            return evaluation_result
            
        except Exception as e:
            self.logger.error(
                f"Error in strategy selection: {e}",
                source_module=self._source_module
            )
            return None
            
    async def _create_selection_context(self) -> StrategySelectionContext:
        """Create context for strategy selection."""
        
        # Get current strategy from configuration
        current_strategy = self.config.get("current_strategy_id", "default")
        
        # Get available strategies
        available_strategies = self.config.get("available_strategies", [current_strategy])
        
        portfolio_state: Dict[str, Any] = {}
        try:
            pm = self.risk_evaluator.portfolio_manager
            if pm is not None:
                portfolio_state = pm.get_current_state()
        except Exception as exc:  # pragma: no cover - defensive catch
            self.logger.error(
                f"Failed to retrieve portfolio state: {exc}",
                source_module=self._source_module,
            )
            portfolio_state = {}
        
        # Assess market conditions
        trading_pairs = self.config.get("trading_pairs", ["XRP/USD", "DOGE/USD"])
        market_conditions = await self.market_monitor.assess_market_conditions(trading_pairs)
        
        risk_budget = Decimal("0")
        try:
            rm = self.risk_evaluator.risk_manager
            if rm is not None:
                risk_budget = rm.get_available_risk_budget()
        except Exception as exc:  # pragma: no cover - defensive catch
            self.logger.error(
                f"Failed to retrieve risk budget: {exc}",
                source_module=self._source_module,
            )
            risk_budget = Decimal("0")
        
        return StrategySelectionContext(
            timestamp=datetime.utcnow(),
            current_strategy_id=current_strategy,
            available_strategies=available_strategies,
            portfolio_state=portfolio_state,
            market_conditions=market_conditions,
            risk_budget_available=risk_budget
        )
        
    def get_current_strategy(self) -> str:
        """Get the currently active strategy."""
        return self.config.get("current_strategy_id", "default")
        
    async def force_strategy_evaluation(self) -> Optional[StrategyEvaluationResult]:
        """Force an immediate strategy evaluation (for manual triggering)."""
        
        self.logger.info(
            "Forced strategy evaluation requested",
            source_module=self._source_module
        )
        
        return await self.run_strategy_selection() 