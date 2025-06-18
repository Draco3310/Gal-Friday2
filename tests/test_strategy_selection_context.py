from datetime import datetime
from decimal import Decimal

import pytest

from gal_friday.strategy_selection import (
    LiquidityCondition,
    MarketConditionSnapshot,
    MarketRegime,
    StrategySelectionSystem,
    TrendState,
)


class DummyPortfolioManager:
    def __init__(self, state):
        self.state = state

    def get_current_state(self):
        return self.state


class DummyRiskManager:
    def __init__(self, budget):
        self.budget = Decimal(budget)

    def get_available_risk_budget(self):
        return self.budget


class DummyMonitoringService:
    pass


@pytest.mark.asyncio
async def test_create_selection_context_dynamic_data(mock_logger):
    portfolio_state = {"total_equity": "75000", "total_exposure_pct": "10"}
    pm = DummyPortfolioManager(portfolio_state)
    rm = DummyRiskManager("15000")

    config = {
        "current_strategy_id": "s1",
        "available_strategies": ["s1", "s2"],
        "trading_pairs": ["XRP/USD"],
    }

    system = StrategySelectionSystem(
        logger=mock_logger,
        config=config,
        risk_manager=rm,
        portfolio_manager=pm,
        monitoring_service=DummyMonitoringService(),
        database_manager=None,
    )

    snapshot = MarketConditionSnapshot(
        timestamp=datetime.utcnow(),
        volatility_regime=MarketRegime.NORMAL_VOLATILITY,
        realized_volatility_24h=0.03,
        implied_volatility=None,
        volatility_percentile=50.0,
        trend_state=TrendState.SIDEWAYS,
        trend_strength=0.3,
        momentum_score=0.0,
        liquidity_condition=LiquidityCondition.NORMAL,
        average_spread_bps=1.0,
        order_book_depth_score=0.9,
        volume_24h_usd=Decimal(1000000),
        volume_percentile=70.0,
        active_sessions=["us"],
        correlation_matrix={},
        systemic_risk_score=0.2,
    )

    async def dummy_assess(_):
        return snapshot

    system.market_monitor.assess_market_conditions = dummy_assess

    context = await system._create_selection_context()

    assert context.portfolio_state == portfolio_state
    assert context.risk_budget_available == Decimal(15000)
    assert context.current_strategy_id == "s1"
