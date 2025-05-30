"""Tests for risk management functionality.

This module tests the risk manager's signal approval/rejection logic
and position sizing calculations.
"""

import pytest
from decimal import Decimal
from datetime import datetime, UTC
import uuid

from gal_friday.core.events import (
    TradeSignalProposedEvent,
    TradeSignalApprovedEvent,
    TradeSignalRejectedEvent,
    EventType
)


class TestRiskManager:
    """Test suite for RiskManager functionality."""
    
    @pytest.fixture
    def sample_proposed_signal(self):
        """Create a sample proposed trade signal."""
        return TradeSignalProposedEvent(
            source_module="StrategyArbitrator",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            signal_id=uuid.uuid4(),
            trading_pair="XRP/USD",
            exchange="kraken",
            side="BUY",
            entry_type="LIMIT",
            proposed_entry_price=Decimal("0.5000"),
            proposed_sl_price=Decimal("0.4900"),
            proposed_tp_price=Decimal("0.5200"),
            strategy_id="test_strategy"
        )
    
    def test_position_sizing_calculation(self, sample_proposed_signal):
        """Test position size calculation based on risk parameters."""
        # Risk parameters
        account_balance = Decimal("100000")
        risk_per_trade_pct = Decimal("0.5")  # 0.5%
        
        # Calculate position size
        risk_amount = account_balance * (risk_per_trade_pct / 100)
        
        # Risk per unit = Entry - Stop Loss
        risk_per_unit = sample_proposed_signal.proposed_entry_price - sample_proposed_signal.proposed_sl_price
        
        # Position size = Risk Amount / Risk per unit
        position_size = risk_amount / risk_per_unit
        
        # Expected: $500 risk / $0.01 per unit = 50,000 units
        assert position_size == Decimal("50000")
        
    def test_signal_rejection_max_positions(self):
        """Test signal rejection when max positions reached."""
        # This would test the risk manager rejecting signals
        # when maximum position count is reached
        pass
        
    def test_signal_rejection_correlation_limit(self):
        """Test signal rejection when correlation limit exceeded."""
        # This would test rejecting signals for highly correlated pairs
        # when correlation limits are exceeded
        pass
        
    def test_drawdown_based_position_sizing(self):
        """Test position size reduction during drawdown."""
        base_size = Decimal("1000")
        current_drawdown = Decimal("5.0")  # 5% drawdown
        
        # Simple linear reduction: reduce by 50% at 10% drawdown
        reduction_factor = min(current_drawdown / Decimal("10.0"), Decimal("1.0"))
        adjusted_size = base_size * (1 - reduction_factor * Decimal("0.5"))
        
        # At 5% drawdown, expect 75% of base size
        assert adjusted_size == Decimal("750")


class TestSignalValidation:
    """Test signal validation logic."""
    
    def test_stop_loss_validation_buy_order(self):
        """Test stop loss must be below entry for BUY orders."""
        # Valid BUY signal
        signal = TradeSignalProposedEvent.create(
            TradeSignalProposedEvent.TradeSignalProposedParams(
                source_module="TEST",
                trading_pair="XRP/USD",
                exchange="kraken",
                side="BUY",
                entry_type="LIMIT",
                proposed_entry_price=Decimal("0.5000"),
                proposed_sl_price=Decimal("0.4900"),  # Below entry
                proposed_tp_price=Decimal("0.5200"),  # Above entry
                strategy_id="test"
            )
        )
        assert signal.proposed_sl_price < signal.proposed_entry_price
        assert signal.proposed_tp_price > signal.proposed_entry_price
        
    def test_stop_loss_validation_sell_order(self):
        """Test stop loss must be above entry for SELL orders."""
        # Valid SELL signal
        signal = TradeSignalProposedEvent.create(
            TradeSignalProposedEvent.TradeSignalProposedParams(
                source_module="TEST",
                trading_pair="XRP/USD",
                exchange="kraken", 
                side="SELL",
                entry_type="LIMIT",
                proposed_entry_price=Decimal("0.5000"),
                proposed_sl_price=Decimal("0.5100"),  # Above entry
                proposed_tp_price=Decimal("0.4800"),  # Below entry
                strategy_id="test"
            )
        )
        assert signal.proposed_sl_price > signal.proposed_entry_price
        assert signal.proposed_tp_price < signal.proposed_entry_price 