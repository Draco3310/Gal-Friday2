import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from gal_friday.models.trade import Trade


def create_trade() -> Trade:
    now = datetime.now(UTC)
    return Trade(
        trade_pk=1,
        trade_id=uuid.uuid4(),
        signal_id=None,
        trading_pair="BTC/USD",
        exchange="kraken",
        strategy_id="test",
        side="BUY",
        entry_order_pk=None,
        exit_order_pk=None,
        entry_timestamp=now,
        exit_timestamp=now,
        quantity=Decimal("1.5"),
        average_entry_price=Decimal("100.0"),
        average_exit_price=Decimal("110.0"),
        total_commission=Decimal("0.1"),
        realized_pnl=Decimal("10.0"),
        realized_pnl_pct=5.0,
        exit_reason="TARGET",
    )


def test_to_dict_has_all_columns():
    trade = create_trade()
    data = trade.to_dict()
    assert set(data.keys()) == set(trade.__table__.columns.keys())


def test_to_dict_type_conversions():
    trade = create_trade()
    data = trade.to_dict()
    assert data["trade_id"] == str(trade.trade_id)
    assert data["entry_timestamp"] == trade.entry_timestamp.isoformat()
    assert data["quantity"] == str(trade.quantity)
    assert isinstance(data["realized_pnl_pct"], float)
