"""Unit tests for Position-Order relationship functionality."""

import pytest
import uuid
from datetime import datetime, UTC
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from gal_friday.dal.models.position import Position
from gal_friday.dal.models.order import Order
from gal_friday.dal.models.trade_signal import TradeSignal


class TestPositionOrderRelationship:
    """Test the relationship between Position and Order models."""

    @pytest.mark.asyncio
    async def test_position_order_relationship_creation(self, db_session_maker):
        """Test creating a position with related orders."""
        async with db_session_maker() as session:
            # Create a trade signal first (required for order)
            signal = TradeSignal(
                id=uuid.uuid4(),
                trading_pair="BTC/USD",
                strategy_id="test_strategy",
                side="BUY",
                entry_price=Decimal("50000.00"),
                status="ACTIVE"
            )
            session.add(signal)
            await session.flush()

            # Create a position
            position = Position(
                id=uuid.uuid4(),
                trading_pair="BTC/USD",
                side="LONG",
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00"),
                current_price=Decimal("50000.00"),
                opened_at=datetime.now(UTC),
                is_active=True
            )
            session.add(position)
            await session.flush()

            # Create orders linked to the position
            order1 = Order(
                id=uuid.uuid4(),
                signal_id=signal.id,
                position_id=position.id,
                trading_pair="BTC/USD",
                exchange="kraken",
                side="BUY",
                order_type="LIMIT",
                quantity=Decimal("0.5"),
                limit_price=Decimal("49500.00"),
                status="FILLED"
            )
            
            order2 = Order(
                id=uuid.uuid4(),
                signal_id=signal.id,
                position_id=position.id,
                trading_pair="BTC/USD",
                exchange="kraken",
                side="BUY",
                order_type="MARKET",
                quantity=Decimal("0.5"),
                status="FILLED"
            )
            
            session.add_all([order1, order2])
            await session.commit()

            # Verify relationships
            await session.refresh(position)
            assert len(position.orders) == 2
            assert order1 in position.orders
            assert order2 in position.orders
            
            # Verify back-reference
            await session.refresh(order1)
            await session.refresh(order2)
            assert order1.position.id == position.id
            assert order2.position.id == position.id

    @pytest.mark.asyncio
    async def test_order_without_position(self, db_session_maker):
        """Test creating an order without a position (position_id is nullable)."""
        async with db_session_maker() as session:
            # Create a trade signal
            signal = TradeSignal(
                id=uuid.uuid4(),
                trading_pair="ETH/USD",
                strategy_id="test_strategy",
                side="SELL",
                entry_price=Decimal("3000.00"),
                status="ACTIVE"
            )
            session.add(signal)
            await session.flush()

            # Create order without position
            order = Order(
                id=uuid.uuid4(),
                signal_id=signal.id,
                position_id=None,  # No position assigned
                trading_pair="ETH/USD",
                exchange="kraken",
                side="SELL",
                order_type="LIMIT",
                quantity=Decimal("2.0"),
                limit_price=Decimal("3100.00"),
                status="PENDING"
            )
            
            session.add(order)
            await session.commit()

            # Verify order was created without position
            await session.refresh(order)
            assert order.position is None
            assert order.position_id is None

    @pytest.mark.asyncio
    async def test_cascade_delete_orphan(self, db_session_maker):
        """Test that deleting a position deletes associated orders due to cascade."""
        async with db_session_maker() as session:
            # Create signal
            signal = TradeSignal(
                id=uuid.uuid4(),
                trading_pair="ADA/USD",
                strategy_id="test_strategy",
                side="BUY",
                entry_price=Decimal("1.50"),
                status="ACTIVE"
            )
            session.add(signal)
            await session.flush()

            # Create position
            position = Position(
                id=uuid.uuid4(),
                trading_pair="ADA/USD",
                side="LONG",
                quantity=Decimal("1000.0"),
                entry_price=Decimal("1.50"),
                current_price=Decimal("1.55"),
                opened_at=datetime.now(UTC),
                is_active=True
            )
            session.add(position)
            await session.flush()

            # Create order
            order = Order(
                id=uuid.uuid4(),
                signal_id=signal.id,
                position_id=position.id,
                trading_pair="ADA/USD",
                exchange="kraken",
                side="BUY",
                order_type="MARKET",
                quantity=Decimal("1000.0"),
                status="FILLED"
            )
            session.add(order)
            await session.commit()

            order_id = order.id
            position_id = position.id

            # Delete the position
            await session.delete(position)
            await session.commit()

            # Verify the order was also deleted due to cascade
            deleted_order = await session.get(Order, order_id)
            assert deleted_order is None

            deleted_position = await session.get(Position, position_id)
            assert deleted_position is None

    @pytest.mark.asyncio
    async def test_foreign_key_constraint(self, db_session_maker):
        """Test that foreign key constraint prevents invalid position_id."""
        async with db_session_maker() as session:
            # Create signal
            signal = TradeSignal(
                id=uuid.uuid4(),
                trading_pair="SOL/USD",
                strategy_id="test_strategy",
                side="BUY",
                entry_price=Decimal("100.00"),
                status="ACTIVE"
            )
            session.add(signal)
            await session.flush()

            # Try to create order with non-existent position_id
            invalid_position_id = uuid.uuid4()
            order = Order(
                id=uuid.uuid4(),
                signal_id=signal.id,
                position_id=invalid_position_id,  # This position doesn't exist
                trading_pair="SOL/USD",
                exchange="kraken",
                side="BUY",
                order_type="LIMIT",
                quantity=Decimal("10.0"),
                limit_price=Decimal("99.00"),
                status="PENDING"
            )
            
            session.add(order)
            
            # This should raise an IntegrityError due to foreign key constraint
            with pytest.raises(IntegrityError):
                await session.commit()

    @pytest.mark.asyncio
    async def test_position_query_with_orders(self, db_session_maker):
        """Test querying positions with their related orders."""
        async with db_session_maker() as session:
            # Create signal
            signal = TradeSignal(
                id=uuid.uuid4(),
                trading_pair="MATIC/USD",
                strategy_id="test_strategy",
                side="BUY",
                entry_price=Decimal("0.80"),
                status="ACTIVE"
            )
            session.add(signal)
            await session.flush()

            # Create position
            position = Position(
                id=uuid.uuid4(),
                trading_pair="MATIC/USD",
                side="LONG",
                quantity=Decimal("1500.0"),
                entry_price=Decimal("0.80"),
                current_price=Decimal("0.82"),
                opened_at=datetime.now(UTC),
                is_active=True
            )
            session.add(position)
            await session.flush()

            # Create multiple orders
            orders_data = [
                {"quantity": Decimal("500.0"), "side": "BUY", "status": "FILLED"},
                {"quantity": Decimal("500.0"), "side": "BUY", "status": "FILLED"},
                {"quantity": Decimal("500.0"), "side": "BUY", "status": "FILLED"},
            ]

            for order_data in orders_data:
                order = Order(
                    id=uuid.uuid4(),
                    signal_id=signal.id,
                    position_id=position.id,
                    trading_pair="MATIC/USD",
                    exchange="kraken",
                    side=order_data["side"],
                    order_type="MARKET",
                    quantity=order_data["quantity"],
                    status=order_data["status"]
                )
                session.add(order)

            await session.commit()

            # Query position with orders
            stmt = select(Position).where(Position.trading_pair == "MATIC/USD")
            result = await session.execute(stmt)
            queried_position = result.scalar_one()

            # Verify the relationship loaded correctly
            assert len(queried_position.orders) == 3
            total_order_quantity = sum(order.quantity for order in queried_position.orders)
            assert total_order_quantity == Decimal("1500.0")

    @pytest.mark.asyncio
    async def test_order_update_position_reference(self, db_session_maker):
        """Test updating an order's position reference."""
        async with db_session_maker() as session:
            # Create signal
            signal = TradeSignal(
                id=uuid.uuid4(),
                trading_pair="DOT/USD",
                strategy_id="test_strategy",
                side="BUY",
                entry_price=Decimal("25.00"),
                status="ACTIVE"
            )
            session.add(signal)
            await session.flush()

            # Create two positions
            position1 = Position(
                id=uuid.uuid4(),
                trading_pair="DOT/USD",
                side="LONG",
                quantity=Decimal("100.0"),
                entry_price=Decimal("25.00"),
                current_price=Decimal("25.50"),
                opened_at=datetime.now(UTC),
                is_active=True
            )
            
            position2 = Position(
                id=uuid.uuid4(),
                trading_pair="DOT/USD",
                side="LONG",
                quantity=Decimal("50.0"),
                entry_price=Decimal("24.50"),
                current_price=Decimal("25.50"),
                opened_at=datetime.now(UTC),
                is_active=True
            )
            
            session.add_all([position1, position2])
            await session.flush()

            # Create order linked to position1
            order = Order(
                id=uuid.uuid4(),
                signal_id=signal.id,
                position_id=position1.id,
                trading_pair="DOT/USD",
                exchange="kraken",
                side="BUY",
                order_type="LIMIT",
                quantity=Decimal("50.0"),
                limit_price=Decimal("24.00"),
                status="PENDING"
            )
            session.add(order)
            await session.commit()

            # Verify initial relationship
            await session.refresh(position1)
            assert len(position1.orders) == 1
            assert order in position1.orders

            # Update order to reference position2
            order.position_id = position2.id
            await session.commit()

            # Refresh and verify the relationship changed
            await session.refresh(position1)
            await session.refresh(position2)
            await session.refresh(order)
            
            assert len(position1.orders) == 0
            assert len(position2.orders) == 1
            assert order in position2.orders
            assert order.position.id == position2.id 