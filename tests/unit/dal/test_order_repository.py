# tests/unit/dal/test_order_repository.py
import pytest
from unittest.mock import MagicMock
import uuid
from datetime import datetime, timedelta, timezone # Changed to timezone.utc for broader compatibility
from decimal import Decimal

# Models and Repository
from gal_friday.dal.models.order import Order
from gal_friday.dal.repositories.order_repository import OrderRepository

# Fixtures from tests/unit/dal/conftest.py will be automatically available
# (db_session, db_setup, db_session_maker, db_engine)
# Fixture from tests/conftest.py for mock_logger (ensure this path is correct)
# Assuming a top-level conftest.py for mock_logger, or adjust path if it's elsewhere.
# For this exercise, assuming mock_logger is available as described.
# from tests.conftest import mock_logger # This is correct if tests/conftest.py exists

# Helper to create sample order data
def sample_order_data(override: dict = None) -> dict:
    data = {
        "id": uuid.uuid4(), # Client-generated UUID for PK
        "signal_id": uuid.uuid4(),
        "trading_pair": "BTC/USD",
        "exchange": "KRAKEN",
        "side": "BUY",
        "order_type": "LIMIT",
        "quantity": Decimal("1.0"),
        "limit_price": Decimal("50000.0"),
        "status": "ACTIVE",
        "exchange_order_id": None,
        "filled_quantity": Decimal("0"),
        "average_fill_price": None,
        "commission": None,
        "created_at": datetime.now(timezone.utc), 
        "updated_at": None,
    }
    if override:
        data.update(override)
    return data

@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_order(db_session_maker, mock_logger, db_setup): # Use db_session_maker
    repo = OrderRepository(session_maker=db_session_maker, logger=mock_logger)
    
    order_data_dict = sample_order_data()
    # BaseRepository.create expects a dict or a model instance.
    
    created_order_model = await repo.create(order_data_dict) 
    
    assert created_order_model is not None
    assert created_order_model.id == order_data_dict["id"] # ID was provided
    assert created_order_model.signal_id == order_data_dict["signal_id"]
    assert created_order_model.trading_pair == "BTC/USD"

    # Verify in DB using a new session
    async with db_session_maker() as session:
        fetched_order = await session.get(Order, created_order_model.id)
        assert fetched_order is not None
        assert fetched_order.signal_id == order_data_dict["signal_id"]

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_order_by_id(db_session_maker, mock_logger, db_setup):
    repo = OrderRepository(session_maker=db_session_maker, logger=mock_logger)
    
    order_data_dict = sample_order_data()
    order_instance = Order(**order_data_dict)
    async with db_session_maker() as session:
        session.add(order_instance)
        await session.commit()
        await session.refresh(order_instance)

    fetched_order = await repo.get_by_id(order_instance.id)
    assert fetched_order is not None
    assert fetched_order.id == order_instance.id
    assert fetched_order.trading_pair == "BTC/USD"

    non_existent_id = uuid.uuid4()
    not_found_order = await repo.get_by_id(non_existent_id)
    assert not_found_order is None

@pytest.mark.unit
@pytest.mark.asyncio
async def test_update_order_status_direct_repo_update(db_session_maker, mock_logger, db_setup):
    # This tests the BaseRepository's update method as used by OrderRepository
    repo = OrderRepository(session_maker=db_session_maker, logger=mock_logger)

    order_data_dict = sample_order_data({"status": "ACTIVE"})
    order_instance = Order(**order_data_dict)
    async with db_session_maker() as session:
        session.add(order_instance)
        await session.commit()
        await session.refresh(order_instance)

    updates = {"status": "FILLED", "filled_quantity": Decimal("1.0")}
    # BaseRepository.update returns the updated model instance or None
    updated_order = await repo.update(order_instance.id, updates) 

    assert updated_order is not None
    assert updated_order.status == "FILLED"
    assert updated_order.filled_quantity == Decimal("1.0")
    assert updated_order.updated_at is not None # BaseRepository.update sets this

    # Verify in DB
    async with db_session_maker() as session:
        refetched_order = await session.get(Order, order_instance.id)
        assert refetched_order is not None
        assert refetched_order.status == "FILLED"

@pytest.mark.unit
@pytest.mark.asyncio
async def test_order_repository_update_order_status_method(db_session_maker, mock_logger, db_setup):
    # This tests the specific update_order_status method in OrderRepository
    repo = OrderRepository(session_maker=db_session_maker, logger=mock_logger)

    order_data_dict = sample_order_data({"status": "ACTIVE"})
    order_instance = Order(**order_data_dict)
    async with db_session_maker() as session:
        session.add(order_instance)
        await session.commit()
        await session.refresh(order_instance)
    
    updated_order = await repo.update_order_status(
        order_id=order_instance.id, 
        status="FILLED", 
        filled_quantity=Decimal("0.5"),
        average_fill_price=Decimal("51000.0")
    )
    assert updated_order is not None
    assert updated_order.status == "FILLED"
    assert updated_order.filled_quantity == Decimal("0.5")
    assert updated_order.average_fill_price == Decimal("51000.0")
    assert updated_order.updated_at is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_order(db_session_maker, mock_logger, db_setup):
    repo = OrderRepository(session_maker=db_session_maker, logger=mock_logger)

    order_data_dict = sample_order_data()
    order_instance = Order(**order_data_dict)
    async with db_session_maker() as session:
        session.add(order_instance)
        await session.commit()
        await session.refresh(order_instance)

    delete_result = await repo.delete(order_instance.id)
    assert delete_result is True

    # Verify not in DB
    async with db_session_maker() as session:
        not_found_order = await session.get(Order, order_instance.id)
        assert not_found_order is None

    delete_non_existent = await repo.delete(uuid.uuid4())
    assert delete_non_existent is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_active_orders(db_session_maker, mock_logger, db_setup):
    repo = OrderRepository(session_maker=db_session_maker, logger=mock_logger)

    active_order_data = sample_order_data({"status": "ACTIVE", "signal_id": uuid.uuid4()})
    filled_order_data = sample_order_data({"status": "FILLED", "signal_id": uuid.uuid4()})
    
    async with db_session_maker() as session:
        session.add_all([Order(**active_order_data), Order(**filled_order_data)])
        await session.commit()

    active_orders = await repo.get_active_orders()
    assert len(active_orders) == 1
    assert active_orders[0].status == "ACTIVE"
    assert active_orders[0].signal_id == active_order_data["signal_id"]

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_recent_orders(db_session_maker, mock_logger, db_setup):
    repo = OrderRepository(session_maker=db_session_maker, logger=mock_logger)

    now = datetime.now(timezone.utc)
    order_new_dict = sample_order_data({"created_at": now - timedelta(hours=1), "signal_id": uuid.uuid4()})
    order_old_dict = sample_order_data({"created_at": now - timedelta(hours=48), "signal_id": uuid.uuid4()})

    async with db_session_maker() as session:
        session.add_all([Order(**order_new_dict), Order(**order_old_dict)])
        await session.commit()
    
    recent_orders = await repo.get_recent_orders(hours=24)
    assert len(recent_orders) == 1
    assert recent_orders[0].signal_id == order_new_dict["signal_id"]

    all_recent_orders = await repo.get_recent_orders(hours=72)
    assert len(all_recent_orders) == 2

@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_orders_by_signal(db_session_maker, mock_logger, db_setup):
    repo = OrderRepository(session_maker=db_session_maker, logger=mock_logger)
    target_signal_id = uuid.uuid4()
    other_signal_id = uuid.uuid4()

    order1_data = sample_order_data({"signal_id": target_signal_id, "created_at": datetime.now(timezone.utc) - timedelta(minutes=2)})
    order2_data = sample_order_data({"signal_id": target_signal_id, "created_at": datetime.now(timezone.utc) - timedelta(minutes=1)})
    order3_data = sample_order_data({"signal_id": other_signal_id})
    
    async with db_session_maker() as session:
        session.add_all([Order(**order1_data), Order(**order2_data), Order(**order3_data)])
        await session.commit()

    signal_orders = await repo.get_orders_by_signal(str(target_signal_id)) # Pass ID as string if needed by repo method
    assert len(signal_orders) == 2
    assert signal_orders[0].id == order1_data["id"] # Check order (ASC by created_at)
    assert signal_orders[1].id == order2_data["id"]
    for order in signal_orders:
        assert order.signal_id == target_signal_id

@pytest.mark.unit
@pytest.mark.asyncio
async def test_find_by_exchange_id(db_session_maker, mock_logger, db_setup):
    repo = OrderRepository(session_maker=db_session_maker, logger=mock_logger)
    target_exchange_id = "EXCH_ID_123"
    
    order_data_target = sample_order_data({"exchange_order_id": target_exchange_id, "signal_id": uuid.uuid4()})
    order_data_other = sample_order_data({"exchange_order_id": "EXCH_ID_456", "signal_id": uuid.uuid4()})

    async with db_session_maker() as session:
        session.add_all([Order(**order_data_target), Order(**order_data_other)])
        await session.commit()

    found_order = await repo.find_by_exchange_id(target_exchange_id)
    assert found_order is not None
    assert found_order.exchange_order_id == target_exchange_id
    assert found_order.id == order_data_target["id"]

    not_found_order = await repo.find_by_exchange_id("NON_EXISTENT_EXCH_ID")
    assert not_found_order is None

# It's good practice to also test edge cases, like empty results for finders,
# or trying to update/delete non-existent orders, some of which are covered.
# Consider testing different statuses for get_active_orders if "ACTIVE" is not the only one.
# The current tests provide good coverage for the main functionalities.
