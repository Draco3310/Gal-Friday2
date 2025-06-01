import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.config_manager import ConfigManager
from gal_friday.dal.connection_pool import DatabaseConnectionPool


@pytest.fixture
def mock_config():
    mock = MagicMock(spec=ConfigManager)
    mock.get.side_effect = lambda key, default=None: {
        "database.connection_string": "postgresql+asyncpg://test:test@localhost/testdb",
        "database.echo_sql": False,
    }.get(key, default)
    mock.get_int.side_effect = lambda key, default=None: { # Added None default
        "database.pool.min_size": 5,
        "database.pool.max_size": 10,
    }.get(key, default if default is not None else 0) # Ensure default for get_int
    mock.get_bool.side_effect = lambda key, default=None: { # Added None default
        "database.echo_sql": False,
    }.get(key, default if default is not None else False) # Ensure default for get_bool
    return mock

@pytest.fixture
def mock_logger():
    # Use a real logger instance but with a mock handler or spy on its methods
    # For simplicity here, MagicMock is fine for checking calls.
    logger_mock = MagicMock(spec=logging.Logger)
    logger_mock.info = MagicMock()
    logger_mock.error = MagicMock()
    logger_mock.exception = MagicMock()
    logger_mock.debug = MagicMock()
    logger_mock.warning = MagicMock()
    return logger_mock

@pytest.mark.asyncio
@patch("gal_friday.dal.connection_pool.create_async_engine", autospec=True)
async def test_initialize_success(mock_create_engine, mock_config, mock_logger):
    mock_engine_instance = AsyncMock()
    mock_create_engine.return_value = mock_engine_instance

    pool = DatabaseConnectionPool(config=mock_config, logger=mock_logger)
    await pool.initialize()

    assert pool.is_initialized()
    assert pool.get_session_maker() is not None

    expected_url = URL.create(
        drivername="postgresql+asyncpg",
        username="test",
        password="test",
        host="localhost",
        database="testdb",
    )
    mock_create_engine.assert_called_once()
    call_args = mock_create_engine.call_args[0]
    call_kwargs = mock_create_engine.call_args[1]

    assert str(call_args[0]) == str(expected_url)
    assert call_kwargs["pool_size"] == 5
    assert call_kwargs["max_overflow"] == 5 # 10 (max_size) - 5 (min_size)
    assert call_kwargs["echo"] is False
    assert call_kwargs["pool_recycle"] == 300
    assert call_kwargs["pool_timeout"] == 10


    mock_logger.info.assert_any_call(
        "SQLAlchemy AsyncEngine initialized",
        source_module="DatabaseConnectionPool",
    )

@pytest.mark.asyncio
async def test_initialize_failure_no_url(mock_config, mock_logger):
    # mock_config.get("database.connection_string") should return "" or None
    # Make sure other keys still return their defaults if get() is called for them.
    def side_effect_for_get(key, default=None):
        if key == "database.connection_string":
            return "" # Simulate missing URL
        return {
            "database.echo_sql": False,
            # Add other keys if get() is called for them during the error path
        }.get(key, default)

    mock_config.get.side_effect = side_effect_for_get

    pool = DatabaseConnectionPool(config=mock_config, logger=mock_logger)
    with pytest.raises(ValueError, match="Database connection string is missing"):
        await pool.initialize()

    assert not pool.is_initialized()
    mock_logger.error.assert_called_with(
        "Database connection string is not configured.",
        source_module="DatabaseConnectionPool",
    )

@pytest.mark.asyncio
@patch("gal_friday.dal.connection_pool.create_async_engine", autospec=True)
async def test_acquire_session(mock_create_engine, mock_config, mock_logger):
    mock_engine_instance = AsyncMock()
    mock_create_engine.return_value = mock_engine_instance

    # This mock_session_instance will be returned by the session_maker when called
    mock_session_instance = AsyncMock(spec=AsyncSession)

    pool = DatabaseConnectionPool(config=mock_config, logger=mock_logger)
    await pool.initialize()

    # To test acquire correctly, we need to control the session it yields.
    # We replace the pool's _session_maker with a MagicMock that returns our mock_session_instance.
    # This is more robust than patching async_sessionmaker globally for this test.
    original_session_maker = pool._session_maker
    mock_sm_instance_for_pool = MagicMock(spec=async_sessionmaker)
    mock_sm_instance_for_pool.return_value = mock_session_instance # When mock_sm_instance_for_pool() is called, it returns mock_session_instance
    pool._session_maker = mock_sm_instance_for_pool

    async with pool.acquire() as session:
        assert session is mock_session_instance
        # Simulate some operation with the session
        await session.execute(MagicMock())

    mock_sm_instance_for_pool.assert_called_once() # Assert our mock session_maker was used
    mock_session_instance.close.assert_called_once() # Assert session.close() was called by context manager
    mock_session_instance.rollback.assert_not_called() # Assert rollback was not called for success path

    pool._session_maker = original_session_maker # Restore original session_maker

@pytest.mark.asyncio
async def test_acquire_before_initialize(mock_config, mock_logger):
    pool = DatabaseConnectionPool(config=mock_config, logger=mock_logger)
    with pytest.raises(RuntimeError, match="Session maker is not initialized. Call initialize() first."):
        async with pool.acquire():
            pass # This code should not be reached

@pytest.mark.asyncio
@patch("gal_friday.dal.connection_pool.create_async_engine", autospec=True)
async def test_close_pool(mock_create_engine, mock_config, mock_logger):
    mock_engine_instance = AsyncMock()
    # We need to mock the dispose method on the instance that will be stored in _engine
    mock_engine_instance.dispose = AsyncMock()
    mock_create_engine.return_value = mock_engine_instance

    pool = DatabaseConnectionPool(config=mock_config, logger=mock_logger)
    await pool.initialize()
    assert pool.is_initialized()

    await pool.close()

    assert not pool.is_initialized()
    assert pool._engine is None # Check engine is reset
    assert pool._session_maker is None # Check session_maker is reset
    mock_engine_instance.dispose.assert_called_once()
    mock_logger.info.assert_any_call( # Use assert_any_call if other info logs might exist
        "SQLAlchemy AsyncEngine disposed",
        source_module="DatabaseConnectionPool",
    )

@pytest.mark.asyncio
@patch("gal_friday.dal.connection_pool.create_async_engine", autospec=True)
async def test_acquire_session_handles_exception(mock_create_engine, mock_config, mock_logger):
    mock_engine_instance = AsyncMock()
    mock_create_engine.return_value = mock_engine_instance

    # This mock_session_instance will be returned by the session_maker
    mock_session_instance = AsyncMock(spec=AsyncSession)
    # Make an operation on the session raise an error
    mock_session_instance.execute = AsyncMock(side_effect=ValueError("DB error"))

    pool = DatabaseConnectionPool(config=mock_config, logger=mock_logger)
    await pool.initialize()

    # Replace the pool's _session_maker with a MagicMock
    original_session_maker = pool._session_maker
    mock_sm_instance_for_pool = MagicMock(spec=async_sessionmaker)
    mock_sm_instance_for_pool.return_value = mock_session_instance
    pool._session_maker = mock_sm_instance_for_pool

    with pytest.raises(ValueError, match="DB error"):
        async with pool.acquire() as session:
            await session.execute(MagicMock()) # This will raise ValueError("DB error")

    mock_session_instance.rollback.assert_called_once() # Assert rollback was called on error
    mock_session_instance.close.assert_called_once() # Assert session.close() was still called

    pool._session_maker = original_session_maker # Restore
