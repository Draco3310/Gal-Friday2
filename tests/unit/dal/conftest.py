# tests/unit/dal/conftest.py
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Import Base and all models to populate Base.metadata
from gal_friday.dal.models.models_base import Base
# Ensure all models are imported here so Base.metadata knows about them
from gal_friday.dal.models import (
    Order, Position, TradeSignal, ModelVersion, 
    ModelDeployment, ReconciliationEvent, PositionAdjustment, 
    Log, Experiment, ExperimentAssignment, ExperimentOutcome, 
    RetrainingJob, DriftDetectionEvent
)

@pytest.fixture(scope="function")
async def db_engine():
    # Using SQLite in-memory for tests
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    yield engine
    await engine.dispose()

@pytest.fixture(scope="function")
async def db_setup(db_engine):
    # Ensures all tables are created and dropped for each test function
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield # Test runs here

    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture(scope="function")
def db_session_maker(db_engine):
    # Provides an async_sessionmaker for the test db engine.
    maker = async_sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)
    return maker

@pytest.fixture(scope="function")
async def db_session(db_session_maker):
    # Provides an AsyncSession for a test, managing its lifecycle.
    async with db_session_maker() as session:
        yield session
