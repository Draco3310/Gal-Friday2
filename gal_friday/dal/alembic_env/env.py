import asyncio
import os
import sys
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool # Import NullPool
from sqlalchemy.engine import Connection

from alembic import context

# Ensure the application's root directory is in the Python path
# This allows importing gal_friday.dal.models etc.
# The path needs to be adjusted based on where alembic is run from relative to the project root.
# If alembic.ini has prepend_sys_path = ., and alembic is run from project root, this might not be strictly needed here.
# However, being explicit can help. Assuming project root is one level up from 'gal_friday/dal'.
# The prepend_sys_path = . in alembic.ini should handle adding /app to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
from gal_friday.dal.models import Base  # Import Base from your models package
# Import all models to ensure they are registered with Base.metadata
from gal_friday.dal.models.order import Order
from gal_friday.dal.models.position import Position
from gal_friday.dal.models.trade_signal import TradeSignal
from gal_friday.dal.models.model_version import ModelVersion
from gal_friday.dal.models.model_deployment import ModelDeployment
from gal_friday.dal.models.reconciliation_event import ReconciliationEvent
from gal_friday.dal.models.position_adjustment import PositionAdjustment
from gal_friday.dal.models.experiment import Experiment
from gal_friday.dal.models.experiment_assignment import ExperimentAssignment
from gal_friday.dal.models.experiment_outcome import ExperimentOutcome
from gal_friday.dal.models.retraining_job import RetrainingJob
from gal_friday.dal.models.drift_detection_event import DriftDetectionEvent

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

from typing import Any # For type hints if needed later for render_item etc.
# Import ConfigManager to get database URL
from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService # For consistency, though not strictly used in template

def get_db_url() -> str: # Added return type
    """Retrieve database URL from ConfigManager."""
    # Assuming ConfigManager can be instantiated without complex dependencies here
    # or that necessary config files are accessible.
    # This might need adjustment based on how ConfigManager loads its config.
    # For simplicity, assuming it can find 'config/config.yaml' from the CWD
    # where alembic is run (usually project root).
    try:
        # Create a dummy logger for ConfigManager if it requires one
        # This is a bit of a hack; ideally ConfigManager is usable without full app setup
        class MinimalLogger:
            def get_logger(self, name): return self
            def info(self, msg, *args, **kwargs): print(f"INFO: {msg}")
            def error(self, msg, *args, **kwargs): print(f"ERROR: {msg}")
            def warning(self, msg, *args, **kwargs): print(f"WARNING: {msg}")
            def exception(self, msg, *args, **kwargs): print(f"EXCEPTION: {msg}")
            def debug(self, msg, *args, **kwargs): print(f"DEBUG: {msg}")

        # logger_service = MinimalLogger() # type: ignore # Not needed for ConfigManager
        config_manager = ConfigManager() # Removed logger argument
        db_url = config_manager.get("database.connection_string")
        if not db_url:
            # Fallback to alembic.ini if ConfigManager doesn't provide the URL
            db_url = config.get_main_option("sqlalchemy.url")
            print(f"Warning: Could not get DB URL from ConfigManager, falling back to alembic.ini URL: {db_url}")
        else:
            print(f"Using DB URL from ConfigManager: {db_url}")
        return db_url
    except Exception as e:
        print(f"Error getting DB URL from ConfigManager: {e}. Falling back to alembic.ini.")
        return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_db_url() # Use helper to get URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        # literal_binds=True, # Not suitable for autogenerate comparison
        dialect_opts={"paramstyle": "named"},
        compare_type=True, # Added for type comparison
    )

    # For autogenerate in offline mode, a transaction is not strictly needed
    # and can cause issues if no DB connection is truly available.
    # The run_migrations() call is what populates the context for diffing.
    context.run_migrations()


def do_run_migrations(connection: Connection) -> None: # Added -> None
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True, # Added for type comparison
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    db_url = get_db_url() # Use helper to get URL

    # For autogenerate without a live DB connection, configure context directly
    # This provides metadata and dialect info without an active connection.
    print("Configuring context for metadata-only autogeneration (no actual DB connection attempt).")
    context.configure(
        connection=None, # Explicitly no connection
        url=db_url,      # Still needed for dialect and other settings
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True, # Good practice for autogenerate
        # For autogenerate, we don't want to output SQL to a buffer like in offline DDL generation
    )
    # No transaction needed here as we are not executing DDLs, just diffing
    context.run_migrations()
    # Original async connect logic removed for this specific "no connection" autogen scenario


async def run_async_migrations() -> None: # Added -> None
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    db_url = get_db_url() # Use helper to get URL

    # For autogenerate without a live DB connection, configure context directly
    # This provides metadata and dialect info without an active connection.
    print("Configuring context for metadata-only autogeneration (no actual DB connection attempt).")
    context.configure(
        connection=None, # Explicitly no connection
        url=db_url,      # Still needed for dialect and other settings
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True, # Good practice for autogenerate
        # For autogenerate, we don't want to output SQL to a buffer like in offline DDL generation
    )
    # No transaction needed here as we are not executing DDLs, just diffing
    context.run_migrations()
    # Original async connect logic removed for this specific "no connection" autogen scenario


def run_migrations_online() -> None: # Added -> None
    """Run migrations in 'online' mode.
    Modified to support metadata-only autogeneration if DB is unavailable by not connecting.
    """
    # The run_async_migrations function itself now handles the "no connection" setup.
    # We still use asyncio.run because run_async_migrations is an async function.
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
