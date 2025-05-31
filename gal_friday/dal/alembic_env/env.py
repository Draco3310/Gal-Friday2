import asyncio
import os
import sys
from logging.config import fileConfig

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool # Import NullPool
from sqlalchemy.engine import Connection

from alembic import context # type: ignore[import-not-found]

# Ensure the application's root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Imports for Alembic hook type hints
from typing import Any, cast, List, Tuple, Union, Optional, Literal
from alembic.runtime.migration import MigrationContext # type: ignore[import-not-found]
from alembic.autogenerate.api import AutogenContext, CompareTypeContext # type: ignore[import-not-found]
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.sql.schema import SchemaItem
from sqlalchemy import Column as SAColumn # Alias to avoid clash

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
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

# Typed dummy functions for Alembic hooks
def process_revision_directives(
    context: MigrationContext, revision: Union[str, Tuple[str, ...]], directives: List[Any]
) -> None:
    print(f"Processing revision {revision} with directives: {directives}")

def render_item(
    type_: str, obj: Any, autogen_context: AutogenContext
) -> Union[str, Literal[False], None]:
    print(f"Rendering item type {type_} for object {obj}")
    return None

def include_object(
    object: SchemaItem, name: Optional[str], type_: str, reflected: bool, compare_to: Optional[Any]
) -> bool:
    print(f"Checking include_object for {type_} {name}")
    return True

def include_name(
    name: Optional[str], type_: str, parent_names: Optional[dict[str, Any]]
) -> bool:
    print(f"Checking include_name for {type_} {name}")
    return True

def include_symbol(
    table_name: str,
    schema_name: Optional[str],
    symbol_type: str,
    is_reflected: bool,
    symbol_name: str
) -> bool:
    print(f"Checking include_symbol for {symbol_type} {symbol_name} in table {schema_name}.{table_name}")
    return True

def compare_type(
    context: CompareTypeContext,
    inspected_column: dict[str, Any],
    metadata_column: SAColumn,
    inspected_type: Any,
    metadata_type: Any,
) -> Union[bool, None]:
    print(f"Comparing type for column {metadata_column.name}: DB {inspected_type} vs Meta {metadata_type}")
    return None

# Import ConfigManager to get database URL
from gal_friday.config_manager import ConfigManager
# from gal_friday.logger_service import LoggerService # Not strictly used here

def get_db_url() -> str:
    try:
        config_manager = ConfigManager()
        db_url = config_manager.get("database.connection_string")
        if not db_url:
            db_url = config.get_main_option("sqlalchemy.url")
            print(f"Warning: DB URL from ConfigManager empty, falling back to alembic.ini URL: {db_url}")
        else:
            print(f"Using DB URL from ConfigManager: {db_url}")
        return cast(str, db_url)
    except Exception as e:
        print(f"Error getting DB URL from ConfigManager: {e}. Falling back to alembic.ini.")
        return cast(str, config.get_main_option("sqlalchemy.url"))


def run_migrations_offline() -> None:
    url = get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        # The following hooks would be configured here if used:
        # process_revision_directives=process_revision_directives,
        # render_item=render_item,
        # include_object=include_object,
        # include_name=include_name,
        # include_symbol=include_symbol, # If this is a custom hook or part of a specific setup
        # compare_type=compare_type,
    )
    context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        # Hooks can also be configured for online mode:
        # process_revision_directives=process_revision_directives,
        # render_item=render_item,
        # include_object=include_object,
        # include_name=include_name,
        # include_symbol=include_symbol,
        # compare_type=compare_type,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    db_url = get_db_url()
    print("Configuring context for metadata-only autogeneration (no actual DB connection attempt).")
    context.configure(
        connection=None,
        url=db_url,
        target_metadata=target_metadata,
        compare_type=True,
        include_schemas=True,
        # Hooks can be configured here too:
        # process_revision_directives=process_revision_directives,
        # render_item=render_item,
        # include_object=include_object,
        # include_name=include_name,
        # include_symbol=include_symbol,
        # compare_type=compare_type,
    )
    context.run_migrations()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
