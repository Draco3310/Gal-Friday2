"""Alembic environment configuration script for Gal-Friday."""
import asyncio
import logging
import sys # os import removed F401
from logging.config import fileConfig
from pathlib import Path

from alembic import context # type: ignore[import-not-found]
from sqlalchemy import Column as SAColumn # Alias to avoid clash
from sqlalchemy import create_engine, pool
from sqlalchemy.ext.asyncio import create_async_engine # Added for F821
from sqlalchemy.engine import Connection
from sqlalchemy.sql.schema import SchemaItem
from typing import Any, Literal, cast

from alembic.autogenerate.api import AutogenContext, CompareTypeContext # type: ignore[import-not-found]
from alembic.runtime.migration import MigrationContext # type: ignore[import-not-found]

from gal_friday.config_manager import ConfigManager
from gal_friday.dal.models import Base

# Ensure the application's root directory is in the Python path
APP_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(APP_ROOT_DIR))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# Base is already imported above

# Import all models to ensure they are registered with Base.metadata

target_metadata = Base.metadata

logger = logging.getLogger(__name__)

# Typed dummy functions for Alembic hooks
def process_revision_directives(
    context: MigrationContext, revision: str | tuple[str, ...], directives: list[Any],
) -> None:
    """Process revision directives."""
    logger.debug(f"Processing revision {revision} with directives: {directives}")

def render_item(
    type_: str, obj: Any, autogen_context: AutogenContext,  # type: ignore[arg-type] # ANN401
) -> str | Literal[False] | None:
    """Render an item for Alembic."""
    logger.debug(f"Rendering item type {type_} for object {obj}")
    return None

def include_object(
    object: SchemaItem, name: str | None, type_: str, reflected: bool, compare_to: Any | None,  # type: ignore[arg-type] # ANN401
) -> bool:
    """Determine if an object should be included in the migration."""
    logger.debug(f"Checking include_object for {type_} {name}")
    return True

def include_name(
    name: str | None, type_: str, parent_names: dict[str, Any] | None,
) -> bool:
    """Determine if a name should be included."""
    logger.debug(f"Checking include_name for {type_} {name}")
    return True

def include_symbol(
    table_name: str,
    schema_name: str | None,
    symbol_type: str, # Literal[...] is more precise but str is fine for a dummy
    is_reflected: bool,
    symbol_name: str,
) -> bool:
    """Determine if a symbol should be included."""
    logger.debug(
        f"Checking include_symbol for {symbol_type} {symbol_name} " # E501
        f"in table {schema_name}.{table_name}", # COM812
    )
    return True

def compare_type(
    context: CompareTypeContext,
    inspected_column: dict[str, Any],
    metadata_column: SAColumn,
    inspected_type: Any,  # type: ignore[arg-type] # ANN401
    metadata_type: Any,  # type: ignore[arg-type] # ANN401
) -> bool | None:
    """Compare database and metadata types."""
    logger.debug(
        f"Comparing type for column {metadata_column.name}: " # E501
        f"DB {inspected_type} vs Meta {metadata_type}", # COM812
    )
    return None

# ConfigManager is already imported above

# from gal_friday.logger_service import LoggerService # Not strictly used here

def get_db_url() -> str:
    """Get the database URL from ConfigManager or alembic.ini."""
    try:
        config_manager = ConfigManager()
        db_url = config_manager.get("database.connection_string")
        if not db_url:
            db_url = config.get_main_option("sqlalchemy.url")
            logger.warning(
                f"DB URL from ConfigManager empty, falling back to alembic.ini URL: {db_url}", # E501, COM812
            )
        else:
            logger.info(f"Using DB URL from ConfigManager: {db_url}")
        return cast("str", db_url)
    except Exception as e:
        logger.error(f"Error getting DB URL from ConfigManager: {e}. Falling back to alembic.ini.")
        return cast("str", config.get_main_option("sqlalchemy.url"))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
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
    """Run migrations using the provided database connection."""
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
    """Run migrations in 'online' mode using an async engine.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    db_url = get_db_url()
    logger.info(
        "Configuring context for metadata-only autogeneration " # E501
        "(no actual DB connection attempt).", # COM812
    )
    # F841: connectable = create_async_engine(db_url, poolclass=pool.NullPool) # Unused

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
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
