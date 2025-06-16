import asyncio
import logging
import os
import sys
from logging.config import fileConfig

from alembic import context  # type: ignore
from sqlalchemy.engine import Connection

# Ensure the application's root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Imports for Alembic hook type hints
from typing import Any, Literal, cast

from alembic.autogenerate.api import (  # type: ignore
    AutogenContext,
    CompareTypeContext)
from alembic.runtime.migration import MigrationContext  # type: ignore
from sqlalchemy import Column as SAColumn  # Alias to avoid clash
from sqlalchemy.sql.schema import SchemaItem

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
from gal_friday.dal.models import Base  # Import Base from your models package

# Import all models to ensure they are registered with Base.metadata

target_metadata = Base.metadata

logger = logging.getLogger(__name__)

# Typed dummy functions for Alembic hooks
def process_revision_directives(
    context: MigrationContext, revision: str | tuple[str, ...], directives: list[Any]) -> None:
    logger.debug(f"Processing revision {revision} with directives: {directives}")

def render_item(
    type_: str, obj: Any, autogen_context: AutogenContext) -> str | Literal[False] | None:
    logger.debug(f"Rendering item type {type_} for object {obj}")
    return None

def include_object(
    object: SchemaItem, name: str | None, type_: str, reflected: bool, compare_to: Any | None) -> bool:
    logger.debug(f"Checking include_object for {type_} {name}")
    return True

def include_name(
    name: str | None, type_: str, parent_names: dict[str, Any] | None) -> bool:
    logger.debug(f"Checking include_name for {type_} {name}")
    return True

# Note: include_symbol is not a standard Alembic hook.
# It appears to be a custom hook that might have been used in a
# specific Alembic extension or customization. Keeping it here
# for compatibility but it won't be called by standard Alembic.
def include_symbol(
    table_name: str,
    schema_name: str | None,
    symbol_type: str,
    is_reflected: bool,
    symbol_name: str) -> bool:
    logger.debug(f"Checking include_symbol for {symbol_type} {symbol_name} in table {schema_name}.{table_name}")
    return True

def compare_type(
    context: CompareTypeContext,
    inspected_column: dict[str, Any],
    metadata_column: SAColumn,
    inspected_type: Any,
    metadata_type: Any) -> bool | None:
    logger.debug(f"Comparing type for column {metadata_column.name}: DB {inspected_type} vs Meta {metadata_type}")
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
            logger.warning(f"DB URL from ConfigManager empty, falling back to alembic.ini URL: {db_url}")
        else:
            logger.info(f"Using DB URL from ConfigManager: {db_url}")
        return cast("str", db_url)
    except Exception as e:
        logger.error(f"Error getting DB URL from ConfigManager: {e}. Falling back to alembic.ini.")
        return cast("str", config.get_main_option("sqlalchemy.url"))


def run_migrations_offline() -> None:
    url = get_db_url()
    
    # Standard Alembic hooks
    standard_hooks = {
        "url": url,
        "target_metadata": target_metadata,
        "dialect_opts": {"paramstyle": "named"},
        "compare_type": True,
        "process_revision_directives": process_revision_directives,
        "render_item": render_item,
        "include_object": include_object,
        "include_name": include_name,
        "compare_type": compare_type,
    }
    
    # Note: include_symbol is not a standard hook and will be ignored
    # by Alembic unless using a custom extension
    
    context.configure(**standard_hooks)
    context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    # Standard Alembic hooks for online mode
    standard_hooks = {
        "connection": connection,
        "target_metadata": target_metadata,
        "compare_type": True,
        "process_revision_directives": process_revision_directives,
        "render_item": render_item,
        "include_object": include_object,
        "include_name": include_name,
        "compare_type": compare_type,
    }
    
    context.configure(**standard_hooks)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    db_url = get_db_url()
    logger.info("Configuring context for metadata-only autogeneration (no actual DB connection attempt).")
    
    # Standard Alembic hooks for async/metadata-only mode
    standard_hooks = {
        "connection": None,
        "url": db_url,
        "target_metadata": target_metadata,
        "compare_type": True,
        "include_schemas": True,
        "process_revision_directives": process_revision_directives,
        "render_item": render_item,
        "include_object": include_object,
        "include_name": include_name,
        "compare_type": compare_type,
    }
    
    context.configure(**standard_hooks)
    context.run_migrations()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()