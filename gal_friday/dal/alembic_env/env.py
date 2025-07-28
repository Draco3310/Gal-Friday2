"""Alembic environment configuration for database migrations.

This module configures the Alembic migration environment for the Gal-Friday
database schema. It handles both online (connected to database) and offline
(SQL generation only) migration modes.
"""

import logging
from logging.config import fileConfig
from pathlib import Path
import sys

from alembic import context
import asyncio
from sqlalchemy.engine import Connection

# Ensure the application's root directory is in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.resolve()))

# Imports for Alembic hook type hints
from typing import Any, Literal, cast

from alembic.autogenerate.api import AutogenContext
from alembic.runtime.migration import MigrationContext
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
    context: MigrationContext,
    revision: str | tuple[str, ...],
    directives: list[Any],
) -> None:
    """Process and potentially modify revision directives before migration.

    Args:
        context: Migration context instance.
        revision: Revision identifier or tuple of identifiers.
        directives: List of migration directive objects to process.
    """
    logger.debug(f"Processing revision {revision} with directives: {directives}")


def render_item(type_: str, obj: Any, autogen_context: AutogenContext) -> str | Literal[False] | None:
    """Render a database object for migration script generation.

    Args:
        type_: Type of the database object.
        obj: Database object to render.
        autogen_context: Autogeneration context.

    Returns:
        Rendered string, False to skip, or None for default rendering.
    """
    logger.debug(f"Rendering item type {type_} for object {obj}")
    return None


def include_object(obj: SchemaItem, name: str | None, type_: str, reflected: bool, compare_to: Any | None) -> bool:
    """Determine if a database object should be included in migration.

    Args:
        obj: Database schema object.
        name: Object name.
        type_: Object type (table, column, etc.).
        reflected: Whether object was reflected from database.
        compare_to: Object to compare against.

    Returns:
        True to include object, False to exclude.
    """
    logger.debug(f"Checking include_object for {type_} {name}")
    return True


def include_name(name: str | None, type_: str, parent_names: dict[str, Any] | None) -> bool:
    """Determine if a named object should be included in migration.

    Args:
        name: Object name.
        type_: Object type.
        parent_names: Parent object names.

    Returns:
        True to include, False to exclude.
    """
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
    symbol_name: str,
) -> bool:
    """Custom hook to determine if a database symbol should be included.

    Note: This is not a standard Alembic hook and requires custom extension.

    Args:
        table_name: Name of the table.
        schema_name: Schema name if applicable.
        symbol_type: Type of symbol.
        is_reflected: Whether symbol was reflected.
        symbol_name: Name of the symbol.

    Returns:
        True to include, False to exclude.
    """
    logger.debug(f"Checking include_symbol for {symbol_type} {symbol_name} in table {schema_name}.{table_name}")
    return True


def compare_type(
    context: Any,  # CompareTypeContext
    inspected_column: dict[str, Any],
    metadata_column: Any,  # SAColumn[Any]
    inspected_type: Any,
    metadata_type: Any,
) -> bool | None:
    """Compare database and metadata column types for differences.

    Args:
        context: Comparison context.
        inspected_column: Column info from database.
        metadata_column: Column from metadata.
        inspected_type: Database column type.
        metadata_type: Metadata column type.

    Returns:
        True if different, False if same, None for default comparison.
    """
    logger.debug(f"Comparing type for column {metadata_column.name}: DB {inspected_type} vs Meta {metadata_type}")
    return None


# Import ConfigManager to get database URL
from gal_friday.config_manager import ConfigManager


def get_db_url() -> str:
    """Get database URL from configuration.

    Attempts to get URL from ConfigManager first, falls back to alembic.ini.

    Returns:
        Database connection URL string.
    """
    try:
        config_manager = ConfigManager()
        db_url = config_manager.get("database.connection_string")
        if not db_url:
            db_url = config.get_main_option("sqlalchemy.url")
            logger.warning(f"DB URL from ConfigManager empty, falling back to alembic.ini URL: {db_url}")
        else:
            logger.info(f"Using DB URL from ConfigManager: {db_url}")
        return cast("str", db_url)
    except Exception:
        logger.exception("Error getting DB URL from ConfigManager: . Falling back to alembic.ini.")
        return cast("str", config.get_main_option("sqlalchemy.url"))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This generates SQL scripts without requiring a database connection.
    Useful for reviewing changes before applying them.
    """
    url = get_db_url()

    # Standard Alembic hooks
    standard_hooks = {
        "url": url,
        "target_metadata": target_metadata,
        "dialect_opts": {"paramstyle": "named"},
        "compare_type": compare_type,
        "process_revision_directives": process_revision_directives,
        "render_item": render_item,
        "include_object": include_object,
        "include_name": include_name,
    }

    # Note: include_symbol is not a standard hook and will be ignored
    # by Alembic unless using a custom extension

    context.configure(**standard_hooks)  # type: ignore[arg-type]
    context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with an active database connection.

    Args:
        connection: Active SQLAlchemy database connection.
    """
    # Standard Alembic hooks for online mode
    standard_hooks = {
        "connection": connection,
        "target_metadata": target_metadata,
        "compare_type": compare_type,
        "process_revision_directives": process_revision_directives,
        "render_item": render_item,
        "include_object": include_object,
        "include_name": include_name,
    }

    context.configure(**standard_hooks)  # type: ignore[arg-type]
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode for metadata-only operations.

    Used for autogeneration without actual database connection.
    """
    db_url = get_db_url()
    logger.info("Configuring context for metadata-only autogeneration (no actual DB connection attempt).")

    # Standard Alembic hooks for async/metadata-only mode
    standard_hooks = {
        "connection": None,
        "url": db_url,
        "target_metadata": target_metadata,
        "compare_type": compare_type,
        "include_schemas": True,
        "process_revision_directives": process_revision_directives,
        "render_item": render_item,
        "include_object": include_object,
        "include_name": include_name,
    }

    context.configure(**standard_hooks)  # type: ignore[arg-type]
    context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with database connection.

    Creates async engine and runs migrations within a connection context.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
