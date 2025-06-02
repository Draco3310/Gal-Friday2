"""Generates DDL (Data Definition Language) statements for database schema.

This script connects to a PostgreSQL database (via a dummy URL, as it only needs the dialect)
and generates SQL statements for creating and dropping tables based on SQLAlchemy models.
It's designed to be run from the command line and outputs DDL to stdout.
Error messages and import status are written to stderr.

The script assumes that all necessary SQLAlchemy models are defined in
`gal_friday.dal.models` and are correctly imported.
"""
import asyncio
import sys
from pathlib import Path

# Ensure the current directory (project root) is in sys.path
# This allows 'from gal_friday...' imports if the script is in the project root
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateTable, DropTable

# Import Base and all models to populate Base.metadata
try:
    from gal_friday.dal.models import Base

    # Import all models to ensure they are registered with Base.metadata
    # These imports are flagged by F401 (unused) but are necessary for SQLAlchemy's
    # metadata registration to work correctly. The noqa directives suppress these warnings.
    from gal_friday.dal.models.drift_detection_event import DriftDetectionEvent  # noqa: F401
    from gal_friday.dal.models.experiment import Experiment  # noqa: F401
    from gal_friday.dal.models.experiment_assignment import (
        ExperimentAssignment,  # noqa: F401
    )
    from gal_friday.dal.models.experiment_outcome import ExperimentOutcome  # noqa: F401
    from gal_friday.dal.models.model_deployment import ModelDeployment  # noqa: F401
    from gal_friday.dal.models.model_version import ModelVersion  # noqa: F401
    from gal_friday.dal.models.order import Order  # noqa: F401
    from gal_friday.dal.models.position import Position  # noqa: F401
    from gal_friday.dal.models.position_adjustment import PositionAdjustment  # noqa: F401
    from gal_friday.dal.models.reconciliation_event import ReconciliationEvent  # noqa: F401
    from gal_friday.dal.models.retraining_job import RetrainingJob  # noqa: F401
    from gal_friday.dal.models.trade_signal import TradeSignal  # noqa: F401
    sys.stderr.write("Successfully imported models.\n")
except ImportError as e:
    sys.stderr.write(f"Error importing models: {e}. Check PYTHONPATH and script location.\n")
    sys.stderr.write(f"Current sys.path: {sys.path!s}\n")
    sys.exit(1)

async def main() -> None:
    """Generates DDL for all tables."""
    # Define a dummy URL; only the dialect matters for DDL compilation without a connection.
    # Ensure the dialect matches your target database (postgresql).
    # Using asyncpg dialect as it's used in the project.
    dummy_url = "postgresql+asyncpg://user:pass@localhost/dbname"
    # The engine is not used directly but is needed for some SQLAlchemy versions
    # to ensure the dialect is correctly loaded.
    _ = create_async_engine(dummy_url, poolclass=NullPool)
    pg_dialect = postgresql.dialect()

    sys.stdout.write("-- UPGRADE DDL --\n")
    for table in Base.metadata.sorted_tables:
        try:
            # Compile DDL using the engine's dialect
            ddl_statement = str(CreateTable(table).compile(dialect=pg_dialect)).strip()
            sys.stdout.write(f"{ddl_statement};\n")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"-- Error compiling CREATE for table {table.name}: {e}\n")


    sys.stdout.write("\n-- DOWNGRADE DDL --\n")
    for table in reversed(Base.metadata.sorted_tables):
        try:
            # Compile DDL using the engine's dialect
            ddl_statement = str(DropTable(table).compile(dialect=pg_dialect)).strip()
            sys.stdout.write(f"{ddl_statement};\n")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"-- Error compiling DROP for table {table.name}: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
