import asyncio
import os
import sys

# Ensure the current directory (project root) is in sys.path
# This allows 'from gal_friday...' imports if the script is in the project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateTable, DropTable

# Import Base and all models to populate Base.metadata
try:
    from gal_friday.dal.models import Base
    from gal_friday.dal.models.drift_detection_event import DriftDetectionEvent
    from gal_friday.dal.models.experiment import Experiment
    from gal_friday.dal.models.experiment_assignment import ExperimentAssignment
    from gal_friday.dal.models.experiment_outcome import ExperimentOutcome
    from gal_friday.dal.models.model_deployment import ModelDeployment
    from gal_friday.dal.models.model_version import ModelVersion
    from gal_friday.dal.models.order import Order
    from gal_friday.dal.models.position import Position
    from gal_friday.dal.models.position_adjustment import PositionAdjustment
    from gal_friday.dal.models.reconciliation_event import ReconciliationEvent
    from gal_friday.dal.models.retraining_job import RetrainingJob
    from gal_friday.dal.models.trade_signal import TradeSignal
    print("Successfully imported models.", file=sys.stderr)
except ImportError as e:
    print(f"Error importing models: {e}. Check PYTHONPATH and script location.", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)

async def main():
    # Define a dummy URL; only the dialect matters for DDL compilation without a connection.
    # Ensure the dialect matches your target database (postgresql).
    # Using asyncpg dialect as it's used in the project.
    dummy_url = "postgresql+asyncpg://user:pass@localhost/dbname"
    engine = create_async_engine(dummy_url, poolclass=NullPool)
    pg_dialect = postgresql.dialect()

    print("-- UPGRADE DDL --")
    for table in Base.metadata.sorted_tables:
        try:
            # Compile DDL using the engine's dialect
            ddl_statement = str(CreateTable(table).compile(dialect=pg_dialect)).strip()
            print(f"{ddl_statement};")
        except Exception as e:
            print(f"-- Error compiling CREATE for table {table.name}: {e}", file=sys.stderr)


    print("\n-- DOWNGRADE DDL --")
    for table in reversed(Base.metadata.sorted_tables):
        try:
            # Compile DDL using the engine's dialect
            ddl_statement = str(DropTable(table).compile(dialect=pg_dialect)).strip()
            print(f"{ddl_statement};")
        except Exception as e:
            print(f"-- Error compiling DROP for table {table.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
