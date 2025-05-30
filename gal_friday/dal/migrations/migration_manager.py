"""Database migration management system."""

from pathlib import Path

import asyncpg

from gal_friday.logger_service import LoggerService


class MigrationManager:
    """Manages database schema migrations."""

    def __init__(self, db_pool: asyncpg.Pool, logger: LoggerService) -> None:
        """Initialize the MigrationManager with a database connection pool and logger.

        Args:
            db_pool: Async database connection pool
            logger: Logger service for logging messages
        """
        self.db_pool = db_pool
        self.logger = logger
        self._source_module = self.__class__.__name__
        self.migrations_dir = Path(__file__).parent / "scripts"

    async def initialize(self) -> None:
        """Create migrations table if not exists."""
        query = """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(query)

    async def get_current_version(self) -> int:
        """Get current schema version."""
        query = "SELECT MAX(version) FROM schema_migrations"

        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval(query)
            return result or 0

    async def get_pending_migrations(self) -> list[tuple[int, Path]]:
        """Get list of pending migrations."""
        current_version = await self.get_current_version()

        migrations = []
        for file in sorted(self.migrations_dir.glob("*.sql")):
            # File format: 001_create_tables.sql
            version = int(file.stem.split("_")[0])
            if version > current_version:
                migrations.append((version, file))

        return migrations

    async def run_migration(self, version: int, migration_file: Path) -> None:
        """Execute a single migration."""
        self.logger.info(
            f"Running migration {version}: {migration_file.name}",
            source_module=self._source_module,
        )

        # Read migration SQL
        sql = migration_file.read_text()

        async with self.db_pool.acquire() as conn, conn.transaction():
            # Execute migration
            await conn.execute(sql)

            # Record migration
            await conn.execute(
                "INSERT INTO schema_migrations (version, name) VALUES ($1, $2)",
                version,
                migration_file.stem,
            )

        self.logger.info(
            f"Migration {version} completed successfully",
            source_module=self._source_module,
        )

    async def run_all_migrations(self) -> None:
        """Run all pending migrations."""
        await self.initialize()

        pending = await self.get_pending_migrations()
        if not pending:
            self.logger.info(
                "No pending migrations",
                source_module=self._source_module,
            )
            return

        for version, file in pending:
            await self.run_migration(version, file)

        self.logger.info(
            f"Completed {len(pending)} migrations",
            source_module=self._source_module,
        )

    async def rollback_migration(self, target_version: int) -> None:
        """Rollback to specific version."""
        current_version = await self.get_current_version()

        if target_version >= current_version:
            self.logger.warning(
                f"Target version {target_version} is not less than current {current_version}",
                source_module=self._source_module,
            )
            return

        # Find rollback scripts
        for version in range(current_version, target_version, -1):
            rollback_file = self.migrations_dir / f"{version:03d}_rollback.sql"
            if rollback_file.exists():
                await self._execute_rollback(version, rollback_file)
            else:
                self.logger.error(
                    f"Rollback script not found for version {version}",
                    source_module=self._source_module,
                )
                raise FileNotFoundError(f"Missing rollback script for version {version}")

    async def _execute_rollback(self, version: int, rollback_file: Path) -> None:
        """Execute rollback script."""
        sql = rollback_file.read_text()

        async with self.db_pool.acquire() as conn, conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                "DELETE FROM schema_migrations WHERE version = $1",
                version,
            )
