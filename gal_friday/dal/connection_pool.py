"""Database connection pool management."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


class DatabaseConnectionPool:
    """Manages database connection pools."""

    def __init__(self, config: ConfigManager, logger: LoggerService) -> None:
        """Initialize the connection pool manager.

        Args:
            config: Configuration manager instance
            logger: Logger service instance
        """
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

        self.postgres_pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize connection pools."""
        async with self._pool_lock:
            if self.postgres_pool is None:
                try:
                    self.postgres_pool = await asyncpg.create_pool(
                        self.config.get("database.connection_string"),
                        min_size=self.config.get_int("database.pool.min_size", 10),
                        max_size=self.config.get_int("database.pool.max_size", 20),
                        max_inactive_connection_lifetime=300,
                        command_timeout=10,
                    )

                    self.logger.info(
                        "Database connection pool initialized",
                        source_module=self._source_module,
                    )
                except Exception:
                    self.logger.exception(
                        "Failed to initialize database pool",
                        source_module=self._source_module,
                    )
                    raise

    async def close(self) -> None:
        """Close all connection pools."""
        if self.postgres_pool:
            await self.postgres_pool.close()
            self.postgres_pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a database connection.

        Yields:
            asyncpg.Connection: A database connection from the pool

        Raises:
            RuntimeError: If the connection pool is not initialized
            asyncpg.PostgresError: If there's an error acquiring a connection
        """
        if not self.postgres_pool:
            raise RuntimeError("Database connection pool is not initialized")

        async with self.postgres_pool.acquire() as conn:
            yield conn

    async def execute_query(self, query: str, *args: object) -> list[dict[str, Any]]:
        """Execute a query and return results.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            List of dictionaries representing the query results

        Raises:
            asyncpg.PostgresError: If there's an error executing the query
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def execute_command(self, command: str, *args: object) -> str:
        """Execute a command (INSERT, UPDATE, DELETE).

        Args:
            command: SQL command string
            *args: Command parameters

        Returns:
            str: Status message from the database

        Raises:
            asyncpg.PostgresError: If there's an error executing the command
        """
        async with self.acquire() as conn:
            return await conn.execute(command, *args)
