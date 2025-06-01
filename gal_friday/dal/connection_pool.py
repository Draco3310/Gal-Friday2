"""Database connection pool management."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from gal_friday.config_manager import ConfigManager

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class DatabaseConnectionPool:
    """Manages SQLAlchemy database engine and sessions."""

    def __init__(self, config: ConfigManager, logger: "LoggerService") -> None:
        """Initialize the connection pool manager.

        Args:
            config: Configuration manager instance
            logger: Logger service instance
        """
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

        self._engine: AsyncEngine | None = None
        self._session_maker: async_sessionmaker[AsyncSession] | None = None
        self._pool_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the SQLAlchemy AsyncEngine."""
        async with self._pool_lock:
            if self._engine is None:
                try:
                    db_url = self.config.get("database.connection_string")
                    if not db_url:
                        self.logger.error(
                            "Database connection string is not configured.",
                            source_module=self._source_module,
                        )
                        raise ValueError("Database connection string is missing.")

                    # Note: Pool size and other asyncpg-specific pool params are
                    # handled differently in SQLAlchemy or have defaults.
                    # `pool_size` and `max_overflow` are common SQLAlchemy pool params.
                    # We can expose these via config if needed.
                    self._engine = create_async_engine(
                        db_url,
                        pool_size=self.config.get_int("database.pool.min_size", 5), # SQLAlchemy uses pool_size
                        max_overflow=self.config.get_int("database.pool.max_size", 10) - self.config.get_int("database.pool.min_size", 5), # max_overflow is additional connections beyond pool_size
                        pool_recycle=300, # Corresponds to max_inactive_connection_lifetime
                        pool_timeout=10, # Corresponds to command_timeout (for connection acquisition)
                        echo=self.config.get_bool("database.echo_sql", False), # Optional: log SQL
                    )
                    self._session_maker = async_sessionmaker(
                        self._engine, expire_on_commit=False, class_=AsyncSession,
                    )
                    self.logger.info(
                        "SQLAlchemy AsyncEngine initialized",
                        source_module=self._source_module,
                    )
                except Exception:
                    self.logger.exception(
                        "Failed to initialize SQLAlchemy AsyncEngine",
                        source_module=self._source_module,
                    )
                    raise

    async def close(self) -> None:
        """Dispose of the SQLAlchemy AsyncEngine."""
        async with self._pool_lock:
            if self._engine:
                await self._engine.dispose()
                self._engine = None
                self._session_maker = None
                self.logger.info(
                    "SQLAlchemy AsyncEngine disposed",
                    source_module=self._source_module,
                )

    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._engine is not None and self._session_maker is not None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[AsyncSession]:
        """Provide an AsyncSession from the session maker.

        Yields:
            AsyncSession: A SQLAlchemy AsyncSession

        Raises:
            RuntimeError: If the session maker is not initialized
        """
        if not self._session_maker:
            self.logger.error(
                "Session maker not initialized. Call initialize() first.",
                source_module=self._source_module,
            )
            raise RuntimeError(
                "Session maker is not initialized. Call initialize() first.",
            )

        session = self._session_maker()
        try:
            yield session
            # Note: Transactions should be handled by the calling code (e.g., repository methods)
            # await session.commit() # Typically not done here, but in the repository
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # execute_query and execute_command are removed as direct session usage is preferred.
    # Repositories will use the acquire() method to get a session and perform operations.

    def get_session_maker(self) -> async_sessionmaker[AsyncSession] | None:
        """Return the async_sessionmaker instance."""
        return self._session_maker
