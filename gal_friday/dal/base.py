"""Base repository pattern for data access."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import asyncpg

from gal_friday.logger_service import LoggerService


# Define a bound TypeVar that must be a BaseEntity
class BaseEntity(ABC):
    """Base class for all database entities."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for database storage."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseEntity":
        """Create entity from database record."""


T = TypeVar("T", bound=BaseEntity)


class BaseRepository(Generic[T], ABC):
    """Base repository with common database operations."""

    def __init__(self, db_pool: asyncpg.Pool, logger: LoggerService, table_name: str) -> None:
        """Initialize base repository.

        Args:
            db_pool: Database connection pool
            logger: Logger service instance
            table_name: Name of the database table
        """
        self.db_pool = db_pool
        self.logger = logger
        self.table_name = table_name
        self._source_module = self.__class__.__name__

    def _quote_identifier(self, identifier: str) -> str:
        """Properly quote SQL identifiers."""
        return f'"{identifier}"'

    async def create(self, data: dict[str, Any]) -> str:
        """Create a new entity.

        Args:
            data: Dictionary of column names to values
        Returns:
            str: The ID of the created entity
        Raises:
            DatabaseError: If the operation fails
        """
        columns = list(data.keys())
        values = list(data.values())
        placeholders = [f"${i+1}" for i in range(len(columns))]

        # Use parameterized query to prevent SQL injection
        query = """
            INSERT INTO {table_name} ({columns})
            VALUES ({placeholders})
            RETURNING id
        """.format(
            table_name=self._quote_identifier(self.table_name),
            columns=", ".join(self._quote_identifier(col) for col in columns),
            placeholders=", ".join(placeholders),
        ).strip()

        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, *values)
                return str(result)
        except Exception:
            self.logger.exception(
                f"Error creating in {self.table_name}",
                source_module=self._source_module,
            )
            raise

    async def update(self, id: str, updates: dict[str, Any]) -> bool:
        """Update an existing entity.

        Args:
            id: ID of the entity to update
            updates: Dictionary of column names to new values
        Returns:
            bool: True if update was successful, False otherwise
        Raises:
            DatabaseError: If the operation fails
        """
        if not updates:
            return False

        set_clauses = [f"{self._quote_identifier(col)} = ${i+2}"
                      for i, col in enumerate(updates.keys())]

        # Use parameterized query to prevent SQL injection
        query = """
            UPDATE {table_name}
            SET {set_clauses}, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
        """.format(
            table_name=self._quote_identifier(self.table_name),
            set_clauses=", ".join(set_clauses),
        ).strip()

        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(query, id, *updates.values())
                # Parse the result string to check if rows were affected
                # PostgreSQL execute returns a string like "UPDATE 1" or "UPDATE 0"
                if isinstance(result, str):
                    return result.split()[-1] != "0"
                return bool(result)
        except Exception:
            self.logger.exception(
                f"Error updating {self.table_name}",
                source_module=self._source_module,
            )
            raise

    async def find_all(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str | None = None,
    ) -> list[T]:
        """Find multiple entities with filtering.

        Args:
            filters: Dictionary of column names to filter values
            limit: Maximum number of results to return
            offset: Number of results to skip
            order_by: Column to order results by
        Returns:
            List[T]: List of found entities
        Raises:
            DatabaseError: If the operation fails
        """
        # Use parameterized query to prevent SQL injection
        query_parts = [f"SELECT * FROM {self._quote_identifier(self.table_name)}"]
        params: list[Any] = []
        param_count = 0

        # Add WHERE clause if filters provided
        if filters:
            where_clauses = []
            for col, value in filters.items():
                param_count += 1
                where_clauses.append(f"{col} = ${param_count}")
                params.append(value)
            query_parts.append(f"WHERE {' AND '.join(where_clauses)}")

        # Add ORDER BY
        if order_by:
            query_parts.append(f"ORDER BY {order_by}")

        # Add pagination
        param_count += 1
        query_parts.append(f"LIMIT ${param_count}")
        params.append(limit)

        param_count += 1
        query_parts.append(f"OFFSET ${param_count}")
        params.append(offset)

        query = " ".join(query_parts)

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [self._row_to_entity(dict(row)) for row in rows]
        except Exception:
            self.logger.exception(
                f"Error finding many in {self.table_name}",
                source_module=self._source_module,
            )
            raise

    async def delete(self, id: str) -> bool:
        """Delete entity by ID.

        Args:
            id: ID of the entity to delete
        Returns:
            bool: True if deletion was successful, False otherwise
        Raises:
            DatabaseError: If the operation fails
        """
        # Use parameterized query to prevent SQL injection
        query = f"""
            DELETE FROM {self._quote_identifier(self.table_name)}
            WHERE id = $1
        """.strip()

        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(query, id)
                # Parse the result string to check if rows were affected
                # PostgreSQL execute returns a string like "DELETE 1" or "DELETE 0"
                if isinstance(result, str):
                    return result.split()[-1] != "0"
                return bool(result)
        except Exception:
            self.logger.exception(
                f"Error deleting from {self.table_name}",
                source_module=self._source_module,
            )
            raise

    async def execute_transaction(
        self,
        operations: list[Callable[[asyncpg.Connection], Any]],
    ) -> None:
        """Execute multiple operations in a transaction.

        Args:
            operations: List of async functions that take a connection
        Raises:
            DatabaseError: If any operation in the transaction fails
        """
        async with self.db_pool.acquire() as conn, conn.transaction():
            for operation in operations:
                await operation(conn)

    @abstractmethod
    def _row_to_entity(self, row: dict[str, Any]) -> T:
        """Convert database row to entity."""
