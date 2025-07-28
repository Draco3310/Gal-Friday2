"""Base repository pattern for data access using SQLAlchemy."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

if TYPE_CHECKING:
    from gal_friday.dal.models import Base
    from gal_friday.logger_service import LoggerService

else:
    pass

# Define a TypeVar bound to the SQLAlchemy Base model
T = TypeVar("T", bound="Base")  # Use string literal for forward reference


class BaseRepository(Generic[T]):
    """Base repository with common SQLAlchemy database operations."""

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        model_class: type[T],
        logger: "LoggerService",  # Use string literal for forward reference
    ) -> None:
        """Initialize base repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            model_class: The SQLAlchemy model class this repository manages.
            logger: Logger service instance.
        """
        self._session_maker = session_maker
        self._model_class = model_class
        self.logger = logger
        self._source_module = self.__class__.__name__

    @property
    def session_maker(self) -> async_sessionmaker[AsyncSession]:
        """Get the session maker for creating new sessions."""
        return self._session_maker

    @property
    def model_class(self) -> type[T]:
        """Get the model class this repository manages."""
        return self._model_class

    async def create(self, data: dict[str, Any] | T) -> T:
        """Create a new entity.

        Args:
            data: Dictionary of column names to values, or a model instance.

        Returns:
            The created entity instance, with all fields populated (including defaults).

        Raises:
            SQLAlchemyError: If the database operation fails.
        """
        try:
            async with self._session_maker() as session:
                instance = self._model_class(**data) if isinstance(data, dict) else data

                session.add(instance)
                await session.commit()  # Commit flushes and expires objects
                await session.refresh(instance)  # Refresh to get server-side defaults like ID, created_at
                self.logger.debug(
                    f"Created new {self._model_class.__name__} with ID {getattr(instance, 'id', None)}",
                    source_module=self._source_module,
                )
                return instance
        except Exception:  # Catch generic Exception for logging, re-raise specific if needed
            self.logger.exception(
                f"Error creating in {self._model_class.__name__}: ",
                source_module=self._source_module,
            )
            raise

    async def get_by_id(self, entity_id: Any) -> T | None:
        """Get an entity by its primary key.

        Args:
            entity_id: The primary key value.

        Returns:
            The entity instance or None if not found.

        Raises:
            SQLAlchemyError: If the database operation fails.
        """
        try:
            async with self._session_maker() as session:
                instance = await session.get(self._model_class, entity_id)
                if instance:
                    self.logger.debug(
                        f"Retrieved {self._model_class.__name__} with ID {entity_id}",
                        source_module=self._source_module,
                    )
                else:
                    self.logger.debug(
                        f"{self._model_class.__name__} with ID {entity_id} not found",
                        source_module=self._source_module,
                    )
                return instance
        except Exception:
            self.logger.exception(
                f"Error getting {self._model_class.__name__} by ID {entity_id}: ",
                source_module=self._source_module,
            )
            raise

    async def update(self, entity_id: Any, updates: dict[str, Any]) -> T | None:
        """Update an existing entity.

        Args:
            entity_id: ID of the entity to update.
            updates: Dictionary of column names to new values.

        Returns:
            The updated entity instance or None if not found.

        Raises:
            SQLAlchemyError: If the operation fails.
        """
        if not updates:
            self.logger.warning(
                f"Update called for {self._model_class.__name__} ID {entity_id} with no updates.",
                source_module=self._source_module,
            )
            return await self.get_by_id(entity_id)  # Return current state if no updates

        try:
            async with self._session_maker() as session:
                entity = await session.get(self._model_class, entity_id)
                if entity:
                    for key, value in updates.items():
                        if hasattr(entity, key):
                            setattr(entity, key, value)
                        else:
                            self.logger.warning(
                                f"Attempted to update non-existent attribute '{key}' on {self._model_class.__name__}",
                                source_module=self._source_module,
                            )
                    if hasattr(entity, "updated_at"):
                        entity.updated_at = datetime.now(UTC)

                    await session.commit()
                    await session.refresh(entity)
                    self.logger.info(
                        f"Updated {self._model_class.__name__} with ID {entity_id}",
                        source_module=self._source_module,
                    )
                    return entity
                self.logger.warning(
                    f"Attempted to update non-existent {self._model_class.__name__} with ID {entity_id}",
                    source_module=self._source_module,
                )
                return None
        except Exception:
            self.logger.exception(
                f"Error updating {self._model_class.__name__} with ID {entity_id}: ",
                source_module=self._source_module,
            )
            raise

    async def find_all(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,  # e.g., "column_name" or "column_name DESC"
    ) -> Sequence[T]:
        """Find multiple entities with filtering, ordering, and pagination.

        Args:
            filters: Dictionary of column names to filter values.
            limit: Maximum number of results to return.
            offset: Number of results to skip.
            order_by: Column to order results by (e.g., "name" or "created_at DESC").

        Returns:
            A sequence of found entities.

        Raises:
            SQLAlchemyError: If the operation fails.
            ValueError: If order_by clause is malformed.
        """
        try:
            async with self._session_maker() as session:
                # Build query
                stmt = self._build_query(filters, order_by, limit, offset)

                # Execute and return results
                result = await session.execute(stmt)
                entities = result.scalars().all()

                self._log_query_results(entities)
                return cast("Sequence[T]", entities)

        except ValueError:
            self._log_value_error()
            raise
        except Exception:
            self._log_general_error()
            raise

    def _build_query(
        self,
        filters: dict[str, Any] | None,
        order_by: str | None,
        limit: int | None,
        offset: int | None,
    ) -> Any:
        """Build the query with filters, ordering, and pagination."""
        stmt = select(self._model_class)

        # Apply filters
        if filters:
            stmt = self._apply_filters(stmt, filters)

        # Apply ordering
        if order_by:
            stmt = self._apply_ordering(stmt, order_by)

        # Apply pagination
        return self._apply_pagination(stmt, limit, offset)

    def _apply_filters(self, stmt: Any, filters: dict[str, Any]) -> Any:
        """Apply filter conditions to the query."""
        for column_name, value in filters.items():
            if hasattr(self._model_class, column_name):
                stmt = stmt.where(getattr(self._model_class, column_name) == value)
            else:
                self.logger.warning(
                    f"Filter key '{column_name}' not found on model {self._model_class.__name__}",
                    source_module=self._source_module,
                )
        return stmt

    def _apply_ordering(self, stmt: Any, order_by: str) -> Any:
        """Apply ordering to the query."""
        parts = order_by.strip().split()
        col_name = parts[0]

        if not hasattr(self._model_class, col_name):
            raise ValueError(f"Invalid order_by column: {col_name} on {self._model_class.__name__}")

        col = getattr(self._model_class, col_name)

        # Check for DESC modifier
        return stmt.order_by(desc(col)) if len(parts) > 1 and parts[1].upper() == "DESC" else stmt.order_by(asc(col))

    def _apply_pagination(self, stmt: Any, limit: int | None, offset: int | None) -> Any:
        """Apply pagination to the query."""
        if limit is not None:
            stmt = stmt.limit(limit)
        if offset is not None:
            stmt = stmt.offset(offset)
        return stmt

    def _log_query_results(self, entities: Sequence[T]) -> None:
        """Log the results of the query."""
        self.logger.debug(
            f"Found {len(entities)} {self._model_class.__name__}(s) with given criteria",
            source_module=self._source_module,
        )

    def _log_value_error(self) -> None:
        """Log ValueError for invalid order_by clause."""
        self.logger.exception(
            f"Invalid order_by clause for {self._model_class.__name__}",
            source_module=self._source_module,
        )

    def _log_general_error(self) -> None:
        """Log general errors during find_all operation."""
        self.logger.exception(
            f"Error finding all {self._model_class.__name__}: ",
            source_module=self._source_module,
        )

    async def delete(self, entity_id: Any) -> bool:
        """Delete entity by ID.

        Args:
            entity_id: ID of the entity to delete.

        Returns:
            True if deletion was successful, False otherwise.

        Raises:
            SQLAlchemyError: If the operation fails.
        """
        try:
            async with self._session_maker() as session:
                entity = await session.get(self._model_class, entity_id)
                if entity:
                    await session.delete(entity)
                    await session.commit()
                    self.logger.info(
                        f"Deleted {self._model_class.__name__} with ID {entity_id}",
                        source_module=self._source_module,
                    )
                    return True
                self.logger.warning(
                    f"Attempted to delete non-existent {self._model_class.__name__} with ID {entity_id}",
                    source_module=self._source_module,
                )
                return False
        except Exception:
            self.logger.exception(
                f"Error deleting {self._model_class.__name__} with ID {entity_id}: ",
                source_module=self._source_module,
            )
            raise

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities matching the given filters.

        Args:
            filters: Dictionary of column names to filter values.

        Returns:
            Number of entities matching the filters.
        """
        try:
            async with self._session_maker() as session:
                stmt = select(func.count()).select_from(self._model_class)

                # Apply filters if provided
                if filters:
                    for column_name, value in filters.items():
                        if hasattr(self._model_class, column_name):
                            stmt = stmt.where(getattr(self._model_class, column_name) == value)
                        else:
                            self.logger.warning(
                                f"Filter key '{column_name}' not found on model {self._model_class.__name__}",
                                source_module=self._source_module,
                            )

                result = await session.execute(stmt)
                count = result.scalar() or 0

                self.logger.debug(
                    f"Counted {count} {self._model_class.__name__}(s) matching filters",
                    source_module=self._source_module,
                )

                return count
        except Exception:
            self.logger.exception(
                f"Error counting {self._model_class.__name__}: ",
                source_module=self._source_module,
            )
            raise

    async def exists(self, entity_id: Any) -> bool:
        """Check if an entity exists by ID.

        Args:
            entity_id: The primary key value.

        Returns:
            True if the entity exists, False otherwise.
        """
        try:
            async with self._session_maker() as session:
                # Use exists() subquery for efficiency
                stmt = select(
                    select(self._model_class)
                    .where(
                        cast("Any", self._model_class).id == entity_id,
                    )
                    .exists(),
                )

                result = await session.execute(stmt)
                exists = result.scalar() or False

                self.logger.debug(
                    f"{self._model_class.__name__} with ID {entity_id} {'exists' if exists else 'does not exist'}",
                    source_module=self._source_module,
                )

                return exists
        except Exception:
            self.logger.exception(
                f"Error checking existence of {self._model_class.__name__} with ID {entity_id}: ",
                source_module=self._source_module,
            )
            raise

    # execute_transaction method is removed as SQLAlchemy sessions handle transactions.
    # Each method like create, update, delete that calls session.commit()
    # is effectively an atomic transaction if the session_maker is configured
    # correctly (which it is by default). For multi-operation transactions,
    # a session can be passed around or a higher-level service can manage it.
