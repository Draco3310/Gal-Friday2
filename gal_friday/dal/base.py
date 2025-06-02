"""Base repository pattern for data access using SQLAlchemy."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import (  # I001: Removed unused _BaseForTypeVar
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
)

from sqlalchemy import asc, desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

if TYPE_CHECKING:
    from gal_friday.dal.models import Base  # F401: _BaseForTypeVar removed
    from gal_friday.logger_service import LoggerService
    # Ensure Base is imported within TYPE_CHECKING for type hinting T
    # from gal_friday.dal.models import Base as _BaseForTypeVar # This alias is unused
else:
    # This path is taken at runtime.
    pass

_BoundType = TypeVar("_BoundType", bound="Base")
T = TypeVar("T", bound=_BoundType)


class BaseRepository(Generic[T]):
    """Base repository with common SQLAlchemy database operations."""

    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        model_class: type[T],
        logger: "LoggerService", # Use string literal for forward reference
    ) -> None:
        """Initialize base repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            model_class: The SQLAlchemy model class this repository manages.
            logger: Logger service instance.
        """
        self.session_maker = session_maker
        self.model_class = model_class
        self.logger = logger
        self._source_module = self.__class__.__name__

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
            async with self.session_maker() as session:
                # SIM108: Use ternary operator
                instance = self.model_class(**data) if isinstance(data, dict) else data

                session.add(instance)
                await session.commit()  # Commit flushes and expires objects
                # Refresh to get server-side defaults like ID, created_at
                await session.refresh(instance)
                self.logger.debug(
                    (
                        f"Created new {self.model_class.__name__} with ID "
                        f"{getattr(instance, 'id', None)}"
                    ),
                    source_module=self._source_module,
                ) # COM812
                return cast("T", instance)
        except Exception as e: # Catch generic Exception for logging, re-raise specific if needed
            self.logger.exception(
                f"Error creating in {self.model_class.__name__}: {e}",
                source_module=self._source_module,
            )
            raise

    async def get_by_id(self, entity_id: Any) -> T | None:  # type: ignore[arg-type] # ANN401
        """Get an entity by its primary key.

        Args:
            entity_id: The primary key value.

        Returns:
            The entity instance or None if not found.

        Raises:
            SQLAlchemyError: If the database operation fails.
        """
        try:
            async with self.session_maker() as session:
                instance = await session.get(self.model_class, entity_id)
                if instance:
                    self.logger.debug(
                        f"Retrieved {self.model_class.__name__} with ID {entity_id}",
                        source_module=self._source_module,
                    )
                else:
                    self.logger.debug(
                        f"{self.model_class.__name__} with ID {entity_id} not found",
                        source_module=self._source_module,
                    )
                return instance
        except Exception as e:
            self.logger.exception(
                f"Error getting {self.model_class.__name__} by ID {entity_id}: {e}",
                source_module=self._source_module,
            )
            raise

    async def update(self, entity_id: Any, updates: dict[str, Any]) -> T | None:  # type: ignore[arg-type] # ANN401
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
                f"Update called for {self.model_class.__name__} ID {entity_id} with no updates.",
                source_module=self._source_module,
            )
            return await self.get_by_id(entity_id) # Return current state if no updates

        try:
            async with self.session_maker() as session:
                entity = await session.get(self.model_class, entity_id)
                if entity:
                    for key, value in updates.items():
                        if hasattr(entity, key):
                            setattr(entity, key, value)
                        else:
                            self.logger.warning(
                                (
                                    f"Attempted to update non-existent attribute '{key}' "
                                    f"on {self.model_class.__name__}"
                                ),
                                source_module=self._source_module,
                            )
                    if hasattr(entity, "updated_at"):
                        # Assuming updated_at is a standard datetime field
                        entity.updated_at = datetime.now(UTC) # B010: Direct assignment


                    await session.commit()
                    await session.refresh(entity)
                    self.logger.info(
                        f"Updated {self.model_class.__name__} with ID {entity_id}",
                        source_module=self._source_module,
                    )
                    return entity
                self.logger.warning(
                    (
                        f"Attempted to update non-existent {self.model_class.__name__} "
                        f"with ID {entity_id}"
                    ),
                    source_module=self._source_module,
                ) # COM812
                return None
        except Exception as e:
            self.logger.exception(
                f"Error updating {self.model_class.__name__} with ID {entity_id}: {e}",
                source_module=self._source_module,
            )
            raise

    async def find_all(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None, # e.g., "column_name" or "column_name DESC"
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
            async with self.session_maker() as session:
                stmt = select(self.model_class)

                if filters:
                    for column_name, value in filters.items():
                        if hasattr(self.model_class, column_name):
                            stmt = stmt.where(getattr(self.model_class, column_name) == value)
                        else:
                            self.logger.warning(
                                (
                                    f"Filter key '{column_name}' not found on model " # E501
                                    f"{self.model_class.__name__}"
                                ),
                                source_module=self._source_module,
                            ) # COM812
                if order_by:
                    parts = order_by.strip().split()
                    col_name = parts[0]
                    if not hasattr(self.model_class, col_name):
                        raise ValueError(
                            f"Invalid order_by column: {col_name} on {self.model_class.__name__}",
                        ) # COM812

                    col = getattr(self.model_class, col_name)
                    if len(parts) > 1 and parts[1].upper() == "DESC":
                        stmt = stmt.order_by(desc(col))
                    else:
                        stmt = stmt.order_by(asc(col))

                if limit is not None:
                    stmt = stmt.limit(limit)
                if offset is not None:
                    stmt = stmt.offset(offset)

                result = await session.execute(stmt)
                entities = result.scalars().all()
                self.logger.debug(
                    f"Found {len(entities)} {self.model_class.__name__}(s) with given criteria",
                    source_module=self._source_module,
                )
                return entities
        except ValueError as ve: # Catch specific ValueError for logging
            self.logger.error(
                f"Invalid order_by clause for {self.model_class.__name__}: {ve}",
                source_module=self._source_module,
            )
            raise
        except Exception as e:
            self.logger.exception(
                f"Error finding all {self.model_class.__name__}: {e}",
                source_module=self._source_module,
            )
            raise

    async def delete(self, entity_id: Any) -> bool:  # type: ignore[arg-type] # ANN401
        """Delete entity by ID.

        Args:
            entity_id: ID of the entity to delete.

        Returns:
            True if deletion was successful, False otherwise.

        Raises:
            SQLAlchemyError: If the operation fails.
        """
        try:
            async with self.session_maker() as session:
                entity = await session.get(self.model_class, entity_id)
                if entity:
                    await session.delete(entity)
                    await session.commit()
                    self.logger.info(
                        f"Deleted {self.model_class.__name__} with ID {entity_id}",
                        source_module=self._source_module,
                    )
                    return True
                self.logger.warning(
                    (
                        f"Attempted to delete non-existent {self.model_class.__name__} "
                        f"with ID {entity_id}"
                    ),
                    source_module=self._source_module,
                ) # COM812
                return False
        except Exception as e:
            self.logger.exception(
                f"Error deleting {self.model_class.__name__} with ID {entity_id}: {e}",
                source_module=self._source_module,
            )
            raise

    # execute_transaction method is removed as SQLAlchemy sessions handle transactions.
    # Each method like create, update, delete that calls session.commit()
    # is effectively an atomic transaction if the session_maker is configured
    # correctly (which it is by default). For multi-operation transactions,
    # a session can be passed around or a higher-level service can manage it.
