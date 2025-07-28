"""Simple session-based repository for direct session usage.

This is a lightweight repository for use cases where you have a direct AsyncSession
and don't need the full features of the comprehensive BaseRepository.
"""

from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class SimpleBaseRepository(Generic[T]):
    """Simple repository with basic CRUD operations for direct session usage."""

    def __init__(self, session: AsyncSession, model: type[T]) -> None:
        """Initialize the instance."""
        self._session = session
        self._model = model

    @property
    def session(self) -> AsyncSession:
        """Get the current session."""
        return self._session

    @property
    def model(self) -> type[T]:
        """Get the model class this repository manages."""
        return self._model

    async def get_by_id(self, entity_id: Any) -> T | None:
        """Get entity by ID."""
        result = await self._session.execute(
            select(self._model).where(self._model.id == entity_id),  # type: ignore[attr-defined]
        )
        return result.scalar_one_or_none()

    async def get(self, entity_id: Any) -> T | None:
        """Get entity by ID (alias for get_by_id)."""
        return await self.get_by_id(entity_id)

    async def get_all(self) -> list[T]:
        """Get all entities."""
        result = await self._session.execute(select(self._model))
        return list(result.scalars().all())

    async def create(self, data: dict[str, Any] | T) -> T:
        """Create new entity."""
        entity = self._model(**data) if isinstance(data, dict) else data

        self._session.add(entity)
        await self._session.flush()
        return entity

    async def update(self, entity_id: Any, updates: dict[str, Any]) -> T | None:
        """Update existing entity."""
        entity = await self.get_by_id(entity_id)
        if entity:
            for key, value in updates.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            await self._session.flush()
            return entity
        return None

    async def delete(self, entity_id: Any) -> bool:
        """Delete entity by ID."""
        entity = await self.get_by_id(entity_id)
        if entity:
            await self._session.delete(entity)
            await self._session.flush()
            return True
        return False
