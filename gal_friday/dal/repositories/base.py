"""Base repository class for common database operations."""

from typing import Any, Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations."""

    def __init__(self, session: AsyncSession, model: type[T]) -> None:
        """Initialize the instance."""
        self.session = session
        self.model = model

    async def get(self, id: Any) -> T | None:
        """Get entity by ID."""
        result = await self.session.execute(
            select(self.model).where(self.model.id == id),  # type: ignore[attr-defined]
        )
        return result.scalar_one_or_none()

    async def get_all(self) -> list[T]:
        """Get all entities."""
        result = await self.session.execute(select(self.model))
        return list(result.scalars().all())

    async def create(self, entity: T) -> T:
        """Create new entity."""
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def update(self, entity: T) -> T:
        """Update existing entity."""
        await self.session.merge(entity)
        await self.session.flush()
        return entity

    async def delete(self, entity: T) -> None:
        """Delete entity."""
        await self.session.delete(entity)
        await self.session.flush()
