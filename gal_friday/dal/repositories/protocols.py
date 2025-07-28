"""Repository protocols for type-safe repository interfaces.

This module defines protocols that repository implementations should follow
to ensure type safety and consistent interfaces across the application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Type variable for model types
T = TypeVar("T")
ModelType = TypeVar("ModelType")


@runtime_checkable
class BaseRepositoryProtocol(Protocol, Generic[T]):
    """Core repository protocol defining basic CRUD operations."""

    async def create(self, data: dict[str, Any] | T) -> T:
        """Create a new entity.

        Args:
            data: Dictionary of column names to values, or a model instance.

        Returns:
            The created entity instance.
        """
        ...

    async def get_by_id(self, entity_id: Any) -> T | None:
        """Get an entity by its primary key.

        Args:
            entity_id: The primary key value.

        Returns:
            The entity instance or None if not found.
        """
        ...

    async def update(self, entity_id: Any, updates: dict[str, Any]) -> T | None:
        """Update an existing entity.

        Args:
            entity_id: ID of the entity to update.
            updates: Dictionary of column names to new values.

        Returns:
            The updated entity instance or None if not found.
        """
        ...

    async def delete(self, entity_id: Any) -> bool:
        """Delete entity by ID.

        Args:
            entity_id: ID of the entity to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        ...


@runtime_checkable
class QueryableRepositoryProtocol(Protocol, Generic[T]):  # type: ignore[misc]
    """Protocol for repositories with advanced query capabilities."""

    async def find_all(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
    ) -> Sequence[T]:
        """Find multiple entities with filtering, ordering, and pagination.

        Args:
            filters: Dictionary of column names to filter values.
            limit: Maximum number of results to return.
            offset: Number of results to skip.
            order_by: Column to order results by.

        Returns:
            A sequence of found entities.
        """
        ...

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entities matching the given filters.

        Args:
            filters: Dictionary of column names to filter values.

        Returns:
            Number of entities matching the filters.
        """
        ...

    async def exists(self, entity_id: Any) -> bool:
        """Check if an entity exists by ID.

        Args:
            entity_id: The primary key value.

        Returns:
            True if the entity exists, False otherwise.
        """
        ...


@runtime_checkable
class AuditableRepositoryProtocol(Protocol, Generic[T]):
    """Protocol for repositories with audit trail support."""

    async def create_with_audit(
        self,
        data: dict[str, Any] | T,
        user_id: str | None = None,
        audit_metadata: dict[str, Any] | None = None,
    ) -> T:
        """Create entity with audit trail.

        Args:
            data: Entity data to create.
            user_id: ID of the user performing the operation.
            audit_metadata: Additional metadata for audit trail.

        Returns:
            The created entity instance.
        """
        ...

    async def update_with_audit(
        self,
        entity_id: Any,
        updates: dict[str, Any],
        user_id: str | None = None,
        audit_metadata: dict[str, Any] | None = None,
    ) -> T | None:
        """Update entity with audit trail.

        Args:
            entity_id: ID of the entity to update.
            updates: Dictionary of column names to new values.
            user_id: ID of the user performing the operation.
            audit_metadata: Additional metadata for audit trail.

        Returns:
            The updated entity instance or None if not found.
        """
        ...

    async def delete_with_audit(
        self,
        entity_id: Any,
        user_id: str | None = None,
        audit_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Delete entity with audit trail.

        Args:
            entity_id: ID of the entity to delete.
            user_id: ID of the user performing the operation.
            audit_metadata: Additional metadata for audit trail.

        Returns:
            True if deletion was successful, False otherwise.
        """
        ...


@runtime_checkable
class TransactionalRepositoryProtocol(Protocol, Generic[T]):  # type: ignore[misc]
    """Protocol for repositories with transaction support."""

    async def begin_transaction(self) -> AsyncSession:
        """Begin a new database transaction.

        Returns:
            An async session for the transaction.
        """
        ...

    async def commit_transaction(self, session: AsyncSession) -> None:
        """Commit the current transaction.

        Args:
            session: The session to commit.
        """
        ...

    async def rollback_transaction(self, session: AsyncSession) -> None:
        """Rollback the current transaction.

        Args:
            session: The session to rollback.
        """
        ...


@runtime_checkable
class SessionManagedRepositoryProtocol(Protocol, Generic[T]):  # type: ignore[misc]
    """Protocol for repositories that manage their own sessions."""

    @property
    def session_maker(self) -> async_sessionmaker[AsyncSession]:
        """Get the session maker for creating new sessions."""
        ...

    @property
    def model_class(self) -> type[T]:
        """Get the model class this repository manages."""
        ...


@runtime_checkable
class DirectSessionRepositoryProtocol(Protocol, Generic[T]):  # type: ignore[misc]
    """Protocol for repositories that use a direct session."""

    @property
    def session(self) -> AsyncSession:
        """Get the current session."""
        ...

    @property
    def model(self) -> type[T]:
        """Get the model class this repository manages."""
        ...


@runtime_checkable
class ComprehensiveRepositoryProtocol(
    BaseRepositoryProtocol[T],
    QueryableRepositoryProtocol[T],
    SessionManagedRepositoryProtocol[T],
    Protocol,
):
    """Comprehensive repository protocol combining all standard capabilities."""


@runtime_checkable
class SimpleRepositoryProtocol(
    BaseRepositoryProtocol[T],
    DirectSessionRepositoryProtocol[T],
    Protocol,
):
    """Simple repository protocol for direct session usage."""


# Type aliases for common repository patterns
RepositoryProtocol = ComprehensiveRepositoryProtocol[T]
LightweightRepositoryProtocol = SimpleRepositoryProtocol[T]


# Repository factory protocol
@runtime_checkable
class RepositoryFactoryProtocol(Protocol):
    """Protocol for repository factory implementations."""

    def create_repository(
        self,
        model_class: type[ModelType],
        session_maker: async_sessionmaker[AsyncSession] | None = None,
    ) -> BaseRepositoryProtocol[ModelType]:
        """Create a repository instance for the given model class.

        Args:
            model_class: The model class to create a repository for.
            session_maker: Optional session maker for the repository.

        Returns:
            A repository instance for the model class.
        """
        ...


# Type guards for repository validation
def is_comprehensive_repository(repo: Any) -> bool:
    """Check if a repository implements the comprehensive protocol."""
    return isinstance(repo, ComprehensiveRepositoryProtocol)


def is_simple_repository(repo: Any) -> bool:
    """Check if a repository implements the simple protocol."""
    return isinstance(repo, SimpleRepositoryProtocol)


def is_auditable_repository(repo: Any) -> bool:
    """Check if a repository implements the auditable protocol."""
    return isinstance(repo, AuditableRepositoryProtocol)


def is_transactional_repository(repo: Any) -> bool:
    """Check if a repository implements the transactional protocol."""
    return isinstance(repo, TransactionalRepositoryProtocol)


def is_queryable_repository(repo: Any) -> bool:
    """Check if a repository implements the queryable protocol."""
    return isinstance(repo, QueryableRepositoryProtocol)


# Repository validation utilities
class RepositoryValidator:
    """Utilities for validating repository implementations."""

    @staticmethod
    def validate_repository_interface(repo: Any, expected_protocol: type) -> bool:
        """Validate that a repository implements the expected protocol.

        Args:
            repo: The repository instance to validate.
            expected_protocol: The protocol the repository should implement.

        Returns:
            True if the repository implements the protocol, False otherwise.
        """
        return isinstance(repo, expected_protocol)

    @staticmethod
    def get_repository_capabilities(repo: Any) -> dict[str, bool]:
        """Get the capabilities of a repository.

        Args:
            repo: The repository instance to analyze.

        Returns:
            Dictionary mapping capability names to whether they're supported.
        """
        return {
            "base_operations": is_comprehensive_repository(repo) or is_simple_repository(repo),
            "comprehensive": is_comprehensive_repository(repo),
            "simple": is_simple_repository(repo),
            "auditable": is_auditable_repository(repo),
            "transactional": is_transactional_repository(repo),
            "queryable": is_queryable_repository(repo),
        }

    @staticmethod
    def validate_repository_requirements(
        repo: Any,
        requires_audit: bool = False,
        requires_transactions: bool = False,
        requires_advanced_queries: bool = False,
    ) -> tuple[bool, list[str]]:
        """Validate that a repository meets specific requirements.

        Args:
            repo: The repository instance to validate.
            requires_audit: Whether audit capabilities are required.
            requires_transactions: Whether transaction capabilities are required.
            requires_advanced_queries: Whether advanced query capabilities are required.

        Returns:
            Tuple of (is_valid, list_of_missing_capabilities).
        """
        missing_capabilities = []

        if not (is_comprehensive_repository(repo) or is_simple_repository(repo)):
            missing_capabilities.append("basic_repository_operations")

        if requires_audit and not is_auditable_repository(repo):
            missing_capabilities.append("audit_capabilities")

        if requires_transactions and not is_transactional_repository(repo):
            missing_capabilities.append("transaction_capabilities")

        if requires_advanced_queries and not is_queryable_repository(repo):
            missing_capabilities.append("advanced_query_capabilities")

        return len(missing_capabilities) == 0, missing_capabilities
