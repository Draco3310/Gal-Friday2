"""Type guards for repository operations and validation.

This module provides type guards to ensure repository instances meet
the required protocols and can be safely used in type-safe contexts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Callable

    from .protocols import (
        AuditableRepositoryProtocol,
        BaseRepositoryProtocol,
        ComprehensiveRepositoryProtocol,
        QueryableRepositoryProtocol,
        SimpleRepositoryProtocol,
        TransactionalRepositoryProtocol,
    )


# Type guards for repository protocol validation
def is_base_repository(obj: Any) -> TypeGuard[BaseRepositoryProtocol[Any]]:
    """Check if an object implements BaseRepositoryProtocol."""
    return (
        hasattr(obj, "create")
        and callable(obj.create)
        and hasattr(obj, "get_by_id")
        and callable(obj.get_by_id)
        and hasattr(obj, "update")
        and callable(obj.update)
        and hasattr(obj, "delete")
        and callable(obj.delete)
    )


def is_queryable_repository(obj: Any) -> TypeGuard[QueryableRepositoryProtocol[Any]]:
    """Check if an object implements QueryableRepositoryProtocol."""
    return (
        hasattr(obj, "find_all")
        and callable(obj.find_all)
        and hasattr(obj, "count")
        and callable(obj.count)
        and hasattr(obj, "exists")
        and callable(obj.exists)
    )


def is_comprehensive_repository(obj: Any) -> TypeGuard[ComprehensiveRepositoryProtocol[Any]]:
    """Check if an object implements ComprehensiveRepositoryProtocol."""
    return (
        is_base_repository(obj)
        and is_queryable_repository(obj)
        and hasattr(obj, "session_maker")
        and hasattr(obj, "model_class")
    )


def is_simple_repository(obj: Any) -> TypeGuard[SimpleRepositoryProtocol[Any]]:
    """Check if an object implements SimpleRepositoryProtocol."""
    return is_base_repository(obj) and hasattr(obj, "session") and hasattr(obj, "model")


def is_auditable_repository(obj: Any) -> TypeGuard[AuditableRepositoryProtocol[Any]]:
    """Check if an object implements AuditableRepositoryProtocol."""
    return (
        hasattr(obj, "create_with_audit")
        and callable(obj.create_with_audit)
        and hasattr(obj, "update_with_audit")
        and callable(obj.update_with_audit)
        and hasattr(obj, "delete_with_audit")
        and callable(obj.delete_with_audit)
    )


def is_transactional_repository(obj: Any) -> TypeGuard[TransactionalRepositoryProtocol[Any]]:
    """Check if an object implements TransactionalRepositoryProtocol."""
    return (
        hasattr(obj, "begin_transaction")
        and callable(obj.begin_transaction)
        and hasattr(obj, "commit_transaction")
        and callable(obj.commit_transaction)
        and hasattr(obj, "rollback_transaction")
        and callable(obj.rollback_transaction)
    )


# Repository validation utilities
class RepositoryTypeGuards:
    """Utility class for repository type validation."""

    @staticmethod
    def validate_repository_type(obj: Any, expected_type: str) -> tuple[bool, list[str]]:
        """Validate that a repository implements the expected type.

        Args:
            obj: The repository object to validate.
            expected_type: The expected repository type ("base", "queryable", "comprehensive",
                "simple", "auditable", "transactional").

        Returns:
            Tuple of (is_valid, list_of_missing_methods).
        """
        type_validators: dict[str, Callable[[Any], bool]] = {
            "base": is_base_repository,
            "queryable": is_queryable_repository,
            "comprehensive": is_comprehensive_repository,
            "simple": is_simple_repository,
            "auditable": is_auditable_repository,
            "transactional": is_transactional_repository,
        }

        if expected_type not in type_validators:
            return False, [f"Unknown repository type: {expected_type}"]

        validator = type_validators[expected_type]
        is_valid = validator(obj)

        if is_valid:
            return True, []

        # Get missing methods for the expected type
        missing_methods = RepositoryTypeGuards._get_missing_methods_for_type(obj, expected_type)
        return False, missing_methods

    @staticmethod
    def _get_missing_methods_for_type(obj: Any, expected_type: str) -> list[str]:
        """Get missing methods for a specific repository type."""
        type_checkers = {
            "base": RepositoryTypeGuards._check_base_methods,
            "queryable": RepositoryTypeGuards._check_queryable_methods,
            "comprehensive": RepositoryTypeGuards._check_comprehensive_methods,
            "simple": RepositoryTypeGuards._check_simple_methods,
            "auditable": RepositoryTypeGuards._check_auditable_methods,
            "transactional": RepositoryTypeGuards._check_transactional_methods,
        }

        checker = type_checkers.get(expected_type)
        return checker(obj) if checker else []

    @staticmethod
    def _check_base_methods(obj: Any) -> list[str]:
        """Check for missing base repository methods."""
        required_methods = ["create", "get_by_id", "update", "delete"]
        return [method for method in required_methods if not hasattr(obj, method) or not callable(getattr(obj, method))]

    @staticmethod
    def _check_queryable_methods(obj: Any) -> list[str]:
        """Check for missing queryable repository methods."""
        required_methods = ["find_all", "count", "exists"]
        return [method for method in required_methods if not hasattr(obj, method) or not callable(getattr(obj, method))]

    @staticmethod
    def _check_comprehensive_methods(obj: Any) -> list[str]:
        """Check for missing comprehensive repository methods."""
        missing = []
        if not is_base_repository(obj):
            missing.append("base_repository_methods")
        if not is_queryable_repository(obj):
            missing.append("queryable_repository_methods")
        if not hasattr(obj, "session_maker"):
            missing.append("session_maker")
        if not hasattr(obj, "model_class"):
            missing.append("model_class")
        return missing

    @staticmethod
    def _check_simple_methods(obj: Any) -> list[str]:
        """Check for missing simple repository methods."""
        missing = []
        if not is_base_repository(obj):
            missing.append("base_repository_methods")
        if not hasattr(obj, "session"):
            missing.append("session")
        if not hasattr(obj, "model"):
            missing.append("model")
        return missing

    @staticmethod
    def _check_auditable_methods(obj: Any) -> list[str]:
        """Check for missing auditable repository methods."""
        required_methods = ["create_with_audit", "update_with_audit", "delete_with_audit"]
        return [method for method in required_methods if not hasattr(obj, method) or not callable(getattr(obj, method))]

    @staticmethod
    def _check_transactional_methods(obj: Any) -> list[str]:
        """Check for missing transactional repository methods."""
        required_methods = ["begin_transaction", "commit_transaction", "rollback_transaction"]
        return [method for method in required_methods if not hasattr(obj, method) or not callable(getattr(obj, method))]

    @staticmethod
    def get_repository_capabilities(obj: Any) -> dict[str, bool]:
        """Get the capabilities of a repository.

        Args:
            obj: The repository object to analyze.

        Returns:
            Dictionary mapping capability names to whether they're supported.
        """
        return {
            "base": is_base_repository(obj),
            "queryable": is_queryable_repository(obj),
            "comprehensive": is_comprehensive_repository(obj),
            "simple": is_simple_repository(obj),
            "auditable": is_auditable_repository(obj),
            "transactional": is_transactional_repository(obj),
        }

    @staticmethod
    def validate_repository_requirements(
        obj: Any,
        requires_base: bool = True,
        requires_queryable: bool = False,
        requires_comprehensive: bool = False,
        requires_simple: bool = False,
        requires_auditable: bool = False,
        requires_transactional: bool = False,
    ) -> tuple[bool, list[str]]:
        """Validate that a repository meets specific requirements.

        Args:
            obj: The repository object to validate.
            requires_base: Whether base capabilities are required.
            requires_queryable: Whether queryable capabilities are required.
            requires_comprehensive: Whether comprehensive capabilities are required.
            requires_simple: Whether simple capabilities are required.
            requires_auditable: Whether auditable capabilities are required.
            requires_transactional: Whether transactional capabilities are required.

        Returns:
            Tuple of (is_valid, list_of_missing_capabilities).
        """
        missing_capabilities = []

        if requires_base and not is_base_repository(obj):
            missing_capabilities.append("base_repository")

        if requires_queryable and not is_queryable_repository(obj):
            missing_capabilities.append("queryable_repository")

        if requires_comprehensive and not is_comprehensive_repository(obj):
            missing_capabilities.append("comprehensive_repository")

        if requires_simple and not is_simple_repository(obj):
            missing_capabilities.append("simple_repository")

        if requires_auditable and not is_auditable_repository(obj):
            missing_capabilities.append("auditable_repository")

        if requires_transactional and not is_transactional_repository(obj):
            missing_capabilities.append("transactional_repository")

        return len(missing_capabilities) == 0, missing_capabilities


# Assertion utilities for runtime validation
class RepositoryAssertions:
    """Assertion utilities for repository validation."""

    @staticmethod
    def assert_base_repository(obj: Any) -> None:
        """Assert that an object is a base repository."""
        if not is_base_repository(obj):
            raise TypeError(f"Expected BaseRepositoryProtocol, got {type(obj)}")

    @staticmethod
    def assert_queryable_repository(obj: Any) -> None:
        """Assert that an object is a queryable repository."""
        if not is_queryable_repository(obj):
            raise TypeError(f"Expected QueryableRepositoryProtocol, got {type(obj)}")

    @staticmethod
    def assert_comprehensive_repository(obj: Any) -> None:
        """Assert that an object is a comprehensive repository."""
        if not is_comprehensive_repository(obj):
            raise TypeError(f"Expected ComprehensiveRepositoryProtocol, got {type(obj)}")

    @staticmethod
    def assert_simple_repository(obj: Any) -> None:
        """Assert that an object is a simple repository."""
        if not is_simple_repository(obj):
            raise TypeError(f"Expected SimpleRepositoryProtocol, got {type(obj)}")

    @staticmethod
    def assert_auditable_repository(obj: Any) -> None:
        """Assert that an object is an auditable repository."""
        if not is_auditable_repository(obj):
            raise TypeError(f"Expected AuditableRepositoryProtocol, got {type(obj)}")

    @staticmethod
    def assert_transactional_repository(obj: Any) -> None:
        """Assert that an object is a transactional repository."""
        if not is_transactional_repository(obj):
            raise TypeError(f"Expected TransactionalRepositoryProtocol, got {type(obj)}")

    @staticmethod
    def assert_repository_capabilities(
        obj: Any,
        requires_base: bool = True,
        requires_queryable: bool = False,
        requires_comprehensive: bool = False,
        requires_simple: bool = False,
        requires_auditable: bool = False,
        requires_transactional: bool = False,
    ) -> None:
        """Assert that a repository meets specific capability requirements."""
        is_valid, missing_capabilities = RepositoryTypeGuards.validate_repository_requirements(
            obj,
            requires_base=requires_base,
            requires_queryable=requires_queryable,
            requires_comprehensive=requires_comprehensive,
            requires_simple=requires_simple,
            requires_auditable=requires_auditable,
            requires_transactional=requires_transactional,
        )

        if not is_valid:
            raise TypeError(
                f"Repository {type(obj)} missing required capabilities: {missing_capabilities}",
            )


# Convenience functions for common validation patterns
def ensure_base_repository(obj: Any) -> BaseRepositoryProtocol[Any]:
    """Ensure an object is a base repository."""
    RepositoryAssertions.assert_base_repository(obj)
    return obj  # type: ignore[no-any-return]


def ensure_queryable_repository(obj: Any) -> QueryableRepositoryProtocol[Any]:
    """Ensure an object is a queryable repository."""
    RepositoryAssertions.assert_queryable_repository(obj)
    return obj  # type: ignore[no-any-return]


def ensure_comprehensive_repository(obj: Any) -> ComprehensiveRepositoryProtocol[Any]:
    """Ensure an object is a comprehensive repository."""
    RepositoryAssertions.assert_comprehensive_repository(obj)
    return obj  # type: ignore[no-any-return]


def ensure_simple_repository(obj: Any) -> SimpleRepositoryProtocol[Any]:
    """Ensure an object is a simple repository."""
    RepositoryAssertions.assert_simple_repository(obj)
    return obj  # type: ignore[no-any-return]


def ensure_auditable_repository(obj: Any) -> AuditableRepositoryProtocol[Any]:
    """Ensure an object is an auditable repository."""
    RepositoryAssertions.assert_auditable_repository(obj)
    return obj  # type: ignore[no-any-return]


def ensure_transactional_repository(obj: Any) -> TransactionalRepositoryProtocol[Any]:
    """Ensure an object is a transactional repository."""
    RepositoryAssertions.assert_transactional_repository(obj)
    return obj  # type: ignore[no-any-return]
