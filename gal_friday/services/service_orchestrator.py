from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


class ServiceProtocol(Protocol):
    """Protocol for services managed by :class:`ServiceOrchestrator`."""

    async def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the service with required dependencies."""

    async def start(self) -> None:
        """Start the service."""


class ServiceOrchestrator:
    """Manage initialization and startup for a collection of services."""

    def __init__(self, services: Sequence[ServiceProtocol]) -> None:
        """Initialize the instance."""
        self.services = list[Any](services)

    async def initialize_all(self, *args: Any, **kwargs: Any) -> None:
        """Initialize all managed services in order."""
        for service in self.services:
            await service.initialize(*args, **kwargs)

    async def start_all(self) -> None:
        """Start all managed services in order."""
        for service in self.services:
            await service.start()
