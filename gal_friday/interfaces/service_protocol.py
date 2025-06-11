from typing import Protocol, runtime_checkable


@runtime_checkable
class ServiceProtocol(Protocol):
    """Protocol for application services with a standard lifecycle."""

    async def initialize(self) -> None:
        """Perform any asynchronous initialization logic."""
        ...

    async def start(self) -> None:
        """Start the service."""
        ...

    async def stop(self) -> None:
        """Stop the service and cleanup resources."""
        ...
