"""Unit tests for the ServiceOrchestrator."""

import pytest

from gal_friday.services.service_orchestrator import ServiceOrchestrator


class DummyService:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    async def initialize(self, *args, **kwargs) -> None:
        self.events.append(f"init_{self.name}")

    async def start(self) -> None:
        self.events.append(f"start_{self.name}")


class FailingService(DummyService):
    async def initialize(self, *args, **kwargs) -> None:
        raise RuntimeError("fail")


@pytest.mark.asyncio
async def test_initialization_and_start_order() -> None:
    events: list[str] = []
    s1 = DummyService("one", events)
    s2 = DummyService("two", events)
    orchestrator = ServiceOrchestrator([s1, s2])

    await orchestrator.initialize_all(None)
    await orchestrator.start_all()

    assert events == ["init_one", "init_two", "start_one", "start_two"]


@pytest.mark.asyncio
async def test_initialize_failure_propagates() -> None:
    events: list[str] = []
    s1 = FailingService("bad", events)
    s2 = DummyService("ok", events)
    orchestrator = ServiceOrchestrator([s1, s2])

    with pytest.raises(RuntimeError):
        await orchestrator.initialize_all(None)

    assert events == []
