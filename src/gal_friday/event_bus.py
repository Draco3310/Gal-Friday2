import asyncio
import logging
from collections import defaultdict
from typing import Callable, Coroutine, Dict, Set, Optional

# Assuming Event and EventType are defined in core.events
# Use TYPE_CHECKING to avoid circular imports if Event needs PubSubManager type hint later
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core.events import Event, EventType
else:
    # Simple placeholders if not type checking (won't actually work but allows syntax check)
    class EventType:
        pass

    class Event:
        event_type: EventType


log = logging.getLogger(__name__)


class PubSubManager:
    """Handles Publish/Subscribe communication between modules using asyncio Queues."""

    def __init__(self):
        # Stores subscribers: Dict[EventType, Set[asyncio.Queue]]
        self._subscribers: Dict["EventType", Set[asyncio.Queue]] = defaultdict(set)
        # Stores listener tasks for cleanup: Dict[asyncio.Queue, asyncio.Task]
        self._listener_tasks: Dict[asyncio.Queue, asyncio.Task] = {}
        log.info("PubSubManager initialized.")

    async def publish(self, event: "Event") -> None:
        """
        Publishes an event to all subscribers of its type.

        Args:
            event: The Event object to publish.
        """
        event_type = event.event_type
        if event_type not in self._subscribers:
            log.debug(
                f"No subscribers for event type {event_type}. "
                f"Event dropped: {event}"
            )
            return

        subscribers = self._subscribers[event_type]
        log.debug(
            f"Publishing event {type(event).__name__} ({event_type}) to "
            f"{len(subscribers)} subscribers."
        )
        # Use asyncio.gather to put the event into all relevant queues concurrently
        # This prevents one slow subscriber from blocking others.
        # We capture potential exceptions during the put operation.
        results = await asyncio.gather(
            *[self._safe_put(queue, event) for queue in subscribers],
            return_exceptions=True
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Get corresponding queue (order is preserved)
                queue = list(subscribers)[i]
                log.error(
                    f"Error putting event {event_type} into a subscriber queue: "
                    f"{result}. Queue: {queue}",
                    exc_info=result,
                )
                # Potential actions: remove subscriber, log error, etc.
                # For now, just log the error.

    async def _safe_put(self, queue: asyncio.Queue, event: "Event"):
        """Safely put an item into a queue, handling potential Full errors or other issues."""
        try:
            # Consider using put_nowait if backpressure is undesirable and dropping is acceptable,
            # but await put provides backpressure if queues are bounded.
            await queue.put(event)
        except asyncio.QueueFull:
            log.warning(
                f"Subscriber queue is full. Event {type(event).__name__} dropped "
                f"for this subscriber. Consider increasing queue size or faster processing."
            )
        except Exception as e:
            log.exception(
                f"Unexpected error putting event {type(event).__name__} into queue.",
                exc_info=e
            )
            raise  # Reraise to be caught by asyncio.gather

    def subscribe(
        self, event_type: "EventType", handler: Callable[["Event"], Coroutine]
    ) -> asyncio.Queue:
        """
        Subscribes a handler coroutine to a specific event type.

        Creates an asyncio.Queue for the subscriber and starts a listener task.

        Args:
            event_type: The EventType to subscribe to.
            handler: The async function (coroutine) to call when an event is received.

        Returns:
            The asyncio.Queue associated with this subscription (useful for unsubscribing).
        """
        # Create a unique queue for this specific subscription instance
        queue = asyncio.Queue()  # Consider adding maxsize if needed
        self._subscribers[event_type].add(queue)

        # Start a listener task that consumes from the queue and calls the handler
        listener_task = asyncio.create_task(self._listener_loop(queue, handler))
        self._listener_tasks[queue] = listener_task

        log.info(f"Handler {handler.__name__} subscribed to {event_type}. Queue: {queue}")
        return queue

    async def _listener_loop(self, queue: asyncio.Queue, handler: Callable[["Event"], Coroutine]):
        """Internal loop that listens on a queue and calls the handler."""
        log.debug(f"Starting listener loop for handler {handler.__name__} on queue {queue}")
        while True:
            try:
                event = await queue.get()
                log.debug(
                    f"Handler {handler.__name__} received event "
                    f"{type(event).__name__} from queue {queue}"
                )
                try:
                    # Call the actual handler coroutine
                    await handler(event)
                except Exception as e:
                    log.exception(
                        f"Error in event handler {handler.__name__} processing event "
                        f"{type(event).__name__}: {e}",
                        exc_info=e,
                    )
                    # Decide if the loop should continue or break on handler error
                finally:
                    queue.task_done()  # Notify the queue that the task is complete
            except asyncio.CancelledError:
                log.info(f"Listener loop for handler {handler.__name__} cancelled.")
                break
            except Exception as e:
                log.exception(
                    f"Unexpected error in listener loop for handler "
                    f"{handler.__name__}. Loop exiting.",
                    exc_info=e,
                )
                break  # Exit loop on unexpected error

    async def unsubscribe(self, event_type: "EventType", queue: asyncio.Queue) -> bool:
        """
        Unsubscribes a specific listener queue from an event type and cancels its task.

        Args:
            event_type: The EventType to unsubscribe from.
            queue: The specific queue returned by the subscribe method.

        Returns:
            True if successfully unsubscribed, False otherwise.
        """
        if event_type in self._subscribers and queue in self._subscribers[event_type]:
            self._subscribers[event_type].remove(queue)
            log.info(f"Removed subscription queue {queue} from {event_type}.")
            if not self._subscribers[event_type]:
                del self._subscribers[event_type]  # Clean up empty set

            # Cancel and remove the corresponding listener task
            if queue in self._listener_tasks:
                task = self._listener_tasks.pop(queue)
                task.cancel()
                try:
                    await task  # Allow cancellation to propagate
                except asyncio.CancelledError:
                    log.debug(f"Listener task for queue {queue} successfully cancelled.")
                except Exception as e:
                    log.exception(
                        f"Error during listener task cancellation for queue {queue}.",
                        exc_info=e
                    )
                log.info(f"Cancelled listener task for queue {queue}.")
            else:
                log.warning(f"No active listener task found for queue {queue} during unsubscribe.")

            return True
        else:
            log.warning(
                f"Attempted to unsubscribe queue {queue} from {event_type}, "
                f"but it was not found."
            )
            return False

    async def stop(self):
        """Stops all listener tasks and clears subscriptions."""
        log.info(
            f"Stopping PubSubManager. Cancelling "
            f"{len(self._listener_tasks)} listener tasks..."
        )
        if not self._listener_tasks:
            log.info("No active listener tasks to stop.")
            return

        # Cancel all tasks concurrently
        tasks_to_cancel = list(self._listener_tasks.values())
        for task in tasks_to_cancel:
            task.cancel()

        results = await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                # Log errors that are not simple cancellations
                log.error(
                    f"Error encountered while stopping a listener task: {result}",
                    exc_info=result
                )

        self._listener_tasks.clear()
        self._subscribers.clear()
        log.info("All listener tasks cancelled and subscriptions cleared. PubSubManager stopped.")

    def _validate_event_type(self, event_type: str) -> bool:
        """Validates that the event type is registered."""
        if event_type not in self._subscribers:
            log.error(
                f"Invalid event type: {event_type}. Must be one of: {list(self._subscribers)}",
                source_module=self.__class__.__name__,
            )
            return False
        return True


# Example Usage (for testing purposes, remove in production)
async def example_subscriber_one(event: "Event"):
    print(f"Subscriber ONE received: {event}")
    await asyncio.sleep(0.1)  # Simulate work


async def example_subscriber_two(event: "Event"):
    print(f"Subscriber TWO received: {event}")
    # Check attribute safely before accessing
    if hasattr(event, "new_state") and event.new_state == "HALTED":  # Example of specific action
        print("Subscriber TWO sees HALT!")
    await asyncio.sleep(0.05)


async def main():
    # Need actual Event/EventType definitions for this example
    # Let's redefine minimal ones here for standalone testing
    from enum import Enum, auto
    from dataclasses import dataclass, field
    import uuid
    from datetime import datetime

    class EventType(Enum):
        SYSTEM_STATE_CHANGE = auto()
        OTHER_EVENT = auto()

    # Use the same pattern as core/events.py for consistency
    class EventMetadata:
        def __init__(self, source_module: str = "Test", event_id: Optional[uuid.UUID] = None, timestamp: Optional[datetime] = None):
            self.source_module = source_module
            self.event_id = event_id if event_id is not None else uuid.uuid4()
            self.timestamp = timestamp if timestamp is not None else datetime.utcnow()
            
    @dataclass(frozen=True)
    class Event:
        source_module: str
        event_id: uuid.UUID
        timestamp: datetime

        @classmethod
        def create_metadata(cls, source_module: str = "Test") -> EventMetadata:
            return EventMetadata(source_module=source_module)

    @dataclass(frozen=True)
    class SystemStateEvent(Event):
        new_state: str
        reason: str
        event_type: EventType = field(default=EventType.SYSTEM_STATE_CHANGE, init=False)

        @classmethod
        def create(cls, new_state: str, reason: str, source_module: str = "Test") -> "SystemStateEvent":
            metadata = Event.create_metadata(source_module)
            return cls(source_module=metadata.source_module, event_id=metadata.event_id, timestamp=metadata.timestamp, new_state=new_state, reason=reason)

    @dataclass(frozen=True)
    class OtherEvent(Event):
        data: str
        event_type: EventType = field(default=EventType.OTHER_EVENT, init=False)
        
        @classmethod
        def create(cls, data: str, source_module: str = "Test") -> "OtherEvent":
            metadata = Event.create_metadata(source_module)
            return cls(source_module=metadata.source_module, event_id=metadata.event_id, timestamp=metadata.timestamp, data=data)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    pubsub = PubSubManager()

    # Subscribe handlers
    sub1_queue = pubsub.subscribe(EventType.SYSTEM_STATE_CHANGE, example_subscriber_one)
    pubsub.subscribe(EventType.SYSTEM_STATE_CHANGE, example_subscriber_two)
    pubsub.subscribe(EventType.OTHER_EVENT, example_subscriber_one)

    print("Publishing events...")
    # Update instantiation to use factory methods
    event1 = SystemStateEvent.create(new_state="RUNNING", reason="Startup", source_module="Main")
    event2 = SystemStateEvent.create(new_state="HALTED", reason="Test Halt", source_module="TestScript")
    event3 = OtherEvent.create(data="Some other data", source_module="AnotherModule")

    await pubsub.publish(event1)
    await pubsub.publish(event3)
    await pubsub.publish(event2)

    # Allow time for events to be processed
    await asyncio.sleep(0.5)

    # Test unsubscribe
    print("\nUnsubscribing Subscriber ONE from SYSTEM_STATE_CHANGE...")
    unsubscribed = await pubsub.unsubscribe(EventType.SYSTEM_STATE_CHANGE, sub1_queue)
    print(f"Unsubscribed successfully: {unsubscribed}")

    print("\nPublishing another SYSTEM_STATE_CHANGE event...")
    event4 = SystemStateEvent.create(new_state="RUNNING", reason="Resumed", source_module="TestScript")
    await pubsub.publish(event4)  # Only subscriber TWO should get this

    await asyncio.sleep(0.5)

    print("\nStopping PubSubManager...")
    await pubsub.stop()
    print("PubSubManager stopped.")


if __name__ == "__main__":
    # asyncio.run(main())
    pass
