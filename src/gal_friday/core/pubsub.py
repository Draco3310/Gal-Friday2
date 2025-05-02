"""Core Pub/Sub event bus implementation using Event objects."""

import asyncio
import logging
from collections import defaultdict
from typing import Callable, Coroutine, Dict, List, Any, TypeVar

from .events import Event, EventType # Import base Event and Enum

# Type variable for specific Event subclasses
E = TypeVar("E", bound=Event)

# Handler now expects a specific Event subclass
EventHandler = Callable[[E], Coroutine[Any, Any, None]]


class PubSubManager:
    """Manages event subscriptions and publishing within the application."""

    def __init__(self, logger: logging.Logger):
        # Subscribers stored by EventType enum member
        self._subscribers: Dict[EventType, List[Callable[[Event], Coroutine[Any, Any, None]]]] = defaultdict(list)
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._logger = logger
        self._consumer_task: asyncio.Task | None = None

    async def publish(self, event: Event) -> None:
        """Publish an event object by putting it onto the internal queue."""
        # Get event_type from the field or attribute
        event_type = getattr(event, 'event_type', None)
        if event_type is None:
            self._logger.warning(f"Event without event_type attribute: {event}")
            return
            
        await self._event_queue.put(event)
        self._logger.debug(f"Published event: {event_type.name} ({event.event_id})")

    def subscribe(self, event_type: EventType, handler: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        """Register a handler coroutine for a specific EventType."""
        # Type hint for handler is broad (Any) because the dict stores handlers for different event types.
        # The dispatcher logic ensures the correct event type is passed.
        self._subscribers[event_type].append(handler)
        self._logger.info(f"Handler {getattr(handler, '__name__', repr(handler))} subscribed to {event_type.name}")

    def unsubscribe(self, event_type: EventType, handler: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        """Remove a handler for a specific EventType."""
        try:
            self._subscribers[event_type].remove(handler)
            self._logger.info(f"Handler {getattr(handler, '__name__', repr(handler))} unsubscribed from {event_type.name}")
        except ValueError:
            self._logger.warning(
                f"Attempted to unsubscribe handler {getattr(handler, '__name__', repr(handler))} from {event_type.name}, "
                f"but it was not found."
            )

    async def _event_consumer(self) -> None:
        """Internal task to consume events from the queue and dispatch them."""
        self._logger.info("Event consumer task started.")
        while True:
            try:
                event: Event = await self._event_queue.get()
                
                # Get the event_type from the event object itself
                event_type = getattr(event, 'event_type', None)
                
                if not isinstance(event_type, EventType):
                    self._logger.warning(f"Received event object with invalid/missing event_type attribute: {event}")
                    self._event_queue.task_done()
                    continue

                handlers = self._subscribers.get(event_type, [])
                if not handlers:
                    self._logger.debug(f"No subscribers for event type: {event_type.name}")

                for handler in handlers:
                    try:
                        # Schedule the handler execution, passing the full event object
                        asyncio.create_task(handler(event))
                    except Exception as e:
                        handler_name = getattr(handler, '__name__', repr(handler))
                        self._logger.error(
                            f"Error scheduling/executing handler {handler_name} for event {event_type.name}: {e}",
                            exc_info=True,
                        )

                self._event_queue.task_done()
            except asyncio.CancelledError:
                self._logger.info("Event consumer task received cancellation request.")
                break
            except Exception as e:
                self._logger.error(f"Critical error in event consumer loop: {e}", exc_info=True)
                # Avoid busy-looping on persistent errors
                await asyncio.sleep(1)

        self._logger.info("Event consumer task stopped.")


    async def start(self) -> None:
        """Start the background event consumer task."""
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._event_consumer())
            self._logger.info("PubSubManager started.")
        else:
            self._logger.warning("PubSubManager already started.")

    async def stop(self) -> None:
        """Stop the background event consumer task gracefully."""
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass # Expected cancellation
            except Exception as e:
                self._logger.error(f"Error during PubSubManager stop: {e}", exc_info=True)
        self._logger.info("PubSubManager stopped.")