"""Core Pub/Sub event bus implementation using Event objects."""

import asyncio
from collections import defaultdict
from collections.abc import Coroutine
import logging
from typing import Any, Callable, Optional, Protocol, TypeVar

# Fix import path for ConfigManager
from ..config_manager import ConfigManager
from .events import Event, EventType  # Import base Event and Enum

# Type variable for specific Event subclasses
E = TypeVar("E", bound=Event)

# Handler now expects a specific Event subclass
EventHandler = Callable[[E], Coroutine[Any, Any, None]]


# Define a Protocol for events with event_type
class EventWithType(Protocol):
    """Protocol for events that have an event_type attribute."""

    event_id: str
    event_type: EventType


class PubSubManager:
    """Manages event subscriptions and publishing within the application."""

    def __init__(self, logger: logging.Logger, config_manager: ConfigManager) -> None:
        """Initialize the PubSub manager.

        Args
        ----
            logger: Logger instance for reporting operations
            config_manager: Configuration manager for fetching settings
        """
        # Subscribers stored by EventType enum member
        self._subscribers: dict[EventType, list[Callable[[Event], Coroutine[Any, Any, None]]]] = (
            defaultdict(list)
        )
        self._logger = logger
        self._config = config_manager  # Store config reference

        # Configure queue size from config
        queue_maxsize = self._config.get_int("pubsub.queue_maxsize", 0)  # Default to 0 (unlimited)
        # Use PriorityQueue with tuple of (priority, event)
        self._event_queue: asyncio.PriorityQueue[tuple[int, Event]] = asyncio.PriorityQueue(
            maxsize=queue_maxsize
        )
        self._logger.info(
            "PubSubManager initialized with queue maxsize: %s",
            "unlimited" if queue_maxsize == 0 else queue_maxsize
        )

        # Configure handler timeout
        self._handler_timeout_s = self._config.get_float("pubsub.handler_timeout_seconds", 10.0)
        self._logger.info("PubSubManager handler timeout: %ss", self._handler_timeout_s)

        # Error handling configuration
        self._consumer_error_sleep_s = self._config.get_float(
            "pubsub.consumer_error_sleep_seconds", 1.0
        )
        self._handler_max_failures = self._config.get_int("pubsub.handler_max_failures", 5)
        self._handler_failure_counts: dict[Callable, int] = defaultdict(int)

        # Metrics tracking
        self._events_published_count = 0
        self._events_processed_count = 0
        self._handler_errors_count = 0
        self._metrics_log_interval_s = self._config.get_float(
            "pubsub.metrics_log_interval_s", 60.0
        )
        self._metrics_task: Optional[asyncio.Task] = None

        self._consumer_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()

    async def publish(self, event: Event) -> None:
        """Publish an event object by putting it onto the internal queue."""
        # Get event_type from the field or attribute
        event_type = getattr(event, "event_type", None)
        if event_type is None:
            self._logger.warning("Event without event_type attribute: %s", event)
            return

        if not isinstance(event_type, EventType):
            self._logger.warning("Event with invalid EventType: %s", event)
            return

        try:
            # Get priority from event_type value (lower value = higher priority)
            priority = int(event_type.value)

            # Put (priority, event) tuple onto the queue
            await self._event_queue.put((priority, event))
            self._events_published_count += 1
            self._logger.debug(
                "Published event: %s (%s) with priority %s",
                event_type.name,
                getattr(event, "event_id", "unknown"),
                priority
            )
        except Exception:
            self._logger.exception("Error publishing event %s", event_type.name)

    def subscribe(
        self, event_type: EventType, handler: Callable[[Any], Coroutine[Any, Any, None]]
    ) -> None:
        """Register a handler coroutine for a specific EventType."""
        # Type hint for handler is broad (Any) because the dict stores handlers for
        # different event types. The dispatcher logic ensures the correct event type is passed.
        self._subscribers[event_type].append(handler)
        handler_name = getattr(handler, "__name__", repr(handler))
        self._logger.info("Handler %s subscribed to %s", handler_name, event_type.name)

    def unsubscribe(
        self, event_type: EventType, handler: Callable[[Any], Coroutine[Any, Any, None]]
    ) -> None:
        """Remove a handler for a specific EventType."""
        try:
            self._subscribers[event_type].remove(handler)
            handler_name = getattr(handler, "__name__", repr(handler))
            self._logger.info("Handler %s unsubscribed from %s", handler_name, event_type.name)
        except ValueError:
            handler_name = getattr(handler, "__name__", repr(handler))
            self._logger.warning(
                "Attempted to unsubscribe handler %s from %s, but it was not found.",
                handler_name,
                event_type.name
            )

    async def _dispatch_event_to_handler(
        self, handler: Callable[[Event], Coroutine[Any, Any, None]], event: Event
    ) -> None:
        """Execute a single handler with timeout and error handling."""
        handler_name = getattr(handler, "__name__", repr(handler))
        event_type = getattr(event, "event_type", None)
        event_type_name = (
            getattr(event_type, "name", "UnknownType") if event_type else "UnknownType"
        )
        event_id = getattr(event, "event_id", "unknown")

        try:
            # Wrap the handler call in wait_for
            await asyncio.wait_for(handler(event), timeout=self._handler_timeout_s)
            # Reset failure count on success
            if (
                handler in self._handler_failure_counts
                and self._handler_failure_counts[handler] > 0
            ):
                self._handler_failure_counts[handler] = 0
                self._logger.debug("Reset failure count for handler %s", handler_name)
        except asyncio.TimeoutError:
            self._logger.exception(
                "Handler %s timed out (> %ss) processing event %s (%s).",
                handler_name, self._handler_timeout_s, event_type_name, event_id
            )
            self._handler_errors_count += 1
            if event_type and isinstance(event_type, EventType):
                self._track_handler_failure(
                    handler, event_type, f"timeout (> {self._handler_timeout_s}s)"
                )
        except Exception as e:
            self._logger.exception(
                "Error executing handler %s for event %s (%s)",
                handler_name, event_type_name, event_id
            )
            self._handler_errors_count += 1
            if event_type and isinstance(event_type, EventType):
                self._track_handler_failure(handler, event_type, str(e))

    def _track_handler_failure(
        self, handler: Callable, event_type: EventType, error_reason: str
    ) -> None:
        """Track handler failures and auto-unsubscribe if threshold is exceeded."""
        handler_name = getattr(handler, "__name__", repr(handler))

        # Increment failure count
        self._handler_failure_counts[handler] += 1
        failure_count = self._handler_failure_counts[handler]

        # Check if handler should be auto-unsubscribed
        if failure_count >= self._handler_max_failures:
            self._logger.critical(
                "Handler %s exceeded maximum failures (%s). Last error: %s. "
                "Auto-unsubscribing from %s.",
                handler_name,
                self._handler_max_failures,
                error_reason,
                event_type.name,
            )
            try:
                self.unsubscribe(event_type, handler)
                # Clear the failure count after unsubscribing
                del self._handler_failure_counts[handler]
            except Exception:
                self._logger.exception("Error auto-unsubscribing handler %s", handler_name)
        else:
            self._logger.warning(
                "Handler %s failure %s/%s for event type %s. Error: %s",
                handler_name,
                failure_count,
                self._handler_max_failures,
                event_type.name,
                error_reason
            )

    async def _event_consumer(self) -> None:
        """Consume events from the queue and dispatch them to handlers."""
        self._logger.info("Event consumer task started.")
        while True:
            try:
                # Get (priority, event) from PriorityQueue
                _, event = await self._event_queue.get()

                # Get the event_type from the event object itself
                event_type = getattr(event, "event_type", None)

                if not isinstance(event_type, EventType):
                    self._logger.warning(
                        "Received event object with invalid/missing event_type attribute: %s",
                        event,
                    )
                    self._event_queue.task_done()
                    continue

                handlers = self._subscribers.get(event_type, [])
                if not handlers:
                    self._logger.debug("No subscribers for event type: %s", event_type.name)

                for handler in handlers:
                    # Schedule handler execution through dispatch wrapper
                    task = asyncio.create_task(self._dispatch_event_to_handler(handler, event))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

                self._event_queue.task_done()
                self._events_processed_count += 1
            except asyncio.CancelledError:
                self._logger.info("Event consumer task received cancellation request.")
                break
            except Exception:
                self._logger.exception("Critical error in event consumer loop")
                # Use configurable sleep time
                await asyncio.sleep(self._consumer_error_sleep_s)

        self._logger.info("Event consumer task stopped.")

    async def _log_metrics_periodically(self) -> None:
        """Log queue metrics at regular intervals."""
        while True:
            await asyncio.sleep(self._metrics_log_interval_s)
            qsize = self._event_queue.qsize()
            self._logger.info(
                "PubSub Metrics: QueueSize=%s, Published=%s, Processed=%s, HandlerErrors=%s",
                qsize,
                self._events_published_count,
                self._events_processed_count,
                self._handler_errors_count
            )

    async def start(self) -> None:
        """Start the background event consumer task."""
        if self._consumer_task is None or self._consumer_task.done():
            self._consumer_task = asyncio.create_task(self._event_consumer())
            if self._metrics_log_interval_s > 0:
                self._metrics_task = asyncio.create_task(self._log_metrics_periodically())
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
                pass  # Expected cancellation
            except Exception:
                self._logger.exception("Error during PubSubManager stop")

        if self._metrics_task and not self._metrics_task.done():
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass  # Expected cancellation
            except Exception:
                self._logger.exception("Error during metrics task stop")

        # Wait for any remaining background tasks to complete
        if self._background_tasks:
            self._logger.info(
                "Waiting for %s background tasks to complete...",
                len(self._background_tasks),
            )
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._logger.info("All background tasks completed.")

        self._logger.info("PubSubManager stopped.")
