# PubSubManager Module Documentation

## Module Overview

The `gal_friday.core.pubsub.py` module provides an asynchronous event bus, `PubSubManager`, designed for decoupled inter-module communication within the Gal-Friday application. It allows different parts of the system to publish events and subscribe to event types they are interested in, without direct knowledge of each other. This promotes modularity and scalability.

## Key Features

-   **Asynchronous Event Handling:** Built on Python's `asyncio` library for non-blocking event processing, suitable for I/O-bound operations.
-   **`EventType` Enum for Defining Event Types:** Events are categorized using an `EventType` enum (expected to be defined in `gal_friday.core.events`). This enum's integer values are used for event prioritization.
-   **Priority Queue for Events:** Events are processed based on the integer value of their `EventType`. Lower integer values signify higher priority.
-   **Subscription Model:** Asynchronous functions (coroutines) can subscribe to specific `EventType`s to handle them when published.
-   **Event Publishing:** Events are placed onto an `asyncio.PriorityQueue` for consumption and dispatch.
-   **Configurable Handler Execution Timeout:** Each event handler is executed with a configurable timeout to prevent indefinite blocking.
-   **Robust Error Handling:** Includes mechanisms for catching exceptions within handlers, tracking handler failures, and automatically unsubscribing persistently problematic handlers after a configurable number of failures.
-   **Configurable Parameters:** Queue size, handler timeout, error sleep times (for the consumer task), and maximum handler failures are configurable via the `ConfigManager`.
-   **Periodic Metrics Logging:** The system can periodically log metrics about the event queue, such as its current size, the total number of events published and processed, and error counts.

## Core Components

-   **`Event` (Base Class - defined in `gal_friday.core.events`):**
    -   This is the base class for all events handled by the `PubSubManager`.
    -   It is assumed to have at least an `event_id: str` attribute (for tracking and logging) and an `event_type: EventType` attribute that specifies its category and priority.
    -   When an `Event` instance is put on the priority queue, its `event_type.value` (integer) is used for prioritization.

-   **`EventType` (Enum - defined in `gal_friday.core.events`):**
    -   An enumeration that defines different types of events in the system.
    -   Each member of the enum should have an integer value, where a lower value indicates higher priority in the event queue. Example: `MY_CRITICAL_EVENT = 1`, `MY_NORMAL_EVENT = 10`.

-   **`EventHandler` (Type Alias):**
    -   Represents the signature for event handler functions.
    -   Defined as: `Callable[[E], Coroutine[Any, Any, None]]` where `E` is a specific subclass of `Event`.
    -   Handlers must be asynchronous functions (coroutines).

## Class `PubSubManager`

The `PubSubManager` is the central class managing the event bus.

### Initialization (`__init__`)

-   **Parameters:**
    -   `logger (logging.Logger)`: An instance of a Python logger for logging messages, warnings, and errors.
    -   `config_manager (ConfigManager)`: An instance of `ConfigManager` used to fetch configuration parameters for the PubSub system.
-   **Actions:**
    -   Initializes a `defaultdict(list)` to store subscribers, mapping `EventType` to a list of `EventHandler` functions.
    -   Creates an `asyncio.PriorityQueue` to hold incoming events. The maximum size of this queue is configurable.
    -   Loads configuration settings from `ConfigManager`, including:
        -   `pubsub.queue_maxsize`: Maximum number of events the queue can hold.
        -   `pubsub.handler_timeout_seconds`: Timeout for individual event handlers.
        -   `pubsub.consumer_error_sleep_seconds`: Time the event consumer sleeps after an unexpected error.
        -   `pubsub.handler_max_failures`: Number of times a handler can fail before being auto-unsubscribed.
        -   `pubsub.metrics_log_interval_s`: Interval for logging queue metrics.
    -   Initializes counters for metrics (published, processed, errors) and a dictionary to track handler failures.
    -   Sets up `asyncio.Event` objects for managing graceful shutdown (`_stop_event`, `_consumer_stopped_event`, `_metrics_stopped_event`).

### Core Methods

-   **`async publish(event: Event)`**:
    -   Puts an `event` onto the internal `asyncio.PriorityQueue`.
    -   The event's priority is determined by `event.event_type.value` (integer). Lower values have higher priority.
    -   Increments the count of published events for metrics.
    -   Logs the publication of an event (typically at DEBUG level).

-   **`subscribe(event_type: EventType, handler: EventHandler)`**:
    -   Registers an asynchronous `handler` coroutine to be called when an event of the specified `event_type` is published.
    -   Adds the handler to the list of subscribers for that `event_type`.
    -   Logs the subscription.

-   **`unsubscribe(event_type: EventType, handler: EventHandler)`**:
    -   Removes a previously registered `handler` for the given `event_type`.
    -   If the handler is found and removed, logs the unsubscription. Otherwise, logs a warning.

-   **`async _dispatch_event_to_handler(handler: EventHandler, event: Event)`**:
    -   An internal method responsible for executing a single `handler` with a given `event`.
    -   Uses `asyncio.wait_for` to run the handler with the configured `handler_timeout_seconds`.
    -   Catches `asyncio.TimeoutError` if the handler exceeds its execution time.
    -   Catches any other `Exception` raised by the handler.
    -   Logs successes, timeouts, or errors during handler execution.
    -   If an error occurs, it calls `_track_handler_failure` to record the failure.

-   **`_track_handler_failure(handler: Callable, event_type: EventType, error_reason: str)`**:
    -   An internal method to manage failures for a specific `handler`.
    -   Increments a failure counter for the `(event_type, handler)` pair.
    -   If the failure count reaches `handler_max_failures`, the handler is automatically unsubscribed from that `event_type`, and a critical error is logged.
    -   This prevents a consistently failing handler from disrupting event processing or consuming resources.

-   **`async _event_consumer()`**:
    -   The main background task that continuously fetches events from the priority queue.
    -   It runs in an infinite loop until the `_stop_event` is set.
    -   When an event is retrieved, it iterates through all subscribed handlers for that event's `EventType`.
    -   For each handler, it creates a new `asyncio.Task` to run `_dispatch_event_to_handler`, allowing concurrent execution of handlers for the same event if multiple handlers are subscribed.
    -   Handles `asyncio.CancelledError` for graceful shutdown.
    -   Includes error handling for unexpected issues during event consumption, with a configurable sleep period (`consumer_error_sleep_seconds`) before retrying.
    -   Signals `_consumer_stopped_event` upon completion.

-   **`async _log_metrics_periodically()`**:
    -   A background task that periodically logs key metrics of the PubSub system.
    -   Metrics include current queue size, total events published, total events processed, and total handler errors.
    -   The logging interval is configured by `pubsub.metrics_log_interval_s`.
    -   Runs until `_stop_event` is set.
    -   Signals `_metrics_stopped_event` upon completion.

-   **`async start()`**:
    -   Starts the PubSub system by creating and scheduling the `_event_consumer` and (if enabled) `_log_metrics_periodically` background tasks using `asyncio.create_task()`.
    -   Logs that the PubSub service has started.

-   **`async stop()`**:
    -   Initiates a graceful shutdown of the PubSub system.
    -   Sets the `_stop_event` to signal background tasks to terminate.
    -   Waits for the `_event_consumer` and `_log_metrics_periodically` tasks to complete by awaiting `_consumer_stopped_event` and `_metrics_stopped_event`.
    -   This ensures that events currently being processed (or tasks already dispatched to handlers) are allowed to finish up to their timeout. New events from the queue will not be processed after shutdown starts.
    -   Logs that the PubSub service is stopping and when it has successfully stopped.

### Configuration Options (loaded from `ConfigManager`)

The `PubSubManager` relies on the `ConfigManager` to fetch its operational parameters, typically prefixed with `pubsub.`:

-   **`pubsub.queue_maxsize` (int):** The maximum number of events that can be held in the priority queue. Defaults to `1000`.
-   **`pubsub.handler_timeout_seconds` (float):** The maximum time (in seconds) allowed for an event handler to process an event. Defaults to `5.0`.
-   **`pubsub.consumer_error_sleep_seconds` (float):** The time (in seconds) the main event consumer loop will sleep if an unexpected error occurs before retrying. Defaults to `5.0`.
-   **`pubsub.handler_max_failures` (int):** The number of times a specific handler can fail for a given event type before it is automatically unsubscribed. Defaults to `3`.
-   **`pubsub.metrics_log_interval_s` (int):** The interval (in seconds) at which PubSub queue metrics are logged. If `0` or negative, metrics logging is disabled. Defaults to `60`.

## Error Handling

The `PubSubManager` incorporates several layers of error handling:

-   **Handler Timeout:** Event handlers that take longer than `pubsub.handler_timeout_seconds` to complete are cancelled, and a timeout error is logged.
-   **Handler Exceptions:** Any exceptions raised within an event handler are caught by `_dispatch_event_to_handler`. The error is logged, and the system continues to process other events and handlers.
-   **Handler Failure Tracking & Auto-Unsubscription:** The `_track_handler_failure` method monitors repeated failures of the same handler for a specific event type. If failures exceed `pubsub.handler_max_failures`, the handler is automatically removed to prevent it from affecting the system further.
-   **Consumer Loop Errors:** The `_event_consumer` task has a general exception handler to catch unexpected errors during its operation. It logs the error and pauses for `pubsub.consumer_error_sleep_seconds` before continuing, preventing rapid failure loops.
-   **Graceful Shutdown:** The `stop()` method ensures that background tasks are signalled to terminate and are waited upon, allowing for cleaner resource release and completion of in-flight work within timeouts.

## Dependencies

-   **`asyncio`:** The core library for asynchronous programming in Python.
-   **`logging`:** Python's standard logging module.
-   **`collections.defaultdict`:** Used for conveniently managing lists of subscribers.
-   **`gal_friday.config_manager.ConfigManager`:** For retrieving configuration parameters.
-   **`gal_friday.core.events.Event`:** The base class for events.
-   **`gal_friday.core.events.EventType`:** The enum used for defining event types and priorities.

## Usage Example

```python
import asyncio
import logging
from enum import Enum
from typing import Coroutine, Any, Callable # For EventHandler type hint clarity
import uuid # For unique event_ids

# Assuming gal_friday.core.events might look like this:
class EventType(Enum):
    def __lt__(self, other): # Allows direct comparison for PriorityQueue
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

class Event:
    event_type: EventType
    def __init__(self, event_id: str = None):
        self.event_id = event_id or str(uuid.uuid4())

    # Required for PriorityQueue if events have same priority and are not otherwise comparable
    def __lt__(self, other):
        if isinstance(other, Event):
            return self.event_type < other.event_type # Primary sort by event type priority
        return NotImplemented


# Local application imports (if PubSubManager is in this structure)
# from gal_friday.core.pubsub import PubSubManager
# from gal_friday.config_manager import ConfigManager

# --- Mocking PubSubManager and ConfigManager if they are not available directly ---
# This is a simplified PubSubManager for the example to run standalone.
# In a real scenario, you would import the actual PubSubManager.
from collections import defaultdict
class PubSubManager:
    EventHandler = Callable[[Event], Coroutine[Any, Any, None]]

    def __init__(self, logger: logging.Logger, config_manager): # config_manager is mocked
        self.logger = logger
        self.config_manager = config_manager
        self._subscribers = defaultdict(list)
        self._event_queue = asyncio.PriorityQueue(
            maxsize=self.config_manager.get_int("pubsub.queue_maxsize", 1000)
        )
        self._handler_timeout = self.config_manager.get_float("pubsub.handler_timeout_seconds", 5.0)
        self._handler_max_failures = self.config_manager.get_int("pubsub.handler_max_failures", 3)
        self._handler_failures = defaultdict(int)
        self._metrics_log_interval = self.config_manager.get_int("pubsub.metrics_log_interval_s", 60)
        self._consumer_error_sleep = self.config_manager.get_float("pubsub.consumer_error_sleep_seconds", 5.0)

        self._stop_event = asyncio.Event()
        self._consumer_task = None
        self._metrics_task = None
        self._consumer_stopped_event = asyncio.Event()
        self._metrics_stopped_event = asyncio.Event()

        self._events_published = 0
        self._events_processed = 0
        self._events_failed = 0


    async def publish(self, event: Event):
        try:
            # Priority is event_type.value, event object
            await self._event_queue.put((event.event_type.value, event))
            self._events_published += 1
            self.logger.debug(f"Published event: {event.event_id} ({event.event_type.name})")
        except Exception as e:
            self.logger.error(f"Failed to publish event {event.event_id}: {e}")

    def subscribe(self, event_type: EventType, handler: EventHandler):
        self.logger.info(f"Subscribing handler {handler.__name__} to {event_type.name}")
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler):
        try:
            self._subscribers[event_type].remove(handler)
            self.logger.info(f"Unsubscribed handler {handler.__name__} from {event_type.name}")
        except ValueError:
            self.logger.warning(f"Handler {handler.__name__} not found for {event_type.name}")

    async def _dispatch_event_to_handler(self, handler: EventHandler, event: Event):
        handler_name = handler.__name__
        try:
            self.logger.debug(f"Dispatching event {event.event_id} to handler {handler_name}")
            await asyncio.wait_for(handler(event), timeout=self._handler_timeout)
            self.logger.debug(f"Handler {handler_name} completed for event {event.event_id}")
            # Reset failures on success if desired (optional feature)
            # if (event.event_type, handler) in self._handler_failures:
            #     del self._handler_failures[(event.event_type, handler)]
        except asyncio.TimeoutError:
            self.logger.error(f"Handler {handler_name} timed out for event {event.event_id}")
            self._track_handler_failure(handler, event.event_type, "timeout")
            self._events_failed +=1
        except Exception as e:
            self.logger.error(f"Handler {handler_name} failed for event {event.event_id}: {e}", exc_info=True)
            self._track_handler_failure(handler, event.event_type, str(e))
            self._events_failed +=1


    def _track_handler_failure(self, handler: EventHandler, event_type: EventType, error_reason: str):
        failure_key = (event_type, handler)
        self._handler_failures[failure_key] += 1
        failures = self._handler_failures[failure_key]
        self.logger.warning(f"Failure {failures}/{self._handler_max_failures} for handler {handler.__name__} on {event_type.name}. Reason: {error_reason}")
        if failures >= self._handler_max_failures:
            self.logger.critical(f"Handler {handler.__name__} reached max failures for {event_type.name}. Unsubscribing.")
            self.unsubscribe(event_type, handler)


    async def _event_consumer(self):
        self.logger.info("Event consumer started.")
        try:
            while not self._stop_event.is_set():
                try:
                    # Wait for an event with a timeout to allow checking _stop_event periodically
                    _priority, event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                    if event:
                        self.logger.debug(f"Processing event: {event.event_id} ({event.event_type.name}) with priority {_priority}")
                        handlers = self._subscribers.get(event.event_type, [])
                        if not handlers:
                            self.logger.debug(f"No handlers for event type {event.event_type.name}")
                        for handler in handlers:
                            asyncio.create_task(self._dispatch_event_to_handler(handler, event))
                        self._event_queue.task_done()
                        self._events_processed += 1
                except asyncio.TimeoutError:
                    continue # Allows checking _stop_event
                except Exception as e:
                    self.logger.exception(f"Error in event consumer loop: {e}. Sleeping for {self._consumer_error_sleep}s.")
                    await asyncio.sleep(self._consumer_error_sleep)
        except asyncio.CancelledError:
            self.logger.info("Event consumer task cancelled.")
        finally:
            self.logger.info("Event consumer stopped.")
            self._consumer_stopped_event.set()

    async def _log_metrics_periodically(self):
        self.logger.info("Metrics logger started.")
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._metrics_log_interval)
                except asyncio.TimeoutError: # This is the normal path for periodic logging
                    queue_size = self._event_queue.qsize()
                    self.logger.info(
                        f"PubSub Metrics: QueueSize={queue_size}, "
                        f"Published={self._events_published}, Processed={self._events_processed}, "
                        f"FailedHandlers={self._events_failed}"
                    )
        except asyncio.CancelledError:
            self.logger.info("Metrics logger task cancelled.")
        finally:
            self.logger.info("Metrics logger stopped.")
            self._metrics_stopped_event.set()


    async def start(self):
        self.logger.info("PubSubManager starting...")
        self._stop_event.clear()
        self._consumer_stopped_event.clear()
        self._metrics_stopped_event.clear()

        self._consumer_task = asyncio.create_task(self._event_consumer())
        if self._metrics_log_interval > 0:
            self._metrics_task = asyncio.create_task(self._log_metrics_periodically())
        else:
            self._metrics_stopped_event.set() # Instantly signal done if not started
        self.logger.info("PubSubManager started.")

    async def stop(self):
        self.logger.info("PubSubManager stopping...")
        self._stop_event.set()

        if self._consumer_task:
            await self._consumer_stopped_event.wait()
        if self._metrics_task:
             await self._metrics_stopped_event.wait()

        # Cancel tasks if they haven't stopped (e.g. stuck in queue.get without timeout)
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            await asyncio.gather(self._consumer_task, return_exceptions=True)
        if self._metrics_task and not self._metrics_task.done():
            self._metrics_task.cancel()
            await asyncio.gather(self._metrics_task, return_exceptions=True)

        self.logger.info("PubSubManager stopped.")

# Mock ConfigManager for example
class MockConfigManager:
    def get_int(self, key, default): return default
    def get_float(self, key, default): return default
    def get(self, key, default): return default # Added for completeness
# --- End of Mocks ---

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define custom EventTypes based on the mocked EventType
class MyEventTypes(EventType):
    MY_EVENT = 1       # Higher priority
    ANOTHER_EVENT = 2  # Lower priority

# Define custom Event classes based on the mocked Event
class MyEvent(Event):
    event_type: EventType = MyEventTypes.MY_EVENT
    def __init__(self, data: str, event_id: str = None):
        super().__init__(event_id)
        self.data = data

class AnotherEvent(Event):
    event_type: EventType = MyEventTypes.ANOTHER_EVENT
    def __init__(self, value: int, event_id: str = None):
        super().__init__(event_id)
        self.value = value

# Define event handlers
async def my_event_handler(event: MyEvent):
    logger.info(f"Handling MY_EVENT ({event.event_id}): {event.data}")
    await asyncio.sleep(0.1) # Simulate I/O-bound work

async def another_event_handler(event: AnotherEvent):
    logger.info(f"Handling ANOTHER_EVENT ({event.event_id}): {event.value}")
    await asyncio.sleep(0.1)

async def failing_handler(event: MyEvent):
    logger.info(f"Failing handler processing event {event.event_id}")
    raise ValueError("This handler intentionally fails")

async def timeout_handler(event: MyEvent):
    logger.info(f"Timeout handler processing event {event.event_id}, will simulate timeout.")
    await asyncio.sleep(10) # Assuming handler_timeout_seconds is less than 10
    logger.info(f"Timeout handler {event.event_id} finished (should not happen if timeout works).")


async def main():
    # Use the mock ConfigManager
    config_manager = MockConfigManager()

    # Adjust config for testing failure/timeout locally if needed
    # config_manager.get_float = lambda key, default: 1.0 if key == "pubsub.handler_timeout_seconds" else default
    # config_manager.get_int = lambda key, default: 1 if key == "pubsub.handler_max_failures" else default


    pubsub = PubSubManager(logger, config_manager)

    # Subscribe handlers
    pubsub.subscribe(MyEventTypes.MY_EVENT, my_event_handler)
    pubsub.subscribe(MyEventTypes.ANOTHER_EVENT, another_event_handler)
    pubsub.subscribe(MyEventTypes.MY_EVENT, failing_handler) # Subscribe failing handler
    pubsub.subscribe(MyEventTypes.MY_EVENT, timeout_handler) # Subscribe timeout handler


    await pubsub.start()

    # Publish some events
    await pubsub.publish(MyEvent(data="Hello from MyEvent!"))
    await pubsub.publish(AnotherEvent(value=123))
    await pubsub.publish(MyEvent(data="Second MyEvent for normal and failing handlers"))
    await pubsub.publish(MyEvent(data="Third MyEvent for timeout handler"))


    # Allow some time for events to be processed and failures to be handled
    logger.info("Waiting for events to be processed and potential auto-unsubscriptions...")
    await asyncio.sleep(3) # Increased sleep to observe auto-unsubscription

    # Publish again to see if failing_handler was unsubscribed
    logger.info("Publishing another MyEvent after potential unsubscription...")
    await pubsub.publish(MyEvent(data="Event after failing_handler should be gone"))

    await asyncio.sleep(1)


    await pubsub.stop()
    logger.info("Main application finished.")

if __name__ == "__main__":
    # To make the example runnable, we'll use the mocked PubSubManager.
    # If you have the actual gal_friday modules, you'd import them directly.
    asyncio.run(main())

```

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `PubSubManager` module.
