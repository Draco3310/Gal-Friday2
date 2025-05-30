# Core Folder (`gal_friday/core`) Documentation

## Folder Overview

The `gal_friday/core` folder is the foundational heart of the Gal-Friday trading system. It houses the essential building blocks, fundamental interfaces (protocols), central services, and core data types that underpin the entire application's architecture. These components are designed to be generic and reusable, providing the critical infrastructure for event-driven communication, system stability, asset and exchange information management, and overall operational integrity. The modules within this folder enable a decoupled, robust, and extensible system design.

## Key Modules and Their Roles

The `core` folder contains several key Python modules, each with a distinct and critical responsibility:

### `events.py`

-   **Purpose:** Defines the comprehensive hierarchy of system events used for inter-service communication. This is the linchpin of Gal-Friday's event-driven architecture.
-   **Key Aspects:**
    -   **Event Definitions:** Contains dataclass definitions for all significant occurrences within the system, such as `MarketDataOHLCVEvent`, `MarketDataL2Event`, `TradeEvent` (raw trades), `FeatureEvent`, `PredictionEvent`, `TradeSignalProposedEvent`, `TradeSignalApprovedEvent`, `TradeSignalRejectedEvent`, `ExecutionReportEvent`, `SystemStateEvent`, `LogEvent`, `PotentialHaltTriggerEvent`, and many others.
    -   **Immutability:** Events are typically defined as frozen dataclasses (`@dataclass(frozen=True)`), ensuring they are immutable once created, which helps in maintaining predictable state flow.
    -   **Validation:** Event dataclasses often include `__post_init__` validation logic to ensure data integrity and correctness at the point of creation.
    -   **`EventType` Enum:** Defines an enumeration (`EventType`) that categorizes all events, used by the `PubSubManager` for topic-based subscription and message routing. Event priorities for the `PubSubManager`'s priority queue are also derived from these enum values.
-   **Importance:** Crucial for enabling decoupled communication between various services. Services react to events they are interested in, rather than having direct dependencies on each other.

### `pubsub.py`

-   **Purpose:** Implements the `PubSubManager`, an asynchronous publish-subscribe messaging system.
-   **Key Aspects:**
    -   **Decoupled Communication:** Allows different services (publishers) to broadcast events without knowing who the subscribers are, and for services (subscribers) to receive events they are interested in without knowing who the publishers are.
    -   **Asynchronous:** Built on `asyncio` for non-blocking message handling.
    -   **Event Filtering:** Uses `EventType` enums to allow subscribers to listen to specific types of events.
    -   **Priority Queue:** Processes events based on the priority defined in their `EventType`.
-   **Importance:** Forms the central message bus of the application, facilitating the event-driven architecture.
-   **(Reference:** See `docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/modules/pubsub.md` for detailed documentation.)

### `asset_registry.py`

-   **Purpose:** Provides data structures and a registry for managing detailed information about tradable assets and the exchanges they trade on.
-   **Key Aspects:**
    -   **`AssetSpecification`:** A dataclass (or similar structure) defining properties of a financial instrument (e.g., "BTC", "AAPL", "ESZ23"), including its type (crypto, stock, future), underlying, quote currency, tick sizes, lot sizes, and contract specifications.
    -   **`ExchangeSpecification`:** A dataclass defining properties of an exchange (e.g., "Kraken", "Binance", "CME"), including its trading hours, fee structures, API endpoints, rate limits, and supported asset types.
    -   **`AssetRegistry` Class:** A central registry that loads and provides access to `AssetSpecification` and `ExchangeSpecification` objects. It can load this data from configuration files or a database.
-   **Importance:** Enables the system to be multi-asset and multi-exchange capable by providing a standardized way to access instrument and exchange details, crucial for order sizing, price rounding, fee calculation, and session management.

### `halt_coordinator.py`

-   **Purpose:** Implements the `HaltCoordinator` and `HaltCondition` classes, which manage the various conditions that can lead to a system-wide trading HALT.
-   **Key Aspects:**
    -   **`HaltCondition`:** Represents a specific reason why the system might need to halt (e.g., "MaxDrawdownExceeded", "APIFailureRateHigh", "DataStale", "ManualHalt"). Each condition can have a status (active/inactive) and associated metadata.
    -   **`HaltCoordinator`:** A central service that aggregates multiple `HaltCondition`s. The system is considered to be in a halt state if any registered critical condition is active.
    -   **State Management:** Allows services like `MonitoringService` or `RiskManager` to update the status of individual halt conditions.
-   **Importance:** Centralizes halt logic, making it easier to manage system safety and stability by providing a clear overview of why a halt might be in effect.

### `halt_recovery.py`

-   **Purpose:** Implements the `HaltRecoveryManager` and `RecoveryCheckItem` classes, providing a structured process for recovering the system after a HALT.
-   **Key Aspects:**
    -   **`RecoveryCheckItem`:** Represents an individual step or verification that needs to be performed manually or automatically before the system can be safely resumed (e.g., "Verify Exchange Connectivity", "Check Open Positions", "Confirm Data Feeds").
    -   **`HaltRecoveryManager`:** Manages a configurable checklist of `RecoveryCheckItem`s. It tracks the completion status of each item.
    -   **Interactive Recovery:** Often used in conjunction with the `CLIService` to allow operators to view the checklist and mark items as complete.
-   **Importance:** Ensures operational safety by enforcing a methodical recovery procedure after a system disruption, reducing the risk of resuming trading under unsafe conditions.

### `event_store.py`

-   **Purpose:** Implements the `EventStore`, a service responsible for persisting all (or selected) system events to a durable storage, typically a database.
-   **Key Aspects:**
    -   **Event Persistence:** Subscribes to a wide range of events (or all events via a wildcard) from `PubSubManager` and writes them to a database table (e.g., using an `EventLog` SQLAlchemy model mapped to a "event_logs" table).
    -   **Asynchronous Writes:** Uses asynchronous database operations to avoid blocking the main event loop.
    -   **In-Memory Cache:** May maintain a short-term in-memory cache of recent events for quick access or for batching writes.
    -   **Event Replay (Potential):** The stored events can be used for replaying system behavior, which is invaluable for debugging, auditing, and detailed post-incident analysis. It could also potentially be used for state reconstruction in some scenarios.
-   **Importance:** Provides a comprehensive audit trail of system activity. Essential for debugging complex issues, understanding system behavior leading up to specific outcomes, and for compliance or analytical purposes.

### `types.py`

-   **Purpose:** Defines core `typing.Protocol`s (interfaces) for major services and common type aliases used throughout the application.
-   **Key Aspects:**
    -   **Service Protocols:** Specifies the expected methods and attributes for key services like `ConfigManagerProtocol`, `LoggerServiceProtocol`, `MarketPriceServiceProtocol`, `ExecutionHandlerProtocol`, `PortfolioManagerProtocol`, etc. This allows for dependency injection and promotes loose coupling, making it easier to swap out implementations (e.g., live vs. simulated services).
    -   **Common Type Aliases:** Defines aliases for frequently used complex types or simple built-in types for semantic clarity (e.g., `PairSymbol = str`, `Price = Decimal`, `Timestamp = datetime`).
-   **Importance:** Enhances type safety and code clarity. By defining clear contracts for services, it facilitates modular design and makes the system easier to test and maintain.

### `placeholder_classes.py`

-   **Purpose:** Contains minimal stub or placeholder implementations of various services and components.
-   **Key Aspects:**
    -   **Development Aid:** Used during development to allow modules to be worked on independently without requiring all dependencies to be fully implemented.
    -   **Type Checking:** Often used in conjunction with `if typing.TYPE_CHECKING:` blocks to provide type hints and prevent circular import errors, while ensuring these stubs are not part of the runtime code if real implementations are available.
-   **Importance:** Streamlines the development process in a complex system by breaking hard dependencies during early stages or for specific testing scenarios.

### `__init__.py`

-   **Purpose:** An empty file that marks the `core` directory as a Python package.
-   **Key Aspects:** Allows modules within the `core` directory to be imported using package notation (e.g., `from gal_friday.core.events import SystemStateEvent`). May also expose selected classes or functions at the package level for convenience.

## Interactions and Importance

The modules within the `gal_friday/core` folder collectively provide the foundational architecture for the Gal-Friday trading system.

-   The **event-driven nature** is primarily facilitated by `events.py` (defining the "what") and `pubsub.py` (defining the "how" of communication). This allows for a highly decoupled system where services can evolve independently.
-   `asset_registry.py` is crucial for **market flexibility**, enabling the system to adapt to different trading instruments and exchanges by providing a centralized source of truth for their specifications.
-   System **stability and operational safety** are significantly enhanced by `halt_coordinator.py` (for managing automated halt conditions) and `halt_recovery.py` (for ensuring safe resumption after halts).
-   `types.py` underpins **robust design** by defining clear contracts (protocols) for services, promoting type safety, and enabling easier testing and dependency injection.
-   `event_store.py` contributes to **system resilience, auditability, and analyzability** by creating a persistent record of all significant system activities.
-   `placeholder_classes.py` is a pragmatic tool for **streamlining development** in a large, multi-component system.

In essence, the `core` modules are not directly involved in generating trading signals or executing trades themselves, but they provide the indispensable infrastructure and contracts that enable all other application services to function cohesively and reliably.

## Adherence to Standards

The design and implementation of components within the `core` folder aim to align with robust software engineering practices, emphasizing modularity, loose coupling, clear interfaces, and testability. While not formally certified, these principles are inspired by best practices found in software development lifecycle standards to ensure a maintainable and scalable system.
