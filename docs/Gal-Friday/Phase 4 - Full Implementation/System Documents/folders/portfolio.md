# Portfolio Folder (`gal_friday/portfolio`) Documentation

## Folder Overview

The `gal_friday/portfolio` folder contains a suite of core services and logic dedicated to the detailed tracking, management, and valuation of the trading portfolio within the Gal-Friday system. These components are fundamental for maintaining an accurate real-time understanding of the system's financial state, including cash balances, open positions, overall equity, and risk metrics like drawdowns. While the main `PortfolioManager` (typically located at the parent `gal_friday/` level) orchestrates these services, the modules within this folder provide the specialized functionalities required for each aspect of portfolio management.

## Key Modules and Their Roles

The `portfolio` folder is composed of several key Python modules, each focusing on a specific area of portfolio management:

### `funds_manager.py` (`FundsManager` class)

-   **Purpose:** Manages the cash balances of various currencies held within the trading portfolio. It is responsible for accurately tracking available and total funds.
-   **Functionality:**
    -   **Balance Tracking:** Maintains a record of total and available balances for each currency (e.g., USD, EUR, BTC, ETH).
    -   **Transaction Handling:**
        -   Processes deposits and withdrawals, updating balances accordingly.
        -   Updates fund balances based on the costs or proceeds from trade executions (fills).
        -   Deducts commission payments from the appropriate currency balance.
    -   **Thread Safety:** Implements mechanisms (e.g., `asyncio.Lock` or thread-safe data structures if accessed by synchronous code via a bridge) to ensure that balance modifications are atomic and thread-safe, preventing race conditions.
    -   **Reconciliation:** Provides methods to compare and reconcile its internal fund balances with balances reported directly by an exchange (obtained via `ExecutionHandler`). Can adjust internal state based on reconciliation results if configured.
-   **Importance:** Provides an accurate and real-time view of the system's liquidity and buying power. Essential for pre-trade checks by `RiskManager` and for overall financial accounting.

### `position_manager.py` (`PositionManager` class, `TradeInfo` dataclass)

-   **Purpose:** Tracks and manages individual trading positions for each asset or trading pair. It maintains the state of all open positions and a history of closed trades.
-   **Key Components & Functionality:**
    -   **`TradeInfo` (Dataclass):** A data structure likely used to represent individual trades that contribute to a position, holding details like entry/exit price, quantity, side, and timestamp. This might be used internally or for constructing a trade log.
    -   **`PositionManager` Class:**
        -   **Position Tracking:** For each trading pair, it tracks:
            -   Current quantity held (positive for long, negative for short).
            -   Average entry price of the open position.
            -   Unrealized Profit & Loss (P&L) based on current market prices (requires interaction with `MarketPriceService` often via `ValuationService`).
            -   Accumulated realized P&L from closed portions of the position or fully closed trades.
        -   **Trade Processing:** Updates position state based on incoming trade execution (fill) data:
            -   Adjusts quantity and average entry price when new fills occur.
            -   Calculates and records realized P&L when a position is closed or reduced.
        -   **Persistence:** Interacts with a `PositionRepository` (from the DAL) to persist position data (e.g., open positions, historical trades) to the database, ensuring state can be recovered.
        -   **History:** Maintains or provides access to a log of all trades that have affected positions.
-   **Importance:** Provides a detailed and accurate record of all trading activities and their impact on asset holdings. This is crucial for P&L calculation, risk exposure assessment, and performance analysis.

### `reconciliation_service.py` (`ReconciliationService` class, `ReconciliationReport`, `PositionDiscrepancy`, etc. dataclasses)

-   **Purpose:** Dedicated to ensuring data integrity and consistency by comparing the Gal-Friday system's internal perception of its portfolio state (cash balances, open positions, potentially open orders) against the actual state reported by the connected exchange(s).
-   **Key Components & Functionality:**
    -   **`PositionDiscrepancy`, `BalanceDiscrepancy` (Dataclasses):** Structures to hold detailed information about detected differences between internal and exchange states.
    -   **`ReconciliationReport` (Dataclass):** A comprehensive report summarizing the findings of a reconciliation cycle, including all detected discrepancies.
    -   **`ReconciliationService` Class:**
        -   **Data Fetching:** Periodically (or on demand) fetches account balances and open position data directly from the exchange using methods provided by an `ExecutionHandler` that conforms to the `ReconcilableExecutionHandler` protocol.
        -   **Comparison Logic:** Compares the fetched exchange data against the internal state maintained by `FundsManager` and `PositionManager`.
        -   **Discrepancy Identification:** Identifies and quantifies any mismatches, such as differences in cash balances, position quantities, or even missing/extra positions or orders.
        -   **Reporting:** Generates a detailed `ReconciliationReport`.
        -   **Auto-Correction (Optional):** Based on configuration, can attempt to automatically correct minor discrepancies by adjusting the internal state to match the exchange. For significant or unresolvable discrepancies, it would typically raise alerts.
        -   **Interaction with DAL:** May log reconciliation reports or discrepancies to the database for auditing.
        -   **Alerting:** Can trigger alerts (via `AlertingSystem`) when significant discrepancies are found that require manual intervention.
-   **Importance:** Acts as a critical safeguard against data drift or synchronization issues between Gal-Friday's internal records and the exchange. This is vital for maintaining accurate accounting, risk management, and preventing trading decisions based on incorrect state information.

### `valuation_service.py` (`ValuationService` class)

-   **Purpose:** Responsible for calculating the current market value of the entire trading portfolio and for tracking key performance and risk metrics, most notably portfolio drawdowns.
-   **Functionality:**
    -   **Portfolio Valuation:**
        -   Takes current fund balances (from `FundsManager`) and open positions (from `PositionManager`) as input.
        -   Uses a `MarketPriceService` to fetch real-time market prices for all assets held in open positions.
        -   Calculates the current market value of each position.
        -   Aggregates the value of all cash balances (converting them to a single `valuation_currency` if necessary, using rates from `MarketPriceService`) and the market value of all positions to determine the total portfolio equity (Net Asset Value - NAV).
    -   **Performance Metrics:**
        -   Tracks peak portfolio equity over various periods.
        -   Calculates current drawdown percentages:
            -   **Total Drawdown:** From the all-time peak equity.
            -   **Daily Drawdown:** From the peak equity recorded since the start of the current trading day (UTC).
            -   **Weekly Drawdown:** From the peak equity recorded since the start of the current trading week.
        -   **Configurable Resets:** Supports configurable reset times/days for daily and weekly drawdown calculations (e.g., daily reset at 00:00 UTC, weekly reset on Monday at 00:00 UTC).
    -   **Portfolio Exposure Calculation:** Calculates the total market exposure of the portfolio, often expressed as a percentage of equity or as the sum of absolute values of positions.
    -   **P&L Tracking:** While `PositionManager` handles realized P&L per trade, `ValuationService` focuses on the overall unrealized P&L reflected in the current equity.
-   **Importance:** Provides the key metrics (equity, drawdown) that are essential for high-level monitoring of the trading strategy's performance and risk profile. Drawdown figures are often critical inputs for risk management and automated halt conditions.

### `__init__.py`

-   **Purpose:** Marks the `portfolio` directory as a Python package.
-   **Key Aspects:**
    -   Allows modules within this directory to be imported using package notation (e.g., `from gal_friday.portfolio.funds_manager import FundsManager`).
    -   Typically exports the key service classes (`FundsManager`, `PositionManager`, `ReconciliationService`, `ValuationService`) to make them directly accessible at the `gal_friday.portfolio` package level, simplifying imports for the main `PortfolioManager` or other services that might need direct access.

### `py.typed`

-   **Purpose:** This is a marker file as defined by PEP 561. Its presence indicates that the `portfolio` package (and its submodules) supports type checking and that type information should be used by type checkers like MyPy.
-   **Importance:** Contributes to code quality and maintainability by enabling static type analysis, which can catch potential type-related errors before runtime.

## Interactions and Importance

The services within the `gal_friday.portfolio` folder are tightly interconnected and form the backbone of the system's awareness of its financial state and performance:

-   **Core Data Providers:** `FundsManager` and `PositionManager` are the primary sources of truth for what the system owns (cash) and what market exposure it has (positions). They are updated based on trading activity (fills).
-   **Valuation and Risk Assessment:** `ValuationService` consumes data from `FundsManager`, `PositionManager`, and `MarketPriceService` to provide a higher-level financial picture, calculating equity and crucial risk metrics like drawdown. These metrics are vital inputs for `RiskManager` and `MonitoringService`.
-   **Integrity Check:** `ReconciliationService` acts as an independent verifier, ensuring that the internal state maintained by `FundsManager` and `PositionManager` aligns with the external reality reported by the exchange. This is crucial for preventing the system from operating on flawed internal data.
-   **Orchestration by Main `PortfolioManager`:** The main `PortfolioManager` (typically located at `gal_friday/portfolio_manager.py`) is the primary consumer and orchestrator of these lower-level portfolio services. It:
    -   Initializes and holds instances of `FundsManager`, `PositionManager`, and `ValuationService`.
    -   Processes `ExecutionReportEvent`s and delegates updates to `FundsManager` and `PositionManager`.
    -   Triggers `ValuationService` to recalculate portfolio metrics after state changes.
    -   May initiate or use `ReconciliationService` for periodic checks.
    -   Provides a unified API (e.g., `get_current_state()`) that aggregates information from these underlying services for other parts of the system.

This modular approach allows for a clear separation of concerns within the complex domain of portfolio management. Each service focuses on a specific aspect, making the system easier to develop, test, and maintain. The accuracy and timeliness of the data managed and processed by these portfolio components are fundamental to all decision-making, risk management, and performance evaluation processes in Gal-Friday.

## Adherence to Standards

The modular design of the `portfolio` folder, with distinct services for funds, positions, valuation, and reconciliation, reflects a strong adherence to the principle of **Separation of Concerns**. This architectural pattern leads to:
-   **Improved Maintainability:** Changes to how funds are managed, for example, can be localized within `FundsManager` without necessarily impacting `PositionManager` or `ValuationService` directly, as long as their interfaces remain stable.
-   **Enhanced Testability:** Each service can be unit-tested more effectively in isolation by mocking its specific dependencies.
-   **Clarity and Understandability:** The responsibilities of each component are clearly defined, making the overall portfolio management logic easier to understand and reason about.
This structure contributes to building a robust and reliable portfolio management subsystem.
