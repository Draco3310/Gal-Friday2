# StrategyPositionTracker Module (`gal_friday/strategy_position_tracker.py`) Documentation

## Module Overview

The `gal_friday.strategy_position_tracker.py` module provides the `StrategyPositionTracker` class, a specialized component designed for tracking open positions, market exposure, and key performance metrics on a **per-strategy basis**. This is distinct from the global portfolio view managed by `PortfolioManager`. The primary purpose of this tracker is to enable more granular risk management, performance analysis, and decision-making tailored to individual trading strategies operating within the Gal-Friday system. It allows each strategy (or the system components managing them) to understand its own footprint and performance independently.

## Key Features

-   **Per-Strategy Position Tracking:**
    -   Maintains a record of currently open positions (quantity, average entry price, current market value, unrealized P&L) specifically attributed to each unique `strategy_id` and `trading_pair`.
-   **Per-Strategy Performance Metrics Calculation:**
    -   Calculates and maintains a set of performance metrics for each tracked strategy, including:
        -   **Strategy-Specific Equity:** Tracks an equity figure based on the cumulative realized P&L attributed to that strategy. This can be thought of as the strategy's "self-contained" performance.
        -   **Strategy-Specific Peak Equity:** Records the highest equity achieved by the strategy.
        -   **Strategy-Specific Drawdown:** Calculates the current drawdown percentage for the strategy based on its own peak equity.
        -   **Total Exposure Value:** Sum of the absolute market values of all open positions held by the strategy.
        -   **Exposure Percentage (of Portfolio):** The strategy's total exposure value expressed as a percentage of the overall portfolio equity (requires the tracker to be updated with global portfolio equity).
        -   **Number of Open Positions:** Count of currently active positions for the strategy.
-   **P&L Attribution:** Allows for realized Profit & Loss (P&L) amounts to be explicitly recorded against a specific strategy, which then updates its equity and drawdown metrics.
-   **Global Portfolio Context:** Can be updated with the total portfolio equity from `PortfolioManager` to accurately calculate the strategy's exposure as a percentage of the whole portfolio.

## Class `StrategyPositionTracker`

### Initialization (`__init__`)

-   **Parameters:**
    -   `logger_service (LoggerService)`: An instance of `LoggerService` for logging activities and potential errors within the tracker.
-   **Actions:**
    -   Initializes internal data structures to store the state:
        -   `_strategy_positions (defaultdict(dict))`: A nested dictionary. Outer key is `strategy_id`, inner key is `trading_pair`, and the value is a dictionary containing position data (e.g., `{'quantity': Decimal, 'average_entry_price': Decimal, 'current_value': Decimal, 'unrealized_pnl': Decimal}`).
        -   `_strategy_performance (defaultdict(dict))`: A dictionary where the key is `strategy_id` and the value is another dictionary holding performance metrics for that strategy (e.g., `{'equity': Decimal, 'peak_equity': Decimal, 'drawdown_pct': Decimal, 'total_exposure_value': Decimal, 'exposure_pct_of_portfolio': Decimal, 'open_positions_count': int}`).
        -   `_total_portfolio_equity (Decimal)`: Stores the latest known total portfolio equity, initialized to a default (e.g., zero or a configured starting capital if this tracker were to manage its own capital allocation, though typically it reflects global equity for exposure calculation).
    -   Stores the `logger_service`.

### Position Management Methods

-   **`add_position(strategy_id: str, trading_pair: str, position_data: dict) -> None`**:
    -   Adds a new position or updates an existing one for the given `strategy_id` and `trading_pair`.
    -   `position_data` is a dictionary expected to contain keys like `quantity` (Decimal, positive for long, negative for short), `average_entry_price` (Decimal), `current_market_price` (Decimal), and potentially `unrealized_pnl` (Decimal).
    -   Calculates `current_value = quantity * current_market_price`.
    -   Updates `_strategy_positions[strategy_id][trading_pair]`.
    -   Calls `_update_strategy_metrics(strategy_id)` to recalculate exposure and other relevant metrics.
    -   Logs the position addition/update.

-   **`remove_position(strategy_id: str, trading_pair: str) -> None`**:
    -   Removes the position for the given `strategy_id` and `trading_pair` from `_strategy_positions[strategy_id]`.
    -   Calls `_update_strategy_metrics(strategy_id)` to reflect the change in exposure.
    -   Logs the position removal. This is typically called when a position is fully closed.

-   **`clear_strategy_positions(strategy_id: str) -> None`**:
    -   Removes all tracked positions for a specific `strategy_id`.
    -   Resets the exposure-related metrics for that strategy by calling `_update_strategy_metrics(strategy_id)`.
    -   Useful if a strategy is being deactivated or reset.

-   **`get_strategy_positions(strategy_id: str) -> dict`**:
    -   Returns a copy of the dictionary of positions currently held by the specified `strategy_id`.
    -   If the `strategy_id` is not tracked, returns an empty dictionary.

### Performance and Metrics Methods

-   **`record_strategy_pnl(strategy_id: str, pnl_amount: Decimal) -> None`**:
    -   Updates the performance metrics for a given `strategy_id` based on a realized `pnl_amount`.
    -   If the strategy is not yet tracked, calls `_initialize_strategy_metrics(strategy_id)`.
    -   Updates `_strategy_performance[strategy_id]['equity'] += pnl_amount`.
    -   Updates `_strategy_performance[strategy_id]['peak_equity'] = max(peak_equity, current_equity)`.
    -   Recalculates `_strategy_performance[strategy_id]['drawdown_pct']`.
    -   Logs the P&L recording and updated equity/drawdown.

-   **`get_strategy_metrics(strategy_id: str) -> Optional[dict]`**:
    -   Returns a copy of the current performance metrics dictionary for the specified `strategy_id` from `_strategy_performance`.
    -   Returns `None` if the strategy is not tracked.

-   **`get_strategy_exposure_details(strategy_id: str) -> Optional[dict]`**:
    -   Combines position information and performance metrics for a given `strategy_id`.
    -   Returns a dictionary containing both `positions: self.get_strategy_positions(strategy_id)` and `performance: self.get_strategy_metrics(strategy_id)`.
    -   Useful for getting a complete snapshot of a single strategy's state.

-   **`_initialize_strategy_metrics(strategy_id: str) -> None`**: (Internal Method)
    -   Called when a strategy is first encountered (e.g., first position added or P&L recorded).
    -   Sets up the initial structure for `_strategy_performance[strategy_id]` with default values (e.g., equity usually starts at 0 or a nominal amount if tracking "paper" capital per strategy, peak_equity, drawdown_pct=0, exposure_value=0, etc.).

-   **`_update_strategy_metrics(strategy_id: str) -> None`**: (Internal Method)
    -   Recalculates metrics that depend on the current state of open positions for the `strategy_id`.
    -   **Total Exposure Value:** Iterates through `_strategy_positions[strategy_id]`, summing the absolute `current_value` of each position. Updates `_strategy_performance[strategy_id]['total_exposure_value']`.
    -   **Open Positions Count:** Updates `_strategy_performance[strategy_id]['open_positions_count']`.
    -   **Exposure Percentage of Portfolio:** If `_total_portfolio_equity` is greater than zero, calculates `(total_exposure_value / _total_portfolio_equity) * 100`. Updates `_strategy_performance[strategy_id]['exposure_pct_of_portfolio']`.

### Global Context Method

-   **`update_portfolio_equity(total_equity: Decimal) -> None`**:
    -   Updates the `_total_portfolio_equity` attribute with the latest overall portfolio equity value (typically obtained from `PortfolioManager.get_current_state()`).
    -   After updating, it iterates through all tracked strategies and calls `_update_strategy_metrics(strategy_id)` for each one to refresh their `exposure_pct_of_portfolio` metric, as this depends on the global portfolio equity.

### Query Method

-   **`get_all_strategy_ids() -> List[str]`**:
    -   Returns a list of all unique `strategy_id`s for which positions or performance metrics are currently being tracked.

## Use Cases & Integration

The `StrategyPositionTracker` is designed to be used by various components within the Gal-Friday system:

-   **Individual Strategy Implementations (if stateful):**
    -   Strategies themselves could use an instance of this tracker (or a similar mechanism) if they need to maintain their own state about open positions they initiated, especially if multiple instances of the same strategy logic run with different parameters (each instance having a unique `strategy_id`).
-   **`StrategyArbitrator`:**
    -   Before proposing a new trade signal from a particular strategy, the `StrategyArbitrator` could query this tracker to:
        -   Check the current exposure of that strategy (`get_strategy_exposure_details`).
        -   Assess its recent performance (e.g., current drawdown via `get_strategy_metrics`).
        -   This information could be used to modulate signal generation (e.g., reduce signal size if strategy drawdown is high or exposure is at a limit).
-   **`RiskManager`:**
    -   Could use the per-strategy exposure data from this tracker to enforce more granular risk limits, such as:
        -   "Do not allow strategy 'MomentumAlpha_BTC' to exceed X% of total portfolio exposure."
        -   "Halt new signals from any strategy whose individual drawdown exceeds Y%."
-   **`PortfolioManager` (Orchestration):**
    -   While `PortfolioManager` tracks the global portfolio, it might instantiate and update a `StrategyPositionTracker` by associating execution reports with the `strategy_id` that originated the signal. When an `ExecutionReportEvent` comes in with a `strategy_id` in its payload, the `PortfolioManager` could call `add_position` or `record_strategy_pnl` on the tracker.
    -   It would also periodically call `update_portfolio_equity()` on the tracker.
-   **Monitoring & Dashboard Services (`gal_friday/monitoring`):**
    -   The data provided by `get_strategy_metrics()` and `get_strategy_exposure_details()` for all tracked strategies can be invaluable for displaying per-strategy performance and risk contribution on a monitoring dashboard. This allows operators to quickly identify underperforming or overly risky strategies.

## Dependencies

-   **Standard Libraries:**
    -   `collections.defaultdict`: Used for initializing `_strategy_positions` and `_strategy_performance`.
    -   `datetime`: For timestamping if positions or metrics include time-sensitive information (though not explicitly shown as primary keys in the described data structures, timestamps of updates would be relevant).
    -   `decimal.Decimal`: For all financial calculations (prices, quantities, P&L, equity) to ensure precision.
-   **Core Application Modules:**
    -   `gal_friday.logger_service.LoggerService`: For logging.

## Adherence to Standards

The `StrategyPositionTracker` component promotes:
-   **Modular Strategy Development:** By allowing strategies to be tracked and assessed independently, it facilitates a more modular approach where different strategies can be added, removed, or tuned with a clearer understanding of their individual impact.
-   **Fine-Grained Risk Control:** Enables the implementation of risk limits and controls at the strategy level, in addition to global portfolio limits.
-   **Enhanced Performance Analysis:** Provides the data needed to dissect overall portfolio performance and attribute P&L and risk to specific strategies.

This focused tracking capability is a key element in building a sophisticated multi-strategy automated trading system where individual component performance is as important as the overall result.
