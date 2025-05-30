# StrategyArbitrator Module Documentation

## Module Overview

The `gal_friday.strategy_arbitrator.py` module serves as the decision-making component that translates predictive signals into actionable trading proposals. It consumes `PredictionEvent`s from the `PubSubManager`, applies a set of configured trading strategy logic (which includes interpreting prediction values, applying thresholds, and validating secondary confirmation rules), calculates appropriate Stop-Loss (SL) and Take-Profit (TP) levels, determines the proposed entry price, and finally generates `TradeSignalProposedEvent`s. These proposed signals are then published for further processing by other modules like the `OrderExecutionManager`.

## Key Features

-   **Consumes Prediction Events:** Listens for `PredictionEvent`s containing model outputs and associated features.
-   **Configurable Trading Strategies:** Allows definition of detailed trading strategies through configuration, specifying how predictions are interpreted and acted upon.
-   **Flexible Prediction Interpretation:** Supports various ways to interpret the primary prediction value from a `PredictionEvent` based on the `prediction_interpretation` setting (e.g., "prob_up", "prob_down", "price_change_pct").
-   **Threshold-Based Signaling:** Uses configurable `buy_threshold` and `sell_threshold` values to determine the primary signal direction (BUY or SELL) based on the interpreted prediction.
-   **Secondary Confirmation Rules:** Supports an additional layer of validation through `confirmation_rules` which evaluate `associated_features` within the `PredictionEvent` against specified conditions.
-   **Stop-Loss (SL) and Take-Profit (TP) Calculation:**
    -   Calculates SL and TP prices based on configured percentages (`sl_pct`, `tp_pct`).
    -   Alternatively, uses a `default_reward_risk_ratio` to determine TP if only SL (or vice-versa) percentage is provided.
-   **Entry Price Determination:**
    -   For `LIMIT` orders, calculates a proposed entry price by applying a `limit_offset_pct` to the current best bid (for BUY) or best ask (for SELL) obtained from the `MarketPriceService`.
    -   For `MARKET` orders, the entry price is determined by the market at the time of execution.
-   **Publishes Trade Signal Proposals:** Generates and publishes `TradeSignalProposedEvent`s containing all necessary information for a potential trade.
-   **Configuration and Event Validation:** Performs validation of strategy configurations at startup and validates incoming `PredictionEvent`s before processing.

## Custom Exceptions

-   **`StrategyConfigurationError(Exception)`**: Raised if the strategy configuration provided to the arbitrator is invalid or incomplete (e.g., missing essential parameters, inconsistent threshold settings).

## Class `StrategyArbitrator`

### Constants

-   **`_CONDITION_OPERATORS: Dict[str, Callable]`**: A dictionary mapping string representations of comparison conditions (e.g., "gt", "lt", "eq", "gte", "lte") to their corresponding functions from Python's `operator` module (e.g., `operator.gt`, `operator.lt`). This is used for evaluating `confirmation_rules`.

### Initialization (`__init__`)

-   **Parameters:**
    -   `config (dict)`: A dictionary containing the configuration specific to the Strategy Arbitrator, typically from `app_config["strategy_arbitrator"]`. This includes a list of strategy configurations.
    -   `pubsub_manager (PubSubManager)`: An instance of `PubSubManager` for event subscription and publication.
    -   `logger_service (LoggerService)`: An instance of `LoggerService` for structured logging.
    -   `market_price_service (MarketPriceService)`: An instance of `MarketPriceService` used to fetch current market prices (e.g., best bid/ask) for entry price calculation.
-   **Actions:**
    -   Stores references to `pubsub_manager`, `logger_service`, and `market_price_service`.
    -   Loads its service-specific configuration. **Note:** The current implementation typically selects and uses only the *first* strategy defined in the `strategies` list within the configuration. This should be considered if multiple strategies are defined.
    -   Calls `_validate_configuration()` to ensure the selected strategy's parameters are valid.
    -   Initializes internal state, including flags for readiness.

### Configuration Validation

-   **`_validate_configuration() -> None`**:
    -   Orchestrates the validation of the loaded strategy configuration.
    -   Calls specific validation methods for different parts of the configuration.
    -   Raises `StrategyConfigurationError` if any part of the configuration is invalid.

-   **`_validate_core_parameters() -> None`**:
    -   Validates essential parameters like `entry_type` (must be "MARKET" or "LIMIT"), and the presence and validity of SL/TP percentages (`sl_pct`, `tp_pct`) or `default_reward_risk_ratio`.

-   **`_validate_prediction_interpretation_config() -> None`**:
    -   Validates the `prediction_interpretation` field (e.g., "prob_up", "prob_down", "price_change_pct").
    -   Ensures that corresponding threshold fields (e.g., `buy_threshold`, `sell_threshold` for "prob_up"; `price_change_buy_threshold_pct`, `price_change_sell_threshold_pct` for "price_change_pct") are present and correctly defined.

-   **`_validate_confirmation_rules_config() -> None`**:
    -   Validates the structure of the `confirmation_rules` list if it's provided.
    -   Ensures each rule in the list is a dictionary containing valid `feature`, `condition` (from `_CONDITION_OPERATORS.keys()`), and `threshold` keys.

### Event Handling

-   **`async start() -> None`**:
    -   Subscribes the `handle_prediction_event` method to `PredictionEvent`s via the `PubSubManager`.
    -   Logs that the Strategy Arbitrator service has started.

-   **`async stop() -> None`**:
    -   Unsubscribes from `PredictionEvent`s.
    -   Logs that the Strategy Arbitrator service is stopping.

-   **`async handle_prediction_event(event: PredictionEvent) -> None`**:
    -   The main handler for incoming `PredictionEvent`s.
    -   Calls `_validate_prediction_event()` to check the integrity of the received event.
    -   If the event is valid and pertains to the strategy's target (e.g., same trading pair, expected model ID if filtered), it calls `_evaluate_strategy()`.
    -   If `_evaluate_strategy()` returns a `TradeSignalProposedEvent`, it's published using `_publish_trade_signal_proposed()`.

### Strategy Evaluation

-   **`_validate_prediction_event(event: PredictionEvent) -> bool`**:
    -   Validates the received `PredictionEvent`.
    -   Checks for presence of essential fields like `prediction_value`, `trading_pair`, `timestamp`, and `associated_features` (if confirmation rules are used).
    -   Ensures data types are as expected.
    -   Returns `True` if the event is valid, `False` otherwise, logging any validation errors.

-   **`async _evaluate_strategy(prediction_event: PredictionEvent) -> Optional[TradeSignalProposedEvent]`**:
    -   The core method where the trading strategy logic is applied to a `PredictionEvent`.
    -   **1. Determine Primary Signal Side:** Calls `_calculate_signal_side(prediction_event)` to interpret the prediction and get a primary "BUY" or "SELL" signal, or `None` if no signal.
    -   **2. Apply Secondary Confirmation:** If a primary signal is generated, calls `_apply_secondary_confirmation(prediction_event, primary_side)`. If confirmation fails, the signal is discarded.
    -   **3. Fetch Current Price:** If both primary signal and secondary confirmation pass, fetches the current market price for the `trading_pair` from `_market_price_service.get_current_price()`. This price is crucial for SL/TP and entry price calculations.
    -   **4. Calculate SL/TP:** Calls `_calculate_sl_tp_prices(primary_side, current_price, prediction_event.trading_pair)`.
    -   **5. Determine Entry Price:** Calls `_determine_entry_price(primary_side, current_price, prediction_event.trading_pair)`.
    -   **6. Construct Event:** If all steps are successful and valid prices are determined, constructs a `TradeSignalProposedEvent` with all relevant details (trading pair, side, entry price, SL, TP, order type, original prediction data, etc.).
    -   Returns the `TradeSignalProposedEvent` or `None` if any step fails or no signal is warranted.

-   **`_calculate_signal_side(prediction_event: PredictionEvent) -> Optional[str]`**:
    -   Interprets `prediction_event.prediction_value` based on the configured `_prediction_interpretation` method.
    -   Calls one of the helper methods:
        -   **`_get_side_from_prob_up(prob_up: float) -> Optional[str]`**: If `prob_up >= _buy_threshold`, returns "BUY".
        -   **`_get_side_from_prob_down(prob_down: float, trading_pair: str) -> Optional[str]`**: If `prob_down >= _sell_threshold`, returns "SELL". (Note: `trading_pair` might be used for logging or context but isn't directly used in the example logic).
        -   **`_get_side_from_price_change_pct(price_change_pct: float) -> Optional[str]`**: If `price_change_pct >= _price_change_buy_threshold_pct`, returns "BUY". If `price_change_pct <= _price_change_sell_threshold_pct` (note: sell threshold for price change is often negative), returns "SELL".
    -   Returns "BUY", "SELL", or `None`.

-   **`_apply_secondary_confirmation(prediction_event: PredictionEvent, primary_side: str) -> bool`**:
    -   Iterates through all rules defined in `_confirmation_rules`.
    -   For each rule, calls `_validate_confirmation_rule()`.
    -   If *any* rule returns `False`, the overall confirmation fails, and this method returns `False`.
    -   If all rules pass (or if there are no confirmation rules), returns `True`.

-   **`_validate_confirmation_rule(rule: dict, features: dict, trading_pair: str, primary_side: str) -> bool`**:
    -   Validates a single confirmation rule against the `features` (from `prediction_event.associated_features`).
    -   Retrieves the feature value from `features` using `rule["feature"]`.
    -   Retrieves the operator function from `_CONDITION_OPERATORS` using `rule["condition"]`.
    -   Compares the feature value with `rule["threshold"]` using the operator.
    -   Returns `True` if the condition is met, `False` otherwise. Handles missing features or invalid rule configurations gracefully by typically returning `False` and logging an error.

### Price Calculation

-   **`_calculate_sl_tp_prices(side: str, current_price: Decimal, trading_pair: str) -> Tuple[Optional[Decimal], Optional[Decimal]]`**:
    -   Calculates Stop-Loss (SL) and Take-Profit (TP) prices.
    -   Calls `_calculate_stop_loss_price_and_risk()` to determine the SL price and the risk amount per unit.
    -   If SL is determined and `_tp_pct` is configured, TP is calculated directly from `current_price` and `_tp_pct`.
    -   If `_tp_pct` is not set but `_default_reward_risk_ratio` and SL (and thus risk) are available, TP is calculated based on the risk and ratio.
    -   Returns a tuple `(sl_price, tp_price)`.

    -   **`_calculate_stop_loss_price_and_risk(side: str, entry_price: Decimal, trading_pair: str) -> Tuple[Optional[Decimal], Optional[Decimal]]`**:
        -   Calculates the SL price based on `entry_price` (or `current_price` if market order) and `_sl_pct`.
        -   For a "BUY" signal, SL = `entry_price * (1 - _sl_pct / 100)`.
        -   For a "SELL" signal, SL = `entry_price * (1 + _sl_pct / 100)`.
        -   Calculates risk: `abs(entry_price - sl_price)`.
        -   Returns `(sl_price, risk_per_unit)`.

    -   **`_calculate_take_profit_price(side: str, entry_price: Decimal, risk_per_unit: Optional[Decimal], trading_pair: str) -> Optional[Decimal]`**:
        -   If `_tp_pct` is defined:
            -   For "BUY": TP = `entry_price * (1 + _tp_pct / 100)`.
            -   For "SELL": TP = `entry_price * (1 - _tp_pct / 100)`.
        -   Else if `_default_reward_risk_ratio` and `risk_per_unit` are defined:
            -   For "BUY": TP = `entry_price + (risk_per_unit * _default_reward_risk_ratio)`.
            -   For "SELL": TP = `entry_price - (risk_per_unit * _default_reward_risk_ratio)`.
        -   Returns the calculated TP price or `None`.

-   **`_determine_entry_price(side: str, current_market_price_details: dict, trading_pair: str) -> Optional[Decimal]`**:
    -   Calculates the proposed entry price.
    -   If `_entry_type` is "MARKET", returns `None` (signifying market execution).
    -   If `_entry_type` is "LIMIT":
        -   Uses `current_market_price_details` (which should contain `best_bid` and `best_ask` from `MarketPriceService`).
        -   For a "BUY" signal: Entry Price = `best_bid * (1 + _limit_offset_pct / 100)`. (Offset aims to place the limit order slightly above best bid to potentially improve fill probability, or could be negative to be more passive).
        -   For a "SELL" signal: Entry Price = `best_ask * (1 - _limit_offset_pct / 100)`. (Offset aims to place slightly below best ask).
        -   The exact logic for `_limit_offset_pct` (positive/negative) depends on whether it's an offset from the perspective of the trader (e.g., willingness to pay more/less) or market maker (e.g., providing liquidity). This documentation assumes it's an aggressive offset to cross the spread slightly or join the best price.
    -   Returns the calculated limit entry price or `None`.

### Publishing

-   **`async _publish_trade_signal_proposed(event: TradeSignalProposedEvent) -> None`**:
    -   Publishes the fully formed `TradeSignalProposedEvent` to the `PubSubManager`.
    -   Logs the details of the proposed signal.

### Configuration (Key options from `strategy_arbitrator.strategies[]` list in app config)

The Strategy Arbitrator is configured via a list of strategy objects. Typically, only the first strategy in this list is used by an instance of the arbitrator.

-   **`id (str)`**: A unique identifier for this specific strategy configuration (e.g., "rsi_momentum_eth_1h").
-   **`prediction_interpretation (str)`**: How to interpret the `prediction_value` from `PredictionEvent`.
    -   `"prob_up"`: Prediction is treated as a probability of price increase.
    -   `"prob_down"`: Prediction is treated as a probability of price decrease.
    -   `"price_change_pct"`: Prediction is treated as an expected percentage change in price.
-   **`buy_threshold (Optional[float/Decimal])`**: Required if `prediction_interpretation` is "prob_up". The prediction value must be >= this for a BUY signal.
-   **`sell_threshold (Optional[float/Decimal])`**: Required if `prediction_interpretation` is "prob_down". The prediction value must be >= this for a SELL signal.
-   **`price_change_buy_threshold_pct (Optional[float/Decimal])`**: Required if `prediction_interpretation` is "price_change_pct". The predicted percentage change must be >= this for a BUY signal.
-   **`price_change_sell_threshold_pct (Optional[float/Decimal])`**: Required if `prediction_interpretation` is "price_change_pct". The predicted percentage change must be <= this for a SELL signal (often a negative value).
-   **`entry_type (str)`**: Order type for the proposed signal.
    -   `"MARKET"`: Propose a market order.
    -   `"LIMIT"`: Propose a limit order.
-   **`sl_pct (Optional[float/Decimal])`**: Stop-loss percentage from the entry price. E.g., `2.0` for a 2% SL.
-   **`tp_pct (Optional[float/Decimal])`**: Take-profit percentage from the entry price. E.g., `4.0` for a 4% TP.
-   **`default_reward_risk_ratio (Optional[float/Decimal])`**: If `tp_pct` is not directly set (or `sl_pct` for shorting if TP is primary), this ratio is used with the calculated risk (from SL) to determine TP. E.g., `2.0` means TP aims for twice the risk.
-   **`confirmation_rules (Optional[List[dict]])`**: A list of secondary confirmation rules. Each rule is a dictionary:
    -   `feature (str)`: The name of the feature key within `PredictionEvent.associated_features`.
    -   `condition (str)`: The comparison operator (e.g., "gt", "lt", "eq", "gte", "lte").
    -   `threshold (float/Decimal/str)`: The value to compare the feature against.
-   **`limit_offset_pct (Optional[float/Decimal])`**: Required if `entry_type` is "LIMIT". Percentage offset from the best bid (for BUY) or best ask (for SELL) to calculate the limit order price. A positive value might mean a more aggressive placement (paying more for buys, asking less for sells).

## Dependencies

-   **`uuid`**: For generating unique event IDs.
-   **`datetime`**: For timestamping.
-   **`decimal.Decimal`**: For precise financial calculations (prices, percentages).
-   **`operator`**: Used for mapping condition strings to actual comparison functions.
-   **`gal_friday.core.events`**: Definitions for `PredictionEvent`, `TradeSignalProposedEvent`.
-   **`gal_friday.core.pubsub.PubSubManager`**: For event-driven communication.
-   **`gal_friday.logger_service.LoggerService`**: For structured logging.
-   **`gal_friday.market_price_service.MarketPriceService`**: To get current market bid/ask prices for limit order calculations.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `StrategyArbitrator` module.
