# Manual Code Review Findings: `strategy_arbitrator.py`

## Review Date: May 5, 2025
## Reviewer: AI Assistant
## File Reviewed: `src/gal_friday/strategy_arbitrator.py`

## Summary

The `strategy_arbitrator.py` module is responsible for consuming prediction events from the Prediction Service, applying configurable trading strategy rules, and generating trade signal proposals. It implements a simple threshold-based strategy that evaluates prediction probabilities to generate BUY or SELL signals.

While the module successfully implements basic functionality for consuming prediction events and producing trade signal proposals, it has several significant gaps when compared to the requirements, particularly in the areas of Stop-Loss/Take-Profit determination, secondary confirmation conditions, and trade exit logic. The implementation is also limited in its error handling and validation capabilities.

## Strengths

1. **Configurable Strategy Parameters**: The module properly loads and validates strategy parameters from configuration, including buy/sell thresholds and entry types.

2. **Clean Event Handling**: Good implementation of event subscription and publishing using the pubsub mechanism, with proper async handling.

3. **Type Validation**: Performs appropriate type checking of incoming events to ensure they are PredictionEvents.

4. **Module Lifecycle Management**: Well-implemented start/stop methods with proper cleanup of event subscriptions.

5. **Strategy ID Tracking**: Maintains and passes along strategy identifiers to enable tracking of which strategy generated a signal.

## Issues Identified

### A. Functional Requirements Gaps

1. **Missing SL/TP Calculation**: While the code has placeholder values and configuration parameters for stop-loss and take-profit percentages (`sl_pct` and `tp_pct`), it doesn't actually calculate the SL/TP prices:
   ```python
   proposed_event = TradeSignalProposedEvent(
       # ...
       # Placeholder - Must be calculated later!
       proposed_sl_price=Decimal("0"),
       # Placeholder - Must be calculated later!
       proposed_tp_price=Decimal("0"),
       # ...
   )
   ```

2. **No Secondary Confirmation Logic**: The code lacks implementation of any secondary confirmation conditions as described in FR-403, relying solely on prediction probability thresholds.

3. **Missing Trade Exit Logic**: No implementation of the trade exit logic described in FR-407 and FR-408 (time-based exits, prediction reversal exits).

4. **No Entry Price Determination**: For limit orders, the module should determine a proposed entry price, but this is not implemented:
   ```python
   proposed_entry_price=None,
   ```

### B. Design & Implementation Issues

1. **Debug Print Statement**: Contains an unnecessary print statement that outputs on module import:
   ```python
   print("Strategy Arbitrator Loaded")
   ```

2. **Unused PredictionPayload Dataclass**: Defines a PredictionPayload dataclass that isn't used in the implementation:
   ```python
   @dataclass
   class PredictionPayload:
       # ...
   ```

3. **Incomplete Error Context**: Error messages in the error handling sections lack sufficient context for proper debugging.

4. **Incorrect Docstrings**: Some docstrings don't accurately reflect the current implementation, such as:
   ```python
   def _evaluate_strategy(
       self, prediction_event: PredictionEvent  # Use PredictionEvent directly
   ) -> Optional[TradeSignalProposedEvent]:
       """
       Evaluates trading strategy based on prediction probabilities.
       Returns TradeSignalProposedEvent if strategy triggers, None otherwise.
       """
   ```
   The comment "Use PredictionEvent directly" suggests this was changed from a different parameter type.

### C. Error Handling Concerns

1. **Limited Validation of Prediction Data**: Minimal validation of prediction event data, which could lead to unexpected behavior with malformed events.

2. **Missing Error Cases**: Doesn't handle common potential errors like missing prediction values or invalid trading pair formats.

3. **Improper Exception Propagation**: Initialization exceptions are propagated directly rather than being logged and handling gracefully:
   ```python
   except KeyError as key_error:
       raise ValueError("Missing required parameter '{key}'".format(key=key_error))
   ```

### D. Configuration & Hardcoding Issues

1. **Binary Prediction Assumption**: The code assumes a binary prediction model with a hard-coded calculation of `prob_down = 1.0 - prob_up`, which may not be appropriate for all models:
   ```python
   prob_up = prediction_event.prediction_value
   prob_down = 1.0 - prob_up  # Assuming binary prediction target
   ```

2. **Missing Validation for Required Configuration**: While some validation exists, it doesn't check all required configuration parameters.

3. **Assumed Prediction Interpretation**: The code assumes that `prediction_value` represents the probability of an upward price movement, which may not be true for all prediction models.

## Recommendations

### High Priority

1. **Implement Proper SL/TP Price Calculation**:
   ```python
   def _calculate_sl_tp_prices(self, side: str, current_price: Decimal) -> tuple[Decimal, Decimal]:
       """
       Calculate stop-loss and take-profit prices based on current price and configured percentages.

       Args:
           side: The trade side ("BUY" or "SELL")
           current_price: The current market price

       Returns:
           Tuple of (stop_loss_price, take_profit_price)
       """
       if self._sl_pct is None or self._tp_pct is None:
           self.logger.warning(
               "SL/TP percentages not configured, using defaults",
               source_module=self._source_module
           )
           sl_pct = Decimal("0.05") if self._sl_pct is None else self._sl_pct
           tp_pct = Decimal("0.10") if self._tp_pct is None else self._tp_pct
       else:
           sl_pct = self._sl_pct
           tp_pct = self._tp_pct

       if side == "BUY":
           sl_price = current_price * (Decimal("1") - sl_pct)
           tp_price = current_price * (Decimal("1") + tp_pct)
       else:  # SELL
           sl_price = current_price * (Decimal("1") + sl_pct)
           tp_price = current_price * (Decimal("1") - tp_pct)

       return sl_price, tp_price
   ```

2. **Add Current Price Retrieval for SL/TP Calculation**:
   ```python
   async def _get_current_price(self, trading_pair: str) -> Optional[Decimal]:
       """
       Retrieve current market price for a trading pair.
       In the MVP, this would typically come from a market price service.

       Args:
           trading_pair: The trading pair to get the price for

       Returns:
           The current price as a Decimal, or None if unavailable
       """
       # Implementation would depend on how the system accesses market data
       # This is a placeholder example that needs to be replaced with actual code
       try:
           # In a real implementation, this would call the market price service
           # For example:
           # price = await self._market_price_service.get_price(trading_pair)
           # For now, we'll simply use a dummy value for illustration
           return Decimal("100.00")  # Dummy value for illustration
       except Exception as e:
           self.logger.error(
               f"Error retrieving current price for {trading_pair}: {e}",
               exc_info=True,
               source_module=self._source_module
           )
           return None
   ```

3. **Implement Secondary Confirmation Logic**:
   ```python
   def _apply_secondary_confirmation(
       self,
       prediction_event: PredictionEvent,
       side: str
   ) -> bool:
       """
       Apply secondary confirmation rules to validate the primary signal.

       Args:
           prediction_event: The prediction event
           side: The proposed trade side ("BUY" or "SELL")

       Returns:
           True if secondary confirmation passes, False otherwise
       """
       # Get confirmation settings from config
       conf_settings = self._mvp_strategy_config.get("confirmation", {})
       min_confidence = Decimal(str(conf_settings.get("min_confidence", 0)))

       # Check confidence if available
       if hasattr(prediction_event, "confidence") and prediction_event.confidence is not None:
           confidence = Decimal(str(prediction_event.confidence))
           if confidence < min_confidence:
               self.logger.info(
                   f"Signal rejected: confidence {confidence} < minimum {min_confidence}",
                   source_module=self._source_module
               )
               return False

       # Add additional confirmation logic here as needed
       # For example, trend alignment, volume confirmation, etc.

       return True
   ```

### Medium Priority

1. **Improve Error Handling**:
   ```python
   def _validate_prediction_event(self, event: PredictionEvent) -> bool:
       """
       Validate that a prediction event contains all required data in the expected format.

       Args:
           event: The prediction event to validate

       Returns:
           True if valid, False otherwise
       """
       try:
           # Check for required fields
           if not hasattr(event, "prediction_value") or event.prediction_value is None:
               self.logger.warning(
                   f"Missing prediction_value in event {event.event_id}",
                   source_module=self._source_module
               )
               return False

           # Validate trading pair format
           if not hasattr(event, "trading_pair") or not isinstance(event.trading_pair, str):
               self.logger.warning(
                   f"Invalid trading_pair in event {event.event_id}",
                   source_module=self._source_module
               )
               return False

           if "/" not in event.trading_pair:
               self.logger.warning(
                   f"Trading pair {event.trading_pair} is not in the expected format (e.g., BTC/USD)",
                   source_module=self._source_module
               )
               return False

           # Validate prediction value range
           if not (0 <= event.prediction_value <= 1):
               self.logger.warning(
                   f"Prediction value {event.prediction_value} is outside expected range [0,1]",
                   source_module=self._source_module
               )
               return False

           return True
       except Exception as e:
           self.logger.error(
               f"Error validating prediction event {event.event_id}: {e}",
               exc_info=True,
               source_module=self._source_module
           )
           return False
   ```

2. **Add Limit Order Price Determination**:
   ```python
   def _determine_limit_price(
       self,
       side: str,
       current_price: Decimal
   ) -> Decimal:
       """
       Determine an appropriate limit price based on the trade side and current market price.

       Args:
           side: The trade side ("BUY" or "SELL")
           current_price: The current market price

       Returns:
           The calculated limit price
       """
       # Get limit price settings from config
       limit_settings = self._mvp_strategy_config.get("limit_settings", {})
       offset_pct = Decimal(str(limit_settings.get("offset_pct", 0.5)))

       # Calculate limit price with offset
       if side == "BUY":
           # For buys, set limit below current price
           return current_price * (Decimal("1") - (offset_pct / 100))
       else:  # SELL
           # For sells, set limit above current price
           return current_price * (Decimal("1") + (offset_pct / 100))
   ```

3. **Create Proper Configuration Validation**:
   ```python
   def _validate_configuration(self) -> None:
       """Validate the strategy configuration and set defaults where appropriate."""
       # Validate required parameters
       required_params = ["buy_threshold", "sell_threshold"]
       for param in required_params:
           if param not in self._mvp_strategy_config:
               raise ValueError(f"Missing required strategy parameter: {param}")

       # Validate parameter types and ranges
       if not (0 <= self._buy_threshold <= 1):
           raise ValueError(f"buy_threshold must be between 0 and 1, got {self._buy_threshold}")

       if not (0 <= self._sell_threshold <= 1):
           raise ValueError(f"sell_threshold must be between 0 and 1, got {self._sell_threshold}")

       # Validate SL/TP parameters if present
       if self._sl_pct is not None and self._sl_pct <= 0:
           raise ValueError(f"sl_pct must be positive, got {self._sl_pct}")

       if self._tp_pct is not None and self._tp_pct <= 0:
           raise ValueError(f"tp_pct must be positive, got {self._tp_pct}")

       # Set defaults for optional parameters
       if self._sl_pct is None:
           self._sl_pct = Decimal("0.05")  # 5% default
           self.logger.info(
               f"No sl_pct configured, using default: {self._sl_pct}",
               source_module=self._source_module
           )

       if self._tp_pct is None:
           self._tp_pct = Decimal("0.10")  # 10% default
           self.logger.info(
               f"No tp_pct configured, using default: {self._tp_pct}",
               source_module=self._source_module
           )
   ```

### Low Priority

1. **Remove Debug Print Statement**:
   ```python
   # Remove this line:
   print("Strategy Arbitrator Loaded")
   ```

2. **Improve Logging with Metrics**:
   ```python
   def _log_strategy_metrics(self) -> None:
       """Log key strategy metrics for monitoring."""
       metrics = {
           "strategy_id": self._strategy_id,
           "buy_threshold": float(self._buy_threshold),
           "sell_threshold": float(self._sell_threshold),
           "sl_pct": float(self._sl_pct) if self._sl_pct is not None else None,
           "tp_pct": float(self._tp_pct) if self._tp_pct is not None else None,
       }

       self.logger.info(
           f"Strategy metrics: {metrics}",
           source_module=self._source_module
       )
   ```

3. **Add Model Type Configuration**:
   ```python
   def _interpret_prediction(self, prediction_event: PredictionEvent) -> tuple[float, float]:
       """
       Interpret prediction values based on the configured model type.

       Args:
           prediction_event: The prediction event

       Returns:
           Tuple of (probability_up, probability_down)
       """
       model_type = self._mvp_strategy_config.get("model_type", "binary_updown")
       raw_prediction = prediction_event.prediction_value

       if model_type == "binary_updown":
           # Model directly predicts P(up)
           prob_up = raw_prediction
           prob_down = 1.0 - prob_up
       elif model_type == "binary_downup":
           # Model directly predicts P(down)
           prob_down = raw_prediction
           prob_up = 1.0 - prob_down
       elif model_type == "regression_pct":
           # Model predicts percentage move, convert to probabilities
           # This is a simplified example
           if raw_prediction > 0:
               prob_up = 0.5 + (raw_prediction / 2)  # Scale to [0.5, 1.0]
               prob_down = 1.0 - prob_up
           else:
               prob_down = 0.5 + (abs(raw_prediction) / 2)  # Scale to [0.5, 1.0]
               prob_up = 1.0 - prob_down
       else:
           self.logger.warning(
               f"Unknown model type: {model_type}, using default binary interpretation",
               source_module=self._source_module
           )
           prob_up = raw_prediction
           prob_down = 1.0 - prob_up

       return prob_up, prob_down
   ```

## Compliance Assessment

The module partially complies with the requirements:

1. **Interface Compliance**: The module generally follows the expected interface for consuming prediction events and publishing trade signal proposals, but lacks implementation of some required functionality.

2. **Strategy Implementation**: Implements the basic threshold-based strategy described in FR-402, but lacks the secondary confirmation conditions required by FR-403.

3. **SL/TP Determination**: Does not calculate preliminary Stop-Loss and Take-Profit levels as required by FR-404, instead using placeholder values.

4. **Trade Signal Format**: The structure of the trade signal proposals matches the expected format, but lacks important calculated values for SL/TP and entry prices.

5. **Trade Exit Logic**: Does not implement the trade exit logic described in FR-407 and FR-408.

## Follow-up Actions

- [ ] Implement proper calculation of Stop-Loss and Take-Profit levels
- [ ] Add secondary confirmation conditions as specified in FR-403
- [ ] Implement limit order entry price determination
- [ ] Improve error handling and validation
- [ ] Add support for non-binary prediction models
- [ ] Implement trade exit logic as described in FR-407 and FR-408
- [ ] Remove debugging print statement
- [ ] Add more comprehensive configuration validation
- [ ] Improve logging with strategy metrics
- [ ] Consider adding market data integration for real-time price access
