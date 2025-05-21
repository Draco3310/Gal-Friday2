# Manual Code Review Findings: `risk_manager.py`

## Review Date: May 5, 2025
## Reviewer: AI Assistant
## File Reviewed: `src/gal_friday/risk_manager.py`

## Summary

The `risk_manager.py` module implements a robust risk control layer for the trading system. It processes proposed trade signals, evaluates them against various risk criteria, and either approves them with appropriate position sizing or rejects them. The module also performs periodic risk monitoring of the portfolio to trigger system halts when necessary.

The implementation effectively enforces key risk limits, calculates position sizes using the fixed fractional risk method, and handles the publishing of approved or rejected trade signals. However, there are some gaps in the implementation of certain required risk checks and potential improvements to be made in error handling and code organization.

## Strengths

1. **Comprehensive Risk Configuration**: The module properly loads and uses a wide range of configurable risk parameters including drawdown limits, exposure limits, and position sizing values.

2. **Fixed Fractional Position Sizing**: Well-implemented position size calculation using the fixed fractional risk method as specified in the requirements.

3. **Proper Stop Loss Validation**: Good validation of stop loss prices relative to entry prices, including minimum distance checks.

4. **Periodic Risk Monitoring**: Effective implementation of periodic portfolio-level risk monitoring with HALT triggering.

5. **Decimal Usage**: Proper use of Decimal throughout for financial calculations to avoid floating-point precision issues.

## Issues Identified

### A. Functional Requirements Gaps

1. **Insufficient Order Validation**: Missing implementation of the "fat finger" check for proposed entry prices as specified in FR-506.

2. **Incomplete Pre-Trade Checks**: While some checks are implemented, the code is missing checks for maximum total portfolio exposure and sufficient balance.

3. **Missing Consecutive Losses Tracking**: No implementation of consecutive losing trades limit tracking as required by FR-504.

4. **Limited TP Price Handling**: The code generates a TP price if none is provided, but there's no validation of provided TP prices against any risk criteria.

### B. Design & Implementation Issues

1. **Conditional Type Imports**: Uses TYPE_CHECKING for avoiding circular imports, but doesn't have a proper runtime import mechanism for PortfolioManager, creating potential runtime issues.

2. **Global Decimal Precision Setting**: Sets a global Decimal precision that could affect other modules:
   ```python
   getcontext().prec = 28
   ```

3. **Unused Data Classes**: The `TradeSignalProposedPayload` and `SystemHaltPayload` classes are defined but never used in the implementation.

4. **Mixed Error Handling**: Some errors are handled by returning boolean/message pairs, while others use exceptions, creating an inconsistent error handling pattern.

### C. Error Handling Concerns

1. **Limited Recovery from Portfolio Manager Errors**: If portfolio state retrieval fails, the module simply skips that check cycle rather than implementing a more robust retry or fallback mechanism.

2. **Missing Error Cases**: The position size calculation doesn't handle cases where exchange-specific minimum/maximum order sizes are violated.

3. **No Robust Handling of Invalid Configurations**: The module assumes valid risk configuration values without thorough validation.

### D. Configuration & Hardcoding Issues

1. **Default Risk Parameter Values**: Critical risk parameters have hardcoded default values that should potentially be more conservative:
   ```python
   self._max_total_drawdown_pct = Decimal(str(limits.get("max_total_drawdown_pct", 15.0)))
   ```

2. **Take Profit Calculation**: The default take profit price calculation uses a hardcoded 2x risk multiplier:
   ```python
   # Set TP at 2x the risk distance or some other reasonable default
   if approved_payload_dict["side"].upper() == "BUY":
       tp_price_value = entry_price + (price_diff * 2)
   else:
       tp_price_value = entry_price - (price_diff * 2)
   ```

## Recommendations

### High Priority

1. **Implement Missing Pre-Trade Checks**:
   ```python
   def _check_portfolio_exposure(self, portfolio_state, new_position_value):
       """Check if the new position would exceed maximum portfolio exposure."""
       current_exposure_pct = Decimal(portfolio_state.get("total_exposure_pct", "0"))

       # Calculate new exposure after potential position
       equity = Decimal(portfolio_state["total_equity"])
       new_exposure_pct = current_exposure_pct + ((new_position_value / equity) * 100)

       if new_exposure_pct > self._max_total_exposure_pct:
           return False, f"MAX_TOTAL_EXPOSURE_LIMIT ({new_exposure_pct:.2f}% > {self._max_total_exposure_pct}%)"
       return True, None
   ```

2. **Implement Consecutive Losses Tracking**:
   ```python
   def _check_consecutive_losses(self):
       """Check if the consecutive loss limit has been reached."""
       # This would require tracking trade outcomes, potentially by
       # subscribing to execution report events and maintaining state
       current_consecutive_losses = self._consecutive_loss_count

       if current_consecutive_losses >= self._max_consecutive_losses:
           return False, f"MAX_CONSECUTIVE_LOSSES_LIMIT ({current_consecutive_losses} >= {self._max_consecutive_losses})"
       return True, None
   ```

3. **Implement Fat Finger Check**:
   ```python
   def _check_fat_finger(self, trading_pair, side, proposed_price):
       """Check if the proposed price is too far from current market price."""
       try:
           # Get current market price (implementation details depend on your market data access)
           current_market_price = self._get_current_market_price(trading_pair)
           if current_market_price is None:
               return False, "MARKET_PRICE_UNAVAILABLE"

           # Calculate deviation
           deviation_pct = abs(proposed_price - current_market_price) / current_market_price * 100
           max_deviation_pct = self._config.get("max_price_deviation_pct", 5.0)

           if deviation_pct > max_deviation_pct:
               return False, f"PRICE_DEVIATION_TOO_HIGH ({deviation_pct:.2f}% > {max_deviation_pct}%)"
           return True, None
       except Exception as e:
           self.logger.error(f"Error in fat finger check: {e}", source_module=self._source_module)
           return False, "FAT_FINGER_CHECK_ERROR"
   ```

### Medium Priority

1. **Improve Portfolio State Handling**:
   ```python
   def _get_portfolio_state_with_retry(self, max_retries=3, retry_delay=1.0):
       """Get portfolio state with retries."""
       retries = 0
       while retries < max_retries:
           try:
               portfolio_state = self._portfolio_manager.get_current_state()
               if portfolio_state:
                   return portfolio_state
           except Exception as e:
               self.logger.warning(
                   f"Error getting portfolio state (attempt {retries+1}/{max_retries}): {e}",
                   source_module=self._source_module
               )
           retries += 1
           if retries < max_retries:
               # Use asyncio.sleep in an async version of this method
               time.sleep(retry_delay)

       # After all retries, log error and return None
       self.logger.error(
           f"Failed to get portfolio state after {max_retries} attempts",
           source_module=self._source_module
       )
       return None
   ```

2. **Fix Decimal Precision Handling**:
   ```python
   # Instead of setting global precision:
   # getcontext().prec = 28

   # Create a local context for precision operations
   def _get_decimal_context(self):
       """Get a decimal context with the configured precision."""
       context = getcontext().copy()
       context.prec = self._config.get("decimal_precision", 28)
       return context

   # Then use it in calculations:
   def _calculate_position_size(self, ...):
       context = self._get_decimal_context()
       with context:
           # Perform calculations...
   ```

3. **Improve Configuration Validation**:
   ```python
   def _validate_config(self):
       """Validate configuration values and set reasonable defaults if needed."""
       errors = []
       warnings = []

       # Validate max drawdown limits
       if self._max_total_drawdown_pct <= 0 or self._max_total_drawdown_pct > 100:
           msg = f"Invalid max_total_drawdown_pct: {self._max_total_drawdown_pct}. Setting to default 15%"
           self.logger.warning(msg, source_module=self._source_module)
           warnings.append(msg)
           self._max_total_drawdown_pct = Decimal("15.0")

       # Continue with other validations...

       if errors:
           error_msg = f"Configuration validation errors: {'; '.join(errors)}"
           self.logger.error(error_msg, source_module=self._source_module)
           raise ValueError(error_msg)

       if warnings:
           self.logger.warning(
               f"Configuration warnings: {'; '.join(warnings)}",
               source_module=self._source_module
           )
   ```

### Low Priority

1. **Implement Order Size Rounding**:
   ```python
   def _round_position_size_to_exchange_precision(self, quantity, trading_pair):
       """Round the position size to the exchange precision."""
       # Get exchange specific precision for the trading pair
       # This could be loaded from configuration or from an exchange info service
       try:
           precision = self._get_trading_pair_precision(trading_pair)
           if precision is not None:
               # Round to precision
               rounded_qty = Decimal(str(quantity)).quantize(
                   Decimal('0.' + '0' * precision),
                   rounding=ROUND_DOWN
               )
               return rounded_qty
       except Exception as e:
           self.logger.warning(
               f"Error rounding position size: {e}. Using unrounded value.",
               source_module=self._source_module
           )
       return quantity
   ```

2. **Improve TP Price Handling**:
   ```python
   def _validate_tp_price(self, side, entry_price, sl_price, tp_price=None):
       """Validate take profit price or generate a reasonable one."""
       if tp_price is None:
           # Calculate a default TP based on risk:reward ratio from config
           risk_reward_ratio = self._config.get("risk_reward_ratio", 2.0)
           price_diff = abs(entry_price - sl_price)

           if side.upper() == "BUY":
               return entry_price + (price_diff * risk_reward_ratio)
           else:
               return entry_price - (price_diff * risk_reward_ratio)
       else:
           # Validate the provided TP price
           if side.upper() == "BUY" and tp_price <= entry_price:
               return None, "INVALID_TP_PRICE (TP <= Entry for BUY)"
           if side.upper() == "SELL" and tp_price >= entry_price:
               return None, "INVALID_TP_PRICE (TP >= Entry for SELL)"
           return tp_price, None
   ```

3. **Add Risk Metrics Tracking**:
   ```python
   def _update_risk_metrics(self, portfolio_state):
       """Update and track risk metrics over time."""
       if not hasattr(self, "_risk_metrics_history"):
           self._risk_metrics_history = []

       # Extract current metrics
       try:
           metrics = {
               "timestamp": datetime.utcnow(),
               "equity": Decimal(portfolio_state["total_equity"]),
               "total_drawdown_pct": Decimal(portfolio_state["total_drawdown_pct"]),
               "daily_drawdown_pct": Decimal(portfolio_state["daily_drawdown_pct"]),
               "weekly_drawdown_pct": Decimal(portfolio_state["weekly_drawdown_pct"]),
               "total_exposure_pct": Decimal(portfolio_state.get("total_exposure_pct", "0"))
           }

           # Store metrics (keeping last N entries)
           self._risk_metrics_history.append(metrics)
           if len(self._risk_metrics_history) > 1000:  # Keep last 1000 entries
               self._risk_metrics_history.pop(0)

           # Log significant changes
           if len(self._risk_metrics_history) > 1:
               prev_metrics = self._risk_metrics_history[-2]
               dd_change = metrics["total_drawdown_pct"] - prev_metrics["total_drawdown_pct"]
               if abs(dd_change) > Decimal("0.5"):  # Log if drawdown changed by more than 0.5%
                   self.logger.info(
                       f"Significant drawdown change: {dd_change:+.2f}% to {metrics['total_drawdown_pct']:.2f}%",
                       source_module=self._source_module
                   )
       except (KeyError, TypeError, ValueError) as e:
           self.logger.error(
               f"Error updating risk metrics: {e}",
               source_module=self._source_module
           )
   ```

## Compliance Assessment

The module partially complies with the architectural requirements:

1. **Interface Implementation**: The implementation mostly conforms to the `RiskManager` interface defined in the interface definitions document, but is missing some required functionality like consecutive losses tracking.

2. **Functional Requirements**: Meets most of FR-503 (drawdown limits) and FR-505 (position sizing), but has incomplete implementation of FR-504 (consecutive losses) and FR-506 (pre-trade checks).

3. **Event Handling**: Properly consumes trade signal proposal events and publishes approved/rejected signals and potential halt triggers.

4. **Risk Methodology**: Implements the fixed fractional risk method for position sizing as required, but could improve validation of inputs and handling of edge cases.

5. **Portfolio State Interaction**: Correctly queries the PortfolioManager for synchronous state information but lacks robust error handling and retry mechanisms.

## Follow-up Actions

- [ ] Implement the missing "fat finger" check for proposed entry prices
- [ ] Add tracking and enforcement of consecutive losses limit
- [ ] Implement checks for total portfolio exposure and sufficient balance
- [ ] Fix the global Decimal precision setting with a context-based approach
- [ ] Enhance error handling, especially for portfolio state retrieval failures
- [ ] Improve configuration validation and provide better defaults
- [ ] Add order size rounding based on exchange precision requirements
- [ ] Implement proper take profit price validation
- [ ] Consider adding risk metrics tracking for monitoring purposes
- [ ] Clean up unused data classes and improve code organization
