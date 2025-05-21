# Manual Code Review Findings: `simulated_execution_handler.py`

## Review Date: May 5, 2025
## Reviewer: AI Assistant
## File Reviewed: `src/gal_friday/simulated_execution_handler.py`

## Summary

The `simulated_execution_handler.py` module implements a simulated version of the execution handler for backtesting purposes. It processes approved trade signals and simulates order execution based on historical price data without actual exchange interaction. The implementation provides realistic simulation of slippage, fees, and order execution dynamics, allowing strategies to be tested under various market conditions without risking real capital.

While the module implements core functionality for simulating market and limit orders, it has several gaps compared to the requirements, particularly in handling stop-loss/take-profit orders, partial fills, and more sophisticated simulation models. Additionally, there are some implementation issues and optimization opportunities that should be addressed.

## Strengths

1. **Clean Interface Compatibility**: The module implements the same interface as the real execution handler, making it interchangeable for backtesting purposes.

2. **Multiple Order Type Support**: Successfully implements simulation for both market and limit orders with appropriate logic for each.

3. **Configurable Simulation Parameters**: Provides configurable parameters for fees, slippage models, and other simulation behaviors.

4. **Realistic Fill Logic**: Implements realistic fill logic for limit orders by checking if the price level was reached in the bar data.

5. **Proper Event Publishing**: Correctly publishes execution report events with appropriate information to maintain the event-driven architecture.

## Issues Identified

### A. Functional Requirements Gaps

1. **Missing Stop-Loss/Take-Profit Handling**: No implementation of SL/TP order monitoring and execution after entry fills as required by FR-606.

2. **No Partial Fills Support**: The implementation always assumes full fills for orders, which doesn't match FR-607 requiring partial fills simulation.

3. **Limited Order Types**: Only market and limit orders are supported; stop orders are mentioned but not implemented.

4. **No Limit Order Timeout**: No implementation of limit order timeout logic as required by FR-605.

### B. Design & Implementation Issues

1. **Inconsistent Error Handling**: Some methods propagate exceptions while others catch and handle them internally, leading to inconsistent error behavior.

2. **Inefficient Bar Data Retrieval**: The `_get_next_bar_data` method doesn't specify a time range or limit, potentially retrieving more data than necessary.

3. **Hard Dependency on Bar Structure**: The code assumes specific structure of bar data (OHLCV) which makes it less flexible if data source changes.

4. **Commented Out Module-Level Logger**: The code has commented out a module-level logger but still refers to it in comments.

### C. Simulation Quality Concerns

1. **Simple Slippage Model**: The slippage model is relatively basic, with just fixed and ATR-based options, missing more sophisticated volume/volatility/market depth considerations.

2. **Limited Market Condition Simulation**: No simulation of extreme market conditions such as gaps, fast markets, or low liquidity scenarios.

3. **No Realistic Latency Simulation**: No implementation of realistic processing delays or latency that would occur in real trading.

4. **Missing Exchange-Specific Behaviors**: Doesn't simulate exchange-specific behaviors or quirks that might affect real trading.

### D. Error Handling & Validation

1. **Limited Order Parameter Validation**: Minimal validation of order parameters before attempting simulation.

2. **Simplistic Error Communication**: Error messages in execution reports lack detail for proper diagnosis.

3. **No Validation Against Exchange Rules**: No validation of order sizes against exchange minimum/maximum requirements.

## Recommendations

### High Priority

1. **Implement SL/TP Order Handling**:
   ```python
   async def _monitor_and_execute_sl_tp(self, entry_fill_event, sl_price, tp_price):
       """Monitors price action and executes SL/TP orders when conditions are met."""
       trading_pair = entry_fill_event.trading_pair
       side = "SELL" if entry_fill_event.side.upper() == "BUY" else "BUY"
       quantity = entry_fill_event.quantity_filled

       # In backtesting, we can look ahead to see if SL/TP would be triggered
       next_bars = self.data_service.get_future_bars(
           trading_pair,
           entry_fill_event.timestamp_exchange,
           limit=50  # Reasonable number to check
       )

       for bar in next_bars:
           # Check if SL triggered (price moved against position)
           sl_triggered = (
               (side == "SELL" and bar["low"] <= sl_price) or
               (side == "BUY" and bar["high"] >= sl_price)
           )

           # Check if TP triggered (price moved in favor of position)
           tp_triggered = (
               (side == "SELL" and bar["high"] >= tp_price) or
               (side == "BUY" and bar["low"] <= tp_price)
           )

           if sl_triggered or tp_triggered:
               # Create and publish execution report for the SL/TP order
               trigger_price = sl_price if sl_triggered else tp_price
               trigger_type = "Stop Loss" if sl_triggered else "Take Profit"

               await self._publish_simulated_report(
                   entry_fill_event,
                   "FILLED",
                   quantity,
                   trigger_price,
                   quantity * trigger_price * self.taker_fee_pct,
                   entry_fill_event.commission_asset,
                   f"{trigger_type} execution",
                   bar.name  # Timestamp of the bar
               )

               self.logger.info(
                   f"{trigger_type} executed at {trigger_price} for {trading_pair}",
                   source_module=self.__class__.__name__
               )
               return True

       # If neither SL nor TP was triggered
       self.logger.info(
           f"Neither SL ({sl_price}) nor TP ({tp_price}) triggered for {trading_pair}",
           source_module=self.__class__.__name__
       )
       return False
   ```

2. **Add Partial Fills Support**:
   ```python
   def _calculate_fill_quantity(self, event, bar_data):
       """Calculate partial or full fill quantity based on order size and liquidity."""
       requested_qty = event.quantity

       # Simple model: Use bar volume to estimate available liquidity
       bar_volume = Decimal(str(bar_data["volume"]))

       # Estimate what percentage of the bar's volume we can realistically fill
       # without excessive slippage (assuming we're not the only trader)
       max_fill_pct = self.config.get_decimal(
           "backtest.max_fill_percentage", Decimal("0.1")
       )

       # Calculate maximum quantity that could be filled based on available volume
       max_fill_qty = bar_volume * max_fill_pct

       if requested_qty <= max_fill_qty:
           # Can fill the entire order
           return requested_qty, "FILLED"
       else:
           # Partial fill
           partial_qty = max_fill_qty
           self.logger.info(
               f"Partial fill: {partial_qty}/{requested_qty} due to volume constraints",
               source_module=self.__class__.__name__
           )
           # Check if minimum fill threshold is met
           min_fill_pct = self.config.get_decimal(
               "backtest.min_fill_percentage", Decimal("0.05")
           )
           if partial_qty < (requested_qty * min_fill_pct):
               # Too small to be worthwhile
               return Decimal("0"), "REJECTED"
           return partial_qty, "PARTIALLY_FILLED"
   ```

3. **Implement Limit Order Timeout**:
   ```python
   async def _simulate_limit_order_with_timeout(self, event, next_bars):
       """Simulates a limit order with timeout consideration."""
       limit_price = event.limit_price
       timeout_bars = self.config.get("backtest.limit_order_timeout_bars", 5)

       for i, bar in enumerate(next_bars[:timeout_bars]):
           filled = self._check_limit_order_fill(event.side, limit_price, bar)
           if filled:
               # Calculate fill details
               # ...existing code...
               return fill_result

       # If we get here, the order timed out without filling
       self.logger.info(
           f"Limit order timed out after {timeout_bars} bars without filling",
           source_module=self.__class__.__name__
       )
       return {
           "status": "REJECTED",
           "quantity": Decimal(0),
           "fill_price": None,
           "commission": Decimal(0),
           "commission_asset": None,
           "error_msg": "Limit order timed out",
           "timestamp": datetime.utcnow()
       }
   ```

### Medium Priority

1. **Improve Slippage Model**:
   ```python
   def _calculate_slippage(self, trading_pair, side, base_price, signal_timestamp, order_size):
       """Enhanced slippage calculation that considers size, volatility, and market depth."""
       slippage = Decimal(0)

       try:
           if self.slippage_model == "fixed":
               slippage = base_price * self.slip_fixed_pct
           elif self.slippage_model == "volatility":
               # Get ATR for the bar the signal was generated on
               atr = self.data_service.get_atr(trading_pair, signal_timestamp)
               if atr is not None and atr > 0:
                   slippage = atr * self.slip_atr_multiplier
               else:
                   self.logger.warning(
                       f"Could not get ATR for {trading_pair} at {signal_timestamp}, using fallback",
                       source_module=self.__class__.__name__
                   )
                   slippage = base_price * self.slip_fixed_pct
           elif self.slippage_model == "market_impact":
               # Get volume to calculate market impact
               avg_volume = self.data_service.get_average_volume(trading_pair, signal_timestamp)
               if avg_volume and avg_volume > 0:
                   # Calculate market impact as percentage of average volume
                   volume_ratio = min(order_size / avg_volume, Decimal("1.0"))
                   # Non-linear impact model: impact increases more than linearly with size
                   impact_factor = self.config.get_decimal("backtest.market_impact_factor", Decimal("0.1"))
                   slippage = base_price * impact_factor * (volume_ratio ** Decimal("1.5"))
               else:
                   self.logger.warning(
                       f"Could not get volume for {trading_pair}, using fallback",
                       source_module=self.__class__.__name__
                   )
                   slippage = base_price * self.slip_fixed_pct
           else:
               self.logger.warning(
                   f"Unknown slippage model: {self.slippage_model}. Using fallback.",
                   source_module=self.__class__.__name__
               )
               slippage = base_price * self.slip_fixed_pct
       except Exception as e:
           self.logger.error(
               f"Error calculating slippage: {e}",
               source_module=self.__class__.__name__,
               exc_info=True
           )
           slippage = base_price * self.slip_fixed_pct  # Fallback

       # Slippage is always adverse
       return abs(slippage)
   ```

2. **Standardize Error Handling**:
   ```python
   async def _handle_exception(self, event, exception, context=""):
       """Standardized handler for exceptions during simulation."""
       error_msg = f"Error during {context}: {str(exception)}"
       self.logger.error(
           error_msg,
           source_module=self.__class__.__name__,
           exc_info=True
       )

       # Publish standardized error report
       await self._publish_simulated_report(
           event,
           "ERROR",
           Decimal(0),
           None,
           Decimal(0),
           None,
           error_msg,
           datetime.utcnow()
       )

       # Return a standardized error result
       return {
           "status": "ERROR",
           "error_msg": error_msg,
           "exception_type": type(exception).__name__
       }
   ```

3. **Add Order Parameter Validation**:
   ```python
   def _validate_order_parameters(self, event):
       """Validates order parameters against exchange rules and logical constraints."""
       errors = []

       # Check for valid trading pair format
       if "/" not in event.trading_pair:
           errors.append(f"Invalid trading pair format: {event.trading_pair}")

       # Validate order type
       valid_order_types = ["MARKET", "LIMIT"]
       if event.order_type.upper() not in valid_order_types:
           errors.append(f"Unsupported order type: {event.order_type}")

       # Validate quantity
       min_quantity = self.config.get_decimal(
           f"backtest.min_order_size.{event.trading_pair}", Decimal("0.001")
       )
       if event.quantity < min_quantity:
           errors.append(f"Order size too small: {event.quantity} < {min_quantity}")

       # Validate limit price for limit orders
       if event.order_type.upper() == "LIMIT" and not event.limit_price:
           errors.append("Limit price missing for limit order")

       # Validate stop price for stop orders
       if event.order_type.upper() in ["STOP", "STOP_LIMIT"] and not hasattr(event, "stop_price"):
           errors.append("Stop price missing for stop order")

       return errors
   ```

### Low Priority

1. **Add Latency Simulation**:
   ```python
   async def _simulate_latency(self):
       """Simulates realistic order processing latency."""
       # Get configuration for latency simulation
       min_latency_ms = self.config.get("backtest.min_latency_ms", 0)
       max_latency_ms = self.config.get("backtest.max_latency_ms", 50)

       if min_latency_ms <= 0 and max_latency_ms <= 0:
           return  # No latency simulation

       # Generate a random latency between min and max
       import random
       latency_ms = random.uniform(min_latency_ms, max_latency_ms)
       latency_s = latency_ms / 1000.0

       if latency_s > 0:
           self.logger.debug(
               f"Simulating latency: {latency_ms:.2f}ms",
               source_module=self.__class__.__name__
           )
           await asyncio.sleep(latency_s)
   ```

2. **Implement Realistic Fill Price Distribution**:
   ```python
   def _calculate_realistic_fill_price(self, bar, side, order_type):
       """Calculates a more realistic fill price based on intra-bar price distribution."""
       # Simple model: Assume prices within a bar follow a distribution
       # weighted toward the bar's open and close
       # This simulates the tendency for more trades to occur at those levels

       # Get the price range
       high, low = Decimal(str(bar["high"])), Decimal(str(bar["low"]))
       open_price, close_price = Decimal(str(bar["open"])), Decimal(str(bar["close"]))

       if order_type.upper() == "MARKET":
           # For market orders, use a weighted average that factors in
           # price movement direction and open/close prices
           if side.upper() == "BUY":
               # For buys, bias toward higher prices
               weights = {"open": 0.3, "high": 0.4, "close": 0.3}
               fill_price = (
                   (open_price * weights["open"]) +
                   (high * weights["high"]) +
                   (close_price * weights["close"])
               )
           else:  # SELL
               # For sells, bias toward lower prices
               weights = {"open": 0.3, "low": 0.4, "close": 0.3}
               fill_price = (
                   (open_price * weights["open"]) +
                   (low * weights["low"]) +
                   (close_price * weights["close"])
               )
       else:  # LIMIT orders - use the limit price directly
           # Handled elsewhere
           fill_price = None

       return fill_price
   ```

3. **Add Performance Metrics Tracking**:
   ```python
   def _track_execution_performance(self, event, processing_time_ms):
       """Tracks performance metrics for execution simulation."""
       if not hasattr(self, "_performance_metrics"):
           self._performance_metrics = {
               "count": 0,
               "total_time_ms": 0,
               "max_time_ms": 0,
               "order_types": {},
               "rejection_reasons": {}
           }

       metrics = self._performance_metrics
       metrics["count"] += 1
       metrics["total_time_ms"] += processing_time_ms
       metrics["max_time_ms"] = max(metrics["max_time_ms"], processing_time_ms)

       # Track by order type
       order_type = event.order_type.upper()
       metrics["order_types"][order_type] = metrics["order_types"].get(order_type, 0) + 1

       # Log performance summary every N executions
       if metrics["count"] % 100 == 0:
           avg_time = metrics["total_time_ms"] / metrics["count"]
           self.logger.info(
               f"Execution performance: {metrics['count']} orders, "
               f"avg={avg_time:.2f}ms, max={metrics['max_time_ms']:.2f}ms",
               source_module=self.__class__.__name__
           )
           self.logger.info(
               f"Order type distribution: {metrics['order_types']}",
               source_module=self.__class__.__name__
           )
   ```

## Compliance Assessment

The module partially complies with the requirements:

1. **Interface Compatibility**: Successfully maintains the same interface as the real execution handler for compatibility during backtesting.

2. **Order Type Support**: Implements market and limit orders but lacks support for stop-loss/take-profit orders and their monitoring.

3. **Simulation Realism**: Provides basic slippage models and realistic fee structures but lacks more sophisticated market impact models and realistic latency.

4. **Event Communication**: Correctly publishes execution report events with standardized formats, maintaining the event-driven architecture.

5. **Configurability**: Provides good configurability for fees and basic slippage models but lacks configuration options for more advanced simulation parameters.

## Follow-up Actions

- [ ] Implement SL/TP order monitoring and execution after entry fills (FR-606)
- [ ] Add support for partial fills simulation (FR-607)
- [ ] Implement limit order timeout logic (FR-605)
- [ ] Enhance slippage models to account for order size and market depth
- [ ] Add realistic latency simulation
- [ ] Implement more sophisticated order matching logic
- [ ] Standardize error handling and improve validation
- [ ] Add simulation of exchange-specific behaviors and limits
- [ ] Enhance bar data manipulation for more realistic price behavior
- [ ] Implement performance tracking for simulation optimization
