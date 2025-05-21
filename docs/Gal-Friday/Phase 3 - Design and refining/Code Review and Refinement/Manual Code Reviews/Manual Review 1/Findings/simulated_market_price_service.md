# Manual Code Review Findings: `simulated_market_price_service.py`

## Review Date: May 5, 2025
## Reviewer: AI Assistant
## File Reviewed: `src/gal_friday/simulated_market_price_service.py`

## Summary

The `simulated_market_price_service.py` module implements a simulated version of the market price service for backtesting purposes. It provides historical price data at specific timestamps during simulation, allowing other system components to access market prices as if they were in a live trading environment. The implementation successfully manages historical data retrieval and handles time-based progression of simulated prices.

While the module provides basic functionality for price simulation, it has several gaps compared to the requirements, particularly in simulating bid/ask spreads, market depth, and more sophisticated market behaviors. Additionally, there are issues with the error handling approach, logging standards, and integration with the broader system architecture.

## Strengths

1. **Clean Interface**: The module provides a straightforward interface for accessing historical prices at specific timestamps.

2. **Time Progression Handling**: Good implementation of time-based progression with the `update_time` method that updates the simulation's current timestamp.

3. **Flexible Price Retrieval**: Successfully implements price lookup using pandas' `asof` function to handle requests for timestamps between available data points.

4. **Self-Conversion Handling**: Properly handles special cases like currency self-conversion (e.g., USD/USD = 1.0).

5. **Decimal Precision**: Uses Decimal type for price values, which is appropriate for financial calculations.

## Issues Identified

### A. Functional Requirements Gaps

1. **Missing Bid/Ask Spread Simulation**: The implementation only provides 'close' prices without simulating realistic bid/ask spreads as required for proper backtesting.

2. **No Market Depth Simulation**: No implementation of order book depth, which would be necessary for realistic slippage modeling.

3. **Limited Price Data**: Only provides the 'close' price, lacking access to the full OHLC data that might be needed by some strategies.

4. **Single Price Lookup Method**: Only implements `get_latest_price` without supporting other interface methods like `get_bid_ask` that may be needed for compatibility with the real market price service.

### B. Design & Implementation Issues

1. **Module-Level Logger**: Uses a module-level logger rather than the injected logger_service approach used by other components:
   ```python
   log = logging.getLogger(__name__)
   ```

2. **Inconsistent Error Handling**: Some errors return None while others log warnings but still attempt to continue processing, potentially leading to inconsistent behavior.

3. **No Interface Alignment Documentation**: Lacks explicit documentation indicating how the interface aligns with the real `MarketPriceService`.

4. **Missing Integration Points**: No clear methods for integration with the `BacktestingEngine` for controlling the simulation.

### C. Error Handling Concerns

1. **Limited Data Validation**: Minimal validation of the historical data structure during initialization:
   ```python
   if not isinstance(df.index, pd.DatetimeIndex):
       log.warning(f"Historical data for {pair} does not have a DatetimeIndex.")
   if "close" not in df.columns:
       log.warning(f"Historical data for {pair} is missing 'close' column.")
   ```

2. **Silent Defaults to None**: When errors occur, the method silently returns None without providing a way to distinguish between "no data available" and "error occurred":
   ```python
   except Exception as e:
       log.exception("Error retrieving latest price...", exc_info=e)
       return None
   ```

3. **No Validation of Simulation Time**: No verification that `update_time` receives a valid datetime object before setting it.

### D. Configuration & Hardcoding Issues

1. **No Configurable Parameters**: No configuration for important simulation parameters like spread calculation or slippage models.

2. **Fixed Column Names**: Hardcoded column name 'close' for price lookup without configuration option:
   ```python
   price = pair_data.iloc[idx_pos]["close"]
   ```

3. **No Configuration for Data Requirements**: No configurable validation of required data fields or formats.

## Recommendations

### High Priority

1. **Implement Bid/Ask Spread Simulation**:
   ```python
   def get_bid_ask(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]:
       """Gets the bid and ask prices for a trading pair at the current simulation time.

       Args:
           trading_pair: The trading pair symbol (e.g., "XRP/USD")

       Returns:
           A tuple of (bid, ask) prices, or None if prices cannot be determined
       """
       close_price = self.get_latest_price(trading_pair)
       if close_price is None:
           return None

       # Get spread parameters from config or use default
       base_spread_pct = self._config.get(
           f"simulation.spread.{trading_pair}",
           self._config.get("simulation.spread.default", 0.1)
       )

       # Calculate spread based on market conditions (e.g., volatility)
       spread_pct = self._adjust_spread_for_volatility(
           trading_pair, base_spread_pct
       )

       # Calculate the half-spread amount
       half_spread = close_price * (spread_pct / 100) / 2

       # Calculate bid and ask
       bid = close_price - half_spread
       ask = close_price + half_spread

       return (bid, ask)
   ```

2. **Align Interface with Real Service**:
   ```python
   def get_price_sync(self, trading_pair: str, use_mid: bool = True) -> Optional[Decimal]:
       """Synchronously gets the latest price for a trading pair, matching the
       interface of the real MarketPriceService.

       Args:
           trading_pair: The trading pair symbol (e.g., "XRP/USD")
           use_mid: If True, returns mid price; if False, returns last traded price

       Returns:
           The price as a Decimal, or None if unavailable
       """
       # For compatibility with the real service interface
       if use_mid:
           bid_ask = self.get_bid_ask(trading_pair)
           if bid_ask is None:
               return None
           bid, ask = bid_ask
           return (bid + ask) / 2
       else:
           return self.get_latest_price(trading_pair)
   ```

3. **Implement Market Depth Simulation**:
   ```python
   def get_order_book_snapshot(self, trading_pair: str, depth: int = 5) -> Optional[Dict[str, List]]:
       """Simulates an order book snapshot based on the current price.

       Args:
           trading_pair: The trading pair symbol
           depth: Number of price levels to generate

       Returns:
           A dictionary with 'bids' and 'asks' lists, or None if unavailable
       """
       close_price = self.get_latest_price(trading_pair)
       if close_price is None:
           return None

       bid_ask = self.get_bid_ask(trading_pair)
       if bid_ask is None:
           return None

       bid, ask = bid_ask

       # Generate simulated order book with declining volume at each level
       bids = []
       asks = []

       # Generate bid levels (price descending)
       base_volume = self._get_base_volume(trading_pair)
       price_increment = bid * Decimal('0.0005')  # 0.05% between levels

       for i in range(depth):
           price = bid - (i * price_increment)
           # Volume decreases at deeper levels
           volume = base_volume * (1 - (i * 0.15))
           bids.append([float(price), float(volume)])

       # Generate ask levels (price ascending)
       price_increment = ask * Decimal('0.0005')  # 0.05% between levels

       for i in range(depth):
           price = ask + (i * price_increment)
           # Volume decreases at deeper levels
           volume = base_volume * (1 - (i * 0.15))
           asks.append([float(price), float(volume)])

       return {
           'bids': bids,
           'asks': asks
       }
   ```

### Medium Priority

1. **Replace Module-Level Logger with Injected LoggerService**:
   ```python
   class SimulatedMarketPriceService:
       """Provides synchronous access to the latest market prices based on historical
       data during a backtest simulation.
       """

       def __init__(self,
                   historical_data: Dict[str, pd.DataFrame],
                   config: Dict[str, Any],
                   logger_service: LoggerService):
           """
           Initializes the service with historical market data.

           Args:
               historical_data: A dictionary where keys are trading pairs (e.g., "XRP/USD")
                                and values are pandas DataFrames containing OHLCV data
                                indexed by timestamp (UTC).
               config: Configuration dictionary
               logger_service: The shared logger service
           """
           self.historical_data = historical_data
           self._config = config
           self.logger = logger_service
           self._current_timestamp: Optional[datetime] = None
           self._source_module = self.__class__.__name__

           # Validate data format minimally
           for pair, df in historical_data.items():
               if not isinstance(df.index, pd.DatetimeIndex):
                   self.logger.warning(
                       f"Historical data for {pair} does not have a DatetimeIndex.",
                       source_module=self._source_module
                   )
               if "close" not in df.columns:  # Assuming we use 'close' price
                   self.logger.warning(
                       f"Historical data for {pair} is missing 'close' column.",
                       source_module=self._source_module
                   )

           self.logger.info(
               "SimulatedMarketPriceService initialized.",
               source_module=self._source_module
           )
   ```

2. **Improve Error Handling and Validation**:
   ```python
   def update_time(self, timestamp: datetime) -> bool:
       """Updates the current simulation time.

       Args:
           timestamp: The current simulation time

       Returns:
           True if successful, False otherwise
       """
       if not isinstance(timestamp, datetime):
           self.logger.error(
               f"Invalid timestamp type: {type(timestamp)}. Expected datetime.",
               source_module=self._source_module
           )
           return False

       self.logger.debug(
           f"Updating simulated time to: {timestamp}",
           source_module=self._source_module
       )
       self._current_timestamp = timestamp
       return True

   def validate_data_requirements(self, trading_pair: str) -> bool:
       """Validates that required data is available for a trading pair.

       Args:
           trading_pair: The trading pair to validate

       Returns:
           True if valid data exists, False otherwise
       """
       if trading_pair not in self.historical_data:
           # Handle special case for self-conversion
           if trading_pair.count("/") == 1:
               base, quote = trading_pair.split("/")
               if base == quote:
                   return True
           return False

       pair_data = self.historical_data[trading_pair]
       if "close" not in pair_data.columns:
           return False

       return True
   ```

3. **Add Configurable Parameters**:
   ```python
   def _load_configuration(self) -> None:
       """Loads simulation parameters from configuration."""
       sim_config = self._config.get("simulation", {})

       # Load spread configuration
       self._default_spread_pct = Decimal(str(
           sim_config.get("default_spread_pct", "0.1")
       ))

       # Load volatility impact on spread
       self._volatility_spread_multiplier = Decimal(str(
           sim_config.get("volatility_spread_multiplier", "1.5")
       ))

       # Load market depth configuration
       self._base_volume = Decimal(str(
           sim_config.get("base_volume", "100.0")
       ))

       # Load volume decay factor for order book simulation
       self._volume_decay_factor = Decimal(str(
           sim_config.get("volume_decay_factor", "0.15")
       ))

       # Price column to use (default to 'close')
       self._price_column = sim_config.get("price_column", "close")

       self.logger.info(
           f"Loaded simulation parameters: spread={self._default_spread_pct}%, "
           f"vol_multiplier={self._volatility_spread_multiplier}",
           source_module=self._source_module
       )
   ```

### Low Priority

1. **Add Volatility-Based Spread Calculation**:
   ```python
   def _calculate_volatility(self, trading_pair: str, lookback_periods: int = 10) -> Optional[Decimal]:
       """Calculates the recent volatility for a trading pair.

       Args:
           trading_pair: The trading pair to calculate volatility for
           lookback_periods: Number of periods to look back

       Returns:
           Volatility as a percentage, or None if cannot be calculated
       """
       if self._current_timestamp is None:
           return None

       if trading_pair not in self.historical_data:
           return None

       try:
           pair_data = self.historical_data[trading_pair]

           # Find the index location of current timestamp
           if self._current_timestamp in pair_data.index:
               current_idx = pair_data.index.get_loc(self._current_timestamp)
           else:
               # Find the nearest timestamp before current time
               nearest_idx = pair_data.index.get_indexer([self._current_timestamp], method='pad')[0]
               if nearest_idx < 0:
                   return None
               current_idx = nearest_idx

           # Get data for volatility calculation
           start_idx = max(0, current_idx - lookback_periods)
           historical_window = pair_data.iloc[start_idx:current_idx+1]["close"]

           if len(historical_window) < 2:
               return None

           # Calculate percentage returns
           returns = historical_window.pct_change().dropna()

           # Calculate volatility as standard deviation of returns
           volatility = Decimal(str(returns.std() * 100))
           return volatility

       except Exception as e:
           self.logger.error(
               f"Error calculating volatility for {trading_pair}: {e}",
               source_module=self._source_module,
               exc_info=True
           )
           return None

   def _adjust_spread_for_volatility(self, trading_pair: str, base_spread_pct: Decimal) -> Decimal:
       """Adjusts the spread based on recent market volatility.

       Args:
           trading_pair: The trading pair
           base_spread_pct: The base spread percentage

       Returns:
           Adjusted spread percentage
       """
       volatility = self._calculate_volatility(trading_pair)
       if volatility is None:
           return base_spread_pct

       # Higher volatility = wider spread, with configurable multiplier
       volatility_factor = 1 + (volatility * self._volatility_spread_multiplier / Decimal("100"))
       adjusted_spread = base_spread_pct * volatility_factor

       # Cap maximum spread to reasonable value
       max_spread_pct = Decimal("2.0")  # 2%
       return min(adjusted_spread, max_spread_pct)
   ```

2. **Implement Support for Multiple Price Types**:
   ```python
   def get_ohlc(self, trading_pair: str) -> Optional[Dict[str, Decimal]]:
       """Gets the OHLC data for a trading pair at the current simulation time.

       Args:
           trading_pair: The trading pair symbol

       Returns:
           Dictionary with OHLC values, or None if unavailable
       """
       if self._current_timestamp is None:
           self.logger.error(
               "Cannot get OHLC: Simulation time not set.",
               source_module=self._source_module
           )
           return None

       pair_data = self.historical_data.get(trading_pair)
       if pair_data is None:
           self.logger.warning(
               f"No historical data found for trading pair: {trading_pair}",
               source_module=self._source_module
           )
           return None

       try:
           # Find the nearest data point
           if self._current_timestamp in pair_data.index:
               # Use exact match
               idx_pos = pair_data.index.get_loc(self._current_timestamp)
               data_point = pair_data.iloc[idx_pos]
           else:
               # Use the latest data point before current time
               timestamp = pair_data.index.asof(self._current_timestamp)
               if timestamp is None:
                   return None
               data_point = pair_data.loc[timestamp]

           # Convert to Decimal dictionary
           ohlc = {
               "open": Decimal(str(data_point["open"])),
               "high": Decimal(str(data_point["high"])),
               "low": Decimal(str(data_point["low"])),
               "close": Decimal(str(data_point["close"])),
               "volume": Decimal(str(data_point["volume"])) if "volume" in data_point else Decimal("0")
           }
           return ohlc

       except Exception as e:
           self.logger.error(
               f"Error retrieving OHLC for {trading_pair}: {e}",
               source_module=self._source_module,
               exc_info=True
           )
           return None
   ```

3. **Add Simulation Status Monitoring**:
   ```python
   def get_simulation_status(self) -> Dict[str, Any]:
       """Gets the current status of the price simulation.

       Returns:
           Dictionary with simulation status information
       """
       status = {
           "current_timestamp": self._current_timestamp,
           "available_pairs": list(self.historical_data.keys()),
           "data_timeframes": {},
           "missing_data_warnings": []
       }

       # For each pair, calculate data coverage
       for pair, df in self.historical_data.items():
           if len(df) > 0:
               status["data_timeframes"][pair] = {
                   "start": df.index[0].isoformat(),
                   "end": df.index[-1].isoformat(),
                   "points": len(df),
                   "interval": self._detect_interval(df)
               }

               # Check for missing data
               if self._has_missing_data(df):
                   status["missing_data_warnings"].append(
                       f"{pair}: Potential gaps detected in historical data"
                   )

       return status

   def _detect_interval(self, df: pd.DataFrame) -> str:
       """Attempts to detect the interval of the dataframe.

       Args:
           df: DataFrane with time-indexed data

       Returns:
           String representing the approximate interval
       """
       if len(df) < 2:
           return "unknown"

       # Get the first few intervals
       intervals = []
       for i in range(min(5, len(df) - 1)):
           diff = df.index[i+1] - df.index[i]
           intervals.append(diff.total_seconds())

       # Calculate average interval
       avg_interval = sum(intervals) / len(intervals)

       # Convert to human-readable format
       if avg_interval < 60:
           return f"{avg_interval:.0f}s"
       elif avg_interval < 3600:
           return f"{avg_interval/60:.0f}m"
       elif avg_interval < 86400:
           return f"{avg_interval/3600:.0f}h"
       else:
           return f"{avg_interval/86400:.0f}d"

   def _has_missing_data(self, df: pd.DataFrame) -> bool:
       """Checks if the dataframe has potential missing data points.

       Args:
           df: DataFrame with time-indexed data

       Returns:
           True if missing data is suspected, False otherwise
       """
       if len(df) < 3:
           return False

       # Calculate common interval
       intervals = [(df.index[i+1] - df.index[i]).total_seconds() for i in range(len(df) - 1)]
       most_common_interval = max(set(intervals), key=intervals.count)

       # Check for intervals significantly larger than the common interval
       for interval in intervals:
           if interval > most_common_interval * 1.5:
               return True

       return False
   ```

## Compliance Assessment

The module partially complies with the architectural requirements:

1. **Interface Compatibility**: While it provides the basic `get_latest_price` functionality, it lacks compatibility with the full interface expected for the `MarketPriceService`, particularly for bid/ask spread simulation and more sophisticated market depth modeling.

2. **Historical Data Handling**: Successfully implements time-based progression and historical data lookup, but does not provide the full range of price data types that might be required.

3. **Integration with Backtesting**: Lacks clear integration points with the `BacktestingEngine` for controlling the simulation and doesn't fully support the requirements specified in FR-1003 for realistic market simulation.

4. **Error Handling**: While the code has basic error handling, it lacks the robustness required for a production-quality simulation component, especially around data validation and error recovery.

5. **Logging**: Uses a module-level logger rather than the injected logger_service pattern used by other components, creating inconsistency in the codebase.

## Follow-up Actions

- [ ] Implement bid/ask spread simulation to more realistically model market conditions
- [ ] Add market depth simulation for proper slippage modeling
- [ ] Replace module-level logger with injected LoggerService for consistency
- [ ] Align interface with the real MarketPriceService for better integration
- [ ] Add configuration options for simulation parameters
- [ ] Implement validation and error handling improvements
- [ ] Add support for multiple price types (OHLC) beyond just 'close'
- [ ] Implement volatility-based spread calculation for more realistic simulation
- [ ] Add simulation status monitoring capabilities
- [ ] Consider implementing market regime simulation for testing strategies under different conditions
