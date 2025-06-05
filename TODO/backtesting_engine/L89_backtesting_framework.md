# Task: Implement comprehensive backtesting framework with performance analytics.

### 1. Context
- **File:** `gal_friday/backtesting_engine.py`
- **Line:** `89`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing a comprehensive backtesting framework with performance analytics.

### 2. Problem Statement
Without a comprehensive backtesting framework, the system cannot evaluate trading strategies against historical data, measure performance metrics, or validate strategy effectiveness before live deployment. This prevents proper strategy development, risk assessment, and performance optimization.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Backtesting Engine Core:** High-performance backtesting execution engine
2. **Build Performance Analytics:** Comprehensive metrics calculation and analysis
3. **Implement Strategy Evaluation:** Framework for testing multiple strategies
4. **Add Risk Analysis:** Risk metrics, drawdown analysis, and risk-adjusted returns
5. **Create Benchmarking System:** Comparison against market indices and benchmarks
6. **Build Reporting Framework:** Detailed reports with visualizations and insights

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import logging

class BacktestMode(str, Enum):
    """Backtesting execution modes"""
    VECTORIZED = "vectorized"      # Fast vectorized backtesting
    EVENT_DRIVEN = "event_driven"  # Realistic event-driven simulation

@dataclass
class BacktestConfig:
    """Configuration for backtesting runs"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: List[str]
    mode: BacktestMode
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005

@dataclass
class PerformanceMetrics:
    """Performance metrics results"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float

class BacktestingFramework:
    """Enterprise-grade backtesting framework with performance analytics"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data and state management
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.portfolio_value = config.initial_capital
        self.positions: Dict[str, float] = {}
        self.cash = config.initial_capital
        
        # Performance tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        
    async def run_backtest(self, strategy: Any, data_loader: Any) -> Dict[str, Any]:
        """
        Run comprehensive backtest with performance analytics
        Replace TODO with full backtesting framework
        """
        
        try:
            start_time = datetime.now()
            self.logger.info(f"Starting backtest for {strategy.name}")
            self.logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
            self.logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
            
            # Load historical data
            await self._load_historical_data(data_loader)
            
            # Initialize strategy
            await self._initialize_strategy(strategy)
            
            # Execute backtesting based on mode
            if self.config.mode == BacktestMode.VECTORIZED:
                await self._run_vectorized_backtest(strategy)
            elif self.config.mode == BacktestMode.EVENT_DRIVEN:
                await self._run_event_driven_backtest(strategy)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            # Create result object
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'strategy_name': strategy.name,
                'config': self.config,
                'performance_metrics': performance_metrics,
                'portfolio_history': pd.DataFrame(self.portfolio_history),
                'trade_history': pd.DataFrame(self.trade_history),
                'execution_time': execution_time
            }
            
            self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            self.logger.info(f"Total return: {performance_metrics.total_return:.2%}")
            self.logger.info(f"Sharpe ratio: {performance_metrics.sharpe_ratio:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise BacktestError(f"Backtesting failed: {e}")
    
    async def _load_historical_data(self, data_loader: Any) -> None:
        """Load historical data for all symbols"""
        
        self.logger.info("Loading historical data")
        
        for symbol in self.config.symbols:
            data = await data_loader.load_data(
                symbol=symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            
            if data.empty:
                raise BacktestError(f"No data available for {symbol}")
            
            self.historical_data[symbol] = data
            self.logger.debug(f"Loaded {len(data)} data points for {symbol}")
    
    async def _run_event_driven_backtest(self, strategy: Any) -> None:
        """Run event-driven backtesting simulation"""
        
        self.logger.info("Running event-driven backtest")
        
        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for data in self.historical_data.values():
            all_timestamps.update(data.index)
        
        timestamps = sorted(all_timestamps)
        
        for timestamp in timestamps:
            # Create market data snapshot for this timestamp
            market_data = {}
            for symbol, data in self.historical_data.items():
                if timestamp in data.index:
                    market_data[symbol] = data.loc[timestamp].to_dict()
            
            # Update strategy with new market data
            signals = await strategy.on_market_data(timestamp, market_data)
            
            # Process signals and execute trades
            await self._process_signals(signals, timestamp, market_data)
            
            # Update portfolio value
            self._update_portfolio_value(timestamp, market_data)
    
    async def _process_signals(self, signals: List[Dict[str, Any]], 
                             timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Process trading signals and execute trades"""
        
        for signal in signals:
            symbol = signal.get('symbol')
            action = signal.get('action')  # 'buy', 'sell', 'hold'
            quantity = signal.get('quantity', 0)
            
            if action in ['buy', 'sell'] and quantity > 0:
                await self._execute_trade(symbol, action, quantity, timestamp, market_data)
    
    async def _execute_trade(self, symbol: str, action: str, quantity: float,
                           timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Execute a trade with realistic costs"""
        
        if symbol not in market_data:
            self.logger.warning(f"No market data for {symbol} at {timestamp}")
            return
        
        price = market_data[symbol].get('close', 0)
        if price <= 0:
            return
        
        # Apply slippage
        if action == 'buy':
            execution_price = price * (1 + self.config.slippage_rate)
        else:
            execution_price = price * (1 - self.config.slippage_rate)
        
        # Calculate trade value and commission
        trade_value = quantity * execution_price
        commission = trade_value * self.config.commission_rate
        
        # Check if we have enough cash for buy orders
        if action == 'buy':
            total_cost = trade_value + commission
            if total_cost > self.cash:
                self.logger.warning(f"Insufficient cash for {symbol} buy order")
                return
            
            self.cash -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        elif action == 'sell':
            current_position = self.positions.get(symbol, 0)
            if quantity > current_position:
                quantity = current_position  # Sell only what we have
            
            if quantity <= 0:
                return
            
            self.cash += trade_value - commission
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        # Record trade
        self.trade_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': execution_price,
            'commission': commission,
            'trade_value': trade_value
        })
        
        self.logger.debug(f"Executed {action} {quantity} {symbol} at ${execution_price:.2f}")
    
    def _update_portfolio_value(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Update portfolio value based on current market prices"""
        
        position_value = 0
        
        for symbol, quantity in self.positions.items():
            if symbol in market_data and quantity > 0:
                price = market_data[symbol].get('close', 0)
                position_value += quantity * price
        
        total_portfolio_value = self.cash + position_value
        
        # Calculate daily return
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1]['total_value']
            daily_return = (total_portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)
        
        # Record portfolio state
        self.portfolio_history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_portfolio_value,
            'positions': self.positions.copy()
        })
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not self.portfolio_history:
            raise BacktestError("No portfolio history available for metrics calculation")
        
        # Convert to pandas for easier calculations
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('timestamp', inplace=True)
        
        returns = pd.Series(self.daily_returns)
        
        # Basic metrics
        initial_value = self.config.initial_capital
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Time-based metrics
        days = len(portfolio_df)
        years = days / 252  # Trading days per year
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Trade analysis
        trades_df = pd.DataFrame(self.trade_history)
        total_trades = len(trades_df)
        win_rate = 0  # Simplified - would need more complex P&L calculation
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            win_rate=win_rate
        )

class BacktestError(Exception):
    """Exception raised for backtesting errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of missing data; robust error recovery during strategy execution; comprehensive logging
- **Configuration:** Configurable commission and slippage models; adjustable risk parameters; flexible benchmark selection
- **Testing:** Unit tests for performance calculations; integration tests with strategy frameworks; validation against known results
- **Dependencies:** Pandas and NumPy for calculations; data loading services; strategy execution framework; visualization libraries

### 4. Acceptance Criteria
- [ ] Backtesting framework supports multiple execution modes (vectorized, event-driven)
- [ ] Performance analytics calculate comprehensive metrics including Sharpe ratio, drawdown, and risk-adjusted returns
- [ ] Strategy evaluation framework handles multiple strategies and parameter optimization
- [ ] Risk analysis provides detailed drawdown analysis and risk decomposition
- [ ] Benchmark comparison includes alpha, beta, and relative performance metrics
- [ ] Trade execution simulation includes realistic costs, slippage, and market impact
- [ ] Performance optimization handles large datasets efficiently without memory issues
- [ ] Reporting framework generates detailed analysis with statistical significance tests
- [ ] Configuration system allows flexible backtesting scenarios and parameter sensitivity analysis
- [ ] TODO placeholder is completely replaced with production-ready implementation