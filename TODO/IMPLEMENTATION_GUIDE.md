# Gal-Friday Trading System: Enterprise Implementation Guide

## Project Overview

**Gal-Friday** is an enterprise-grade algorithmic trading system built in Python that handles real-time market data, portfolio management, risk assessment, and automated trading execution across multiple exchanges. This guide provides comprehensive implementation instructions for the **90 detailed implementation tickets** found in the `TODO/` directory.

## Implementation Ticket Structure

Each TODO ticket follows a standardized format:
- **Context**: File location, line number, and current state
- **Problem Statement**: Detailed description of the issue
- **Proposed Solution**: Enterprise-grade implementation with pseudocode
- **Key Considerations & Dependencies**: Technical requirements and integration points
- **Acceptance Criteria**: Specific deliverables and success metrics

## Technical Architecture

### Core Technology Stack (from pyproject.toml & requirements.txt)
- **Python Version**: 3.11+ (strict requirement)
- **Database**: PostgreSQL with SQLAlchemy 2.0.41 + asyncpg 0.30.0 + Alembic 1.16.1
- **Web Framework**: FastAPI 0.115.12 with WebSockets 15.0.1
- **ML Stack**: XGBoost 3.0.2, scikit-learn 1.6.1, TensorFlow 2.15.0
- **Data Processing**: pandas 2.3.0, NumPy 1.26.4, pandas_ta 0.3.14b0
- **Cloud Services**: Google Cloud (Storage + Secret Manager), AWS (aioboto3)
- **Monitoring**: Rich 14.0.0 for CLI, python-json-logger 3.3.0

### Project Structure Context (All Modules with TODO Tickets)
```
gal_friday/
├── Core Application Files
│   ├── main.py                    (51KB, 1209 lines) - Application entry point
│   ├── config_manager.py          (17KB, 468 lines) - Configuration system
│   ├── exceptions.py              (13KB, 404 lines) - Error handling framework
│   ├── database.py                (3.7KB, 131 lines) - Database connections
│   └── health_check.py            (22KB, 652 lines) - System health monitoring
│
├── Trading & Execution (CRITICAL)
│   ├── execution_handler.py       (127KB, 3182 lines) - Core trading execution
│   ├── simulated_execution_handler.py (108KB, 2574 lines) - Backtesting execution
│   ├── risk_manager.py           (87KB, 1924 lines) - Risk controls & validation
│   └── execution/                 - Execution interfaces and adapters
│
├── Portfolio Management
│   ├── portfolio_manager.py       (49KB, 1252 lines) - Portfolio operations
│   └── portfolio/
│       ├── position_manager.py    (25KB, 513 lines) - Position tracking
│       ├── reconciliation_service.py (33KB, 744 lines) - Trade reconciliation
│       ├── funds_manager.py       (12KB, 324 lines) - Fund management
│       └── valuation_service.py   (30KB, 792 lines) - Portfolio valuation
│
├── Data Processing & ML
│   ├── feature_engine.py          (115KB, 2241 lines) - ML feature processing
│   ├── prediction_service.py      (64KB, 1595 lines) - ML predictions
│   ├── backtesting_engine.py      (56KB, 1407 lines) - Strategy backtesting
│   ├── data_ingestor.py           (70KB, 1969 lines) - Data ingestion
│   └── predictors/                - ML model implementations
│       └── xgboost_predictor/     - XGBoost-specific implementations
│
├── Market Data Services
│   ├── market_price_service.py    (6.4KB, 183 lines) - Real-time pricing
│   ├── simulated_market_price_service.py (63KB, 1650 lines) - Simulated data
│   ├── kraken_historical_data_service.py (58KB, 1520 lines) - Historical data
│   ├── historical_data_service.py (2.4KB, 79 lines) - Data service interface
│   ├── exchange_info_service.py   (2.1KB, 77 lines) - Exchange metadata
│   └── market_price/              - Market data processing modules
│
├── Strategy & Decision Making
│   ├── strategy_arbitrator.py     (43KB, 1019 lines) - Strategy coordination
│   ├── strategy_position_tracker.py (9.4KB, 258 lines) - Position tracking
│   └── interfaces/                - Strategy interfaces
│
├── Infrastructure & Services
│   ├── cli_service.py             (30KB, 869 lines) - Command line interface
│   ├── cli_service_mocks.py       (3.9KB, 136 lines) - CLI testing mocks
│   ├── monitoring_service.py      (72KB, 1739 lines) - System monitoring
│   ├── logger_service.py          (53KB, 1326 lines) - Logging infrastructure
│   └── dashboard_service/         - Real-time dashboard (TODO implementation)
│
├── Core Framework
│   └── core/
│       ├── events.py              (45KB, 1272 lines) - Event system
│       ├── event_store.py         (15KB, 451 lines) - Event persistence
│       ├── pubsub.py              (13KB, 322 lines) - Pub/Sub messaging
│       ├── types.py               (4.5KB, 178 lines) - Core type definitions
│       ├── feature_registry_client.py (12KB, 277 lines) - Feature registry
│       ├── feature_models.py      (3.1KB, 67 lines) - Feature model definitions
│       ├── asset_registry.py      (11KB, 309 lines) - Asset management
│       ├── halt_coordinator.py    (7.5KB, 195 lines) - System halt coordination
│       └── halt_recovery.py       (5.1KB, 149 lines) - Recovery mechanisms
│
├── Data Access Layer (DAL)
│   └── dal/
│       ├── base.py                (11KB, 274 lines) - Database base classes
│       ├── connection_pool.py     (5.2KB, 136 lines) - Connection management
│       ├── influxdb_client.py     (7.1KB, 217 lines) - Time-series database
│       ├── models/                - Database models
│       ├── repositories/          - Data repositories
│       ├── migrations/            - Database migrations
│       └── alembic_env/           - Alembic migration environment
│
├── Utilities & Stubs
│   ├── talib_stubs.py             (1.5KB, 57 lines) - Technical analysis stubs
│   ├── typer_stubs.py             (3.5KB, 145 lines) - CLI framework stubs
│   ├── backtest_historical_data_provider.py (1KB, 39 lines) - Backtest data
│   └── utils/                     - Utility modules
│
└── Model Lifecycle & Training
    ├── model_lifecycle/           - ML model management
    ├── model_training/            - Model training pipelines
    └── models/                    - Trained model storage

TODO Implementation Tickets by Module:
├── TODO/main/                     - Main application improvements (L709, L922, L923)
├── TODO/risk_manager/             - Risk management (8 tickets: L1814-L1917)
├── TODO/execution_handler/        - Execution improvements (5 tickets: L317-L3075)
├── TODO/portfolio_manager/        - Portfolio enhancements (L709)
├── TODO/position_manager/         - Position management (L160, L372)
├── TODO/feature_engine/           - Feature processing (L92, L234)
├── TODO/backtesting_engine/       - Backtesting framework (L89)
├── TODO/prediction_service/       - ML pipeline (L156)
├── TODO/monitoring_service/       - Monitoring enhancements (L234)
├── TODO/dashboard_service/        - Real-time dashboard (L178)
├── TODO/cli_service/              - CLI improvements (L868)
├── TODO/config_validator/         - Configuration validation (L142)
├── TODO/logger_service/           - Logging enhancements (L164)
├── TODO/simulated_market_price_service/ - Market simulation (L173, L298)
├── TODO/websocket_client/         - WebSocket processing (L532, L541)
├── TODO/strategy_arbitrator/      - Strategy coordination (L191, L310)
├── TODO/strategy_interface/       - Strategy framework (L373, L462)
├── TODO/reconciliation_service/   - Trade reconciliation (L613)
├── TODO/reconciliation_repository/ - Reconciliation data (L41, L86)
├── TODO/xgboost_predictor/        - XGBoost implementation (L369-373)
├── TODO/performance_optimizer/    - Performance improvements (L728)
├── TODO/experiment_manager/       - Experiment management (L230)
├── TODO/gap_detector/             - Data gap detection (L216)
├── TODO/fill/                     - Order fill processing (L66)
├── TODO/position/                 - Position modeling (L43)
├── TODO/model_repository/         - Model storage (L103)
├── TODO/feature_registry_client/  - Feature registry (L37)
├── TODO/feature_models/           - Feature modeling (L49)
├── TODO/execution_handler_interface/ - Execution interfaces (L147-148)
├── TODO/cli_service_mocks/        - CLI testing (L46-78)
├── TODO/talib_stubs/              - Technical analysis stubs (L1, L10-56)
└── TODO/typer_stubs/              - CLI framework stubs (L1)
```

## Code Quality Standards (from pyproject.toml)

### Mandatory Requirements
- **Line Length**: 99 characters maximum
- **Type Checking**: `disallow_untyped_defs = true` (mypy strict mode)
- **Test Coverage**: Minimum 80% coverage requirement
- **Linting**: Ruff with comprehensive rule set (E, F, I, W, C90, N, D, UP, B, ANN, S, RUF)
- **Documentation**: Google-style docstrings mandatory

### Type Safety Pattern
```python
# All functions must have complete type annotations
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

async def process_trading_signal(
    self, 
    signal: TradingSignal, 
    risk_params: RiskParameters,
    execution_context: ExecutionContext
) -> ValidationResult:
    """Process trading signal with comprehensive risk validation.
    
    Args:
        signal: The trading signal to validate and execute
        risk_params: Risk parameters for validation
        execution_context: Current market and portfolio context
        
    Returns:
        ValidationResult with approval status and risk metrics
        
    Raises:
        RiskValidationError: If signal violates risk limits
        ExecutionError: If order placement fails
        ConfigurationError: If required configuration is missing
    """
```

## Critical Implementation Tickets by Priority

### Priority 1: Risk Management & Execution (CRITICAL)

#### Risk Manager Module (`gal_friday/risk_manager.py` - 87KB, 1924 lines)

**🎫 TODO/risk_manager/L1814_execution_report_handling.md**
- **Task**: Implement execution report handling for tracking consecutive losses
- **Implementation**: ExecutionReport Pydantic model, risk metrics tracking, state persistence
- **Key Dependencies**: Persistence service, PubSub system, PortfolioManager integration
- **Acceptance Criteria**: Real-time risk metrics, automated threshold responses, system restart survival

**🎫 TODO/risk_manager/L1845_signal_rejection.md**
- **Task**: Signal validation and rejection logic
- **Implementation**: Multi-layer validation pipeline with configurable rules

**🎫 TODO/risk_manager/L1859_signal_approval.md**
- **Task**: Signal approval workflow with risk checks
- **Implementation**: Approval state machine with audit trails

**🎫 TODO/risk_manager/L1870_initial_validation.md**
- **Task**: Initial signal validation before processing
- **Implementation**: Input validation, format checking, basic risk screening

**🎫 TODO/risk_manager/L1882_market_price_checks.md**
- **Task**: Market price validation against risk parameters
- **Implementation**: Real-time price validation, volatility checks, market hours validation

**🎫 TODO/risk_manager/L1907_position_sizing.md**
- **Task**: Position sizing with portfolio limits
- **Implementation**: Kelly criterion, portfolio percentage limits, correlation adjustments

**🎫 TODO/risk_manager/L1917_lot_size_calculation.md**
- **Task**: Lot size calculation with exchange requirements
- **Implementation**: Exchange-specific lot size rules, precision handling

#### Execution Handler Module (`gal_friday/execution_handler.py` - 127KB, 3182 lines)

**🎫 TODO/execution_handler/L317_kraken_adapter.md**
- **Task**: Refactor using Kraken adapter implementing BaseExecutionAdapter interface
- **Implementation**: Exchange abstraction layer, adapter pattern, factory pattern
- **Key Dependencies**: Exchange API libraries, rate limiting, authentication modules
- **Acceptance Criteria**: Multi-exchange support, consistent interface, proper error handling

**🎫 TODO/execution_handler/L1115_exchange_adapter.md**
- **Task**: Exchange adapter abstraction and interface standardization
- **Implementation**: Abstract base class, capability discovery, configuration management

**🎫 TODO/execution_handler/L3075_async_order_processing.md**
- **Task**: Async order processing with WebSocket integration
- **Implementation**: High-performance async processing, WebSocket real-time updates

**🎫 TODO/execution_handler/L550_configurable_cancellation.md**
- **Task**: Configurable order cancellation with safety checks
- **Implementation**: Safety validation, cancellation policies, audit logging

**🎫 TODO/execution_handler/L551_order_state_tracking.md**
- **Task**: Order state tracking with execution reports
- **Implementation**: State machine, transition validation, persistence

### Priority 2: Data Processing & Analytics (HIGH)

#### Feature Engine Module (`gal_friday/feature_engine.py` - 115KB, 2241 lines)

**🎫 TODO/feature_engine/L92_feature_spec_fields.md**
- **Task**: Expand FeatureSpec with multiple outputs
- **Implementation**: Enhanced feature specification, multiple output support

**🎫 TODO/feature_engine/L234_feature_extraction.md**
- **Task**: Robust feature extraction with error handling
- **Implementation**: Malformed data handling, validation, recovery strategies

#### Backtesting Engine Module (`gal_friday/backtesting_engine.py` - 56KB, 1407 lines)

**🎫 TODO/backtesting_engine/L89_backtesting_framework.md**
- **Task**: Comprehensive backtesting framework with performance analytics
- **Implementation**: Vectorized and event-driven modes, performance metrics calculation
- **Key Dependencies**: Historical data services, strategy frameworks, visualization libraries
- **Acceptance Criteria**: Multiple execution modes, comprehensive metrics, benchmarking system

#### Portfolio Management

**🎫 TODO/portfolio_manager/L709_placeholder_trade_history.md**
- **Task**: Replace hardcoded trade history with configurable retrieval
- **Implementation**: Flexible trade history system, filtering, aggregation

**🎫 TODO/position_manager/L160_validation_logic.md**
- **Task**: Position validation logic implementation
- **Implementation**: Position consistency checks, reconciliation

**🎫 TODO/position_manager/L372_trading_pair_split.md**
- **Task**: Trading pair split handling
- **Implementation**: Symbol mapping, split adjustments, historical data correction

### Priority 3: Infrastructure & Services (MEDIUM)

#### Dashboard Service

**🎫 TODO/dashboard_service/L178_real_time_dashboard.md**
- **Task**: Real-time dashboard with live trading metrics
- **Implementation**: WebSocket streaming, portfolio visualization, trading metrics display
- **Key Dependencies**: FastAPI, WebSocket support, JavaScript charting libraries
- **Acceptance Criteria**: Real-time updates, responsive design, multi-user support

#### CLI Service

**🎫 TODO/cli_service/L868_main_guard.md**
- **Task**: Implement main guard with proper CLI service initialization
- **Implementation**: Argument parsing, service lifecycle, health checks

**🎫 TODO/cli_service_mocks/L46-78_placeholder_mock_classes.md**
- **Task**: Replace placeholder mock classes with production implementations
- **Implementation**: Proper test doubles, realistic behavior, comprehensive mocking

#### Configuration & Validation

**🎫 TODO/config_validator/L142_placeholder_warning.md**
- **Task**: Replace placeholder warnings with formal validation errors
- **Implementation**: Structured error reporting, configuration guidance, remediation suggestions

#### Market Data Services

**🎫 TODO/simulated_market_price_service/L298_simulation_engine.md**
- **Task**: Real-time simulation engine with configurable speeds
- **Implementation**: Market simulation, configurable time acceleration, realistic market behavior

**🎫 TODO/gap_detector/L216_interpolation.md**
- **Task**: Price interpolation for gap detection
- **Implementation**: Gap detection algorithms, interpolation strategies, data quality validation

**🎫 TODO/websocket_client/L532_placeholder_order_book.md**
- **Task**: Order book processing implementation
- **Implementation**: Real-time order book updates, depth calculation, market data normalization

## Implementation Patterns from TODO Tickets

### Error Handling Pattern (from risk_manager tickets)
```python
# Pattern from TODO/risk_manager/L1814_execution_report_handling.md
class TradingSystemError(Exception):
    """Base exception for trading system errors"""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc)
        self.correlation_id = self._generate_correlation_id()
```

### Adapter Pattern (from execution_handler tickets)
```python
# Pattern from TODO/execution_handler/L317_kraken_adapter.md
class BaseExecutionAdapter(ABC):
    """Abstract base class for exchange execution adapters"""
    
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place an order on the exchange"""
        pass
    
    @abstractmethod
    async def cancel_order(self, exchange_order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        pass
```

### Configuration Pattern (from config_validator tickets)
```python
# Pattern from TODO/config_validator/L142_placeholder_warning.md
@dataclass
class ValidationError:
    """Structured validation error with guidance"""
    code: str
    severity: ValidationSeverity
    category: ValidationCategory
    title: str
    message: str
    remediation: Optional[str] = None
    documentation_url: Optional[str] = None
```

## Testing Requirements (from TODO acceptance criteria)

### Performance Testing Pattern
```python
# Pattern from multiple TODO tickets
@pytest.mark.asyncio
async def test_risk_validation_performance():
    """Test both correctness AND performance requirements"""
    start_time = time.perf_counter()
    result = await risk_manager.validate_signal(signal)
    duration = time.perf_counter() - start_time
    
    assert result.is_valid
    assert duration < 0.010  # 10ms requirement from TODO tickets
```

### Integration Testing Pattern
```python
# Pattern from execution_handler TODO tickets
@pytest.mark.integration
async def test_end_to_end_trading_workflow():
    """Integration test for complete trading workflow"""
    # Test signal -> risk validation -> execution -> portfolio update
    # Based on acceptance criteria from multiple TODO tickets
```

## Implementation Order (Based on TODO Dependencies)

### Phase 1: Foundation (Week 1-2)
1. **TODO/config_validator/** - Configuration validation framework
2. **TODO/logger_service/** - Enhanced logging infrastructure  
3. **TODO/monitoring_service/** - Observability framework

### Phase 2: Core Trading (Week 3-4)
4. **TODO/risk_manager/** - All 8 risk management tickets (L1814, L1845, L1859, L1870, L1882, L1907, L1917, etc.)
5. **TODO/execution_handler/** - All 5 execution tickets (L317, L1115, L3075, L550, L551)
6. **TODO/portfolio_manager/** - Position management tickets

### Phase 3: Data & Analytics (Week 5-6)
7. **TODO/feature_engine/** - Feature processing tickets (L92, L234)
8. **TODO/backtesting_engine/** - Backtesting framework (L89)
9. **TODO/prediction_service/** - ML pipeline tickets

### Phase 4: Infrastructure (Week 7-8)
10. **TODO/dashboard_service/** - Real-time dashboard (L178)
11. **TODO/cli_service/** - CLI improvements (L868, mock classes)
12. **TODO/websocket_client/** - WebSocket processing

## Success Criteria (Aggregated from all TODO tickets)

### Functional Requirements
- [ ] **Complete TODO Replacement**: All 90 TODO tickets implemented according to their specifications
- [ ] **Type Safety**: Full mypy compliance with strict mode enabled
- [ ] **Error Handling**: Comprehensive exception handling per TODO specifications
- [ ] **Performance**: Meet latency/throughput requirements specified in individual tickets

### Quality Requirements
- [ ] **Test Coverage**: 80%+ coverage with tests specified in TODO acceptance criteria
- [ ] **Documentation**: Complete docstrings following TODO specifications
- [ ] **Security**: Input validation and secure credential handling per TODO requirements

### Integration Requirements
- [ ] **Database**: Proper SQLAlchemy 2.0.41 integration with transactions
- [ ] **WebSocket**: Real-time data streaming per dashboard and websocket TODO tickets
- [ ] **Exchange APIs**: Multi-exchange support per execution_handler TODO tickets

## Getting Started

1. **Review TODO Tickets**: Start by reading the specific TODO tickets for your assigned module
2. **Follow TODO Specifications**: Each ticket contains detailed pseudocode and acceptance criteria
3. **Implement Dependencies First**: Check "Key Considerations & Dependencies" in each TODO ticket
4. **Test Against Acceptance Criteria**: Use the specific criteria listed in each TODO ticket

---

**Note**: This implementation guide references the 90 detailed TODO tickets in the `/TODO` directory. Each ticket contains enterprise-grade specifications with pseudocode, dependencies, and acceptance criteria. Always refer to the specific TODO ticket for detailed implementation requirements rather than implementing generic solutions.
