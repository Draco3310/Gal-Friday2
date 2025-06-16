# Critical Priority Solutions - Gal-Friday Trading Bot

**Document Version:** 1.0  
**Date:** 2025-01-15  
**Purpose:** Detailed implementation solutions for critical priority issues identified in code review

---

## Executive Summary

This document provides precise, actionable solutions for the four critical priority issues that must be resolved before the Gal-Friday trading bot can be deployed to production:

1. **Complete HALT Mechanism Implementation**
2. **Comprehensive Test Suite (80%+ coverage)**
3. **Secure API Credential Management**
4. **Complete Monitoring System**

Each solution includes detailed implementation steps, code examples, and verification procedures.

---

## 1. Complete HALT Mechanism Implementation

### Current State Analysis

The monitoring service has a basic HALT framework but critical gaps exist:
- HALT triggers are not fully connected to monitoring events
- No automated position closure on HALT (configurable behavior exists but not implemented)
- Missing market volatility detection
- No external HALT command interface beyond CLI
- Manual intervention requirement not properly implemented

### Solution Architecture

#### 1.1 HALT Trigger Integration

**Implementation Steps:**

1. **Create Central HALT Coordinator**
```python
# gal_friday/core/halt_coordinator.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from decimal import Decimal
import uuid

@dataclass
class HaltCondition:
    """Represents a condition that can trigger a HALT."""
    condition_id: str
    name: str
    threshold: Any
    current_value: Any
    is_triggered: bool
    timestamp: datetime
    
class HaltCoordinator:
    """Central coordinator for all HALT conditions and triggers."""
    
    def __init__(self, config_manager, pubsub_manager, logger_service):
        self.config = config_manager
        self.pubsub = pubsub_manager
        self.logger = logger_service
        
        # Track all HALT conditions
        self.conditions: Dict[str, HaltCondition] = {}
        self._is_halted = False
        self._halt_reason = ""
        self._halt_source = ""
        self._halt_timestamp: Optional[datetime] = None
        
        # Configure HALT conditions from config
        self._initialize_conditions()
        
    def _initialize_conditions(self):
        """Initialize all HALT conditions from configuration."""
        # Drawdown conditions
        self.register_condition(
            "max_total_drawdown",
            "Maximum Total Drawdown",
            self.config.get_decimal("risk.limits.max_total_drawdown_pct", "15.0")
        )
        self.register_condition(
            "max_daily_drawdown",
            "Maximum Daily Drawdown", 
            self.config.get_decimal("risk.limits.max_daily_drawdown_pct", "2.0")
        )
        self.register_condition(
            "max_consecutive_losses",
            "Maximum Consecutive Losses",
            self.config.get_int("risk.limits.max_consecutive_losses", 5)
        )
        # Market conditions
        self.register_condition(
            "max_volatility",
            "Maximum Market Volatility",
            self.config.get_decimal("monitoring.max_volatility_threshold", "5.0")
        )
        # System conditions
        self.register_condition(
            "api_error_rate",
            "API Error Rate Threshold",
            self.config.get_int("monitoring.max_api_errors_per_minute", 10)
        )
        self.register_condition(
            "data_staleness",
            "Market Data Staleness",
            self.config.get_int("monitoring.max_data_staleness_seconds", 60)
        )
```

2. **Enhance Monitoring Service Integration**
```python
# Modifications to monitoring_service.py
async def _check_all_halt_conditions(self) -> None:
    """Comprehensive check of all HALT conditions."""
    
    # 1. Drawdown checks
    await self._check_drawdown_conditions()
    
    # 2. Market volatility checks
    await self._check_market_volatility()
    
    # 3. System health checks
    await self._check_system_health()
    
    # 4. API connectivity checks
    await self._check_api_connectivity()
    
    # 5. Data freshness checks
    await self._check_market_data_freshness()
    
    # 6. Position risk checks
    await self._check_position_risk()

async def _check_market_volatility(self) -> None:
    """Check market volatility and trigger HALT if needed."""
    for pair in self.config.get_list("trading.pairs", []):
        # Calculate rolling volatility
        volatility = await self._calculate_volatility(pair)
        
        if volatility is not None:
            threshold = self.config.get_decimal(
                "monitoring.max_volatility_threshold", "5.0"
            )
            
            if volatility > threshold:
                reason = (
                    f"Market volatility for {pair} ({volatility:.2f}%) "
                    f"exceeds threshold ({threshold}%)"
                )
                await self.trigger_halt(
                    reason=reason,
                    source="AUTO: Market Volatility"
                )
                break
```

#### 1.2 Automated Position Closure

**Implementation:**

1. **Position Close Command Handler**
```python
# gal_friday/execution_handler.py additions
async def handle_close_position_command(self, event: ClosePositionCommand) -> None:
    """Handle emergency position closure during HALT."""
    try:
        self.logger.warning(
            f"Processing emergency position closure for {event.trading_pair}",
            source_module=self._source_module,
            context={
                "quantity": str(event.quantity),
                "side": event.side,
                "reason": "HALT triggered"
            }
        )
        
        # Create market order for immediate execution
        order_params = {
            "pair": self._get_kraken_pair_name(event.trading_pair),
            "type": event.side.lower(),
            "ordertype": "market",
            "volume": str(event.quantity),
            "validate": False  # Skip validation for emergency orders
        }
        
        # Add emergency flag for special handling
        order_params["userref"] = "HALT_CLOSE"
        
        # Execute order with priority handling
        result = await self._place_order_with_priority(order_params)
        
        if result and not result.get("error"):
            self.logger.info(
                f"Emergency close order placed successfully",
                source_module=self._source_module,
                context={"order_id": result.get("txid")}
            )
        else:
            self.logger.critical(
                f"Failed to place emergency close order",
                source_module=self._source_module,
                context={"error": result.get("error") if result else "No response"}
            )
            
    except Exception as e:
        self.logger.critical(
            "Critical error during emergency position closure",
            source_module=self._source_module,
            exc_info=True
        )
```

2. **Priority Order Placement**
```python
async def _place_order_with_priority(self, params: dict) -> dict:
    """Place order with priority handling for emergencies."""
    # Bypass normal rate limiting for emergency orders
    if params.get("userref") == "HALT_CLOSE":
        # Direct placement without rate limit wait
        return await self._make_private_request("/0/private/AddOrder", params)
    else:
        # Normal rate-limited placement
        await self.rate_limiter.wait_for_private_capacity()
        return await self._make_private_request("/0/private/AddOrder", params)
```

#### 1.3 External HALT Interface

**Implementation:**

1. **REST API Endpoint**
```python
# gal_friday/api/halt_endpoints.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class HaltRequest(BaseModel):
    reason: str
    source: str = "External API"
    
class HaltStatus(BaseModel):
    is_halted: bool
    halt_reason: Optional[str]
    halt_timestamp: Optional[datetime]
    halt_source: Optional[str]

@app.post("/api/v1/halt")
async def trigger_halt(
    request: HaltRequest,
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
):
    """Trigger system HALT via API."""
    try:
        await monitoring_service.trigger_halt(
            reason=request.reason,
            source=request.source
        )
        return {"status": "success", "message": "HALT triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/resume")
async def resume_trading(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
):
    """Resume trading after HALT."""
    try:
        await monitoring_service.trigger_resume(source="External API")
        return {"status": "success", "message": "Trading resumed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/halt/status")
async def get_halt_status(
    monitoring_service: MonitoringService = Depends(get_monitoring_service)
) -> HaltStatus:
    """Get current HALT status."""
    return HaltStatus(
        is_halted=monitoring_service.is_halted(),
        halt_reason=monitoring_service.get_halt_reason(),
        halt_timestamp=monitoring_service.get_halt_timestamp(),
        halt_source=monitoring_service.get_halt_source()
    )
```

2. **WebSocket HALT Notifications**
```python
# gal_friday/api/websocket_handler.py
@app.websocket("/ws/halt-status")
async def halt_status_websocket(websocket: WebSocket):
    """Real-time HALT status updates via WebSocket."""
    await websocket.accept()
    
    # Subscribe to system state changes
    async def handle_state_change(event):
        await websocket.send_json({
            "type": "halt_status",
            "is_halted": event.new_state == "HALTED",
            "state": event.new_state,
            "reason": event.reason,
            "timestamp": event.timestamp.isoformat()
        })
    
    pubsub.subscribe(EventType.SYSTEM_STATE_CHANGE, handle_state_change)
    
    try:
        # Keep connection alive
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pubsub.unsubscribe(EventType.SYSTEM_STATE_CHANGE, handle_state_change)
```

#### 1.4 Manual Intervention Requirements

**Implementation:**

1. **HALT Recovery Checklist**
```python
# gal_friday/core/halt_recovery.py
@dataclass
class RecoveryCheckItem:
    """Single item in recovery checklist."""
    item_id: str
    description: str
    is_completed: bool = False
    completed_by: Optional[str] = None
    completed_at: Optional[datetime] = None
    
class HaltRecoveryManager:
    """Manages the recovery process after a HALT."""
    
    def __init__(self, config_manager, logger_service):
        self.config = config_manager
        self.logger = logger_service
        self.checklist: List[RecoveryCheckItem] = []
        self._initialize_checklist()
        
    def _initialize_checklist(self):
        """Initialize recovery checklist based on HALT reason."""
        self.checklist = [
            RecoveryCheckItem(
                "review_halt_reason",
                "Review and understand the HALT trigger reason"
            ),
            RecoveryCheckItem(
                "check_market_conditions",
                "Verify current market conditions are acceptable"
            ),
            RecoveryCheckItem(
                "review_positions",
                "Review all open positions and their P&L"
            ),
            RecoveryCheckItem(
                "verify_api_connectivity",
                "Confirm API connectivity to exchange"
            ),
            RecoveryCheckItem(
                "check_account_balance",
                "Verify account balance matches expectations"
            ),
            RecoveryCheckItem(
                "review_risk_parameters",
                "Review and potentially adjust risk parameters"
            ),
            RecoveryCheckItem(
                "confirm_resume",
                "Confirm decision to resume trading"
            )
        ]
    
    def get_incomplete_items(self) -> List[RecoveryCheckItem]:
        """Get list of incomplete checklist items."""
        return [item for item in self.checklist if not item.is_completed]
    
    def complete_item(self, item_id: str, completed_by: str) -> bool:
        """Mark a checklist item as complete."""
        for item in self.checklist:
            if item.item_id == item_id:
                item.is_completed = True
                item.completed_by = completed_by
                item.completed_at = datetime.now(UTC)
                return True
        return False
    
    def is_recovery_complete(self) -> bool:
        """Check if all recovery items are complete."""
        return all(item.is_completed for item in self.checklist)
```

2. **Enhanced CLI for Recovery**
```python
# Addition to cli_service.py
@app.command()
def recovery_status():
    """Show HALT recovery checklist status."""
    if not cli.recovery_manager:
        print("Recovery manager not initialized")
        return
        
    table = Table(title="HALT Recovery Checklist")
    table.add_column("Status", style="cyan")
    table.add_column("Item", style="white")
    table.add_column("Completed By", style="green")
    
    for item in cli.recovery_manager.checklist:
        status = "✓" if item.is_completed else "✗"
        completed_by = item.completed_by or "-"
        table.add_row(status, item.description, completed_by)
    
    console.print(table)
    
    if cli.recovery_manager.is_recovery_complete():
        console.print("\n[green]All recovery items complete. Safe to resume.[/green]")
    else:
        incomplete = len(cli.recovery_manager.get_incomplete_items())
        console.print(f"\n[yellow]{incomplete} items remaining.[/yellow]")

@app.command()
def complete_recovery_item(
    item_id: str,
    completed_by: str = typer.Option(..., prompt="Your name")
):
    """Mark a recovery checklist item as complete."""
    if cli.recovery_manager.complete_item(item_id, completed_by):
        print(f"✓ Item '{item_id}' marked complete by {completed_by}")
    else:
        print(f"✗ Item '{item_id}' not found")
```

### Verification Procedures

1. **Unit Tests for HALT Mechanism**
```python
# tests/test_halt_mechanism.py
import pytest
from datetime import datetime
from decimal import Decimal

@pytest.mark.asyncio
async def test_halt_trigger_on_max_drawdown():
    """Test HALT triggers correctly on max drawdown breach."""
    # Setup
    monitoring = create_test_monitoring_service()
    portfolio_state = {
        "total_drawdown_pct": Decimal("16.0"),  # Exceeds 15% limit
        "initial_equity": Decimal("100000"),
        "current_equity": Decimal("84000")
    }
    
    # Execute
    await monitoring._check_drawdown_conditions(portfolio_state)
    
    # Verify
    assert monitoring.is_halted()
    assert "drawdown" in monitoring.get_halt_reason().lower()

@pytest.mark.asyncio
async def test_position_closure_on_halt():
    """Test positions are closed when HALT behavior is 'close'."""
    # Setup
    monitoring = create_test_monitoring_service(halt_behavior="close")
    execution = create_test_execution_handler()
    
    # Mock open positions
    positions = {
        "XRP/USD": {"quantity": "1000", "side": "BUY"},
        "DOGE/USD": {"quantity": "-5000", "side": "SELL"}
    }
    
    # Execute
    await monitoring.trigger_halt("Test HALT", "TEST")
    
    # Verify close commands were published
    close_commands = get_published_events(ClosePositionCommand)
    assert len(close_commands) == 2
    assert any(cmd.trading_pair == "XRP/USD" for cmd in close_commands)
    assert any(cmd.trading_pair == "DOGE/USD" for cmd in close_commands)
```

### Implementation Timeline

1. **Week 1**: Implement HaltCoordinator and enhance monitoring integration
2. **Week 2**: Complete automated position closure mechanism
3. **Week 3**: Add external API interface and WebSocket notifications
4. **Week 4**: Implement recovery checklist and manual intervention tools
5. **Week 5**: Comprehensive testing and documentation

---

## 2. Comprehensive Test Suite Implementation (80%+ Coverage)

### Current State Analysis

- No unit tests exist (0% coverage)
- Testing framework is configured but unused
- No integration or performance tests
- No test data fixtures or mocks

### Solution Architecture

#### 2.1 Test Structure and Organization

**Directory Structure:**
```
tests/
├── unit/
│   ├── core/
│   │   ├── test_events.py
│   │   ├── test_pubsub.py
│   │   └── test_asset_registry.py
│   ├── test_data_ingestor.py
│   ├── test_feature_engine.py
│   ├── test_prediction_service.py
│   ├── test_strategy_arbitrator.py
│   ├── test_risk_manager.py
│   ├── test_execution_handler.py
│   ├── test_portfolio_manager.py
│   ├── test_monitoring_service.py
│   └── test_logger_service.py
├── integration/
│   ├── test_data_flow.py
│   ├── test_signal_lifecycle.py
│   ├── test_halt_scenarios.py
│   └── test_error_recovery.py
├── performance/
│   ├── test_latency.py
│   ├── test_throughput.py
│   └── test_resource_usage.py
├── fixtures/
│   ├── market_data.py
│   ├── models.py
│   └── configs.py
└── conftest.py
```

#### 2.2 Core Test Fixtures

**conftest.py:**
```python
# tests/conftest.py
import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, UTC
from typing import Dict, Any
import uuid

from gal_friday.config_manager import ConfigManager
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "trading": {
            "pairs": ["XRP/USD", "DOGE/USD"],
            "mode": "paper"
        },
        "risk": {
            "limits": {
                "max_total_drawdown_pct": 15.0,
                "max_daily_drawdown_pct": 2.0,
                "max_consecutive_losses": 5
            },
            "sizing": {
                "risk_per_trade_pct": 0.5
            }
        },
        "monitoring": {
            "check_interval_seconds": 1,  # Fast for tests
            "max_data_staleness_seconds": 5
        }
    }

@pytest.fixture
async def pubsub_manager():
    """Create a test PubSubManager instance."""
    pubsub = PubSubManager()
    await pubsub.start()
    yield pubsub
    await pubsub.stop()

@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    class MockLogger:
        def __init__(self):
            self.messages = []
            
        def log(self, level, message, **kwargs):
            self.messages.append({
                "level": level,
                "message": message,
                "kwargs": kwargs
            })
            
        def info(self, message, **kwargs):
            self.log("INFO", message, **kwargs)
            
        def warning(self, message, **kwargs):
            self.log("WARNING", message, **kwargs)
            
        def error(self, message, **kwargs):
            self.log("ERROR", message, **kwargs)
            
        def critical(self, message, **kwargs):
            self.log("CRITICAL", message, **kwargs)
            
    return MockLogger()

@pytest.fixture
def sample_market_data():
    """Provide sample market data for tests."""
    return {
        "XRP/USD": {
            "bids": [
                (Decimal("0.5000"), Decimal("1000")),
                (Decimal("0.4999"), Decimal("2000")),
                (Decimal("0.4998"), Decimal("1500"))
            ],
            "asks": [
                (Decimal("0.5001"), Decimal("1000")),
                (Decimal("0.5002"), Decimal("2000")),
                (Decimal("0.5003"), Decimal("1500"))
            ],
            "timestamp": datetime.now(UTC)
        }
    }
```

#### 2.3 Unit Test Examples

**1. Data Ingestor Tests:**
```python
# tests/unit/test_data_ingestor.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
import json
from decimal import Decimal

from gal_friday.data_ingestor import DataIngestor
from gal_friday.core.events import MarketDataL2Event

class TestDataIngestor:
    """Test suite for DataIngestor."""
    
    @pytest.fixture
    def data_ingestor(self, test_config, pubsub_manager, mock_logger):
        """Create DataIngestor instance for testing."""
        config = Mock()
        config.get.side_effect = lambda k, default=None: test_config.get(k, default)
        
        return DataIngestor(
            config=config,
            pubsub_manager=pubsub_manager,
            logger_service=mock_logger
        )
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, data_ingestor):
        """Test WebSocket connection establishment."""
        with patch('websockets.connect') as mock_connect:
            mock_ws = AsyncMock()
            mock_connect.return_value = mock_ws
            
            success = await data_ingestor._establish_connection()
            
            assert success
            assert mock_connect.called
            assert data_ingestor._connection is not None
    
    @pytest.mark.asyncio
    async def test_order_book_parsing(self, data_ingestor):
        """Test L2 order book message parsing."""
        # Sample Kraken book message
        book_msg = {
            "channel": "book",
            "type": "update",
            "data": [{
                "symbol": "XRP/USD",
                "bids": [["0.5000", "1000.0"], ["0.4999", "2000.0"]],
                "asks": [["0.5001", "1000.0"], ["0.5002", "2000.0"]],
                "checksum": 123456789
            }]
        }
        
        # Process message
        await data_ingestor._process_message(json.dumps(book_msg))
        
        # Verify internal book state
        book_state = data_ingestor._l2_books.get("XRP/USD")
        assert book_state is not None
        assert len(book_state["bids"]) == 2
        assert Decimal("0.5000") in book_state["bids"]
    
    @pytest.mark.asyncio
    async def test_reconnection_logic(self, data_ingestor):
        """Test automatic reconnection on connection loss."""
        with patch('websockets.connect') as mock_connect:
            # First connection fails
            mock_connect.side_effect = [
                ConnectionError("Connection failed"),
                AsyncMock()  # Second attempt succeeds
            ]
            
            # Should retry and succeed
            success = await data_ingestor._reconnect_with_backoff()
            
            assert success
            assert mock_connect.call_count == 2
    
    @pytest.mark.asyncio
    async def test_checksum_validation(self, data_ingestor):
        """Test order book checksum validation."""
        # Set up book state
        data_ingestor._l2_books["XRP/USD"] = {
            "bids": {Decimal("0.5000"): Decimal("1000")},
            "asks": {Decimal("0.5001"): Decimal("1000")},
            "checksum": None
        }
        
        # Calculate checksum
        checksum = data_ingestor._calculate_book_checksum(
            data_ingestor._l2_books["XRP/USD"]
        )
        
        assert checksum is not None
        assert isinstance(checksum, int)
```

**2. Risk Manager Tests:**
```python
# tests/unit/test_risk_manager.py
import pytest
from decimal import Decimal
from uuid import uuid4

from gal_friday.risk_manager import RiskManager
from gal_friday.core.events import TradeSignalProposedEvent

class TestRiskManager:
    """Test suite for RiskManager."""
    
    @pytest.fixture
    def risk_manager(self, test_config, pubsub_manager, mock_logger):
        """Create RiskManager instance for testing."""
        portfolio_manager = Mock()
        portfolio_manager.get_current_state.return_value = {
            "current_equity": Decimal("100000"),
            "total_drawdown_pct": Decimal("1.5"),
            "positions": {}
        }
        
        market_price_service = Mock()
        market_price_service.get_latest_price = AsyncMock(
            return_value=Decimal("0.5000")
        )
        
        exchange_info_service = Mock()
        exchange_info_service.get_tick_size = Mock(
            return_value=Decimal("0.0001")
        )
        
        return RiskManager(
            config=test_config,
            pubsub_manager=pubsub_manager,
            portfolio_manager=portfolio_manager,
            logger_service=mock_logger,
            market_price_service=market_price_service,
            exchange_info_service=exchange_info_service
        )
    
    @pytest.mark.asyncio
    async def test_position_sizing_calculation(self, risk_manager):
        """Test correct position size calculation."""
        # Create trade signal
        signal = TradeSignalProposedEvent(
            signal_id=uuid4(),
            trading_pair="XRP/USD",
            side="BUY",
            entry_type="LIMIT",
            proposed_entry_price="0.5000",
            proposed_sl_price="0.4900",
            proposed_tp_price="0.5200"
        )
        
        # Calculate position size
        # Risk = 0.5% of 100k = $500
        # Stop distance = 0.5000 - 0.4900 = 0.0100
        # Position size = 500 / 0.0100 = 50,000 XRP
        
        result = risk_manager._calculate_position_size(
            signal,
            Decimal("100000"),
            Decimal("0.5000"),
            Decimal("0.4900")
        )
        
        assert result.is_valid
        assert result.quantity == Decimal("50000")
    
    @pytest.mark.asyncio
    async def test_drawdown_limit_check(self, risk_manager):
        """Test drawdown limit enforcement."""
        # Set high drawdown
        risk_manager._portfolio_manager.get_current_state.return_value = {
            "current_equity": Decimal("85000"),
            "initial_equity": Decimal("100000"),
            "total_drawdown_pct": Decimal("15.0")  # At limit
        }
        
        # Check should trigger
        should_halt = await risk_manager._check_drawdown_limits({})
        
        assert should_halt
        assert "drawdown" in risk_manager._last_rejection_reason.lower()
    
    @pytest.mark.asyncio
    async def test_fat_finger_protection(self, risk_manager):
        """Test fat finger price validation."""
        # Signal with price far from market
        signal = TradeSignalProposedEvent(
            signal_id=uuid4(),
            trading_pair="XRP/USD",
            side="BUY",
            entry_type="LIMIT",
            proposed_entry_price="0.6000",  # 20% above market
            proposed_sl_price="0.5900"
        )
        
        # Market price is 0.5000
        is_valid = await risk_manager._validate_fat_finger(
            signal,
            Decimal("0.5000")
        )
        
        assert not is_valid
```

**3. Integration Test Example:**
```python
# tests/integration/test_signal_lifecycle.py
import pytest
from decimal import Decimal
import asyncio

class TestSignalLifecycle:
    """Test complete signal flow from prediction to execution."""
    
    @pytest.mark.asyncio
    async def test_full_signal_flow(self, integrated_system):
        """Test signal flows correctly through all components."""
        # 1. Inject market data
        await integrated_system.inject_market_data({
            "XRP/USD": {
                "bids": [(Decimal("0.5000"), Decimal("1000"))],
                "asks": [(Decimal("0.5001"), Decimal("1000"))]
            }
        })
        
        # 2. Inject prediction
        await integrated_system.inject_prediction({
            "XRP/USD": {
                "probability_up": 0.75,  # Strong buy signal
                "model_id": "test_model"
            }
        })
        
        # 3. Wait for signal propagation
        await asyncio.sleep(0.1)
        
        # 4. Verify signal was generated
        signals = integrated_system.get_generated_signals()
        assert len(signals) == 1
        assert signals[0].side == "BUY"
        
        # 5. Verify risk checks passed
        approved_signals = integrated_system.get_approved_signals()
        assert len(approved_signals) == 1
        
        # 6. Verify order was placed
        orders = integrated_system.get_placed_orders()
        assert len(orders) == 1
        assert orders[0].pair == "XRP/USD"
```

#### 2.4 Test Coverage Strategy

**1. Coverage Configuration:**
```toml
# pyproject.toml.old additions
[tool.coverage.run]
source = ["gal_friday"]
omit = [
    "tests/*",
    "*/__init__.py",
    "gal_friday/talib_stubs.py",
    "gal_friday/typer_stubs.py"
]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "pass"
]
precision = 2
fail_under = 80
```

**2. Test Execution Script:**
```bash
#!/bin/bash
# scripts/run_tests.sh

echo "Running Gal-Friday Test Suite"
echo "============================="

# Run unit tests with coverage
echo "Running unit tests..."
pytest tests/unit/ -v --cov=gal_friday --cov-report=term-missing

# Run integration tests
echo "Running integration tests..."
pytest tests/integration/ -v --cov-append

# Run performance tests (optional)
if [ "$1" == "--perf" ]; then
    echo "Running performance tests..."
    pytest tests/performance/ -v
fi

# Generate HTML coverage report
coverage html

# Check coverage threshold
coverage report --fail-under=80

echo "Test suite complete. Coverage report available in htmlcov/index.html"
```

#### 2.5 Mock Framework

**1. Exchange Mock:**
```python
# tests/fixtures/mock_exchange.py
class MockKrakenAPI:
    """Mock Kraken API for testing."""
    
    def __init__(self):
        self.orders = {}
        self.balances = {
            "USD": Decimal("100000"),
            "XRP": Decimal("0"),
            "DOGE": Decimal("0")
        }
        self.order_counter = 0
        
    async def add_order(self, params):
        """Mock order placement."""
        order_id = f"TEST-{self.order_counter}"
        self.order_counter += 1
        
        self.orders[order_id] = {
            "status": "open",
            "pair": params["pair"],
            "type": params["type"],
            "ordertype": params["ordertype"],
            "volume": params["volume"],
            "price": params.get("price"),
            "timestamp": datetime.now(UTC)
        }
        
        return {"error": [], "result": {"txid": [order_id]}}
    
    async def query_orders(self, txid):
        """Mock order query."""
        if txid in self.orders:
            return {"error": [], "result": {txid: self.orders[txid]}}
        return {"error": ["Order not found"]}
```

**2. Model Mock:**
```python
# tests/fixtures/mock_models.py
class MockPredictionModel:
    """Mock ML model for testing."""
    
    def __init__(self, default_prediction=0.5):
        self.default_prediction = default_prediction
        self.prediction_count = 0
        
    def predict(self, features):
        """Generate mock predictions."""
        self.prediction_count += 1
        
        # Add some variation based on features
        base = self.default_prediction
        if "rsi" in features and features["rsi"] < 30:
            base += 0.1  # Oversold, likely to go up
        elif "rsi" in features and features["rsi"] > 70:
            base -= 0.1  # Overbought, likely to go down
            
        return min(max(base, 0.0), 1.0)
```

### Verification and Metrics

**1. Coverage Targets by Module:**
- Core modules (events, pubsub): 95%+
- Business logic (risk, strategy): 90%+
- External interfaces (execution, data): 80%+
- Utilities and helpers: 70%+

**2. Test Execution Time Targets:**
- Unit tests: < 30 seconds
- Integration tests: < 2 minutes
- Full suite: < 5 minutes

**3. Continuous Integration:**
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests with coverage
        run: |
          pytest --cov=gal_friday --cov-report=xml --cov-report=term
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

---

## 3. Secure API Credential Management

### Current State Analysis

- API keys stored in plain text configuration files
- No encryption at rest
- No secrets management integration
- Keys accessible to anyone with file system access
- No key rotation mechanism

### Solution Architecture

#### 3.1 Multi-Layer Security Approach

**Security Layers:**
1. Environment variables for development
2. AWS Secrets Manager / HashiCorp Vault for production
3. Encrypted configuration files as fallback
4. Runtime memory protection

#### 3.2 Implementation Details

**1. Secrets Manager Integration:**
```python
# gal_friday/security/secrets_manager.py
import os
import json
import base64
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import boto3
from botocore.exceptions import ClientError

class SecretsManager:
    """Secure management of API credentials and sensitive data."""
    
    def __init__(self, config_manager, logger_service):
        self.config = config_manager
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        
        # Determine secrets backend
        self.backend = self._determine_backend()
        self._secrets_cache: Dict[str, Any] = {}
        self._encryption_key: Optional[bytes] = None
        
    def _determine_backend(self) -> str:
        """Determine which secrets backend to use."""
        # Priority order: AWS > Vault > Encrypted File > Environment
        if os.getenv("AWS_SECRETS_MANAGER_ENABLED") == "true":
            return "aws"
        elif os.getenv("VAULT_ADDR"):
            return "vault"
        elif os.path.exists(".secrets.enc"):
            return "encrypted_file"
        else:
            return "environment"
    
    async def get_api_credentials(self, exchange: str) -> Dict[str, str]:
        """Retrieve API credentials for specified exchange."""
        cache_key = f"{exchange}_credentials"
        
        # Check cache first
        if cache_key in self._secrets_cache:
            return self._secrets_cache[cache_key]
        
        # Retrieve based on backend
        if self.backend == "aws":
            creds = await self._get_from_aws(f"gal-friday/{exchange}")
        elif self.backend == "vault":
            creds = await self._get_from_vault(f"secret/gal-friday/{exchange}")
        elif self.backend == "encrypted_file":
            creds = await self._get_from_encrypted_file(exchange)
        else:
            creds = self._get_from_environment(exchange)
        
        # Validate credentials
        if not self._validate_credentials(creds):
            raise ValueError(f"Invalid credentials retrieved for {exchange}")
        
        # Cache in memory (with encryption)
        self._secrets_cache[cache_key] = self._encrypt_in_memory(creds)
        
        return creds
    
    async def _get_from_aws(self, secret_name: str) -> Dict[str, str]:
        """Retrieve secrets from AWS Secrets Manager."""
        try:
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId=secret_name)
            
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            else:
                # Binary secret
                decoded = base64.b64decode(response['SecretBinary'])
                return json.loads(decoded)
                
        except ClientError as e:
            self.logger.error(
                f"Failed to retrieve secret from AWS: {e}",
                source_module=self._source_module
            )
            raise
    
    async def _get_from_vault(self, path: str) -> Dict[str, str]:
        """Retrieve secrets from HashiCorp Vault."""
        import hvac
        
        try:
            client = hvac.Client(
                url=os.getenv("VAULT_ADDR"),
                token=os.getenv("VAULT_TOKEN")
            )
            
            response = client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data']
            
        except Exception as e:
            self.logger.error(
                f"Failed to retrieve secret from Vault: {e}",
                source_module=self._source_module
            )
            raise
    
    async def _get_from_encrypted_file(self, exchange: str) -> Dict[str, str]:
        """Retrieve secrets from encrypted local file."""
        try:
            # Load encryption key from environment
            key = self._get_or_create_encryption_key()
            fernet = Fernet(key)
            
            # Read encrypted file
            with open('.secrets.enc', 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt
            decrypted = fernet.decrypt(encrypted_data)
            all_secrets = json.loads(decrypted.decode())
            
            return all_secrets.get(exchange, {})
            
        except Exception as e:
            self.logger.error(
                f"Failed to read encrypted secrets file: {e}",
                source_module=self._source_module
            )
            raise
    
    def _get_from_environment(self, exchange: str) -> Dict[str, str]:
        """Retrieve secrets from environment variables."""
        prefix = f"{exchange.upper()}_"
        return {
            "api_key": os.getenv(f"{prefix}API_KEY", ""),
            "api_secret": os.getenv(f"{prefix}API_SECRET", "")
        }
    
    def _validate_credentials(self, creds: Dict[str, str]) -> bool:
        """Validate credential format and content."""
        if not creds.get("api_key") or not creds.get("api_secret"):
            return False
        
        # Basic format validation
        api_key = creds["api_key"]
        api_secret = creds["api_secret"]
        
        # Check minimum lengths
        if len(api_key) < 20 or len(api_secret) < 40:
            return False
        
        # Verify base64 encoding for secret
        try:
            base64.b64decode(api_secret)
        except Exception:
            return False
        
        return True
    
    def _encrypt_in_memory(self, data: Dict[str, str]) -> Dict[str, bytes]:
        """Encrypt sensitive data in memory."""
        if not self._encryption_key:
            self._encryption_key = Fernet.generate_key()
        
        fernet = Fernet(self._encryption_key)
        return {
            k: fernet.encrypt(v.encode()) for k, v in data.items()
        }
    
    def _decrypt_from_memory(self, encrypted: Dict[str, bytes]) -> Dict[str, str]:
        """Decrypt sensitive data from memory."""
        if not self._encryption_key:
            raise ValueError("No encryption key available")
        
        fernet = Fernet(self._encryption_key)
        return {
            k: fernet.decrypt(v).decode() for k, v in encrypted.items()
        }
```

**2. Secure Configuration Loader:**
```python
# gal_friday/config_manager.py modifications
class SecureConfigManager(ConfigManager):
    """Enhanced config manager with security features."""
    
    def __init__(self, config_path: str, secrets_manager: SecretsManager, **kwargs):
        super().__init__(config_path, **kwargs)
        self.secrets_manager = secrets_manager
        self._sensitive_keys = {
            "kraken.api_key",
            "kraken.secret_key",
            "database.password",
            "influxdb.token"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with automatic secret resolution."""
        # Check if this is a sensitive key
        if key in self._sensitive_keys:
            # Retrieve from secrets manager
            try:
                if key.startswith("kraken."):
                    creds = asyncio.run(
                        self.secrets_manager.get_api_credentials("kraken")
                    )
                    if key == "kraken.api_key":
                        return creds["api_key"]
                    elif key == "kraken.secret_key":
                        return creds["api_secret"]
                elif key == "database.password":
                    db_creds = asyncio.run(
                        self.secrets_manager.get_database_credentials()
                    )
                    return db_creds["password"]
            except Exception as e:
                self.logger.error(
                    f"Failed to retrieve secret for {key}: {e}",
                    source_module=self._source_module
                )
                return default
        
        # Non-sensitive keys use normal config
        return super().get(key, default)
```

**3. Credential Rotation Support:**
```python
# gal_friday/security/credential_rotator.py
import asyncio
from datetime import datetime, timedelta, UTC
from typing import Optional

class CredentialRotator:
    """Manages automatic credential rotation."""
    
    def __init__(
        self, 
        secrets_manager: SecretsManager,
        execution_handler: ExecutionHandler,
        logger_service: LoggerService
    ):
        self.secrets = secrets_manager
        self.execution = execution_handler
        self.logger = logger_service
        self._rotation_task: Optional[asyncio.Task] = None
        self._last_rotation: Optional[datetime] = None
        
    async def start(self):
        """Start the credential rotation monitor."""
        self._rotation_task = asyncio.create_task(
            self._rotation_monitor()
        )
        self.logger.info(
            "Credential rotation monitor started",
            source_module=self.__class__.__name__
        )
    
    async def stop(self):
        """Stop the credential rotation monitor."""
        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass
    
    async def _rotation_monitor(self):
        """Monitor and trigger credential rotation."""
        while True:
            try:
                # Check rotation schedule (e.g., every 90 days)
                if self._should_rotate():
                    await self._perform_rotation()
                
                # Check for compromise indicators
                if await self._check_compromise_indicators():
                    self.logger.critical(
                        "Potential credential compromise detected! Triggering rotation.",
                        source_module=self.__class__.__name__
                    )
                    await self._perform_emergency_rotation()
                
                # Sleep for 1 hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(
                    f"Error in credential rotation monitor: {e}",
                    source_module=self.__class__.__name__
                )
    
    def _should_rotate(self) -> bool:
        """Check if credentials should be rotated."""
        if not self._last_rotation:
            return True
        
        rotation_interval = timedelta(days=90)
        return datetime.now(UTC) - self._last_rotation > rotation_interval
    
    async def _perform_rotation(self):
        """Perform credential rotation."""
        self.logger.info(
            "Starting credential rotation",
            source_module=self.__class__.__name__
        )
        
        try:
            # 1. Generate new credentials (implementation depends on exchange)
            new_creds = await self._generate_new_credentials()
            
            # 2. Test new credentials
            if not await self._test_credentials(new_creds):
                raise ValueError("New credentials failed validation")
            
            # 3. Update secrets manager
            await self.secrets.update_credentials("kraken", new_creds)
            
            # 4. Update execution handler
            await self.execution.update_credentials(new_creds)
            
            # 5. Deactivate old credentials
            await self._deactivate_old_credentials()
            
            self._last_rotation = datetime.now(UTC)
            self.logger.info(
                "Credential rotation completed successfully",
                source_module=self.__class__.__name__
            )
            
        except Exception as e:
            self.logger.error(
                f"Credential rotation failed: {e}",
                source_module=self.__class__.__name__
            )
            raise
```

**4. Memory Protection:**
```python
# gal_friday/security/memory_protection.py
import ctypes
import sys
from typing import Any

class SecureString:
    """Secure string that clears memory on deletion."""
    
    def __init__(self, value: str):
        self._value = value
        self._address = id(value)
    
    def get(self) -> str:
        """Get the string value."""
        return self._value
    
    def __del__(self):
        """Securely clear the string from memory."""
        try:
            if sys.platform == "win32":
                # Windows
                ctypes.memset(self._address, 0, len(self._value))
            else:
                # Unix-like
                import os
                os.system(f"echo '' > /proc/{os.getpid()}/mem")
        except Exception:
            # Best effort - may not always work
            pass

class MemoryProtectedDict:
    """Dictionary that securely handles sensitive data."""
    
    def __init__(self):
        self._data: Dict[str, SecureString] = {}
    
    def set(self, key: str, value: str):
        """Set a secure value."""
        self._data[key] = SecureString(value)
    
    def get(self, key: str) -> Optional[str]:
        """Get a secure value."""
        if key in self._data:
            return self._data[key].get()
        return None
    
    def clear(self):
        """Clear all secure data."""
        for key in list(self._data.keys()):
            del self._data[key]
```

#### 3.3 Configuration File Security

**1. Encrypted Configuration:**
```python
# scripts/encrypt_config.py
#!/usr/bin/env python3
"""Encrypt sensitive configuration data."""

import json
import getpass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

def derive_key(password: str, salt: bytes) -> bytes:
    """Derive encryption key from password."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def encrypt_secrets():
    """Encrypt secrets file."""
    # Get master password
    password = getpass.getpass("Enter master password: ")
    confirm = getpass.getpass("Confirm password: ")
    
    if password != confirm:
        print("Passwords don't match!")
        return
    
    # Generate salt
    salt = os.urandom(16)
    
    # Derive key
    key = derive_key(password, salt)
    fernet = Fernet(key)
    
    # Read plaintext secrets
    with open('.secrets.json', 'r') as f:
        secrets = json.load(f)
    
    # Encrypt
    encrypted = fernet.encrypt(json.dumps(secrets).encode())
    
    # Save encrypted file with salt
    with open('.secrets.enc', 'wb') as f:
        f.write(salt + encrypted)
    
    # Delete plaintext file
    os.remove('.secrets.json')
    
    print("Secrets encrypted successfully!")
    print(f"Salt: {base64.b64encode(salt).decode()}")
    print("Store this salt and password securely!")

if __name__ == "__main__":
    encrypt_secrets()
```

**2. Runtime Decryption:**
```python
# gal_friday/security/config_decryptor.py
class ConfigDecryptor:
    """Decrypt configuration at runtime."""
    
    def __init__(self, password: Optional[str] = None):
        self.password = password or os.getenv("GAL_FRIDAY_MASTER_PASSWORD")
        if not self.password:
            raise ValueError("No master password provided")
    
    def decrypt_file(self, filepath: str) -> Dict[str, Any]:
        """Decrypt configuration file."""
        with open(filepath, 'rb') as f:
            data = f.read()
        
        # Extract salt (first 16 bytes)
        salt = data[:16]
        encrypted = data[16:]
        
        # Derive key
        key = self._derive_key(self.password, salt)
        fernet = Fernet(key)
        
        # Decrypt
        decrypted = fernet.decrypt(encrypted)
        return json.loads(decrypted.decode())
```

#### 3.4 Access Control and Audit

**1. Credential Access Logging:**
```python
# gal_friday/security/audit_logger.py
class SecurityAuditLogger:
    """Log all security-related events."""
    
    def __init__(self, logger_service: LoggerService):
        self.logger = logger_service
        self._source = "SecurityAudit"
    
    def log_credential_access(
        self,
        accessor: str,
        resource: str,
        success: bool,
        reason: Optional[str] = None
    ):
        """Log credential access attempt."""
        self.logger.info(
            "Credential access attempt",
            source_module=self._source,
            context={
                "accessor": accessor,
                "resource": resource,
                "success": success,
                "reason": reason,
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
    
    def log_rotation_event(
        self,
        resource: str,
        success: bool,
        error: Optional[str] = None
    ):
        """Log credential rotation event."""
        self.logger.info(
            "Credential rotation",
            source_module=self._source,
            context={
                "resource": resource,
                "success": success,
                "error": error,
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
```

### Implementation Steps

1. **Phase 1 - Environment Variables (Week 1)**
   - Move all credentials to environment variables
   - Update configuration to read from env
   - Document environment setup

2. **Phase 2 - Secrets Manager Integration (Week 2-3)**
   - Implement AWS Secrets Manager client
   - Add Vault support
   - Create secrets migration script

3. **Phase 3 - Encryption Layer (Week 4)**
   - Implement file encryption/decryption
   - Add memory protection
   - Create key management tools

4. **Phase 4 - Access Control (Week 5)**
   - Implement audit logging
   - Add credential rotation
   - Create monitoring alerts

### Verification Procedures

```python
# tests/security/test_secrets_manager.py
import pytest
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_credential_retrieval():
    """Test secure credential retrieval."""
    secrets = SecretsManager(Mock(), Mock())
    
    with patch.dict(os.environ, {
        "KRAKEN_API_KEY": "test_key_12345678901234567890",
        "KRAKEN_API_SECRET": base64.b64encode(b"test_secret" * 4).decode()
    }):
        creds = await secrets.get_api_credentials("kraken")
        
        assert creds["api_key"] == "test_key_12345678901234567890"
        assert "test_secret" not in str(creds)  # Should be encrypted

@pytest.mark.asyncio
async def test_credential_validation():
    """Test credential validation."""
    secrets = SecretsManager(Mock(), Mock())
    
    # Invalid credentials
    assert not secrets._validate_credentials({})
    assert not secrets._validate_credentials({"api_key": "short"})
    
    # Valid credentials
    valid = {
        "api_key": "a" * 30,
        "api_secret": base64.b64encode(b"b" * 40).decode()
    }
    assert secrets._validate_credentials(valid)
```

---

## 4. Complete Monitoring System

### Current State Analysis

- Basic monitoring structure exists but incomplete
- Limited metrics collection
- No real-time dashboards
- Insufficient alerting mechanisms
- No performance tracking

### Solution Architecture

#### 4.1 Comprehensive Metrics Collection

**1. System Metrics Collector:**
```python
# gal_friday/monitoring/metrics_collector.py
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import Dict, List, Any
import psutil
import asyncio

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_received_mb: float
    open_file_descriptors: int
    thread_count: int
    
@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_profit: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    current_drawdown: float
    max_drawdown: float
    total_pnl: float
    
@dataclass
class LatencyMetrics:
    """System latency measurements."""
    timestamp: datetime
    data_ingestion_ms: float
    feature_calculation_ms: float
    prediction_ms: float
    risk_check_ms: float
    order_placement_ms: float
    total_signal_latency_ms: float

class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self, pubsub_manager, logger_service, influx_client):
        self.pubsub = pubsub_manager
        self.logger = logger_service
        self.influx = influx_client
        self._collection_task: Optional[asyncio.Task] = None
        self._latency_buffer: List[Dict[str, Any]] = []
        self._trading_stats: Dict[str, Any] = {}
        
    async def start(self):
        """Start metrics collection."""
        self._collection_task = asyncio.create_task(
            self._collect_metrics_loop()
        )
        
        # Subscribe to events for latency tracking
        self.pubsub.subscribe(
            EventType.MARKET_DATA_L2,
            self._track_data_latency
        )
        self.pubsub.subscribe(
            EventType.FEATURE_CALCULATED,
            self._track_feature_latency
        )
        self.pubsub.subscribe(
            EventType.PREDICTION_GENERATED,
            self._track_prediction_latency
        )
        
    async def _collect_metrics_loop(self):
        """Main metrics collection loop."""
        while True:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                await self._store_metrics("system", system_metrics)
                
                # Collect trading metrics
                trading_metrics = await self._collect_trading_metrics()
                await self._store_metrics("trading", trading_metrics)
                
                # Process latency buffer
                if self._latency_buffer:
                    latency_metrics = self._process_latency_buffer()
                    await self._store_metrics("latency", latency_metrics)
                
                # Sleep for collection interval
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(
                    f"Error in metrics collection: {e}",
                    source_module=self.__class__.__name__
                )
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        # Get process info
        process = psutil.Process()
        
        # Get system info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        net_io = psutil.net_io_counters()
        
        return SystemMetrics(
            timestamp=datetime.now(UTC),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=process.memory_info().rss / 1024 / 1024,
            disk_io_read_mb=disk_io.read_bytes / 1024 / 1024,
            disk_io_write_mb=disk_io.write_bytes / 1024 / 1024,
            network_sent_mb=net_io.bytes_sent / 1024 / 1024,
            network_received_mb=net_io.bytes_recv / 1024 / 1024,
            open_file_descriptors=len(process.open_files()),
            thread_count=process.num_threads()
        )
```

**2. Real-time Dashboard:**
```python
# gal_friday/monitoring/dashboard.py
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import json
import asyncio
from typing import Set

app = FastAPI()

class DashboardManager:
    """Manages real-time dashboard connections."""
    
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
        self.connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start dashboard broadcasting."""
        self._broadcast_task = asyncio.create_task(
            self._broadcast_metrics()
        )
    
    async def connect(self, websocket: WebSocket):
        """Handle new dashboard connection."""
        await websocket.accept()
        self.connections.add(websocket)
        
        # Send initial state
        await websocket.send_json({
            "type": "initial_state",
            "data": await self._get_current_state()
        })
    
    async def disconnect(self, websocket: WebSocket):
        """Handle dashboard disconnection."""
        self.connections.discard(websocket)
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to all connected dashboards."""
        while True:
            if self.connections:
                metrics_data = {
                    "type": "metrics_update",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "system": self.metrics.get_latest_system_metrics(),
                    "trading": self.metrics.get_latest_trading_metrics(),
                    "latency": self.metrics.get_latest_latency_metrics()
                }
                
                # Broadcast to all connections
                disconnected = set()
                for websocket in self.connections:
                    try:
                        await websocket.send_json(metrics_data)
                    except Exception:
                        disconnected.add(websocket)
                
                # Remove disconnected clients
                self.connections -= disconnected
            
            await asyncio.sleep(1)  # Update every second

# Dashboard HTML template
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Gal-Friday Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a1a; 
            color: #fff;
            margin: 0;
            padding: 20px;
        }
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #888;
            font-size: 0.9em;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-good { background: #4CAF50; }
        .status-warning { background: #FF9800; }
        .status-critical { background: #F44336; }
        .chart-container {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>Gal-Friday Monitoring Dashboard</h1>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div class="metric-value">
                <span class="status-indicator status-good"></span>
                <span id="system-status">RUNNING</span>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">CPU Usage</div>
            <div class="metric-value" id="cpu-usage">0%</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Memory Usage</div>
            <div class="metric-value" id="memory-usage">0%</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Total P&L</div>
            <div class="metric-value" id="total-pnl">$0.00</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value" id="win-rate">0%</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Current Drawdown</div>
            <div class="metric-value" id="drawdown">0%</div>
        </div>
    </div>
    
    <div class="chart-container">
        <canvas id="latency-chart"></canvas>
    </div>
    
    <div class="chart-container">
        <canvas id="pnl-chart"></canvas>
    </div>
    
    <script>
        // WebSocket connection
        const ws = new WebSocket('ws://localhost:8000/ws/dashboard');
        
        // Chart setup
        const latencyChart = new Chart(document.getElementById('latency-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total Signal Latency (ms)',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
        
        // Handle incoming messages
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            
            if (data.type === 'metrics_update') {
                // Update metric cards
                document.getElementById('cpu-usage').textContent = 
                    data.system.cpu_percent.toFixed(1) + '%';
                document.getElementById('memory-usage').textContent = 
                    data.system.memory_percent.toFixed(1) + '%';
                document.getElementById('total-pnl').textContent = 
                    '$' + data.trading.total_pnl.toFixed(2);
                document.getElementById('win-rate').textContent = 
                    data.trading.win_rate.toFixed(1) + '%';
                document.getElementById('drawdown').textContent = 
                    data.trading.current_drawdown.toFixed(2) + '%';
                
                // Update charts
                updateLatencyChart(data.latency);
            }
        };
        
        function updateLatencyChart(latencyData) {
            // Add new data point
            latencyChart.data.labels.push(new Date().toLocaleTimeString());
            latencyChart.data.datasets[0].data.push(latencyData.total_signal_latency_ms);
            
            // Keep only last 50 points
            if (latencyChart.data.labels.length > 50) {
                latencyChart.data.labels.shift();
                latencyChart.data.datasets[0].data.shift();
            }
            
            latencyChart.update();
        }
    </script>
</body>
</html>
"""

@app.get("/dashboard")
async def get_dashboard():
    """Serve the monitoring dashboard."""
    return HTMLResponse(content=dashboard_html)

@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """Handle dashboard WebSocket connections."""
    await dashboard_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        await dashboard_manager.disconnect(websocket)
```

#### 4.2 Advanced Alerting System

**1. Alert Manager:**
```python
# gal_friday/monitoring/alert_manager.py
from enum import Enum
from typing import List, Dict, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertChannel(Enum):
    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"

@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    name: str
    condition: str
    threshold: Any
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 15
    
class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, config_manager, logger_service):
        self.config = config_manager
        self.logger = logger_service
        self.alerts: Dict[str, Alert] = {}
        self._last_alert_times: Dict[str, datetime] = {}
        self._initialize_alerts()
        
    def _initialize_alerts(self):
        """Initialize alert definitions."""
        # Critical alerts
        self.register_alert(Alert(
            alert_id="high_drawdown",
            name="High Drawdown Alert",
            condition="drawdown_pct > threshold",
            threshold=10.0,
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.EMAIL, AlertChannel.LOG],
            cooldown_minutes=30
        ))
        
        self.register_alert(Alert(
            alert_id="api_errors",
            name="API Error Rate Alert",
            condition="api_errors_per_minute > threshold",
            threshold=5,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.WEBHOOK],
            cooldown_minutes=15
        ))
        
        self.register_alert(Alert(
            alert_id="high_latency",
            name="High Latency Alert",
            condition="avg_latency_ms > threshold",
            threshold=500,
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
            cooldown_minutes=10
        ))
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alerts against current metrics."""
        for alert_id, alert in self.alerts.items():
            if self._should_trigger_alert(alert, metrics):
                await self._send_alert(alert, metrics)
    
    def _should_trigger_alert(self, alert: Alert, metrics: Dict[str, Any]) -> bool:
        """Check if alert should be triggered."""
        # Check cooldown
        if alert.alert_id in self._last_alert_times:
            last_alert = self._last_alert_times[alert.alert_id]
            if datetime.now(UTC) - last_alert < timedelta(minutes=alert.cooldown_minutes):
                return False
        
        # Evaluate condition
        try:
            # Simple threshold comparison for now
            if "drawdown" in alert.alert_id:
                return metrics.get("current_drawdown_pct", 0) > alert.threshold
            elif "api_errors" in alert.alert_id:
                return metrics.get("api_errors_per_minute", 0) > alert.threshold
            elif "latency" in alert.alert_id:
                return metrics.get("avg_latency_ms", 0) > alert.threshold
        except Exception:
            return False
        
        return False
    
    async def _send_alert(self, alert: Alert, metrics: Dict[str, Any]):
        """Send alert through configured channels."""
        self._last_alert_times[alert.alert_id] = datetime.now(UTC)
        
        message = self._format_alert_message(alert, metrics)
        
        for channel in alert.channels:
            try:
                if channel == AlertChannel.LOG:
                    await self._send_log_alert(alert, message)
                elif channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert, message)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert, message)
                elif channel == AlertChannel.SMS:
                    await self._send_sms_alert(alert, message)
            except Exception as e:
                self.logger.error(
                    f"Failed to send alert via {channel}: {e}",
                    source_module=self.__class__.__name__
                )
    
    async def _send_email_alert(self, alert: Alert, message: str):
        """Send alert via email."""
        smtp_config = self.config.get("alerting.smtp", {})
        
        msg = MIMEMultipart()
        msg['From'] = smtp_config.get("from_address")
        msg['To'] = smtp_config.get("to_address")
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
        
        msg.attach(MIMEText(message, 'plain'))
        
        with smtplib.SMTP(smtp_config.get("host"), smtp_config.get("port")) as server:
            if smtp_config.get("use_tls"):
                server.starttls()
            if smtp_config.get("username"):
                server.login(
                    smtp_config.get("username"),
                    smtp_config.get("password")
                )
            server.send_message(msg)
```

#### 4.3 Health Checks and SLAs

**1. Health Check Service:**
```python
# gal_friday/monitoring/health_checker.py
@dataclass
class HealthCheckResult:
    """Result of a health check."""
    check_name: str
    is_healthy: bool
    message: str
    latency_ms: float
    metadata: Dict[str, Any]

class HealthChecker:
    """Performs system health checks."""
    
    def __init__(self, config_manager, logger_service):
        self.config = config_manager
        self.logger = logger_service
        self._checks: List[Callable] = []
        self._register_checks()
        
    def _register_checks(self):
        """Register all health checks."""
        self._checks.extend([
            self._check_api_connectivity,
            self._check_database_connectivity,
            self._check_market_data_freshness,
            self._check_model_availability,
            self._check_memory_usage,
            self._check_disk_space
        ])
    
    async def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        results = []
        
        for check in self._checks:
            start_time = datetime.now(UTC)
            try:
                result = await check()
                latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
                result.latency_ms = latency_ms
                results.append(result)
            except Exception as e:
                results.append(HealthCheckResult(
                    check_name=check.__name__,
                    is_healthy=False,
                    message=f"Check failed with error: {str(e)}",
                    latency_ms=0,
                    metadata={}
                ))
        
        return results
    
    async def _check_api_connectivity(self) -> HealthCheckResult:
        """Check Kraken API connectivity."""
        try:
            # Simple API call to check connectivity
            response = await self._make_test_api_call()
            
            return HealthCheckResult(
                check_name="API Connectivity",
                is_healthy=response is not None,
                message="API is accessible" if response else "API unreachable",
                latency_ms=0,
                metadata={"endpoint": "Kraken API"}
            )
        except Exception as e:
            return HealthCheckResult(
                check_name="API Connectivity",
                is_healthy=False,
                message=str(e),
                latency_ms=0,
                metadata={}
            )
```

### Implementation Timeline

1. **Week 1**: Implement metrics collection framework
2. **Week 2**: Build real-time dashboard
3. **Week 3**: Create alerting system
4. **Week 4**: Add health checks and SLA monitoring
5. **Week 5**: Integration testing and performance tuning

### Verification Procedures

```python
# tests/monitoring/test_monitoring_system.py
@pytest.mark.asyncio
async def test_metrics_collection():
    """Test metrics are collected correctly."""
    collector = MetricsCollector(Mock(), Mock(), Mock())
    
    # Collect system metrics
    metrics = await collector._collect_system_metrics()
    
    assert metrics.cpu_percent >= 0
    assert metrics.memory_percent >= 0
    assert metrics.timestamp is not None

@pytest.mark.asyncio
async def test_alert_triggering():
    """Test alerts trigger correctly."""
    alert_manager = AlertManager(Mock(), Mock())
    
    # Test high drawdown alert
    metrics = {"current_drawdown_pct": 12.0}  # Above 10% threshold
    
    triggered = alert_manager._should_trigger_alert(
        alert_manager.alerts["high_drawdown"],
        metrics
    )
    
    assert triggered
```

---

## Summary and Next Steps

This document has provided detailed solutions for all four critical priority issues:

1. **HALT Mechanism**: Complete implementation with all triggers, automated position closure, external interfaces, and recovery procedures
2. **Test Suite**: Comprehensive testing framework with fixtures, mocks, and CI/CD integration targeting 80%+ coverage
3. **API Security**: Multi-layer security with secrets management, encryption, rotation, and audit logging
4. **Monitoring System**: Real-time metrics, dashboards, alerting, and health checks

### Immediate Action Items

1. Begin HALT mechanism implementation (highest risk if not addressed)
2. Set up basic test structure and write initial unit tests
3. Move API credentials to environment variables (quick security win)
4. Deploy basic monitoring dashboard for visibility

### Success Metrics

- HALT mechanism: All specified triggers working, <1s response time
- Test coverage: >80% with all critical paths covered
- Security: Zero plaintext credentials, audit trail complete
- Monitoring: <5 minute detection time for critical issues

With these solutions implemented, Gal-Friday will meet production-ready standards for safety, security, and operational excellence. 