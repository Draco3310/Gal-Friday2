# Task: Make reconciliation type dynamic and configurable instead of hardcoded "full".

### 1. Context
- **File:** `gal_friday/portfolio/reconciliation_service.py`
- **Line:** `613`
- **Keyword/Pattern:** `"Example"`
- **Current State:** The code contains a hardcoded reconciliation type "full" instead of using dynamic, configurable reconciliation strategies.

### 2. Problem Statement
The hardcoded reconciliation type "full" severely limits the flexibility of the reconciliation service, preventing different reconciliation strategies based on operational needs, data availability, or performance requirements. This inflexibility makes it impossible to perform targeted reconciliations, incremental updates, or optimize reconciliation performance for different scenarios without code changes.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Reconciliation Strategy Enum:** Define all supported reconciliation types and strategies
2. **Implement Strategy Pattern:** Create pluggable reconciliation strategies for different scenarios
3. **Add Configuration Management:** Support dynamic reconciliation type selection via configuration
4. **Build Strategy Factory:** Implement factory pattern for strategy creation and management
5. **Add Performance Optimization:** Different strategies optimized for different use cases
6. **Create Monitoring and Metrics:** Track reconciliation performance per strategy type

#### b. Pseudocode or Implementation Sketch
```python
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import logging

class ReconciliationType(str, Enum):
    """Supported reconciliation types"""
    FULL = "full"
    INCREMENTAL = "incremental" 
    DELTA = "delta"
    POSITION_ONLY = "position_only"
    BALANCE_ONLY = "balance_only"
    TRADE_ONLY = "trade_only"
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"

@dataclass
class ReconciliationConfig:
    """Configuration for reconciliation strategies"""
    reconciliation_type: ReconciliationType
    max_discrepancy_threshold: float = 0.01
    auto_resolve_threshold: float = 0.001
    include_pending_trades: bool = True
    historical_lookback_hours: int = 24
    enable_alerts: bool = True
    batch_size: int = 1000
    timeout_seconds: int = 300
    retry_attempts: int = 3

@dataclass 
class ReconciliationResult:
    """Result of reconciliation process"""
    reconciliation_id: str
    type: ReconciliationType
    status: str  # 'completed', 'failed', 'partial'
    start_time: datetime
    end_time: datetime
    total_records_processed: int
    discrepancies_found: int
    auto_resolved_count: int
    manual_resolution_required: int
    summary: Dict[str, Any]
    errors: List[str]

class BaseReconciliationStrategy(ABC):
    """Abstract base for reconciliation strategies"""
    
    def __init__(self, config: ReconciliationConfig, data_service, exchange_service):
        self.config = config
        self.data_service = data_service
        self.exchange_service = exchange_service
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    async def execute_reconciliation(self) -> ReconciliationResult:
        """Execute the reconciliation strategy"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the strategy name"""
        pass
    
    async def _validate_prerequisites(self) -> bool:
        """Validate that prerequisites for reconciliation are met"""
        try:
            # Check data service connectivity
            if not await self.data_service.is_connected():
                self.logger.error("Data service not available")
                return False
            
            # Check exchange service connectivity  
            if not await self.exchange_service.is_connected():
                self.logger.error("Exchange service not available")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Prerequisites validation failed: {e}")
            return False

class FullReconciliationStrategy(BaseReconciliationStrategy):
    """Complete reconciliation of all positions, balances, and trades"""
    
    def get_strategy_name(self) -> str:
        return "Full Reconciliation"
    
    async def execute_reconciliation(self) -> ReconciliationResult:
        """Execute full reconciliation"""
        
        reconciliation_id = f"full_{int(datetime.now().timestamp())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting full reconciliation {reconciliation_id}")
            
            # Validate prerequisites
            if not await self._validate_prerequisites():
                raise ValueError("Prerequisites not met for full reconciliation")
            
            # Get all data from both sources
            internal_data = await self._get_internal_data()
            exchange_data = await self._get_exchange_data()
            
            # Perform comprehensive comparison
            discrepancies = await self._compare_all_data(internal_data, exchange_data)
            
            # Attempt auto-resolution
            auto_resolved = await self._auto_resolve_discrepancies(discrepancies)
            
            # Generate reconciliation result
            result = ReconciliationResult(
                reconciliation_id=reconciliation_id,
                type=ReconciliationType.FULL,
                status='completed',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                total_records_processed=len(internal_data) + len(exchange_data),
                discrepancies_found=len(discrepancies),
                auto_resolved_count=len(auto_resolved),
                manual_resolution_required=len(discrepancies) - len(auto_resolved),
                summary=self._generate_summary(internal_data, exchange_data, discrepancies),
                errors=[]
            )
            
            self.logger.info(f"Full reconciliation {reconciliation_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Full reconciliation {reconciliation_id} failed: {e}")
            return ReconciliationResult(
                reconciliation_id=reconciliation_id,
                type=ReconciliationType.FULL,
                status='failed',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                total_records_processed=0,
                discrepancies_found=0,
                auto_resolved_count=0,
                manual_resolution_required=0,
                summary={},
                errors=[str(e)]
            )

class IncrementalReconciliationStrategy(BaseReconciliationStrategy):
    """Reconcile only changes since last reconciliation"""
    
    def get_strategy_name(self) -> str:
        return "Incremental Reconciliation"
    
    async def execute_reconciliation(self) -> ReconciliationResult:
        """Execute incremental reconciliation"""
        
        reconciliation_id = f"incremental_{int(datetime.now().timestamp())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Get timestamp of last reconciliation
            last_reconciliation = await self._get_last_reconciliation_timestamp()
            
            # Get only changed data since last reconciliation
            internal_changes = await self._get_internal_changes_since(last_reconciliation)
            exchange_changes = await self._get_exchange_changes_since(last_reconciliation)
            
            # Compare changes
            discrepancies = await self._compare_incremental_data(internal_changes, exchange_changes)
            
            # Auto-resolve where possible
            auto_resolved = await self._auto_resolve_discrepancies(discrepancies)
            
            result = ReconciliationResult(
                reconciliation_id=reconciliation_id,
                type=ReconciliationType.INCREMENTAL,
                status='completed',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                total_records_processed=len(internal_changes) + len(exchange_changes),
                discrepancies_found=len(discrepancies),
                auto_resolved_count=len(auto_resolved),
                manual_resolution_required=len(discrepancies) - len(auto_resolved),
                summary=self._generate_incremental_summary(internal_changes, exchange_changes),
                errors=[]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Incremental reconciliation failed: {e}")
            return self._create_error_result(reconciliation_id, start_time, e)

class RealTimeReconciliationStrategy(BaseReconciliationStrategy):
    """Continuous real-time reconciliation for critical operations"""
    
    def get_strategy_name(self) -> str:
        return "Real-Time Reconciliation"
    
    async def execute_reconciliation(self) -> ReconciliationResult:
        """Execute real-time reconciliation"""
        
        reconciliation_id = f"realtime_{int(datetime.now().timestamp())}"
        start_time = datetime.now(timezone.utc)
        
        # Real-time reconciliation focuses on recent trades and positions
        cutoff_time = start_time - timedelta(minutes=15)  # Last 15 minutes
        
        try:
            # Get recent data only
            recent_internal = await self._get_recent_internal_data(cutoff_time)
            recent_exchange = await self._get_recent_exchange_data(cutoff_time)
            
            # Fast comparison for real-time processing
            discrepancies = await self._fast_compare_data(recent_internal, recent_exchange)
            
            # Immediate alerts for critical discrepancies
            critical_discrepancies = [d for d in discrepancies if d.severity == 'critical']
            if critical_discrepancies:
                await self._send_immediate_alerts(critical_discrepancies)
            
            result = ReconciliationResult(
                reconciliation_id=reconciliation_id,
                type=ReconciliationType.REAL_TIME,
                status='completed',
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                total_records_processed=len(recent_internal) + len(recent_exchange),
                discrepancies_found=len(discrepancies),
                auto_resolved_count=0,  # Real-time doesn't auto-resolve
                manual_resolution_required=len(discrepancies),
                summary={'critical_issues': len(critical_discrepancies)},
                errors=[]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Real-time reconciliation failed: {e}")
            return self._create_error_result(reconciliation_id, start_time, e)

class ReconciliationStrategyFactory:
    """Factory for creating reconciliation strategies"""
    
    _strategies = {
        ReconciliationType.FULL: FullReconciliationStrategy,
        ReconciliationType.INCREMENTAL: IncrementalReconciliationStrategy,
        ReconciliationType.REAL_TIME: RealTimeReconciliationStrategy,
        # Add more strategies as needed
    }
    
    @classmethod
    def create_strategy(cls, config: ReconciliationConfig, 
                       data_service, exchange_service) -> BaseReconciliationStrategy:
        """Create reconciliation strategy based on configuration"""
        
        strategy_class = cls._strategies.get(config.reconciliation_type)
        if not strategy_class:
            available_types = list(cls._strategies.keys())
            raise ValueError(
                f"Unsupported reconciliation type: {config.reconciliation_type}. "
                f"Available types: {available_types}"
            )
        
        return strategy_class(config, data_service, exchange_service)
    
    @classmethod
    def get_supported_types(cls) -> List[ReconciliationType]:
        """Get list of supported reconciliation types"""
        return list(cls._strategies.keys())

class ConfigurableReconciliationService:
    """Enhanced reconciliation service with configurable strategies"""
    
    def __init__(self, config_manager, data_service, exchange_service):
        self.config_manager = config_manager
        self.data_service = data_service
        self.exchange_service = exchange_service
        self.logger = logging.getLogger(__name__)
        
    async def perform_reconciliation(self, reconciliation_type: Optional[ReconciliationType] = None) -> ReconciliationResult:
        """
        Perform reconciliation with configurable type
        Replace: hardcoded "full" reconciliation
        """
        
        try:
            # Get reconciliation configuration
            reconciliation_config = self._load_reconciliation_config(reconciliation_type)
            
            self.logger.info(
                f"Starting {reconciliation_config.reconciliation_type.value} reconciliation"
            )
            
            # Create appropriate strategy
            strategy = ReconciliationStrategyFactory.create_strategy(
                reconciliation_config,
                self.data_service,
                self.exchange_service
            )
            
            # Execute reconciliation
            result = await strategy.execute_reconciliation()
            
            # Log results
            self._log_reconciliation_result(result)
            
            # Store results for auditing
            await self._store_reconciliation_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Reconciliation failed: {e}")
            raise
    
    def _load_reconciliation_config(self, reconciliation_type: Optional[ReconciliationType]) -> ReconciliationConfig:
        """Load reconciliation configuration from config manager"""
        
        # Use provided type or get from configuration
        if not reconciliation_type:
            config_type_str = self.config_manager.get('reconciliation.default_type', 'full')
            reconciliation_type = ReconciliationType(config_type_str)
        
        # Load strategy-specific configuration
        config_dict = self.config_manager.get('reconciliation', {})
        
        return ReconciliationConfig(
            reconciliation_type=reconciliation_type,
            max_discrepancy_threshold=config_dict.get('max_discrepancy_threshold', 0.01),
            auto_resolve_threshold=config_dict.get('auto_resolve_threshold', 0.001),
            include_pending_trades=config_dict.get('include_pending_trades', True),
            historical_lookback_hours=config_dict.get('historical_lookback_hours', 24),
            enable_alerts=config_dict.get('enable_alerts', True),
            batch_size=config_dict.get('batch_size', 1000),
            timeout_seconds=config_dict.get('timeout_seconds', 300),
            retry_attempts=config_dict.get('retry_attempts', 3)
        )
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of strategy creation failures; fallback to safe default strategies; comprehensive error logging and reporting
- **Configuration:** Dynamic configuration loading; runtime strategy switching; environment-specific reconciliation settings
- **Testing:** Unit tests for each reconciliation strategy; integration tests for strategy factory; performance tests for different reconciliation types
- **Dependencies:** Integration with ConfigManager for dynamic configuration; data and exchange services for reconciliation data; monitoring system for performance tracking

### 4. Acceptance Criteria
- [ ] Hardcoded "full" reconciliation type is completely replaced with configurable strategy selection
- [ ] Multiple reconciliation strategies (full, incremental, real-time) are implemented and tested
- [ ] ReconciliationStrategyFactory supports dynamic strategy creation based on configuration
- [ ] Configuration allows runtime selection of reconciliation type and parameters
- [ ] Each reconciliation strategy is optimized for its specific use case and performance requirements
- [ ] Comprehensive logging and monitoring tracks reconciliation performance per strategy
- [ ] Error handling ensures graceful fallback when unsupported strategies are requested
- [ ] Integration tests verify end-to-end reconciliation workflow with different strategies
- [ ] Documentation explains when to use each reconciliation strategy type 