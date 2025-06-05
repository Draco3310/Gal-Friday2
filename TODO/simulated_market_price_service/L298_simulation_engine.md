# Task: Implement real-time price simulation engine with configurable replay speeds.

### 1. Context
- **File:** `gal_friday/simulated_market_price_service.py`
- **Line:** `298`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing a real-time price simulation engine with configurable replay speeds.

### 2. Problem Statement
Without proper real-time price simulation engine, the system cannot provide realistic market data replay for backtesting and strategy development. This prevents accurate simulation of trading conditions, timing analysis, and performance evaluation under various market scenarios and replay speeds.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Simulation Engine Core:** High-performance engine for real-time data replay
2. **Build Speed Control System:** Configurable replay speeds with accurate timing
3. **Implement Event Scheduling:** Precise timing of market events and data delivery
4. **Add Market Simulation:** Realistic market behavior simulation with microstructure
5. **Create Performance Optimization:** Efficient data streaming with minimal latency
6. **Build Monitoring Dashboard:** Real-time monitoring of simulation performance

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime, timezone, timedelta
import time
import heapq
from collections import deque

class SimulationSpeed(str, Enum):
    """Simulation replay speeds"""
    REAL_TIME = "1x"
    FAST_2X = "2x"
    FAST_5X = "5x"
    FAST_10X = "10x"
    FAST_100X = "100x"
    MAX_SPEED = "max"

class SimulationState(str, Enum):
    """Simulation engine states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class SimulationEvent:
    """Simulation event with timing information"""
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    priority: int = 0

class RealTimeSimulationEngine:
    """Enterprise-grade real-time price simulation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Simulation state
        self.state = SimulationState.STOPPED
        self.current_sim_time = None
        self.start_real_time = None
        
        # Event management
        self.event_queue = []  # Priority queue for events
        self.event_buffer = deque(maxlen=config.get('buffer_size', 10000))
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Speed multiplier calculation
        self.speed_multiplier = self._calculate_speed_multiplier()
        
        # Simulation task
        self.simulation_task: Optional[asyncio.Task] = None
    
    async def start_simulation(self) -> None:
        """
        Start real-time simulation engine
        Replace TODO with comprehensive simulation system
        """
        
        if self.state != SimulationState.STOPPED:
            raise SimulationError(f"Cannot start simulation in state: {self.state}")
        
        try:
            self.logger.info("Starting real-time simulation engine")
            
            # Load initial data
            await self._load_simulation_data()
            
            # Initialize timing
            self.start_real_time = time.time()
            self.current_sim_time = self.config.get('start_time')
            
            # Start simulation loop
            self.simulation_task = asyncio.create_task(self._simulation_loop())
            
            self.state = SimulationState.RUNNING
            self.logger.info("Simulation started successfully")
            
        except Exception as e:
            self.state = SimulationState.ERROR
            self.logger.error(f"Failed to start simulation: {e}")
            raise SimulationError(f"Simulation start failed: {e}")
    
    async def stop_simulation(self) -> None:
        """Stop simulation gracefully"""
        
        if self.state not in [SimulationState.RUNNING, SimulationState.PAUSED]:
            self.logger.warning(f"Cannot stop simulation in state: {self.state}")
            return
        
        self.logger.info("Stopping simulation")
        
        if self.simulation_task:
            self.simulation_task.cancel()
            try:
                await self.simulation_task
            except asyncio.CancelledError:
                pass
        
        self.state = SimulationState.STOPPED
        self.logger.info("Simulation stopped")
    
    async def set_speed(self, speed: SimulationSpeed) -> None:
        """Change simulation speed during runtime"""
        
        self.config['speed'] = speed
        self.speed_multiplier = self._calculate_speed_multiplier()
        
        self.logger.info(f"Simulation speed changed to {speed.value} ({self.speed_multiplier}x)")
    
    async def _simulation_loop(self) -> None:
        """Main simulation loop"""
        
        try:
            end_time = self.config.get('end_time')
            
            while (self.state in [SimulationState.RUNNING, SimulationState.PAUSED] and
                   self.current_sim_time < end_time):
                
                # Handle pause state
                while self.state == SimulationState.PAUSED:
                    await asyncio.sleep(0.1)
                
                # Process events for current time
                await self._process_current_events()
                
                # Advance simulation time
                await self._advance_simulation_time()
                
                # Yield control to allow other tasks
                await asyncio.sleep(0)
                
        except asyncio.CancelledError:
            self.logger.info("Simulation loop cancelled")
            raise
        except Exception as e:
            self.state = SimulationState.ERROR
            self.logger.error(f"Error in simulation loop: {e}")
            raise
    
    async def _process_current_events(self) -> None:
        """Process all events scheduled for current simulation time"""
        
        events_processed = 0
        
        # Process events from queue
        while (self.event_queue and 
               self.event_queue[0].timestamp <= self.current_sim_time):
            
            event = heapq.heappop(self.event_queue)
            
            try:
                await self._dispatch_event(event)
                events_processed += 1
                
            except Exception as e:
                self.logger.error(f"Error processing event {event.event_type}: {e}")
        
        if events_processed > 0:
            self.logger.debug(f"Processed {events_processed} events at {self.current_sim_time}")
    
    async def _dispatch_event(self, event: SimulationEvent) -> None:
        """Dispatch event to registered handlers"""
        
        event_type = event.event_type
        
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                        
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def _advance_simulation_time(self) -> None:
        """Advance simulation time with proper speed control"""
        
        if self.config.get('speed') == SimulationSpeed.MAX_SPEED:
            # Run as fast as possible
            self.current_sim_time += timedelta(seconds=1)
            return
        
        # Calculate time advancement
        real_time_elapsed = time.time() - self.start_real_time
        expected_sim_time = self.config.get('start_time') + timedelta(
            seconds=real_time_elapsed * self.speed_multiplier
        )
        
        # If we're ahead of schedule, wait
        if self.current_sim_time >= expected_sim_time:
            # Calculate sleep time to maintain proper speed
            ahead_by = (self.current_sim_time - expected_sim_time).total_seconds()
            sleep_time = ahead_by / self.speed_multiplier
            
            if sleep_time > 0:
                await asyncio.sleep(min(sleep_time, 0.1))  # Cap sleep time
        
        # Advance simulation time
        time_step = timedelta(seconds=self._get_time_step_seconds())
        self.current_sim_time += time_step
    
    async def _load_simulation_data(self) -> None:
        """Load simulation data and create events"""
        
        self.logger.info("Loading simulation data")
        
        symbols = self.config.get('symbols', [])
        
        for symbol in symbols:
            # Load historical data for symbol
            data_points = await self._load_historical_data(symbol)
            
            # Create events for each data point
            for data_point in data_points:
                event = SimulationEvent(
                    timestamp=data_point['timestamp'],
                    event_type='price_update',
                    data={
                        'symbol': symbol,
                        'price_data': data_point
                    }
                )
                
                heapq.heappush(self.event_queue, event)
        
        self.logger.info(f"Loaded {len(self.event_queue)} simulation events")
    
    def _calculate_speed_multiplier(self) -> float:
        """Calculate speed multiplier based on configuration"""
        
        speed_map = {
            SimulationSpeed.REAL_TIME: 1.0,
            SimulationSpeed.FAST_2X: 2.0,
            SimulationSpeed.FAST_5X: 5.0,
            SimulationSpeed.FAST_10X: 10.0,
            SimulationSpeed.FAST_100X: 100.0,
            SimulationSpeed.MAX_SPEED: float('inf')
        }
        
        return speed_map.get(self.config.get('speed', SimulationSpeed.REAL_TIME), 1.0)
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register event handler for specific event type"""
        
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for event type: {event_type}")

class SimulationError(Exception):
    """Exception raised for simulation errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of timing errors; recovery from event processing failures; comprehensive error logging
- **Configuration:** Configurable replay speeds; adjustable buffer sizes; symbol-specific simulation settings
- **Testing:** Unit tests for timing accuracy; integration tests with data sources; performance tests for high-speed simulation
- **Dependencies:** Asyncio for concurrency; historical data loader; event handling framework; performance monitoring tools

### 4. Acceptance Criteria
- [ ] Real-time simulation engine accurately replays historical data with configurable speeds
- [ ] Event scheduling maintains precise timing for market events and data delivery
- [ ] Speed control system allows dynamic adjustment of replay speed during simulation
- [ ] Performance optimization handles high-frequency data without memory leaks or latency issues
- [ ] State management provides proper start, stop, pause, and resume functionality
- [ ] Event handling framework supports extensible event types and handlers
- [ ] Metrics monitoring tracks simulation performance and identifies bottlenecks
- [ ] Buffer management prevents memory overflow during long-running simulations
- [ ] Timing accuracy maintains synchronization between real time and simulation time
- [ ] TODO placeholder is completely replaced with production-ready implementation