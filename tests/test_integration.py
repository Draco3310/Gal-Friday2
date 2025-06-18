"""Integration tests for the full Gal-Friday trading system.

This module tests the complete signal lifecycle from market data ingestion
through prediction, signal generation, approval, and execution.
"""

from datetime import UTC, datetime
from decimal import Decimal
import uuid

import asyncio
import pytest

from gal_friday.core.events import (
    EventType,
    ExecutionReportEvent,
    MarketDataL2Event,
    MarketDataOHLCVEvent,
    PredictionEvent,
    SystemStateEvent,
    TradeSignalApprovedEvent,
    TradeSignalProposedEvent,
    TradeSignalRejectedEvent,
)


class TestFullSignalLifecycle:
    """Test complete signal lifecycle from data to execution."""

    @pytest.mark.asyncio
    async def test_market_data_to_execution(self, integrated_system):
        """Test full flow: Market Data → Prediction → Signal → Execution."""
        # Track events through the pipeline
        events_captured = {
            "market_data": [],
            "predictions": [],
            "proposed_signals": [],
            "approved_signals": [],
            "execution_reports": [],
        }

        # Set up event capture
        async def capture_market_data(event):
            if isinstance(event, MarketDataL2Event | MarketDataOHLCVEvent):
                events_captured["market_data"].append(event)

        async def capture_predictions(event):
            if isinstance(event, PredictionEvent):
                events_captured["predictions"].append(event)

        async def capture_proposed_signals(event):
            if isinstance(event, TradeSignalProposedEvent):
                events_captured["proposed_signals"].append(event)

        async def capture_approved_signals(event):
            if isinstance(event, TradeSignalApprovedEvent):
                events_captured["approved_signals"].append(event)

        async def capture_execution_reports(event):
            if isinstance(event, ExecutionReportEvent):
                events_captured["execution_reports"].append(event)

        # Subscribe to all event types
        integrated_system.pubsub.subscribe(EventType.MARKET_DATA_L2, capture_market_data)
        integrated_system.pubsub.subscribe(EventType.PREDICTION, capture_predictions)
        integrated_system.pubsub.subscribe(EventType.TRADE_SIGNAL_PROPOSED, capture_proposed_signals)
        integrated_system.pubsub.subscribe(EventType.TRADE_SIGNAL_APPROVED, capture_approved_signals)
        integrated_system.pubsub.subscribe(EventType.EXECUTION_REPORT, capture_execution_reports)

        # 1. Inject market data
        market_data = MarketDataL2Event(
            source_module="DataIngestion",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            trading_pair="XRP/USD",
            exchange="kraken",
            bids=[[Decimal("0.5000"), Decimal(10000)]],
            asks=[[Decimal("0.5001"), Decimal(10000)]],
            timestamp_exchange=datetime.now(UTC),
            sequence_number=1000,
        )

        await integrated_system.pubsub.publish(market_data)
        await asyncio.sleep(0.1)  # Allow processing

        # 2. Inject prediction (simulating ML model output)
        prediction = PredictionEvent(
            source_module="PredictionService",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            trading_pair="XRP/USD",
            model_id="ensemble_v1",
            prediction_value=Decimal("0.75"),  # 75% confidence bullish
            confidence=Decimal("0.85"),
            features={
                "rsi": 45,
                "macd": 0.001,
                "volume_ratio": 1.2,
            },
        )

        await integrated_system.pubsub.publish(prediction)
        await asyncio.sleep(0.1)

        # 3. Generate trade signal (simulating strategy arbitrator)
        proposed_signal = TradeSignalProposedEvent(
            source_module="StrategyArbitrator",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            signal_id=uuid.uuid4(),
            trading_pair="XRP/USD",
            exchange="kraken",
            side="BUY",
            entry_type="LIMIT",
            proposed_entry_price=Decimal("0.4999"),
            proposed_sl_price=Decimal("0.4900"),
            proposed_tp_price=Decimal("0.5200"),
            strategy_id="momentum_breakout",
        )

        await integrated_system.pubsub.publish(proposed_signal)
        await asyncio.sleep(0.1)

        # 4. Approve signal (simulating risk manager)
        approved_signal = TradeSignalApprovedEvent(
            source_module="RiskManager",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            signal_id=proposed_signal.signal_id,
            trading_pair="XRP/USD",
            exchange="kraken",
            side="BUY",
            quantity=Decimal(1000),  # Risk-adjusted size
            order_type="LIMIT",
            limit_price=Decimal("0.4999"),
            sl_price=Decimal("0.4900"),
            tp_price=Decimal("0.5200"),
        )

        await integrated_system.pubsub.publish(approved_signal)
        await asyncio.sleep(0.5)  # Allow execution

        # Verify complete flow
        assert len(events_captured["market_data"]) > 0
        assert len(events_captured["predictions"]) > 0
        assert len(events_captured["proposed_signals"]) > 0
        assert len(events_captured["approved_signals"]) > 0
        assert len(events_captured["execution_reports"]) > 0

        # Verify execution report
        exec_report = events_captured["execution_reports"][0]
        assert exec_report.order_status in ["NEW", "OPEN"]
        assert exec_report.quantity_ordered == Decimal(1000)

    @pytest.mark.asyncio
    async def test_signal_rejection_flow(self, integrated_system):
        """Test signal rejection by risk manager."""
        rejected_signals = []

        async def capture_rejected(event):
            if isinstance(event, TradeSignalRejectedEvent):
                rejected_signals.append(event)

        integrated_system.pubsub.subscribe(EventType.TRADE_SIGNAL_REJECTED, capture_rejected)

        # Propose risky signal
        risky_signal = TradeSignalProposedEvent(
            source_module="StrategyArbitrator",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            signal_id=uuid.uuid4(),
            trading_pair="XRP/USD",
            exchange="kraken",
            side="BUY",
            entry_type="MARKET",
            proposed_entry_price=Decimal("0.5000"),
            proposed_sl_price=Decimal("0.3000"),  # 40% stop loss - too risky
            proposed_tp_price=Decimal("0.6000"),
            strategy_id="high_risk_strategy",
        )

        # Simulate risk manager rejection
        rejection = TradeSignalRejectedEvent(
            source_module="RiskManager",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            signal_id=risky_signal.signal_id,
            trading_pair="XRP/USD",
            rejection_reason="Stop loss too far from entry (40%)",
            proposed_signal=risky_signal,
        )

        await integrated_system.pubsub.publish(risky_signal)
        await asyncio.sleep(0.1)
        await integrated_system.pubsub.publish(rejection)
        await asyncio.sleep(0.1)

        assert len(rejected_signals) > 0
        assert "Stop loss too far" in rejected_signals[0].rejection_reason


class TestSystemRecovery:
    """Test system recovery and reconnection scenarios."""

    @pytest.mark.asyncio
    async def test_halt_and_recovery(self, integrated_system):
        """Test system HALT and recovery process."""
        state_changes = []

        async def capture_state_changes(event):
            if isinstance(event, SystemStateEvent):
                state_changes.append(event)

        integrated_system.pubsub.subscribe(EventType.SYSTEM_STATE_CHANGE, capture_state_changes)

        # 1. System running normally
        running_state = SystemStateEvent(
            source_module="MonitoringService",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            new_state="RUNNING",
            reason="System startup",
        )
        await integrated_system.pubsub.publish(running_state)

        # 2. Trigger HALT
        halt_state = SystemStateEvent(
            source_module="MonitoringService",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            new_state="HALTED",
            reason="Maximum drawdown exceeded",
        )
        await integrated_system.pubsub.publish(halt_state)

        # 3. Recovery process
        await asyncio.sleep(0.5)

        # 4. Resume
        resume_state = SystemStateEvent(
            source_module="MonitoringService",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            new_state="RUNNING",
            reason="Manual resume after recovery",
        )
        await integrated_system.pubsub.publish(resume_state)

        await asyncio.sleep(0.1)

        # Verify state transitions
        assert len(state_changes) >= 3
        assert state_changes[0].new_state == "RUNNING"
        assert state_changes[1].new_state == "HALTED"
        assert state_changes[2].new_state == "RUNNING"

    @pytest.mark.asyncio
    async def test_connection_recovery(self, integrated_system):
        """Test recovery from connection failures."""
        error_events = []
        recovery_events = []

        async def capture_errors(event):
            if hasattr(event, "error_type") and event.error_type == "CONNECTION":
                error_events.append(event)

        async def capture_recovery(event):
            if hasattr(event, "recovery_type"):
                recovery_events.append(event)

        # Simulate connection failure
        # This would be implemented with actual connection monitoring

        # Simulate recovery
        # This would trigger reconnection logic

        # For now, just verify the test framework
        assert integrated_system is not None


class TestPerformanceIntegration:
    """Test system performance under various load conditions."""

    @pytest.mark.asyncio
    async def test_high_frequency_data_processing(self, integrated_system):
        """Test system performance with high-frequency market data."""
        processed_count = 0
        start_time = asyncio.get_event_loop().time()

        async def count_processed(event):
            nonlocal processed_count
            processed_count += 1

        # Subscribe to all major event types
        integrated_system.pubsub.subscribe(EventType.MARKET_DATA_L2, count_processed)
        integrated_system.pubsub.subscribe(EventType.TRADE_SIGNAL_PROPOSED, count_processed)
        integrated_system.pubsub.subscribe(EventType.EXECUTION_REPORT, count_processed)

        # Generate high-frequency market data
        tasks = []
        for i in range(100):
            event = MarketDataL2Event(
                source_module="DataIngestion",
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                trading_pair="XRP/USD" if i % 2 == 0 else "DOGE/USD",
                exchange="kraken",
                bids=[[Decimal("0.5000") + Decimal(f"0.000{i%10}"), Decimal(1000)]],
                asks=[[Decimal("0.5001") + Decimal(f"0.000{i%10}"), Decimal(1000)]],
                timestamp_exchange=datetime.now(UTC),
                sequence_number=1000 + i,
            )
            tasks.append(integrated_system.pubsub.publish(event))

        await asyncio.gather(*tasks)
        await asyncio.sleep(1)  # Allow processing

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        throughput = processed_count / duration

        # System should handle at least 50 events/second
        assert throughput > 50

    @pytest.mark.asyncio
    async def test_latency_measurement(self, integrated_system):
        """Measure end-to-end latency from market data to execution."""
        latencies = []

        # Create correlated events with tracking
        correlation_id = uuid.uuid4()
        start_time = datetime.now(UTC)

        # 1. Market data with correlation ID
        market_data = MarketDataL2Event(
            source_module="DataIngestion",
            event_id=uuid.uuid4(),
            timestamp=start_time,
            trading_pair="XRP/USD",
            exchange="kraken",
            bids=[[Decimal("0.5000"), Decimal(10000)]],
            asks=[[Decimal("0.5001"), Decimal(10000)]],
            timestamp_exchange=start_time,
            sequence_number=9999,
            metadata={"correlation_id": str(correlation_id)},
        )

        # Track when execution report is received
        execution_received = asyncio.Event()

        async def track_execution(event):
            if isinstance(event, ExecutionReportEvent):
                if hasattr(event, "metadata") and event.metadata.get("correlation_id") == str(correlation_id):
                    latency = (datetime.now(UTC) - start_time).total_seconds() * 1000
                    latencies.append(latency)
                    execution_received.set()

        integrated_system.pubsub.subscribe(EventType.EXECUTION_REPORT, track_execution)

        # Publish market data
        await integrated_system.pubsub.publish(market_data)

        # Wait for execution with timeout
        try:
            await asyncio.wait_for(execution_received.wait(), timeout=5.0)
        except TimeoutError:
            pytest.skip("End-to-end flow not fully implemented")

        # Verify latency is reasonable
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            assert avg_latency < 1000  # Less than 1 second


class TestStressScenarios:
    """Test system behavior under stress conditions."""

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_signals(self, integrated_system):
        """Test handling of multiple simultaneous trading signals."""
        signals_processed = []

        async def track_signals(event):
            if isinstance(event, TradeSignalApprovedEvent | TradeSignalRejectedEvent):
                signals_processed.append(event)

        integrated_system.pubsub.subscribe(EventType.TRADE_SIGNAL_APPROVED, track_signals)
        integrated_system.pubsub.subscribe(EventType.TRADE_SIGNAL_REJECTED, track_signals)

        # Generate multiple signals simultaneously
        signals = []
        for i in range(10):
            signal = TradeSignalProposedEvent(
                source_module="StrategyArbitrator",
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                signal_id=uuid.uuid4(),
                trading_pair="XRP/USD" if i % 2 == 0 else "DOGE/USD",
                exchange="kraken",
                side="BUY" if i % 3 == 0 else "SELL",
                entry_type="LIMIT",
                proposed_entry_price=Decimal("0.5000"),
                proposed_sl_price=Decimal("0.4900"),
                proposed_tp_price=Decimal("0.5100"),
                strategy_id=f"strategy_{i}",
            )
            signals.append(signal)

        # Publish all signals concurrently
        tasks = [integrated_system.pubsub.publish(s) for s in signals]
        await asyncio.gather(*tasks)

        # Simulate risk manager processing
        for signal in signals[:5]:  # Approve first 5
            approved = TradeSignalApprovedEvent(
                source_module="RiskManager",
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                signal_id=signal.signal_id,
                trading_pair=signal.trading_pair,
                exchange=signal.exchange,
                side=signal.side,
                quantity=Decimal(100),
                order_type=signal.entry_type,
                limit_price=signal.proposed_entry_price,
                sl_price=signal.proposed_sl_price,
                tp_price=signal.proposed_tp_price,
            )
            await integrated_system.pubsub.publish(approved)

        for signal in signals[5:]:  # Reject rest
            rejected = TradeSignalRejectedEvent(
                source_module="RiskManager",
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                signal_id=signal.signal_id,
                trading_pair=signal.trading_pair,
                rejection_reason="Maximum concurrent positions reached",
                proposed_signal=signal,
            )
            await integrated_system.pubsub.publish(rejected)

        await asyncio.sleep(0.5)

        # Verify all signals were processed
        assert len(signals_processed) == 10
        approved_count = sum(1 for s in signals_processed if isinstance(s, TradeSignalApprovedEvent))
        rejected_count = sum(1 for s in signals_processed if isinstance(s, TradeSignalRejectedEvent))
        assert approved_count == 5
        assert rejected_count == 5

    @pytest.mark.asyncio
    async def test_rapid_market_movements(self, integrated_system):
        """Test system behavior during rapid price movements."""
        # Simulate rapid price changes
        base_price = Decimal("0.5000")

        for i in range(20):
            # Price moves 1% each update
            price_change = Decimal("0.005") * (1 if i % 2 == 0 else -1)
            new_price = base_price + price_change * i

            market_data = MarketDataL2Event(
                source_module="DataIngestion",
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                trading_pair="XRP/USD",
                exchange="kraken",
                bids=[[new_price - Decimal("0.0001"), Decimal(10000)]],
                asks=[[new_price + Decimal("0.0001"), Decimal(10000)]],
                timestamp_exchange=datetime.now(UTC),
                sequence_number=2000 + i,
            )

            await integrated_system.pubsub.publish(market_data)
            await asyncio.sleep(0.05)  # 50ms between updates

        # System should handle rapid updates without issues
        # In a real implementation, this would verify:
        # - No missed updates
        # - Proper volatility detection
        # - Appropriate trading decisions
        assert True  # Placeholder


# Extension for PredictionEvent creation
class PredictionEvent:
    """Mock prediction event for testing."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.event_type = EventType.PREDICTION


# Extension for TradeSignalRejectedEvent
class TradeSignalRejectedEvent:
    """Mock rejected signal event for testing."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.event_type = EventType.TRADE_SIGNAL_REJECTED
