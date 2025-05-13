"""
Integration tests for the end-to-end trading flow.

These tests verify that the full trading pipeline works correctly, including:
- Market data ingestion
- Feature calculation
- Prediction generation
- Strategy signal creation
- Risk assessment
- Order execution
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gal_friday.core.events import EventType, TradeSignalApprovedEvent
from gal_friday.core.pubsub import PubSubManager
from gal_friday.event_bus import FillEvent, MarketDataEvent
from gal_friday.feature_engine import FeatureEngine
from gal_friday.interfaces.execution_handler_interface import ExecutionHandlerInterface
from gal_friday.logger_service import LoggerService
from gal_friday.portfolio_manager import PortfolioManager
from gal_friday.prediction_service import PredictionService
from gal_friday.risk_manager import RiskManager
from gal_friday.strategy_arbitrator import StrategyArbitrator


class MockExecutionHandler(ExecutionHandlerInterface):
    """A mock execution handler for testing the trading flow."""

    def __init__(self, pubsub):
        """Initialize the mock execution handler.

        Args
        ----
            pubsub: The publish-subscribe manager to use for event communication.
        """
        self.pubsub = pubsub
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.is_running = False

    async def start(self) -> None:
        """Start the execution handler."""
        self.is_running = True

    async def stop(self) -> None:
        """Stop the execution handler."""
        self.is_running = False

    async def handle_trade_signal_approved(self, event: TradeSignalApprovedEvent) -> None:
        """Handle an approved trade signal by simulating order execution."""
        # Create a mock order ID
        order_id = f"mock-order-{datetime.now().timestamp()}"

        # Simulate order placement
        execution_report = FillEvent(
            event_id=f"exec-{order_id}",
            signal_id=event.event_id,
            exchange_order_id=order_id,
            client_order_id=f"client-{order_id}",
            trading_pair=event.trading_pair,
            exchange="kraken",
            order_status="NEW",
            order_type=event.order_type,
            side=event.side,
            quantity_ordered=event.quantity,
            quantity_filled="0",
            average_fill_price=None,
            limit_price=getattr(event, "limit_price", None),
            stop_price=None,
            commission=None,
            commission_asset=None,
        )

        # Publish the execution report
        await asyncio.sleep(0.1)  # Simulate network delay
        self.pubsub.publish(execution_report)

        # Simulate order fill after a short delay
        await asyncio.sleep(0.2)

        # Determine fill price (either limit price or a simulated market price)
        fill_price = getattr(event, "limit_price", "0.5123")

        # Create a fill execution report
        fill_report = FillEvent(
            event_id=f"fill-{order_id}",
            signal_id=event.event_id,
            exchange_order_id=order_id,
            client_order_id=f"client-{order_id}",
            trading_pair=event.trading_pair,
            exchange="kraken",
            order_status="FILLED",
            order_type=event.order_type,
            side=event.side,
            quantity_ordered=event.quantity,
            quantity_filled=event.quantity,
            average_fill_price=fill_price,
            limit_price=getattr(event, "limit_price", None),
            stop_price=None,
            commission="0.001",
            commission_asset="USD",
        )

        # Publish the fill report
        self.pubsub.publish(fill_report)

    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Cancel an order."""
        if exchange_order_id in self.orders:
            # Get order details with defaults for safety
            order_details = self.orders[exchange_order_id]
            trading_pair = order_details.get("trading_pair", "BTC/USD")
            order_type = order_details.get("order_type", "LIMIT")
            side = order_details.get("side", "BUY")
            quantity = order_details.get("quantity", "1.0")
            
            # Simulate order cancellation
            cancel_report = FillEvent(
                event_id=f"cancel-{exchange_order_id}",
                signal_id=order_details.get("signal_id"),
                exchange_order_id=exchange_order_id,
                client_order_id=order_details.get("client_order_id"),
                trading_pair=trading_pair,
                exchange="kraken",
                order_status="CANCELED",
                order_type=order_type,
                side=side,
                quantity_ordered=quantity,
                quantity_filled="0",
                average_fill_price=None,
                limit_price=order_details.get("limit_price"),
                stop_price=None,
                commission=None,
                commission_asset=None,
            )

            # Publish the cancellation report
            self.pubsub.publish(cancel_report)
            return True

        return False


class EventCapture:
    """Utility class to capture and verify events in the integration tests."""

    def __init__(self):
        """Initialize an empty event capture container."""
        self.events = []

    def capture_event(self, event):
        """Store an event for later analysis."""
        self.events.append(event)

    def get_events_of_type(self, event_type):
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def clear(self):
        """Clear stored events."""
        self.events = []


@pytest.fixture
def pubsub():
    """Create a PubSubManager for testing."""
    return PubSubManager()


@pytest.fixture
def event_capture():
    """Create an EventCapture instance for testing."""
    return EventCapture()


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config_manager = MagicMock()

    # Set up the config to return appropriate values for different modules
    def get_config_section(section_path, default=None):
        if section_path == "prediction_service":
            return {
                "process_pool_workers": 2,
                "models": [{
                    "model_id": "test_model_xrp",
                    "trading_pair": "XRP/USD",
                    "model_path": "mock/path/model.joblib",
                    "model_type": "sklearn",
                    "model_feature_names": ["rsi_14", "macd", "spread_pct"],
                    "prediction_target": "prob_price_up_0.1pct_5min",
                }],
            }
        elif section_path == "strategy":
            return {
                "signal_threshold": 0.65,
                "signal_lookback_bars": 3,
                "price_precision": 4,
                "quantity_precision": 1,
                "fixed_sl_pct": 0.01,
                "fixed_tp_pct": 0.02,
            }
        elif section_path == "risk_management":
            return {
                "max_drawdown_pct": 15.0,
                "daily_drawdown_pct": 2.0,
                "weekly_drawdown_pct": 5.0,
                "risk_per_trade_pct": 0.5,
                "max_exposure_pct": 25.0,
                "max_single_exposure_pct": 10.0,
                "max_consecutive_losses": 5,
            }
        elif section_path == "portfolio":
            return {"reconciliation_interval_sec": 3600}

        return default

    config_manager.get.side_effect = get_config_section
    return config_manager


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock(spec=LoggerService)


@pytest.fixture
def mock_market_data_event():
    """Create a mock market data event for testing."""
    return MarketDataEvent(
        event_id="market-data-1",
        timestamp=datetime.now(),
        trading_pair="XRP/USD",
        exchange="kraken",
        price=Decimal("0.5123"),
        bid=Decimal("0.5120"),
        ask=Decimal("0.5125"),
        volume=Decimal("10000.0"),
    )


@pytest.mark.asyncio
async def test_end_to_end_trading_flow(
    pubsub, event_capture, mock_config, mock_logger, mock_market_data_event
):
    """Test the full end-to-end trading flow from market data to order execution."""
    # Set up event capturing
    for event_type in [
        EventType.MARKET_DATA_RAW,
        EventType.FEATURE_CALCULATED,
        EventType.PREDICTION_GENERATED,
        EventType.TRADE_SIGNAL_PROPOSED,
        EventType.TRADE_SIGNAL_APPROVED,
        EventType.EXECUTION_REPORT,
    ]:
        pubsub.subscribe(event_type, event_capture.capture_event)

    # Mock components to focus testing on the flow
    with patch(
        "gal_friday.feature_engine.FeatureEngine._calculate_indicators"
    ) as mock_calc_indicators:
        with patch("gal_friday.prediction_service.ProcessPoolExecutor") as mock_pool:
            # Set up mock feature calculation
            mock_calc_indicators.return_value = pd.DataFrame(
                {"rsi_14": [65.0], "macd": [0.001], "spread_pct": [0.002]}
            )

            # Set up mock prediction
            mock_future = MagicMock()
            mock_future.result.return_value = {"prediction": 0.75, "confidence": 0.8}
            mock_pool_instance = MagicMock()
            mock_pool_instance.submit.return_value = mock_future
            mock_pool.return_value = mock_pool_instance

            # Initialize components
            feature_engine = FeatureEngine(mock_config, pubsub, mock_logger)

            # Mock the process pool for prediction service
            process_pool = MagicMock()
            process_pool.submit.return_value = mock_future

            prediction_service = PredictionService(
                config=mock_config,
                pubsub=pubsub,
                logger=mock_logger,
                process_pool=process_pool,
            )

            strategy = StrategyArbitrator(mock_config, pubsub, mock_logger)

            # Mock initial portfolio state
            portfolio = PortfolioManager(mock_config, pubsub, mock_logger)
            portfolio._equity = Decimal("100000.0")  # $100k starting capital
            portfolio._positions = {}  # No positions

            risk_manager = RiskManager(mock_config, pubsub, mock_logger, portfolio)

            # Create a mock execution handler
            execution_handler = MockExecutionHandler(pubsub)

            # Start all components
            await feature_engine.start()
            await prediction_service.start()
            await strategy.start()
            await portfolio.start()
            await risk_manager.start()
            await execution_handler.start()

            try:
                # Publish a market data event to start the flow
                pubsub.publish(mock_market_data_event)

                # Wait for the full flow to complete
                await asyncio.sleep(1.0)

                # Verify event flow
                market_data_events = event_capture.get_events_of_type(EventType.MARKET_DATA_RAW)
                feature_events = event_capture.get_events_of_type(EventType.FEATURE_CALCULATED)
                prediction_events = event_capture.get_events_of_type(
                    EventType.PREDICTION_GENERATED
                )
                signal_proposed_events = event_capture.get_events_of_type(
                    EventType.TRADE_SIGNAL_PROPOSED
                )
                signal_approved_events = event_capture.get_events_of_type(
                    EventType.TRADE_SIGNAL_APPROVED
                )
                execution_events = event_capture.get_events_of_type(EventType.EXECUTION_REPORT)

                # Verify each step in the flow occurred
                assert len(market_data_events) >= 1, "No market data events captured"
                assert len(feature_events) >= 1, "No feature events captured"
                assert len(prediction_events) >= 1, "No prediction events captured"
                assert len(signal_proposed_events) >= 1, "No trade signal proposed events captured"
                assert len(signal_approved_events) >= 1, "No trade signal approved events captured"
                assert (
                    len(execution_events) >= 2
                ), "Not enough execution events captured"  # NEW and FILLED

                # Verify the chain of events is consistent
                trading_pair = market_data_events[0].trading_pair
                assert feature_events[0].trading_pair == trading_pair
                assert prediction_events[0].trading_pair == trading_pair
                assert signal_proposed_events[0].trading_pair == trading_pair
                assert signal_approved_events[0].trading_pair == trading_pair
                assert execution_events[0].trading_pair == trading_pair

                # Verify the execution resulted in a portfolio update
                filled_event = [e for e in execution_events if e.order_status == "FILLED"][0]
                assert filled_event is not None
                assert filled_event.quantity_filled is not None

            finally:
                # Stop all components
                await execution_handler.stop()
                await risk_manager.stop()
                await portfolio.stop()
                await strategy.stop()
                await prediction_service.stop()
                await feature_engine.stop()


@pytest.mark.asyncio
async def test_trading_flow_with_rejected_signal(
    pubsub, event_capture, mock_config, mock_logger, mock_market_data_event
):
    """Test the trading flow when a signal is rejected by risk management."""
    # Set up event capturing
    for event_type in [
        EventType.MARKET_DATA_RAW,
        EventType.FEATURE_CALCULATED,
        EventType.PREDICTION_GENERATED,
        EventType.TRADE_SIGNAL_PROPOSED,
        EventType.TRADE_SIGNAL_REJECTED,
    ]:
        pubsub.subscribe(event_type, event_capture.capture_event)

    # Mock components to focus testing on the flow
    with patch(
        "gal_friday.feature_engine.FeatureEngine._calculate_indicators"
    ) as mock_calc_indicators:
        with patch("gal_friday.prediction_service.ProcessPoolExecutor") as mock_pool:
            # Set up mock feature calculation
            mock_calc_indicators.return_value = pd.DataFrame(
                {"rsi_14": [65.0], "macd": [0.001], "spread_pct": [0.002]}
            )

            # Set up mock prediction
            mock_future = MagicMock()
            mock_future.result.return_value = {"prediction": 0.75, "confidence": 0.8}
            mock_pool_instance = MagicMock()
            mock_pool_instance.submit.return_value = mock_future
            mock_pool.return_value = mock_pool_instance

            # Initialize components
            feature_engine = FeatureEngine(mock_config, pubsub, mock_logger)

            # Mock the process pool for prediction service
            process_pool = MagicMock()
            process_pool.submit.return_value = mock_future

            prediction_service = PredictionService(
                config=mock_config,
                pubsub=pubsub,
                logger=mock_logger,
                process_pool=process_pool,
            )

            strategy = StrategyArbitrator(mock_config, pubsub, mock_logger)

            # Set up a portfolio manager that makes the risk manager reject trades
            # due to a simulated maximum exposure
            portfolio = PortfolioManager(mock_config, pubsub, mock_logger)
            portfolio._equity = Decimal("100000.0")

            # Simulate a large existing position that would cause risk limits to be hit
            portfolio._positions = {
                "XRP/USD": {
                    "quantity": Decimal("50000"),
                    "average_entry_price": Decimal("0.5"),
                    "current_price": Decimal("0.51"),
                    "unrealized_pnl": Decimal("500"),
                }
            }

            # Override the risk manager's get_current_state method to use our test portfolio state
            risk_manager = RiskManager(mock_config, pubsub, mock_logger, portfolio)

            # Start all components
            await feature_engine.start()
            await prediction_service.start()
            await strategy.start()
            await portfolio.start()
            await risk_manager.start()

            try:
                # Publish a market data event to start the flow
                pubsub.publish(mock_market_data_event)

                # Wait for the flow to complete
                await asyncio.sleep(1.0)

                # Verify event flow
                market_data_events = event_capture.get_events_of_type(EventType.MARKET_DATA_RAW)
                feature_events = event_capture.get_events_of_type(EventType.FEATURE_CALCULATED)
                prediction_events = event_capture.get_events_of_type(
                    EventType.PREDICTION_GENERATED
                )
                signal_proposed_events = event_capture.get_events_of_type(
                    EventType.TRADE_SIGNAL_PROPOSED
                )
                signal_rejected_events = event_capture.get_events_of_type(
                    EventType.TRADE_SIGNAL_REJECTED
                )

                # Verify each step in the flow occurred
                assert len(market_data_events) >= 1, "No market data events captured"
                assert len(feature_events) >= 1, "No feature events captured"
                assert len(prediction_events) >= 1, "No prediction events captured"
                assert len(signal_proposed_events) >= 1, "No trade signal proposed events captured"
                assert len(signal_rejected_events) >= 1, "No trade signal rejected events captured"

                # Verify the chain of events is consistent
                trading_pair = market_data_events[0].trading_pair
                assert feature_events[0].trading_pair == trading_pair
                assert prediction_events[0].trading_pair == trading_pair
                assert signal_proposed_events[0].trading_pair == trading_pair
                assert signal_rejected_events[0].trading_pair == trading_pair

                # Verify the rejection has a reason
                assert signal_rejected_events[0].reason is not None
                assert len(signal_rejected_events[0].reason) > 0

            finally:
                # Stop all components
                await risk_manager.stop()
                await portfolio.stop()
                await strategy.stop()
                await prediction_service.stop()
                await feature_engine.stop()
