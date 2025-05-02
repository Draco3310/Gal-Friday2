import logging
from typing import (  # noqa: F401
    Optional,
    Any,
    TYPE_CHECKING,  # Used in type hints and conditional imports
    Dict,
    List,
    Tuple,
    Union,
)
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
import pandas as pd  # noqa: F401 # Used in type hints for pd.Series
from dataclasses import dataclass, field  # noqa: F401 # Used in placeholder classes
import decimal  # noqa: F401 # Used in error handling
import uuid  # Add missing uuid import
import asyncio  # Required for test coroutines

# Set Decimal precision
getcontext().prec = 28

if TYPE_CHECKING:
    from .core.events import (
        Event,  # Base class for events
        EventType,  # Event type enumeration
        TradeSignalApprovedEvent,
        ExecutionReportEvent
    )
    from .core.pubsub import PubSubManager
    from .config_manager import ConfigManager
    from .logger_service import LoggerService
    from .historical_data_service import HistoricalDataService
else:
    # Import Event-related classes from core placeholders
    from .core.placeholder_classes import (  # noqa: F401 # Required for event system
        Event,
        EventType,
        TradeSignalApprovedEvent,
        ExecutionReportEvent,
        ConfigManager,
        PubSubManager,
        HistoricalDataService,
    )

# log = logging.getLogger(__name__) # Removed module-level logger


class SimulatedExecutionHandler:
    """Simulates order execution based on historical OHLCV data for backtesting."""

    def __init__(
        self,
        config_manager: "ConfigManager",
        pubsub_manager: "PubSubManager",
        data_service: "HistoricalDataService",
        logger_service: "LoggerService",
    ):  # Added logger_service
        self.config = config_manager
        self.pubsub = pubsub_manager
        self.data_service = data_service
        self.logger = logger_service  # Assigned injected logger

        # Load configuration
        self.taker_fee_pct = self.config.get_decimal(
            "backtest.commission_taker_pct",
            Decimal("0.0026")
        )
        self.maker_fee_pct = self.config.get_decimal(
            "backtest.commission_maker_pct",
            Decimal("0.0016")
        )
        self.slippage_model = self.config.get(
            "backtest.slippage_model",
            "volatility"
        )
        self.slip_atr_multiplier = self.config.get_decimal(
            "backtest.slippage_atr_multiplier",
            Decimal("0.1")
        )
        self.slip_fixed_pct = self.config.get_decimal(
            "backtest.slippage_fixed_pct",
            Decimal("0.0005")
        )
        self.valuation_currency = self.config.get(
            "portfolio.valuation_currency",
            "USD"
        ).upper()

        self.logger.info(
            "SimulatedExecutionHandler initialized.",
            source_module=self.__class__.__name__
        )
        self.logger.info(
            (
                f" Fees: Taker={self.taker_fee_pct*100:.4f}%, "
                f"Maker={self.maker_fee_pct*100:.4f}% (Maker not used in MVP)"
            ),
            source_module=self.__class__.__name__,
        )
        self.logger.info(
            (
                f" Slippage: Model={self.slippage_model}, "
                f"ATR Mult={self.slip_atr_multiplier}, "
                f"Fixed Pct={self.slip_fixed_pct*100:.4f}%"
            ),
            source_module=self.__class__.__name__,
        )

    async def start(self) -> None:  # Interface consistency
        # log.info("SimulatedExecutionHandler started.") # Replaced
        self.logger.info(
            "SimulatedExecutionHandler started.", source_module=self.__class__.__name__
        )
        # No external connections to establish

    async def stop(self) -> None:  # Interface consistency
        # log.info("SimulatedExecutionHandler stopped.") # Replaced
        self.logger.info(
            "SimulatedExecutionHandler stopped.", source_module=self.__class__.__name__
        )
        # No external connections to close

    async def handle_trade_signal_approved(self, event: "TradeSignalApprovedEvent") -> None:
        """Processes an approved signal, simulates fill based on next bar data."""
        self.logger.debug(
            (
                f"SimExec received approved signal: "
                f"{event.signal_id} at {event.timestamp}"
            ),
            source_module=self.__class__.__name__,
        )

        # Get next bar data
        next_bar = await self._get_next_bar_data(event)
        if (next_bar is None):
            return

        try:
            # Initialize simulation parameters
            fill_result = await self._simulate_order_fill(event, next_bar)
            if not fill_result:
                return

            # Publish the execution report
            await self._publish_simulated_report(
                event,
                fill_result["status"],
                fill_result["quantity"],
                fill_result["fill_price"],
                fill_result["commission"],
                fill_result["commission_asset"],
                fill_result.get("error_msg"),
                fill_result["timestamp"],
            )

        except Exception as e:
            error_msg = (
                f"Error during fill simulation for signal {event.signal_id}. "
                f"Event: {event}, Next Bar: {next_bar}"
            )
            self.logger.error(
                error_msg,
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            await self._publish_simulated_report(
                event,
                "ERROR",
                Decimal(0),
                None,
                Decimal(0),
                None,
                f"Simulation error: {e}",
            )

    async def _get_next_bar_data(
        self,
        event: "TradeSignalApprovedEvent",
    ) -> Optional[pd.Series]:
        """Gets and validates the next bar data for simulation."""
        try:
            next_bar = self.data_service.get_next_bar(event.trading_pair, event.timestamp)

            if next_bar is None:
                error_msg = (
                    f"No next bar data found for {event.trading_pair} after "
                    f"{event.timestamp}. Cannot simulate fill for {event.signal_id}."
                )
                self.logger.warning(
                    error_msg,
                    source_module=self.__class__.__name__,
                )
                next_time = event.timestamp + timedelta(minutes=1)
                error_detail = f"No historical data for fill simulation at {next_time}"
                await self._publish_simulated_report(
                    event,
                    "REJECTED",
                    Decimal(0),
                    None,
                    Decimal(0),
                    None,
                    error_detail,
                )
                return None

            return next_bar
        except Exception as e:
            self.logger.error(
                f"Error retrieving next bar data: {e}",
                source_module=self.__class__.__name__,
                exc_info=True
            )
            return None

    async def _simulate_order_fill(
        self,
        event: "TradeSignalApprovedEvent",
        next_bar: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """Simulates order fill based on order type and market conditions."""
        fill_qty = event.quantity  # Assume full fill for MVP
        commission_pct = self.taker_fee_pct  # Assume taker for MVP
        
        # Convert index name to datetime if needed
        if hasattr(next_bar, 'name') and next_bar.name is not None:
            # Make sure fill_timestamp is a datetime object, not just a Hashable index
            if isinstance(next_bar.name, datetime):
                fill_timestamp = next_bar.name
            else:
                # Try to convert to datetime or use current time
                # First convert to a specific type that to_datetime can handle
                try:
                    # Convert the Hashable to a string first if needed
                    name_str = str(next_bar.name)
                    fill_timestamp = pd.to_datetime(name_str)
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"Could not convert bar index {next_bar.name} to datetime. Using current time.",
                        source_module=self.__class__.__name__,
                    )
                    fill_timestamp = datetime.utcnow()
        else:
            self.logger.warning(
                "Bar has no name/index. Using current time for fill timestamp.",
                source_module=self.__class__.__name__,
            )
            fill_timestamp = datetime.utcnow()

        # Handle different order types
        if event.order_type.upper() == "MARKET":
            return await self._simulate_market_order(
                event,
                next_bar,
                fill_qty,
                commission_pct,
                fill_timestamp,
            )
        elif event.order_type.upper() == "LIMIT":
            return await self._simulate_limit_order(
                event,
                next_bar,
                fill_qty,
                commission_pct,
                fill_timestamp,
            )
        else:
            error_msg = (
                f"Unsupported order type '{event.order_type}' for simulation. "
                f"Signal {event.signal_id}."
            )
            self.logger.error(
                error_msg,
                source_module=self.__class__.__name__,
            )
            await self._publish_simulated_report(
                event,
                "REJECTED",
                Decimal(0),
                None,
                Decimal(0),
                None,
                f"Unsupported order type: {event.order_type}",
            )
            return None

    async def _simulate_market_order(
        self,
        event: "TradeSignalApprovedEvent",
        next_bar: pd.Series,
        fill_qty: Decimal,
        commission_pct: Decimal,
        fill_timestamp: datetime,
    ) -> dict:
        """Simulates a market order fill."""
        fill_price_base = next_bar["open"]  # Assume fill at next bar's open
        slippage = self._calculate_slippage(
            event.trading_pair,
            event.side,
            Decimal(fill_price_base),
            event.timestamp
        )

        # Calculate fill price with slippage
        simulated_fill_price = (
            Decimal(fill_price_base) + slippage
            if event.side.upper() == "BUY"
            else Decimal(fill_price_base) - slippage
        )

        self.logger.debug(
            (
                f" Market fill sim: Base={fill_price_base}, "
                f"Slip={slippage:.6f}, Final={simulated_fill_price:.6f}"
            ),
            source_module=self.__class__.__name__,
        )

        # Calculate commission
        fill_value = fill_qty * simulated_fill_price
        commission_amount = abs(fill_value * commission_pct)
        _, quote_asset = event.trading_pair.split("/")
        commission_asset = quote_asset.upper()

        # Convert fill_timestamp if needed to ensure it's a datetime object
        timestamp = fill_timestamp
        if not isinstance(timestamp, datetime):
            self.logger.warning(
                f"Converting fill_timestamp from {type(timestamp)} to datetime",
                source_module=self.__class__.__name__,
            )
            # Default to current time if conversion fails
            timestamp = datetime.utcnow()

        return {
            "status": "FILLED",
            "quantity": fill_qty,
            "fill_price": simulated_fill_price,
            "commission": commission_amount,
            "commission_asset": commission_asset,
            "timestamp": timestamp,
        }

    async def _simulate_limit_order(
        self,
        event: "TradeSignalApprovedEvent",
        next_bar: pd.Series,
        fill_qty: Decimal,
        commission_pct: Decimal,
        fill_timestamp: datetime,
    ) -> Optional[dict]:
        """Simulates a limit order fill."""
        limit_price = event.limit_price
        if limit_price is None:
            error_msg = (
                f"Limit price missing for signal {event.signal_id}. "
                "Cannot simulate."
            )
            self.logger.error(
                error_msg,
                source_module=self.__class__.__name__,
            )
            await self._publish_simulated_report(
                event,
                "REJECTED",
                Decimal(0),
                None,
                Decimal(0),
                None,
                "Limit price missing",
            )
            return None

        # Check if order would be filled
        filled = self._check_limit_order_fill(event.side, limit_price, next_bar)
        if not filled:
            bar_info = (
                f"Limit={limit_price}, "
                f"Bar H/L={next_bar['high']}/{next_bar['low']}"
            )
            self.logger.debug(
                f" Limit order {event.signal_id} NOT filled. {bar_info}",
                source_module=self.__class__.__name__,
            )
            await self._publish_simulated_report(
                event,
                "REJECTED",
                Decimal(0),
                None,
                Decimal(0),
                None,
                "Limit price not reached",
            )
            return None

        # Calculate commission
        fill_value = fill_qty * limit_price
        commission_amount = abs(fill_value * commission_pct)
        _, quote_asset = event.trading_pair.split("/")
        commission_asset = quote_asset.upper()

        self.logger.debug(
            f" Limit fill sim: Price={limit_price:.6f}",
            source_module=self.__class__.__name__,
        )

        # Convert fill_timestamp if needed to ensure it's a datetime object
        timestamp = fill_timestamp
        if not isinstance(timestamp, datetime):
            self.logger.warning(
                f"Converting fill_timestamp from {type(timestamp)} to datetime",
                source_module=self.__class__.__name__,
            )
            # Default to current time if conversion fails
            timestamp = datetime.utcnow()

        return {
            "status": "FILLED",
            "quantity": fill_qty,
            "fill_price": limit_price,
            "commission": commission_amount,
            "commission_asset": commission_asset,
            "timestamp": timestamp,
        }

    def _check_limit_order_fill(
        self,
        side: str,
        limit_price: Decimal,
        next_bar: pd.Series,
    ) -> bool:
        """Checks if a limit order would be filled based on price levels."""
        try:
            if side.upper() == "BUY":
                return bool(next_bar["low"] <= limit_price)
            else:  # SELL
                return bool(next_bar["high"] >= limit_price)
        except Exception as e:
            self.logger.error(
                f"Error checking limit order fill: {e}",
                source_module=self.__class__.__name__,
                exc_info=True
            )
            return False

    def _calculate_slippage(
        self,
        trading_pair: str,
        side: str,
        base_price: Decimal,
        signal_timestamp: datetime
    ) -> Decimal:
        """Calculates slippage based on configured model."""
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
                        (
                            f"Could not get ATR for {trading_pair} at {signal_timestamp} "
                            "for slippage calc. Using zero."
                        ),
                        source_module=self.__class__.__name__,
                    )
            else:
                self.logger.warning(
                    f"Unknown slippage model: {self.slippage_model}. Using zero.",
                    source_module=self.__class__.__name__,
                )
        except Exception as e:
            self.logger.error(
                f"Error calculating slippage: {e}",
                source_module=self.__class__.__name__,
                exc_info=True
            )
            
        # Slippage is always adverse
        return abs(slippage)

    async def _publish_simulated_report(
        self,
        originating_event: "TradeSignalApprovedEvent",
        status: str,
        qty_filled: Decimal,
        avg_price: Optional[Decimal],
        commission: Decimal,
        commission_asset: Optional[str],
        error_msg: Optional[str] = None,
        fill_timestamp: Optional[datetime] = None,
    ) -> None:
        """Helper to create and publish a simulated ExecutionReportEvent."""
        try:
            # Generate unique simulation order ID
            timestamp_micros = int(datetime.utcnow().timestamp() * 1e6)
            exchange_order_id = (
                f"sim_{originating_event.signal_id}_{timestamp_micros}"
            )
            client_order_id = f"sim_{originating_event.signal_id}"

            report = ExecutionReportEvent(
                source_module=self.__class__.__name__,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                signal_id=originating_event.signal_id,
                exchange_order_id=exchange_order_id,
                client_order_id=client_order_id,
                trading_pair=originating_event.trading_pair,
                exchange=originating_event.exchange,
                order_status=status,
                order_type=originating_event.order_type,
                side=originating_event.side,
                quantity_ordered=Decimal(originating_event.quantity),
                quantity_filled=qty_filled,
                average_fill_price=avg_price,
                limit_price=originating_event.limit_price,
                stop_price=None,  # Not applicable for MVP fills
                commission=commission,
                commission_asset=commission_asset,
                timestamp_exchange=(
                    fill_timestamp if fill_timestamp else datetime.utcnow()
                ),
                error_message=error_msg,
            )
            await self.pubsub.publish(report)
            self.logger.debug(
                f"Published simulated report: {status} for {report.signal_id}",
                source_module=self.__class__.__name__,
            )
        except Exception as e:
            self.logger.error(
                (
                    f"Failed to publish simulated execution report for "
                    f"signal {originating_event.signal_id}: {e}"
                ),
                source_module=self.__class__.__name__,
                exc_info=True,
            )


# Example Usage
async def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a simple logger for this example
    logger = logging.getLogger("sim_exec_example")
    
    # Use placeholders with proper implementations
    config = ConfigManager()
    pubsub = PubSubManager(logger=logger)
    
    # Create a concrete implementation of HistoricalDataService with required methods
    class MockHistoricalDataService(HistoricalDataService):
        def get_next_bar(self, trading_pair: str, timestamp: datetime) -> Optional[pd.Series]:
            # Return a dummy Series with required columns
            idx = pd.date_range(
                start="2023-01-01 00:01:00",
                periods=5,
                freq="1min",
                tz="UTC"
            )
            dummy_data = {
                'open': [0.495, 0.496, 0.497, 0.498, 0.499],
                'high': [0.50, 0.505, 0.51, 0.515, 0.52],
                'low': [0.4951, 0.4952, 0.4953, 0.4954, 0.4955],
                'close': [0.497, 0.498, 0.499, 0.50, 0.51],
                'volume': [1000, 1100, 1200, 1300, 1400]
            }
            df = pd.DataFrame(dummy_data, index=idx)
            # Return the second row which would be the "next" bar
            return df.iloc[1]
            
        def get_atr(self, trading_pair: str, timestamp: datetime, period: int = 14) -> Optional[Decimal]:
            # Return a dummy ATR value
            return Decimal("0.0025")
            
        async def get_historical_ohlcv(
            self, 
            trading_pair: str, 
            start_time: datetime, 
            end_time: datetime, 
            interval: str
        ) -> Optional[pd.DataFrame]:
            # Required abstract method implementation
            return pd.DataFrame()
            
        async def get_historical_trades(
            self, 
            trading_pair: str, 
            start_time: datetime, 
            end_time: datetime
        ) -> Optional[pd.DataFrame]:
            # Required abstract method implementation
            return pd.DataFrame()
    
    # Use our mock implementation
    data_service = MockHistoricalDataService()
    
    # Create a proper LoggerService implementation
    from .logger_service import LoggerService
    
    class MockLoggerService(LoggerService):
        def __init__(self) -> None:
            pass
            
        def log(self, level: int, msg: str, source_module: Optional[str] = None, 
                context: Optional[Dict[Any, Any]] = None, exc_info: Optional[bool] = None) -> None:
            logger.log(level, f"[{source_module}] {msg}", exc_info=exc_info)
            
        def info(self, msg: str, source_module: Optional[str] = None, 
                 context: Optional[Dict[Any, Any]] = None) -> None:
            logger.info(f"[{source_module}] {msg}")
            
        def debug(self, msg: str, source_module: Optional[str] = None, 
                  context: Optional[Dict[Any, Any]] = None) -> None:
            logger.debug(f"[{source_module}] {msg}")
            
        def warning(self, msg: str, source_module: Optional[str] = None, 
                    context: Optional[Dict[Any, Any]] = None) -> None:
            logger.warning(f"[{source_module}] {msg}")
            
        def error(self, msg: str, source_module: Optional[str] = None, 
                  context: Optional[Dict[Any, Any]] = None, exc_info: Optional[bool] = None) -> None:
            logger.error(f"[{source_module}] {msg}", exc_info=exc_info)
            
        def critical(self, msg: str, source_module: Optional[str] = None, 
                     context: Optional[Dict[Any, Any]] = None, exc_info: Optional[bool] = None) -> None:
            logger.critical(f"[{source_module}] {msg}", exc_info=exc_info)
    
    logger_service = MockLoggerService()
    
    sim_exec = SimulatedExecutionHandler(config, pubsub, data_service, logger_service)
    await sim_exec.start()

    # Ensure proper datetime objects are created for fill_timestamp attribute
    test_datetime = datetime(2023, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)

    # --- Test Market Order Buy --- #
    signal_market_buy = TradeSignalApprovedEvent(
        source_module="TestModule",
        event_id=uuid.uuid4(),
        timestamp=datetime.utcnow(),
        signal_id=uuid.uuid4(),  # Use UUID instead of string
        trading_pair="XRP/USD",
        exchange="kraken",
        side="BUY",
        order_type="MARKET",
        quantity=Decimal("500"),
        limit_price=None,
        sl_price=Decimal("0.48"),  # Add required SL price
        tp_price=Decimal("0.52")   # Add required TP price
    )
    print("\n--- Test Market Buy ---")
    await sim_exec.handle_trade_signal_approved(signal_market_buy)

    # --- Test Limit Order Buy (Should Fill) --- #
    signal_limit_buy_fill = TradeSignalApprovedEvent(
        source_module="TestModule",
        event_id=uuid.uuid4(),
        timestamp=datetime.utcnow(),
        signal_id=uuid.uuid4(),
        trading_pair="XRP/USD",
        exchange="kraken",
        side="BUY",
        order_type="LIMIT",
        quantity=Decimal("200"),
        limit_price=Decimal("0.496"),  # Below next bar's low (0.4951)
        sl_price=Decimal("0.48"),
        tp_price=Decimal("0.52")
    )
    print("\n--- Test Limit Buy (Fill) ---")
    await sim_exec.handle_trade_signal_approved(signal_limit_buy_fill)

    # --- Test Limit Order Buy (Should NOT Fill) --- #
    signal_limit_buy_nofill = TradeSignalApprovedEvent(
        source_module="TestModule",
        event_id=uuid.uuid4(),
        timestamp=datetime.utcnow(),
        signal_id=uuid.uuid4(),
        trading_pair="XRP/USD",
        exchange="kraken",
        side="BUY",
        order_type="LIMIT",
        quantity=Decimal("200"),
        limit_price=Decimal("0.40"),  # Way below market
        sl_price=Decimal("0.38"),
        tp_price=Decimal("0.45")
    )
    print("\n--- Test Limit Buy (No Fill) ---")
    await sim_exec.handle_trade_signal_approved(signal_limit_buy_nofill)

    await sim_exec.stop()


if __name__ == "__main__":
    import pytz  # Need pytz for timezone in example
    asyncio.run(main())
