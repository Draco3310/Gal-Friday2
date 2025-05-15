"""Position management functionality for the portfolio system."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Optional

from ..exceptions import DataValidationError
from ..logger_service import LoggerService


@dataclass
class TradeInfo:
    """Represents information about a single trade."""

    timestamp: datetime
    trade_id: str
    side: str  # "BUY" or "SELL"
    quantity: Decimal
    price: Decimal
    fee: Decimal = Decimal(0)
    fee_currency: Optional[str] = None
    commission: Decimal = Decimal(0)
    commission_asset: Optional[str] = None


@dataclass
class PositionInfo:
    """Maintains information about a trading position."""

    trading_pair: str
    base_asset: str
    quote_asset: str
    quantity: Decimal = Decimal(0)
    average_entry_price: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    trade_history: list[TradeInfo] = field(default_factory=list)
    last_updated: Optional[datetime] = None


class PositionManager:
    """
    Manages trading positions, including updates from trades and PnL calculations.
    """

    def __init__(self, logger_service: LoggerService) -> None:
        """
        Initialize the position manager.

        Args
        ----
            logger_service: Service for logging.
        """
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self._positions: dict[str, PositionInfo] = {}
        self._lock = asyncio.Lock()

    @property
    def positions(self) -> dict[str, PositionInfo]:
        """Return all positions as a read-only copy."""
        return self._positions.copy()

    def get_open_positions(self) -> list[PositionInfo]:
        """Return a list of open positions (non-zero quantity)."""
        return [pos for pos in self._positions.values() if pos.quantity != 0]

    def get_position(self, trading_pair: str) -> Optional[PositionInfo]:
        """Get a specific position by trading pair."""
        return self._positions.get(trading_pair)

    async def initialize_positions(
        self,
        initial_positions: dict[str, dict[str, Any]],
        split_symbol_func: Callable[[str], tuple[str, str]],
    ) -> None:
        """
        Initialize positions from a dictionary (e.g., from config or database).

        Args
        ----
            initial_positions: Dict where keys are trading_pair and values are dicts
                               of position attributes (quantity, average_entry_price, etc.).
            split_symbol_func: A function to split trading_pair into base and quote.
        """
        async with self._lock:
            self._positions.clear()
            for pair, pos_data in initial_positions.items():
                try:
                    base, quote = split_symbol_func(pair)
                    position = PositionInfo(
                        trading_pair=pair,
                        base_asset=base,
                        quote_asset=quote,
                        quantity=Decimal(str(pos_data.get("quantity", 0))),
                        average_entry_price=Decimal(str(pos_data.get("average_entry_price", 0))),
                        realized_pnl=Decimal(str(pos_data.get("realized_pnl", 0))),
                        last_updated=datetime.now().astimezone()
                    )
                    self._positions[pair] = position
                except ValueError:
                    self.logger.exception(
                        "Invalid trading pair format in initial_positions: %s", pair,
                        source_module=self._source_module,
                    )
                except Exception as e:
                    self.logger.exception(
                        "Error loading initial position for %s: %s", pair, str(e),
                        source_module=self._source_module,
                    )
            self.logger.info("Initialized %d positions.", len(self._positions), source_module=self._source_module)

    async def update_position_for_trade(
        self,
        trading_pair: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        timestamp: datetime,
        trade_id: str,
        fee: Decimal = Decimal(0),
        fee_currency: Optional[str] = None,
        commission: Decimal = Decimal(0),
        commission_asset: Optional[str] = None,
    ) -> tuple[Decimal, Optional[PositionInfo]]:
        """
        Update position based on trade execution.

        Calculates realized PnL for SELL trades that reduce or close a position.

        Args
        ----
            trading_pair: The trading pair (e.g., "BTC/USD").
            side: "BUY" or "SELL".
            quantity: Amount of base asset traded.
            price: Execution price.
            timestamp: Time of the trade.
            trade_id: Unique identifier for the trade.
            fee: Trading fee amount.
            fee_currency: Currency of the fee.
            commission: Additional commission paid.
            commission_asset: Asset in which commission is paid.

        Returns
        -------
            A tuple containing: (realized_pnl_for_this_trade, updated_position_info_or_None).
            Returns (Decimal(0), None) if trade validation fails.
        """
        try:
            self._validate_trade_params(trading_pair, side, quantity, price)
        except DataValidationError as e:
            self.logger.error("Trade validation failed for %s: %s", trading_pair, e, source_module=self._source_module)
            return Decimal(0), None

        async with self._lock:
            position = await self._get_or_create_position(trading_pair)
            realized_pnl_trade = Decimal(0)

            trade_record = self._create_trade_record(
                timestamp, trade_id, side, quantity, price,
                fee, fee_currency, commission, commission_asset
            )
            position.trade_history.append(trade_record)

            if side == "BUY":
                new_total_cost = (position.average_entry_price * position.quantity) + (price * quantity)
                position.quantity += quantity
                if position.quantity != 0:
                    position.average_entry_price = new_total_cost / position.quantity
                else:
                    position.average_entry_price = Decimal(0)
            elif side == "SELL":
                if position.quantity < quantity:
                    self.logger.warning(
                        "Selling %s of %s, but current position is %s. Position will go negative or this is a short sale.",
                        quantity, trading_pair, position.quantity,
                        source_module=self._source_module
                    )

                realized_pnl_trade = (price - position.average_entry_price) * quantity
                position.realized_pnl += realized_pnl_trade
                position.quantity -= quantity
                if position.quantity == 0:
                    position.average_entry_price = Decimal(0)

            position.last_updated = timestamp
            self._positions[trading_pair] = position
            self.logger.info(
                "Updated position for %s: Side %s, Qty %s, Price %s. New Pos Qty: %s, AEP: %s. PnL This Trade: %s",
                trading_pair, side, quantity, price, position.quantity, position.average_entry_price, realized_pnl_trade,
                source_module=self._source_module
            )
            return realized_pnl_trade, position

    def _validate_trade_params(self, trading_pair: str, side: str, quantity: Decimal, price: Decimal) -> None:
        """Validate trade parameters."""
        if side not in ("BUY", "SELL"):
            raise DataValidationError("Invalid trade side")

        if quantity <= Decimal(0):
            raise DataValidationError("Invalid quantity")

        if price <= Decimal(0):
            raise DataValidationError("Invalid price")

    async def _get_or_create_position(self, trading_pair: str) -> PositionInfo:
        """Retrieve an existing position or create a new one if not found."""
        if trading_pair not in self._positions:
            try:
                parts = trading_pair.split("/")
                if len(parts) != 2:
                    raise ValueError("Trading pair format must be BASE/QUOTE")
                base_asset, quote_asset = parts[0].strip(), parts[1].strip()
                if not base_asset or not quote_asset:
                     raise ValueError("Base or quote asset cannot be empty")

                self._positions[trading_pair] = PositionInfo(
                    trading_pair=trading_pair, base_asset=base_asset, quote_asset=quote_asset
                )
                self.logger.info("Created new position for %s", trading_pair, source_module=self._source_module)
            except ValueError as e:
                self.logger.error("Failed to parse trading pair %s to create position: %s", trading_pair, e, source_module=self._source_module)
                raise DataValidationError(f"Invalid trading pair format for position creation: {trading_pair}") from e
        return self._positions[trading_pair]

    def _create_trade_record(
        self,
        timestamp: datetime,
        trade_id: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal,
        fee_currency: Optional[str],
        commission: Decimal,
        commission_asset: Optional[str],
    ) -> TradeInfo:
        """Helper to create a TradeInfo object."""
        return TradeInfo(
            timestamp=timestamp,
            trade_id=trade_id,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            fee_currency=fee_currency,
            commission=commission,
            commission_asset=commission_asset,
        )

    async def get_total_realized_pnl(self, trading_pair: Optional[str] = None) -> Decimal:
        """Get total realized PnL for a specific pair or all pairs."""
        async with self._lock:
            if trading_pair:
                pos = self._positions.get(trading_pair)
                return pos.realized_pnl if pos else Decimal(0)
            return sum(pos.realized_pnl for pos in self._positions.values())
