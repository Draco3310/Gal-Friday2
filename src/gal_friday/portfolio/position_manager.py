"""Position management functionality for the portfolio system."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..exceptions import DataValidationError


@dataclass
class TradeInfo:
    """Represents a single completed trade."""

    timestamp: datetime
    side: str  # "BUY" or "SELL"
    quantity: Decimal
    price: Decimal
    commission: Decimal = Decimal(0)
    commission_asset: str = ""


@dataclass
class PositionInfo:
    """Represents a position in a trading pair."""

    trading_pair: str
    base_asset: str
    quote_asset: str
    quantity: Decimal = Decimal(0)
    average_entry_price: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    trade_history: List[TradeInfo] = field(default_factory=list)


class PositionManager:
    """
    Manages trading positions and their updates.

    Handles position creation, updates based on trades, and calculates
    position values based on current market prices.
    """

    def __init__(self, logger_service: Any) -> None:
        """
        Initialize the position manager.

        Args:
            logger_service: Logger service for logging
        """
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self._positions: Dict[str, PositionInfo] = {}
        self._lock = asyncio.Lock()

    @property
    def positions(self) -> Dict[str, PositionInfo]:
        """Get all positions (read-only copy)."""
        return self._positions.copy()

    def get_open_positions(self) -> List[PositionInfo]:
        """Returns a list of open positions (non-zero quantity)."""
        return [pos for pos in self._positions.values() if pos.quantity != 0]

    def get_position(self, trading_pair: str) -> Optional[PositionInfo]:
        """
        Get position information for a specific trading pair.

        Args:
            trading_pair: Trading pair symbol (e.g., "BTC/USD")

        Returns:
            Position info or None if no position exists
        """
        return self._positions.get(trading_pair)

    async def initialize_positions(
        self,
        initial_positions: Dict[str, Dict[str, Any]],
        split_symbol_func: Callable[[str], Tuple[str, str]],
    ) -> None:
        """
        Initialize positions from configuration.

        Args:
            initial_positions: Dictionary of initial positions from config
            split_symbol_func: Function to split trading pair into base/quote
        """
        async with self._lock:
            for pair, pos_data in initial_positions.items():
                try:
                    base, quote = split_symbol_func(pair)
                    self._positions[pair] = PositionInfo(
                        trading_pair=pair,
                        base_asset=base,
                        quote_asset=quote,
                        quantity=Decimal(str(pos_data.get("quantity", 0))),
                        average_entry_price=Decimal(str(pos_data.get("average_entry_price", 0))),
                    )
                except ValueError:
                    self.logger.error(
                        f"Invalid trading pair format in initial_positions: {pair}",
                        source_module=self._source_module,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error loading initial position for {pair}: {e}",
                        source_module=self._source_module,
                        exc_info=True,
                    )

    async def update_position_for_trade(
        self,
        trading_pair: str,
        base_asset: str,
        quote_asset: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        cost_or_proceeds: Decimal,
        timestamp: datetime,
        commission: Decimal = Decimal(0),
        commission_asset: Optional[str] = None,
    ) -> Tuple[Decimal, Optional[PositionInfo]]:
        """
        Updates position information based on trade execution.

        Args:
            trading_pair: Trading pair symbol
            base_asset: Base asset symbol
            quote_asset: Quote asset symbol
            side: Trade side ("BUY" or "SELL")
            quantity: Quantity that was filled
            price: Execution price
            cost_or_proceeds: Total cost or proceeds in quote currency
            timestamp: Execution timestamp
            commission: Commission amount
            commission_asset: Commission asset symbol

        Returns:
            Tuple of (realized_pnl, updated_position)

        Raises:
            DataValidationError: If invalid trade data is provided
        """
        # Validate input parameters
        self._validate_trade_parameters(side, quantity, price)

        # Get or create the position
        position = await self._get_or_create_position(trading_pair, base_asset, quote_asset)

        # Create and add trade record
        trade = self._create_trade_record(
            timestamp, side, quantity, price, commission, commission_asset
        )
        position.trade_history.append(trade)

        # Calculate position update based on trade side and current position
        realized_pnl = Decimal(0)

        if side == "BUY":
            realized_pnl = await self._handle_buy_trade(
                position, quantity, cost_or_proceeds, price
            )
        elif side == "SELL":
            realized_pnl = await self._handle_sell_trade(position, quantity, price)

        return realized_pnl, position

    def _validate_trade_parameters(self, side: str, quantity: Decimal, price: Decimal) -> None:
        """Validates trade parameters."""
        if side not in ("BUY", "SELL"):
            raise DataValidationError(f"Invalid trade side: {side}")

        if quantity <= Decimal(0):
            raise DataValidationError(f"Invalid quantity: {quantity}")

        if price <= Decimal(0):
            raise DataValidationError(f"Invalid price: {price}")

    async def _get_or_create_position(
        self, trading_pair: str, base_asset: str, quote_asset: str
    ) -> PositionInfo:
        """Gets existing position or creates a new one if it doesn't exist."""
        async with self._lock:
            position = self._positions.get(trading_pair)
            if position is None:
                position = PositionInfo(
                    trading_pair=trading_pair, base_asset=base_asset, quote_asset=quote_asset
                )
                self._positions[trading_pair] = position
            return position

    def _create_trade_record(
        self,
        timestamp: datetime,
        side: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
        commission_asset: Optional[str],
    ) -> TradeInfo:
        """Creates a trade record for the trade history."""
        return TradeInfo(
            timestamp=timestamp,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            commission_asset=commission_asset if commission_asset else "",
        )

    async def _handle_buy_trade(
        self,
        position: PositionInfo,
        quantity: Decimal,
        cost: Decimal,
        price: Decimal,
    ) -> Decimal:
        """Handles a BUY trade for a position."""
        realized_pnl = Decimal(0)
        async with self._lock:
            if position.quantity >= 0:
                # Increasing long position or opening a new long position
                await self._increase_long_position(position, quantity, cost)
            else:
                # Reducing or closing a short position, potentially opening a long
                realized_pnl = await self._reduce_short_position(position, quantity, price)

        return realized_pnl

    async def _handle_sell_trade(
        self,
        position: PositionInfo,
        quantity: Decimal,
        price: Decimal,
    ) -> Decimal:
        """Handles a SELL trade for a position."""
        realized_pnl = Decimal(0)
        async with self._lock:
            if position.quantity > 0:
                # Reducing or closing a long position, potentially opening a short
                realized_pnl = await self._reduce_long_position(position, quantity, price)
            else:
                # Increasing short position or opening a new short position
                await self._increase_short_position(position, quantity, price)

        return realized_pnl

    async def _increase_long_position(
        self, position: PositionInfo, quantity: Decimal, cost: Decimal
    ) -> None:
        """Increases a long position."""
        # Calculate new average entry price
        new_total_quantity = position.quantity + quantity
        total_cost = (position.quantity * position.average_entry_price) + cost

        # Update position
        position.quantity = new_total_quantity
        position.average_entry_price = (
            total_cost / new_total_quantity if new_total_quantity > 0 else Decimal(0)
        )

    async def _reduce_short_position(
        self, position: PositionInfo, quantity: Decimal, price: Decimal
    ) -> Decimal:
        """Reduces a short position, potentially flipping to long."""
        realized_pnl = Decimal(0)

        # Calculate realized P&L: (short entry price - close price) * quantity_closed
        quantity_closed = min(abs(position.quantity), quantity)
        realized_pnl = (position.average_entry_price - price) * quantity_closed
        position.realized_pnl += realized_pnl

        # Update quantity
        new_quantity = position.quantity + quantity

        # If we've closed the entire short and opened a long
        if new_quantity > 0:
            # Reset average entry price for the new long position
            excess_quantity = quantity - abs(position.quantity)
            position.average_entry_price = price
            position.quantity = excess_quantity
        else:
            # Still short, or closed exactly
            position.quantity = new_quantity
            # Avg price remains same if just reducing, or becomes 0 if closed
            if new_quantity == 0:
                position.average_entry_price = Decimal(0)

        return realized_pnl

    async def _reduce_long_position(
        self, position: PositionInfo, quantity: Decimal, price: Decimal
    ) -> Decimal:
        """Reduces a long position, potentially flipping to short."""
        realized_pnl = Decimal(0)

        # Calculate realized P&L: (close price - long entry price) * quantity_closed
        quantity_closed = min(position.quantity, quantity)
        realized_pnl = (price - position.average_entry_price) * quantity_closed
        position.realized_pnl += realized_pnl

        # Update quantity
        new_quantity = position.quantity - quantity

        # If we've closed the entire long and opened a short
        if new_quantity < 0:
            # Reset average entry price for the new short position
            excess_quantity = quantity - position.quantity
            position.average_entry_price = price
            position.quantity = -excess_quantity  # Store short as negative
        else:
            # Still long, or closed exactly
            position.quantity = new_quantity
            # Avg price remains same if just reducing, or becomes 0 if closed
            if new_quantity == 0:
                position.average_entry_price = Decimal(0)

        return realized_pnl

    async def _increase_short_position(
        self, position: PositionInfo, quantity: Decimal, price: Decimal
    ) -> None:
        """Increases a short position."""
        # Calculate new average entry price for the short
        new_total_quantity = abs(position.quantity) + quantity
        # For shorts, we weight by quantity and adjust the sign later
        total_cost = (abs(position.quantity) * position.average_entry_price) + (quantity * price)

        # Update position
        position.average_entry_price = (
            total_cost / new_total_quantity if new_total_quantity > 0 else Decimal(0)
        )
        position.quantity = -new_total_quantity  # Store short as negative
