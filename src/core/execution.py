import logging
import uuid
from typing import Optional
from datetime import datetime
from src.core.interfaces import Signal
from src.core.database import DatabaseHandler
from src.core.exchange import ExchangeManager

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, db: DatabaseHandler, exchange: ExchangeManager):
        self.db = db
        self.exchange = exchange

    async def process_signal(self, signal: Signal):
        """
        Processes a trading signal, executes orders, and updates the database.
        """
        logger.info(f"Processing signal: {signal}")

        if signal.action == 'HOLD':
            return

        # Check for existing open position
        open_positions = self.db.get_open_positions(signal.symbol)
        # For V1, assume one position per symbol
        position = open_positions[0] if open_positions else None

        if signal.action == 'BUY':
            if position:
                logger.info(f"Position already open for {signal.symbol}. ignoring BUY.")
                # In V2, we might add to position (scale in)
                return

            await self._open_position(signal)

        elif signal.action == 'SELL':
            if not position:
                logger.info(f"No open position for {signal.symbol} to SELL.")
                return

            await self._close_position(position, signal)

    async def _open_position(self, signal: Signal):
        """
        Executes entry order and creates Parent/Child records.
        """
        qty = signal.quantity or 0.001 # Default or from config
        # In real app, fetch price from signal or ticker

        logger.info(f"Opening position for {signal.symbol} size {qty}")

        # 1. Execute on Exchange
        try:
            # Using market order for simplicity in V1
            order = await self.exchange.create_order(signal.symbol, 'market', 'buy', qty)
            avg_price = order.get('average') or order.get('price') or signal.price or 0.0
        except Exception as e:
            logger.error(f"Failed to execute BUY order: {e}")
            return

        # 2. Create Parent Position (Open)
        position_id = str(uuid.uuid4())
        self.db.create_position(position_id, signal.symbol, qty, signal.stop_loss)

        # 3. Create Child Execution (Entry)
        execution_id = str(uuid.uuid4())
        self.db.add_execution(execution_id, position_id, qty, float(avg_price), "ENTRY")

        logger.info(f"Position {position_id} opened successfully.")

    async def _close_position(self, position: dict, signal: Signal):
        """
        Executes exit order and updates Parent/Child records.
        Handles Scaling Out if quantity < remaining_qty.
        """
        remaining_qty = float(position['remaining_qty'])
        qty_to_sell = signal.quantity if signal.quantity else remaining_qty

        # Cap sell quantity
        if qty_to_sell > remaining_qty:
            qty_to_sell = remaining_qty

        logger.info(f"Selling {qty_to_sell} of {signal.symbol} (Remaining: {remaining_qty})")

        # 1. Execute on Exchange
        try:
            order = await self.exchange.create_order(signal.symbol, 'market', 'sell', qty_to_sell)
            avg_price = order.get('average') or order.get('price') or signal.price or 0.0
        except Exception as e:
            logger.error(f"Failed to execute SELL order: {e}")
            return

        # 2. Create Child Execution (Exit)
        execution_id = str(uuid.uuid4())
        reason = signal.reason or "TP/SL"
        self.db.add_execution(execution_id, position['id'], qty_to_sell, float(avg_price), reason)

        # 3. Update Parent Position
        new_remaining = remaining_qty - qty_to_sell
        new_status = position['status']

        if new_remaining <= 0: # Floating point tolerance might be needed
            new_remaining = 0
            new_status = 'CLOSED'
        elif new_remaining < remaining_qty:
            new_status = 'PARTIAL' # Or keep OPEN? The prompt said "PARTIAL" status in schema.

        self.db.update_position(position['id'], remaining_qty=new_remaining, status=new_status)
        logger.info(f"Position {position['id']} updated. Status: {new_status}, Remaining: {new_remaining}")
