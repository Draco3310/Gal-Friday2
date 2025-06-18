"""Order-Position Integration Service.

This module provides seamless integration between order processing workflows
and position management, automatically establishing relationships when orders
affect positions.
"""

import asyncio
from datetime import datetime, UTC
from decimal import Decimal
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..dal.models import Order
from uuid import UUID

from ..core.events import ExecutionReportEvent
from ..dal.repositories.order_repository import OrderRepository
from ..dal.repositories.position_repository import PositionRepository
from ..logger_service import LoggerService
from ..portfolio.position_manager import PositionManager


class OrderPositionIntegrationService:
    """Service for integrating order processing with position management."""
    
    def __init__(
        self,
        order_repository: OrderRepository,
        position_repository: PositionRepository,
        position_manager: PositionManager,
        logger: LoggerService):
        """Initialize the integration service.
        
        Args:
            order_repository: Repository for order data
            position_repository: Repository for position data
            position_manager: Manager for position operations
            logger: Logger service
        """
        self.order_repository = order_repository
        self.position_repository = position_repository
        self.position_manager = position_manager
        self.logger = logger
        self._source_module = self.__class__.__name__

    async def process_execution_report(self, execution_report: ExecutionReportEvent) -> bool:
        """Process execution report and establish position-order relationships.
        
        This method is called when an execution report is received to automatically
        link orders to positions and update position data.
        
        Args:
            execution_report: The execution report to process
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Only process filled or partially filled orders
            if execution_report.order_status not in ["FILLED", "PARTIALLY_FILLED"]:
                self.logger.debug(
                    f"Skipping execution report for order {execution_report.client_order_id} "
                    f"with status {execution_report.order_status}",
                    source_module=self._source_module)
                return True

            self.logger.info(
                f"Processing execution report for order {execution_report.client_order_id} "
                f"({execution_report.trading_pair}, {execution_report.side}, "
                f"{execution_report.quantity_filled} @ {execution_report.average_fill_price})",
                source_module=self._source_module)

            # Extract order information
            order_id = execution_report.client_order_id or execution_report.exchange_order_id
            if not order_id:
                self.logger.error(
                    "Execution report missing order ID - cannot establish position relationship",
                    source_module=self._source_module)
                return False

            # Update position and establish relationship
            # Ensure price is not None
            if execution_report.average_fill_price is None:
                self.logger.error(
                    f"Execution report for order {order_id} missing average fill price",
                    source_module=self._source_module)
                return False
            
            # Ensure trade_id is not None
            trade_id = execution_report.exchange_order_id or execution_report.client_order_id
            if not trade_id:
                self.logger.error(
                    f"Execution report for order {order_id} missing trade ID",
                    source_module=self._source_module)
                return False
            
            realized_pnl, updated_position = await self.position_manager.update_position_for_trade(
                trading_pair=execution_report.trading_pair,
                side=execution_report.side,
                quantity=execution_report.quantity_filled,
                price=execution_report.average_fill_price,
                timestamp=execution_report.timestamp,
                trade_id=trade_id,
                order_id=order_id,  # This establishes the relationship
                commission=execution_report.commission or Decimal(0),
                commission_asset=execution_report.commission_asset)

            if updated_position:
                self.logger.info(
                    f"Successfully processed execution report - Order {order_id} linked to "
                    f"position {updated_position.id}, realized PnL: {realized_pnl}",
                    source_module=self._source_module)
                return True
            else:
                self.logger.error(
                    f"Failed to update position for order {order_id}",
                    source_module=self._source_module)
                return False

        except Exception as e:
            self.logger.exception(
                f"Error processing execution report for order "
                f"{getattr(execution_report, 'client_order_id', 'unknown')}: {e}",
                source_module=self._source_module)
            return False

    async def link_existing_order_to_position(
        self, 
        order_id: str | UUID, 
        position_id: str | UUID,
        verify_consistency: bool = True
    ) -> bool:
        """Manually link an existing order to a position.
        
        This method can be used for data migration or manual corrections.
        
        Args:
            order_id: The ID of the order to link
            position_id: The ID of the position to link to
            verify_consistency: Whether to verify the link makes sense
            
        Returns:
            True if linking was successful, False otherwise
        """
        try:
            order_id_str = str(order_id)
            position_id_str = str(position_id)

            # Verify entities exist if requested
            if verify_consistency:
                if not await self._verify_order_position_consistency(order_id_str, position_id_str):
                    return False

            # Establish the link
            updated_order = await self.order_repository.link_order_to_position(
                order_id_str, position_id_str
            )

            if updated_order:
                self.logger.info(
                    f"Successfully linked order {order_id_str} to position {position_id_str}",
                    source_module=self._source_module)
                return True
            else:
                self.logger.error(
                    f"Failed to link order {order_id_str} to position {position_id_str}",
                    source_module=self._source_module)
                return False

        except Exception as e:
            self.logger.exception(
                f"Error linking order {order_id} to position {position_id}: {e}",
                source_module=self._source_module)
            return False

    async def unlink_order_from_position(self, order_id: str | UUID) -> bool:
        """Remove position link from an order.
        
        Args:
            order_id: The ID of the order to unlink
            
        Returns:
            True if unlinking was successful, False otherwise
        """
        try:
            order_id_str = str(order_id)
            
            updated_order = await self.order_repository.unlink_order_from_position(order_id_str)
            
            if updated_order:
                self.logger.info(
                    f"Successfully unlinked order {order_id_str} from position",
                    source_module=self._source_module)
                return True
            else:
                self.logger.warning(
                    f"Order {order_id_str} not found or already unlinked",
                    source_module=self._source_module)
                return False

        except Exception as e:
            self.logger.exception(
                f"Error unlinking order {order_id} from position: {e}",
                source_module=self._source_module)
            return False

    async def reconcile_order_position_relationships(
        self, 
        hours_back: int = 24,
        auto_fix: bool = False
    ) -> dict[str, Any]:
        """Reconcile order-position relationships for recent data.
        
        This method identifies and optionally fixes inconsistencies in
        order-position relationships.
        
        Args:
            hours_back: Number of hours to look back for reconciliation
            auto_fix: Whether to automatically fix safe issues
            
        Returns:
            Dictionary containing reconciliation results
        """
        try:
            self.logger.info(
                f"Starting order-position relationship reconciliation (last {hours_back} hours)",
                source_module=self._source_module)

            results: dict[str, Any] = {
                "reconciliation_timestamp": datetime.now(UTC),
                "hours_checked": hours_back,
                "issues_found": [],
                "issues_fixed": 0,
                "orders_checked": 0,
                "positions_checked": 0,
            }

            # Get unlinked filled orders
            unlinked_orders = await self.order_repository.get_unlinked_filled_orders(hours_back)
            results["orders_checked"] = len(unlinked_orders)

            for order in unlinked_orders:
                issue = {
                    "type": "unlinked_filled_order",
                    "order_id": str(order.id),
                    "trading_pair": order.trading_pair,
                    "status": order.status,
                    "filled_quantity": str(order.filled_quantity) if order.filled_quantity else "0",
                }
                results["issues_found"].append(issue)

                # Auto-fix if requested and it's safe
                if auto_fix and await self._can_auto_fix_unlinked_order(order):
                    if await self._auto_fix_unlinked_order(order):
                        issue["auto_fixed"] = "true"
                        results["issues_fixed"] += 1

            # Check for orphaned position references
            orphaned_count = await self._check_orphaned_position_references(results, auto_fix)
            results["issues_fixed"] += orphaned_count

            self.logger.info(
                f"Reconciliation completed - Found {len(results['issues_found'])} issues, "
                f"fixed {results['issues_fixed']}",
                source_module=self._source_module)

            return results

        except Exception as e:
            self.logger.exception(
                f"Error during order-position reconciliation: {e}",
                source_module=self._source_module)
            return {"error": str(e)}

    async def get_position_audit_trail(self, position_id: str | UUID) -> dict[str, Any]:
        """Get complete audit trail for a position.
        
        Args:
            position_id: The ID of the position
            
        Returns:
            Dictionary containing position audit trail
        """
        try:
            position_id_str = str(position_id)

            # Get position details
            position = await self.position_repository.get_by_id(position_id_str)
            if not position:
                return {"error": f"Position {position_id_str} not found"}

            # Get all orders linked to this position
            orders = await self.order_repository.get_orders_by_position(position_id_str)

            # Build audit trail
            audit_trail: dict[str, Any] = {
                "position_id": position_id_str,
                "trading_pair": position.trading_pair,
                "side": position.side,
                "current_quantity": str(position.quantity),
                "entry_price": str(position.entry_price),
                "realized_pnl": str(position.realized_pnl) if position.realized_pnl else "0",
                "is_active": position.is_active,
                "opened_at": position.opened_at.isoformat() if position.opened_at else None,
                "closed_at": position.closed_at.isoformat() if position.closed_at else None,
                "contributing_orders": []
            }

            # Add order details
            total_order_quantity = Decimal(0)
            for order in sorted(orders, key=lambda x: x.created_at or datetime.min):
                order_info = {
                    "order_id": str(order.id),
                    "side": order.side,
                    "quantity": str(order.quantity),
                    "filled_quantity": str(order.filled_quantity) if order.filled_quantity else "0",
                    "average_fill_price": str(order.average_fill_price) if order.average_fill_price else None,
                    "status": order.status,
                    "created_at": order.created_at.isoformat() if order.created_at else None,
                }
                audit_trail["contributing_orders"].append(order_info)
                
                if order.filled_quantity:
                    total_order_quantity += order.filled_quantity

            # Add summary statistics
            audit_trail["summary"] = {
                "total_contributing_orders": len(orders),
                "total_order_quantity": str(total_order_quantity),
                "quantity_consistency": abs(position.quantity - total_order_quantity) < Decimal('0.00000001'),
            }

            return audit_trail

        except Exception as e:
            self.logger.exception(
                f"Error generating audit trail for position {position_id}: {e}",
                source_module=self._source_module)
            return {"error": str(e)}

    async def _verify_order_position_consistency(self, order_id: str, position_id: str) -> bool:
        """Verify that linking an order to a position makes sense."""
        try:
            # Get order and position details
            order = await self.order_repository.get_by_id(order_id)
            position = await self.position_repository.get_by_id(position_id)

            if not order or not position:
                self.logger.error(
                    f"Order {order_id} or position {position_id} not found for consistency check",
                    source_module=self._source_module)
                return False

            # Check trading pair consistency
            if order.trading_pair != position.trading_pair:
                self.logger.error(
                    f"Trading pair mismatch - Order: {order.trading_pair}, Position: {position.trading_pair}",
                    source_module=self._source_module)
                return False

            # Check if order is in a filled state
            if order.status not in ["FILLED", "PARTIALLY_FILLED"]:
                self.logger.warning(
                    f"Order {order_id} status is {order.status} - may not affect position",
                    source_module=self._source_module)

            return True

        except Exception as e:
            self.logger.exception(
                f"Error verifying order-position consistency: {e}",
                source_module=self._source_module)
            return False

    async def _can_auto_fix_unlinked_order(self, order: "Order") -> bool:
        """Determine if an unlinked order can be safely auto-fixed."""
        # Only auto-fix orders that are clearly filled and recent
        if order.status not in ["FILLED", "PARTIALLY_FILLED"]:
            return False
        
        if not order.filled_quantity or order.filled_quantity <= 0:
            return False
        
        # Check if order is recent enough (within last 7 days)
        if order.created_at:
            age_days = (datetime.now(UTC) - order.created_at).days
            if age_days > 7:
                return False
        
        return True

    async def _auto_fix_unlinked_order(self, order: "Order") -> bool:
        """Attempt to auto-fix an unlinked order by finding/creating appropriate position."""
        try:
            # Look for an existing active position for the same trading pair
            position = await self.position_repository.get_position_by_pair(order.trading_pair)
            
            if position:
                # Link to existing position
                updated_order = await self.order_repository.link_order_to_position(
                    str(order.id), str(position.id)
                )
                
                if updated_order:
                    self.logger.info(
                        f"Auto-fixed: Linked order {order.id} to existing position {position.id}",
                        source_module=self._source_module)
                    return True
            
            # If no position exists, we could create one, but that's more complex
            # For now, just log that manual intervention is needed
            self.logger.warning(
                f"Cannot auto-fix order {order.id} - no suitable position found",
                source_module=self._source_module)
            return False
            
        except Exception as e:
            self.logger.exception(
                f"Error auto-fixing unlinked order {order.id}: {e}",
                source_module=self._source_module)
            return False

    async def _check_orphaned_position_references(self, results: dict[str, Any], auto_fix: bool) -> int:
        """Check for and optionally fix orphaned position references."""
        fixed_count = 0
        
        try:
            # This would use a more complex query to find orphaned references
            # For now, placeholder implementation
            self.logger.debug("Checking for orphaned position references")
            
            # Implementation would go here to find orders referencing non-existent positions
            # and auto-fix them if requested
            
        except Exception as e:
            self.logger.exception(
                f"Error checking orphaned position references: {e}",
                source_module=self._source_module)
        
        return fixed_count 