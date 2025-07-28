"""Order persistence service for managing order state in database."""

from typing import Any, cast


class OrderPersistenceService:
    """Service for persisting and retrieving order state from database."""

    def __init__(self, repository: Any, logger: Any) -> None:
        """Initialize the persistence service.

        Args:
            repository: Order repository for database operations
            logger: Logger instance for logging operations
        """
        self.repository = repository
        self.logger = logger

    async def save_order_state(self, order_id: str, order_data: dict[str, Any]) -> None:
        """Save order state to database.

        Args:
            order_id: Unique identifier for the order
            order_data: Order data to persist

        Raises:
            Exception: If save operation fails
        """
        try:
            await self.repository.save_order_state(order_id, order_data)
        except Exception:
            self.logger.exception("Failed to save order state: ")
            raise

    async def load_order_state(self, order_id: str) -> dict[str, Any] | None:
        """Load order state from database.

        Args:
            order_id: Unique identifier for the order

        Returns:
            Order data if found, None otherwise
        """
        try:
            result = await self.repository.load_order_state(order_id)
            return cast("dict[str, Any] | None", result)
        except Exception:
            self.logger.exception("Failed to load order state: ")
            return None

    async def get_active_orders(self) -> list[dict[str, Any]]:
        """Get list of active orders from database.

        Returns:
            List of active order data dictionaries
        """
        try:
            result = await self.repository.get_active_orders()
            return cast("list[dict[str, Any]]", result)
        except Exception:
            self.logger.exception("Failed to get active orders: ")
            return []
