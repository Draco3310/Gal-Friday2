"""Interface definition for executing trades on exchanges."""

import abc

from ..core.events import TradeSignalApprovedEvent


class ExecutionHandlerInterface(abc.ABC):
    """Abstract Base Class for components that execute trades on exchanges.

    Implementations should:
    1. Handle connection setup/teardown via start()/stop().
    2. Process approved trade signals.
    3. Translate internal trade signals to exchange-specific formats.
    4. Place, cancel, and monitor orders.
    5. Report execution status back to the system.
    6. Implement proper error handling and recovery mechanisms.
    """

    @abc.abstractmethod
    async def start(self) -> None:
        """Initialize the service and establish connections to the exchange.

        Should be called once during application startup to set up connections,
        load essential information, and subscribe to events.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def stop(self) -> None:
        """Clean up resources and close connections.

        Should be called once during application shutdown to properly close
        connections, clean up resources, and unsubscribe from events.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def handle_trade_signal_approved(self, event: TradeSignalApprovedEvent) -> None:
        """Process an approved trade signal by placing the corresponding order.

        Args:
        ----
            event: The approved trade signal event containing trade details
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Cancel an open order on the exchange.

        Args:
        ----
            exchange_order_id: The exchange-specific order ID to cancel

        Returns:
        -------
            True if cancellation was successful or initiated, False otherwise
        """
        raise NotImplementedError
