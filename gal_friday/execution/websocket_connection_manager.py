"""WebSocket connection management and resilience."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from gal_friday.logger_service import LoggerService


class ConnectionHealth(Enum):
    """Connection health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ConnectionMetrics:
    """WebSocket connection metrics."""
    messages_received: int = 0
    messages_sent: int = 0
    last_message_time: datetime | None = None
    connection_time: datetime | None = None
    disconnections: int = 0
    errors: int = 0

    @property
    def uptime_seconds(self) -> float:
        """Calculate connection uptime."""
        if self.connection_time:
            return (datetime.now(UTC) - self.connection_time).total_seconds()
        return 0

    @property
    def message_rate(self) -> float:
        """Calculate messages per second."""
        if self.uptime_seconds > 0:
            return self.messages_received / self.uptime_seconds
        return 0


class WebSocketConnectionManager:
    """Manages WebSocket connection health and recovery."""

    # Constants for connection health thresholds
    ERROR_RATE_THRESHOLD = 0.1  # 10% error rate threshold

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the WebSocket connection manager.

        Args:
            logger: Logger service instance for logging messages
        """
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Connection tracking
        self.connections: dict[str, ConnectionMetrics] = {}
        self.health_check_interval = 10.0
        self.unhealthy_threshold = 30.0  # No messages for 30 seconds

        # Recovery strategies
        self.recovery_attempts: dict[str, int] = {}
        self.max_recovery_attempts = 5

    def register_connection(self, connection_id: str) -> None:
        """Register a new connection for monitoring."""
        self.connections[connection_id] = ConnectionMetrics(
            connection_time=datetime.now(UTC))
        self.recovery_attempts[connection_id] = 0

    def record_message(self, connection_id: str, direction: str = "received") -> None:
        """Record message activity."""
        if connection_id in self.connections:
            metrics = self.connections[connection_id]

            if direction == "received":
                metrics.messages_received += 1
            else:
                metrics.messages_sent += 1

            metrics.last_message_time = datetime.now(UTC)

    def record_error(self, connection_id: str) -> None:
        """Record connection error."""
        if connection_id in self.connections:
            self.connections[connection_id].errors += 1

    def record_disconnection(self, connection_id: str) -> None:
        """Record disconnection event."""
        if connection_id in self.connections:
            self.connections[connection_id].disconnections += 1
            self.recovery_attempts[connection_id] += 1

    def check_health(self, connection_id: str) -> ConnectionHealth:
        """Check connection health."""
        if connection_id not in self.connections:
            return ConnectionHealth.UNHEALTHY

        metrics = self.connections[connection_id]

        # Check last message time
        if metrics.last_message_time:
            time_since_message = (datetime.now(UTC) - metrics.last_message_time).total_seconds()

            if time_since_message > self.unhealthy_threshold:
                return ConnectionHealth.UNHEALTHY
            if time_since_message > self.unhealthy_threshold / 2:
                return ConnectionHealth.DEGRADED

        # Check error rate
        if metrics.messages_received > 0:
            error_rate = metrics.errors / metrics.messages_received
            if error_rate > self.ERROR_RATE_THRESHOLD:
                return ConnectionHealth.DEGRADED

        return ConnectionHealth.HEALTHY

    def should_reconnect(self, connection_id: str) -> bool:
        """Determine if connection should be reconnected."""
        attempts = self.recovery_attempts.get(connection_id, 0)
        return attempts < self.max_recovery_attempts

    def reset_recovery_attempts(self, connection_id: str) -> None:
        """Reset recovery attempts after successful reconnection."""
        self.recovery_attempts[connection_id] = 0

    async def monitor_connections(self) -> None:
        """Monitor all connections and trigger recovery."""
        while True:
            try:
                for conn_id, metrics in self.connections.items():
                    health = self.check_health(conn_id)

                    if health == ConnectionHealth.UNHEALTHY:
                        self.logger.warning(
                            f"Connection {conn_id} is unhealthy",
                            source_module=self._source_module,
                            context={
                                "metrics": {
                                    "uptime": metrics.uptime_seconds,
                                    "messages": metrics.messages_received,
                                    "errors": metrics.errors,
                                },
                            })

                await asyncio.sleep(self.health_check_interval)

            except Exception:
                self.logger.exception(
                    "Error monitoring connections",
                    source_module=self._source_module)