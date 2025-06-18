"""HALT Coordinator for Gal-Friday trading system.

Central coordinator for all HALT conditions and triggers, managing the system's
safety mechanisms and emergency procedures.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from gal_friday.config_manager import ConfigManager
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService


@dataclass
class HaltCondition:
    """Represents a condition that can trigger a HALT."""
    condition_id: str
    name: str
    threshold: int | float | Decimal | str | bool
    current_value: int | float | Decimal | str | bool | None # current_value can be None initially
    is_triggered: bool
    timestamp: datetime


class HaltCoordinator:
    """Central coordinator for all HALT conditions and triggers."""

    def __init__(self, config_manager: ConfigManager, pubsub_manager: PubSubManager, logger_service: LoggerService) -> None:
        """Initialize the instance."""
        self.config = config_manager
        self.pubsub = pubsub_manager
        self.logger = logger_service
        self._source_module = self.__class__.__name__

        # Track all HALT conditions
        self.conditions: dict[str, HaltCondition] = {}
        self._is_halted = False
        self._halt_reason = ""
        self._halt_source = ""
        self._halt_timestamp: datetime | None = None

        # Configure HALT conditions from config
        self._initialize_conditions()

    def _initialize_conditions(self) -> None:
        """Initialize all HALT conditions from configuration."""
        # Drawdown conditions
        self.register_condition(
            "max_total_drawdown",
            "Maximum Total Drawdown",
            self.config.get_decimal("risk.limits.max_total_drawdown_pct", Decimal("15.0")))
        self.register_condition(
            "max_daily_drawdown",
            "Maximum Daily Drawdown",
            self.config.get_decimal("risk.limits.max_daily_drawdown_pct", Decimal("2.0")))
        self.register_condition(
            "max_consecutive_losses",
            "Maximum Consecutive Losses",
            self.config.get_int("risk.limits.max_consecutive_losses", 5))
        # Market conditions
        self.register_condition(
            "max_volatility",
            "Maximum Market Volatility",
            self.config.get_decimal("monitoring.max_volatility_threshold", Decimal("5.0")))
        # System conditions
        self.register_condition(
            "api_error_rate",
            "API Error Rate Threshold",
            self.config.get_int("monitoring.max_api_errors_per_minute", 10))
        self.register_condition(
            "data_staleness",
            "Market Data Staleness",
            self.config.get_int("monitoring.max_data_staleness_seconds", 60))

        self.logger.info(
            f"Initialized {len(self.conditions)} HALT conditions",
            source_module=self._source_module,
            context={"conditions": list[Any](self.conditions.keys())})

    def register_condition(self, condition_id: str, name: str, threshold: float | Decimal | str | bool) -> None:
        """Register a new HALT condition."""
        self.conditions[condition_id] = HaltCondition(
            condition_id=condition_id,
            name=name,
            threshold=threshold,
            current_value=None, # Remains None until first update
            is_triggered=False,
            timestamp=datetime.now(UTC))

    def update_condition(self, condition_id: str, current_value: float | Decimal | str | bool) -> bool:
        """Update a condition's current value and check if triggered.

        Returns:
            bool: True if condition is triggered
        """
        if condition_id not in self.conditions:
            self.logger.warning(
                f"Unknown condition ID: {condition_id}",
                source_module=self._source_module)
            return False

        condition = self.conditions[condition_id]
        condition.current_value = current_value
        condition.timestamp = datetime.now(UTC)

        # Check if condition is triggered based on type
        was_triggered = condition.is_triggered

        if isinstance(condition.threshold, int | float | Decimal) and isinstance(current_value, int | float | Decimal):
            # Numeric comparison - both values must be numeric
            condition.is_triggered = current_value > condition.threshold
        elif isinstance(condition.threshold, bool) and isinstance(current_value, bool):
            # Boolean comparison - both values must be boolean
            condition.is_triggered = current_value == condition.threshold
        elif isinstance(condition.threshold, str) and isinstance(current_value, str):
            # String comparison - both values must be strings
            condition.is_triggered = current_value == condition.threshold
        else:
            # Type[Any] mismatch or other comparison - log warning and don't trigger
            self.logger.warning(
                f"Type[Any] mismatch in condition '{condition.name}': threshold type {type(condition.threshold).__name__} vs current_value type {type(current_value).__name__}",
                source_module=self._source_module)
            condition.is_triggered = False

        # Log if condition state changed
        if condition.is_triggered != was_triggered:
            self.logger.warning(
                f"HALT condition '{condition.name}' state changed: {was_triggered} -> {condition.is_triggered}",
                source_module=self._source_module,
                context={
                    "condition_id": condition_id,
                    "current_value": str(current_value),
                    "threshold": str(condition.threshold),
                })

        return condition.is_triggered

    def check_all_conditions(self) -> list[HaltCondition]:
        """Check all conditions and return list[Any] of triggered ones."""
        triggered = []
        for condition in self.conditions.values():
            if condition.is_triggered:
                triggered.append(condition)
        return triggered

    def get_halt_status(self) -> dict[str, Any]:
        """Get current HALT status and conditions."""
        return {
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "halt_source": self._halt_source,
            "halt_timestamp": self._halt_timestamp.isoformat() if self._halt_timestamp else None,
            "conditions": {
                cid: {
                    "name": c.name,
                    "threshold": str(c.threshold),
                    "current_value": str(c.current_value) if c.current_value is not None else None,
                    "is_triggered": c.is_triggered,
                    "last_updated": c.timestamp.isoformat(),
                }
                for cid, c in self.conditions.items()
            },
        }

    def set_halt_state(self, is_halted: bool, reason: str = "", source: str = "") -> None:
        """Set the HALT state."""
        self._is_halted = is_halted
        self._halt_reason = reason
        self._halt_source = source
        self._halt_timestamp = datetime.now(UTC) if is_halted else None

    def clear_halt_state(self) -> None:
        """Clear the HALT state."""
        self._is_halted = False
        self._halt_reason = ""
        self._halt_source = ""
        self._halt_timestamp = None

        # Reset all condition triggers
        for condition in self.conditions.values():
            condition.is_triggered = False
