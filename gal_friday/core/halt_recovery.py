"""HALT Recovery Manager for Gal-Friday trading system.

Manages the recovery process after a HALT, including checklist management
and manual intervention requirements.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


@dataclass
class RecoveryCheckItem:
    """Single item in recovery checklist."""
    item_id: str
    description: str
    is_completed: bool = False
    completed_by: str | None = None
    completed_at: datetime | None = None


class HaltRecoveryManager:
    """Manages the recovery process after a HALT."""

    def __init__(self, config_manager: ConfigManager, logger_service: LoggerService):
        self.config = config_manager
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self.checklist: list[RecoveryCheckItem] = []
        self._initialize_checklist()

    def _initialize_checklist(self) -> None:
        """Initialize recovery checklist based on HALT reason."""
        self.checklist = [
            RecoveryCheckItem(
                "review_halt_reason",
                "Review and understand the HALT trigger reason",
            ),
            RecoveryCheckItem(
                "check_market_conditions",
                "Verify current market conditions are acceptable",
            ),
            RecoveryCheckItem(
                "review_positions",
                "Review all open positions and their P&L",
            ),
            RecoveryCheckItem(
                "verify_api_connectivity",
                "Confirm API connectivity to exchange",
            ),
            RecoveryCheckItem(
                "check_account_balance",
                "Verify account balance matches expectations",
            ),
            RecoveryCheckItem(
                "review_risk_parameters",
                "Review and potentially adjust risk parameters",
            ),
            RecoveryCheckItem(
                "confirm_resume",
                "Confirm decision to resume trading",
            ),
        ]

        self.logger.info(
            f"Initialized recovery checklist with {len(self.checklist)} items",
            source_module=self._source_module,
        )

    def get_incomplete_items(self) -> list[RecoveryCheckItem]:
        """Get list of incomplete checklist items."""
        return [item for item in self.checklist if not item.is_completed]

    def complete_item(self, item_id: str, completed_by: str) -> bool:
        """Mark a checklist item as complete.
        
        Args:
            item_id: ID of the checklist item
            completed_by: Name of person completing the item
            
        Returns:
            bool: True if item was found and marked complete
        """
        for item in self.checklist:
            if item.item_id == item_id:
                item.is_completed = True
                item.completed_by = completed_by
                item.completed_at = datetime.now(UTC)

                self.logger.info(
                    f"Recovery item '{item_id}' completed by {completed_by}",
                    source_module=self._source_module,
                    context={
                        "item_description": item.description,
                        "completed_at": item.completed_at.isoformat(),
                    },
                )
                return True

        self.logger.warning(
            f"Recovery item '{item_id}' not found",
            source_module=self._source_module,
        )
        return False

    def is_recovery_complete(self) -> bool:
        """Check if all recovery items are complete."""
        return all(item.is_completed for item in self.checklist)

    def reset_checklist(self) -> None:
        """Reset all checklist items to incomplete state."""
        for item in self.checklist:
            item.is_completed = False
            item.completed_by = None
            item.completed_at = None

        self.logger.info(
            "Recovery checklist reset",
            source_module=self._source_module,
        )

    def get_checklist_status(self) -> dict:
        """Get current status of recovery checklist.
        
        Returns:
            dict: Status information including completed count and items
        """
        completed_items = [item for item in self.checklist if item.is_completed]
        incomplete_items = [item for item in self.checklist if not item.is_completed]

        return {
            "total_items": len(self.checklist),
            "completed_count": len(completed_items),
            "incomplete_count": len(incomplete_items),
            "is_complete": self.is_recovery_complete(),
            "items": [
                {
                    "id": item.item_id,
                    "description": item.description,
                    "is_completed": item.is_completed,
                    "completed_by": item.completed_by,
                    "completed_at": item.completed_at.isoformat() if item.completed_at else None,
                }
                for item in self.checklist
            ],
        }
