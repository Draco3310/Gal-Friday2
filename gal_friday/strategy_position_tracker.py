"""Strategy position tracking for risk management.

This module provides position and performance tracking functionality for individual
trading strategies, enabling strategy-specific risk management.
"""

from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Union

# Type aliases
LoggerService = Any


class StrategyPositionTracker:
    """Track positions and performance metrics grouped by strategy.

    This class maintains a record of positions and calculates performance metrics
    for each trading strategy, enabling strategy-specific risk constraints.
    """

    def __init__(self, logger_service: LoggerService) -> None:
        """Initialize the strategy position tracker.

        Args:
            logger_service: The application logger service
        """
        self.logger = logger_service
        self._source_module = self.__class__.__name__

        # Add proper type annotations for class variables
        self._strategy_positions: dict[str, dict[str, Any]] = defaultdict(dict)

        # Define the type of performance metrics
        self._strategy_performance: dict[str, dict[str, Union[Decimal, int, datetime]]] = \
            defaultdict(lambda: {
            "peak_equity": Decimal("0"),
            "current_equity": Decimal("0"),
            "drawdown_pct": Decimal("0"),
            "total_exposure_value": Decimal("0"),
            "exposure_pct": Decimal("0"),
            "position_count": 0,
            "last_updated": datetime.utcnow(),
        })
        self._portfolio_equity = Decimal("0")  # Total portfolio equity for reference

    def add_position(
        self,
        strategy_id: str,
        trading_pair: str,
        position_data: dict[str, Any],
    ) -> None:
        """Add or update a position for a strategy.

        Args:
            strategy_id: Identifier for the strategy
            trading_pair: The trading pair symbol
            position_data: Position details and metadata
        """
        self._strategy_positions[strategy_id][trading_pair] = position_data
        self._update_strategy_metrics(strategy_id)

    def remove_position(self, strategy_id: str, trading_pair: str) -> None:
        """Remove a position from a strategy's tracking.

        Args:
            strategy_id: Identifier for the strategy
            trading_pair: The trading pair symbol to remove
        """
        if trading_pair in self._strategy_positions[strategy_id]:
            del self._strategy_positions[strategy_id][trading_pair]
            self._update_strategy_metrics(strategy_id)

    def clear_strategy_positions(self, strategy_id: str) -> None:
        """Clear all positions for a strategy.

        Args:
            strategy_id: Identifier for the strategy to clear
        """
        if strategy_id in self._strategy_positions:
            self._strategy_positions[strategy_id] = {}
            self._update_strategy_metrics(strategy_id)

    def get_strategy_positions(self, strategy_id: str) -> dict[str, Any]:
        """Get all positions for a strategy.

        Args:
            strategy_id: Identifier for the strategy

        Returns
        -------
            Dictionary of trading pairs to their position data
        """
        return self._strategy_positions.get(strategy_id, {})

    def get_strategy_metrics(self, strategy_id: str) -> dict[str, Any]:
        """Get performance metrics for a strategy.

        Args:
            strategy_id: Identifier for the strategy

        Returns
        -------
            Dictionary of performance metrics
        """
        return self._strategy_performance.get(strategy_id, {})

    def update_portfolio_equity(self, total_equity: Decimal) -> None:
        """Update the total portfolio equity reference value.

        Args:
            total_equity: Current total portfolio equity
        """
        self._portfolio_equity = total_equity
        # Recalculate exposure percentages for all strategies
        for strategy_id in self._strategy_positions:
            self._update_strategy_metrics(strategy_id)

    def record_strategy_pnl(self, strategy_id: str, pnl_amount: Decimal) -> None:
        """Record a P&L event for a strategy to update performance metrics.

        Args:
            strategy_id: Identifier for the strategy
            pnl_amount: Profit/loss amount (positive for profit, negative for loss)
        """
        if strategy_id not in self._strategy_performance:
            self._initialize_strategy_metrics(strategy_id)

        metrics = self._strategy_performance[strategy_id]

        # Update current equity
        current_equity = Decimal(str(metrics["current_equity"])) + pnl_amount
        metrics["current_equity"] = current_equity

        # Update peak equity if new high
        peak_equity = metrics["peak_equity"]
        if isinstance(peak_equity, Decimal):
            metrics["peak_equity"] = max(peak_equity, current_equity)
        else:
            metrics["peak_equity"] = max(Decimal(str(peak_equity)), current_equity)

        # Calculate drawdown
        peak_equity = metrics["peak_equity"]
        # Ensure peak_equity is a Decimal before comparison
        if isinstance(peak_equity, Decimal) and peak_equity > Decimal("0"):
            drawdown = (peak_equity - current_equity) / peak_equity * 100
            metrics["drawdown_pct"] = drawdown

        metrics["last_updated"] = datetime.utcnow()
        self.logger.debug(
            "Strategy %s P&L recorded: %s, current equity: %s, drawdown: %.2f%%",
            strategy_id,
            pnl_amount,
            current_equity,
            float(
                metrics["drawdown_pct"] if isinstance(metrics["drawdown_pct"], Decimal)
                else Decimal(str(
                    metrics["drawdown_pct"] if metrics["drawdown_pct"] is not None else 0
                ))
            ),
            source_module=self._source_module,
        )

    def get_all_strategy_ids(self) -> list[str]:
        """Get list of all strategy IDs currently being tracked.

        Returns
        -------
            List of strategy IDs
        """
        # Return all strategy IDs from both positions and performance tracking
        all_ids = set(self._strategy_positions.keys()).union(
            self._strategy_performance.keys()
        )
        return list(all_ids)

    def _initialize_strategy_metrics(self, strategy_id: str) -> None:
        """Initialize metrics for a new strategy.

        Args:
            strategy_id: Identifier for the strategy
        """
        self._strategy_performance[strategy_id] = {
            "peak_equity": Decimal("0"),
            "current_equity": Decimal("0"),
            "drawdown_pct": Decimal("0"),
            "total_exposure_value": Decimal("0"),
            "exposure_pct": Decimal("0"),
            "position_count": 0,
            "last_updated": datetime.utcnow(),
        }

    def _update_strategy_metrics(self, strategy_id: str) -> None:
        """Update performance metrics for a strategy based on current positions.

        Args:
            strategy_id: Identifier for the strategy
        """
        if strategy_id not in self._strategy_performance:
            self._initialize_strategy_metrics(strategy_id)

        metrics = self._strategy_performance[strategy_id]
        positions = self._strategy_positions.get(strategy_id, {})

        # Calculate total position value
        total_exposure_value = Decimal("0")
        for position in positions.values():
            # Position value might be stored directly or calculated from qty * price
            position_value = position.get("position_value")
            if position_value is None and "quantity" in position and "price" in position:
                quantity = Decimal(str(position["quantity"]))
                price = Decimal(str(position["price"]))
                position_value = quantity * price

            if position_value is not None:
                total_exposure_value += Decimal(str(position_value))

        # Update metrics
        metrics["total_exposure_value"] = total_exposure_value
        metrics["position_count"] = len(positions)

        # Calculate exposure percentage if we have portfolio equity
        if self._portfolio_equity > Decimal("0"):
            exposure_pct = (total_exposure_value / self._portfolio_equity) * 100
            metrics["exposure_pct"] = exposure_pct

        metrics["last_updated"] = datetime.utcnow()

    def get_strategy_exposure_details(self, strategy_id: str) -> dict[str, Any]:
        """Get detailed exposure information for a strategy.

        Args:
            strategy_id: Identifier for the strategy

        Returns
        -------
            Dictionary with exposure details
        """
        metrics = self.get_strategy_metrics(strategy_id)
        positions = self.get_strategy_positions(strategy_id)

        # Create a summary dictionary
        return {
            "strategy_id": strategy_id,
            "total_exposure_value": metrics.get("total_exposure_value", Decimal("0")),
            "exposure_pct": metrics.get("exposure_pct", Decimal("0")),
            "position_count": metrics.get("position_count", 0),
            "positions": positions,
            "drawdown_pct": metrics.get("drawdown_pct", Decimal("0")),
            "current_equity": metrics.get("current_equity", Decimal("0")),
            "peak_equity": metrics.get("peak_equity", Decimal("0")),
        }
