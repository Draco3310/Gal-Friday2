"""Repository for A/B testing experiment data."""

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import asyncpg

from gal_friday.dal.base import BaseRepository
from gal_friday.model_lifecycle.experiment_manager import ExperimentConfig, ExperimentStatus

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class ExperimentRepository(BaseRepository):
    """Repository for experiment data persistence."""

    def __init__(self, db_pool: asyncpg.Pool, logger: "LoggerService") -> None:
        """Initialize the experiment repository.

        Args:
            db_pool: Async database connection pool
            logger: Logger service instance
        """
        super().__init__(db_pool, logger, "experiments")

    async def save_experiment(self, config: ExperimentConfig) -> None:
        """Save experiment configuration."""
        query = """
            INSERT INTO experiments (
                experiment_id, name, description, control_model_id,
                treatment_model_id, allocation_strategy, traffic_split,
                start_time, end_time, min_samples_per_variant,
                primary_metric, secondary_metrics, confidence_level,
                minimum_detectable_effect, max_loss_threshold,
                status, config_data
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            ON CONFLICT (experiment_id) DO UPDATE SET
                status = EXCLUDED.status,
                config_data = EXCLUDED.config_data
        """

        await self.db_pool.execute(
            query,
            config.experiment_id,
            config.name,
            config.description,
            config.control_model_id,
            config.treatment_model_id,
            config.allocation_strategy.value,
            float(config.traffic_split),
            config.start_time,
            config.end_time,
            config.min_samples_per_variant,
            config.primary_metric,
            json.dumps(config.secondary_metrics),
            float(config.confidence_level),
            float(config.minimum_detectable_effect),
            float(config.max_loss_threshold) if config.max_loss_threshold else None,
            ExperimentStatus.CREATED.value,
            json.dumps(config.to_dict()),
        )

    async def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get experiment by ID."""
        query = """
            SELECT * FROM experiments WHERE experiment_id = $1
        """

        row = await self.db_pool.fetchrow(query, experiment_id)
        if row:
            return dict(row)
        return None

    async def get_active_experiments(self) -> list[dict[str, Any]]:
        """Get all active experiments."""
        query = """
            SELECT * FROM experiments
            WHERE status IN ('created', 'running')
              AND (end_time IS NULL OR end_time > $1)
            ORDER BY start_time DESC
        """

        rows = await self.db_pool.fetch(query, datetime.now(UTC))
        return [dict(row) for row in rows]

    async def record_assignment(self,
                              experiment_id: str,
                              variant: str,
                              event_id: str,
                              timestamp: datetime) -> None:
        """Record variant assignment for an event."""
        query = """
            INSERT INTO experiment_assignments (
                experiment_id, event_id, variant, assigned_at
            ) VALUES ($1, $2, $3, $4)
            ON CONFLICT (experiment_id, event_id) DO NOTHING
        """

        await self.db_pool.execute(
            query,
            experiment_id,
            event_id,
            variant,
            timestamp,
        )

    async def get_assignment(self,
                           experiment_id: str,
                           event_id: str) -> dict[str, Any] | None:
        """Get variant assignment for an event."""
        query = """
            SELECT * FROM experiment_assignments
            WHERE experiment_id = $1 AND event_id = $2
        """

        row = await self.db_pool.fetchrow(query, experiment_id, event_id)
        if row:
            return dict(row)
        return None

    async def save_outcome(self,
                         experiment_id: str,
                         event_id: str,
                         variant: str,
                         outcome: dict[str, Any],
                         timestamp: datetime) -> None:
        """Save prediction outcome for experiment analysis."""
        query = """
            INSERT INTO experiment_outcomes (
                experiment_id, event_id, variant, outcome_data,
                correct_prediction, signal_generated, trade_return,
                recorded_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        await self.db_pool.execute(
            query,
            experiment_id,
            event_id,
            variant,
            json.dumps(outcome),
            outcome.get("correct_prediction", False),
            outcome.get("signal_generated", False),
            float(outcome.get("return", 0)),
            timestamp,
        )

    async def get_experiment_performance(self,
                                       experiment_id: str) -> dict[str, dict[str, Any]]:
        """Get aggregated performance metrics for experiment variants."""
        query = """
            SELECT
                variant,
                COUNT(*) as sample_count,
                SUM(CASE WHEN correct_prediction THEN 1 ELSE 0 END) as correct_predictions,
                SUM(CASE WHEN signal_generated THEN 1 ELSE 0 END) as signals_generated,
                SUM(trade_return) as total_return,
                AVG(CASE WHEN correct_prediction THEN 1 ELSE 0 END) as accuracy
            FROM experiment_outcomes
            WHERE experiment_id = $1
            GROUP BY variant
        """

        rows = await self.db_pool.fetch(query, experiment_id)

        performance = {}
        for row in rows:
            performance[row["variant"]] = {
                "sample_count": row["sample_count"],
                "correct_predictions": row["correct_predictions"],
                "signals_generated": row["signals_generated"],
                "total_return": float(row["total_return"]),
                "accuracy": float(row["accuracy"]) if row["accuracy"] else 0,
            }

        return performance

    async def save_results(self, experiment_id: str, results: dict[str, Any]) -> None:
        """Save final experiment results."""
        query = """
            UPDATE experiments SET
                status = $2,
                completion_reason = $3,
                results = $4,
                completed_at = $5
            WHERE experiment_id = $1
        """

        await self.db_pool.execute(
            query,
            experiment_id,
            results.get("status", ExperimentStatus.COMPLETED.value),
            results.get("completion_reason"),
            json.dumps(results),
            datetime.now(UTC),
        )

    async def get_experiment_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get experiment history for analysis."""
        query = """
            SELECT
                e.*,
                (SELECT COUNT(DISTINCT event_id)
                 FROM experiment_assignments
                 WHERE experiment_id = e.experiment_id) as total_assignments
            FROM experiments e
            WHERE e.created_at > NOW() - INTERVAL '%s days'
            ORDER BY e.created_at DESC
        """

        rows = await self.db_pool.fetch(query, days)
        return [dict(row) for row in rows]
