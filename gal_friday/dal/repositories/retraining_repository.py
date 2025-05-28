"""Repository for model retraining jobs and drift metrics."""

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import asyncpg

from gal_friday.dal.base import BaseRepository
from gal_friday.model_lifecycle.retraining_pipeline import RetrainingJob

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class RetrainingRepository(BaseRepository):
    """Repository for retraining job persistence."""

    def __init__(self, db_pool: asyncpg.Pool, logger: "LoggerService") -> None:
        """Initialize the retraining repository.

        Args:
            db_pool: Database connection pool
            logger: Logger service instance
        """
        super().__init__(db_pool, logger, "retraining_jobs")

    async def save_job(self, job: RetrainingJob) -> None:
        """Save retraining job."""
        query = """
            INSERT INTO retraining_jobs (
                job_id, model_id, model_name, trigger,
                drift_metrics, status, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """

        await self.db_pool.execute(
            query,
            job.job_id,
            job.model_id,
            job.model_name,
            job.trigger.value,
            json.dumps([m.__dict__ for m in job.drift_metrics]),
            job.status,
            datetime.now(UTC),
        )

    async def update_job_status(self, job: RetrainingJob) -> None:
        """Update job status and results."""
        query = """
            UPDATE retraining_jobs SET
                status = $2,
                start_time = $3,
                end_time = $4,
                new_model_id = $5,
                performance_comparison = $6,
                error_message = $7,
                updated_at = $8
            WHERE job_id = $1
        """

        await self.db_pool.execute(
            query,
            job.job_id,
            job.status,
            job.start_time,
            job.end_time,
            job.new_model_id,
            json.dumps(job.performance_comparison) if job.performance_comparison else None,
            job.error_message,
            datetime.now(UTC),
        )

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get retraining job by ID."""
        query = """
            SELECT * FROM retraining_jobs WHERE job_id = $1
        """

        row = await self.db_pool.fetchrow(query, job_id)
        if row:
            return dict(row)
        return None

    async def get_recent_jobs(self, days: int = 7) -> list[dict[str, Any]]:
        """Get recent retraining jobs."""
        query = """
            SELECT * FROM retraining_jobs
            WHERE created_at > $1
            ORDER BY created_at DESC
        """

        cutoff = datetime.now(UTC) - timedelta(days=days)
        rows = await self.db_pool.fetch(query, cutoff)

        return [dict(row) for row in rows]

    async def get_jobs_by_model(self, model_id: str) -> list[dict[str, Any]]:
        """Get all retraining jobs for a model."""
        query = """
            SELECT * FROM retraining_jobs
            WHERE model_id = $1
            ORDER BY created_at DESC
        """

        rows = await self.db_pool.fetch(query, model_id)
        return [dict(row) for row in rows]

    async def save_drift_detection(self,
                                 model_id: str,
                                 drift_type: str,
                                 metric_name: str,
                                 drift_score: float,
                                 is_significant: bool,
                                 details: dict[str, Any]) -> None:
        """Save drift detection event."""
        query = """
            INSERT INTO drift_detection_events (
                model_id, drift_type, metric_name,
                drift_score, is_significant, details,
                detected_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """

        await self.db_pool.execute(
            query,
            model_id,
            drift_type,
            metric_name,
            drift_score,
            is_significant,
            json.dumps(details),
            datetime.now(UTC),
        )

    async def get_drift_history(self,
                              model_id: str,
                              days: int = 30) -> list[dict[str, Any]]:
        """Get drift detection history for a model."""
        query = """
            SELECT * FROM drift_detection_events
            WHERE model_id = $1 AND detected_at > $2
            ORDER BY detected_at DESC
        """

        cutoff = datetime.now(UTC) - timedelta(days=days)
        rows = await self.db_pool.fetch(query, model_id, cutoff)

        return [dict(row) for row in rows]

    async def get_retraining_metrics(self) -> dict[str, Any]:
        """Get aggregated retraining metrics."""
        query = """
            WITH job_stats AS (
                SELECT
                    COUNT(*) as total_jobs,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_jobs,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_jobs,
                    COUNT(CASE WHEN status = 'running' THEN 1 END) as running_jobs,
                    AVG(EXTRACT(EPOCH FROM (end_time - start_time))) as avg_duration_seconds
                FROM retraining_jobs
                WHERE created_at > NOW() - INTERVAL '30 days'
            ),
            trigger_stats AS (
                SELECT
                    trigger,
                    COUNT(*) as count
                FROM retraining_jobs
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY trigger
            ),
            drift_stats AS (
                SELECT
                    drift_type,
                    COUNT(*) as detections,
                    COUNT(CASE WHEN is_significant THEN 1 END) as significant_detections
                FROM drift_detection_events
                WHERE detected_at > NOW() - INTERVAL '30 days'
                GROUP BY drift_type
            )
            SELECT
                (SELECT row_to_json(job_stats) FROM job_stats) as job_statistics,
                (SELECT json_agg(row_to_json(trigger_stats))
                 FROM trigger_stats) as trigger_distribution,
                (SELECT json_agg(row_to_json(drift_stats))
                 FROM drift_stats) as drift_statistics
        """

        row = await self.db_pool.fetchrow(query)

        return {
            "job_statistics": row["job_statistics"] or {},
            "trigger_distribution": row["trigger_distribution"] or [],
            "drift_statistics": row["drift_statistics"] or [],
        }
