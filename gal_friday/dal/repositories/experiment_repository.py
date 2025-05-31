"""Repository for A/B testing experiment data using SQLAlchemy."""

import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Sequence

from sqlalchemy import select, func, update as sqlalchemy_update, cast, Numeric, Integer
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert # For ON CONFLICT DO UPDATE/NOTHING

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.models.experiment import Experiment
from gal_friday.dal.models.experiment_assignment import ExperimentAssignment
from gal_friday.dal.models.experiment_outcome import ExperimentOutcome
# ExperimentConfig and ExperimentStatus would now likely be service-layer or domain models.

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class ExperimentRepository(BaseRepository[Experiment]):
    """Repository for Experiment data persistence using SQLAlchemy."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], logger: "LoggerService"
    ) -> None:
        """Initialize the experiment repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            logger: Logger service instance.
        """
        super().__init__(session_maker, Experiment, logger)

    async def save_experiment(self, experiment_data: dict[str, Any]) -> Experiment:
        """Saves an experiment. Uses INSERT ... ON CONFLICT DO UPDATE."""
        # Ensure experiment_id is a UUID
        if isinstance(experiment_data.get("experiment_id"), str):
            experiment_data["experiment_id"] = uuid.UUID(experiment_data["experiment_id"])
        elif "experiment_id" not in experiment_data:
             experiment_data["experiment_id"] = uuid.uuid4()


        # Convert datetime strings to datetime objects if necessary, ensure timezone aware
        for key in ["start_time", "end_time", "created_at", "completed_at"]:
            if key in experiment_data and isinstance(experiment_data[key], str):
                dt_obj = datetime.fromisoformat(experiment_data[key])
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=timezone.utc)
                experiment_data[key] = dt_obj
            elif key in experiment_data and isinstance(experiment_data[key], datetime) and experiment_data[key].tzinfo is None:
                 experiment_data[key] = experiment_data[key].replace(tzinfo=timezone.utc)


        # Prepare values for insert/update
        # SQLAlchemy's pg_insert handles type conversions for basic types
        # For complex types like enums previously, ensure they are passed as their DB value (e.g. string)
        # allocation_strategy.value -> allocation_strategy (if it's a string in DB)
        # traffic_split, confidence_level, min_detectable_effect, max_loss_threshold to Decimal if not already

        async with self.session_maker() as session:
            # This is a simplified version. A full upsert might require more specific handling
            # of which columns to update on conflict.
            # For now, let's assume BaseRepository.create or a direct session.merge could work
            # if the PK is set. Or using pg_insert for full ON CONFLICT control.

            # For simplicity, let's use get then update/create. A true upsert is more complex with ORM.
            existing_exp = await session.get(Experiment, experiment_data["experiment_id"])
            if existing_exp:
                # Update existing
                for key, value in experiment_data.items():
                    if hasattr(existing_exp, key):
                        setattr(existing_exp, key, value)
                if hasattr(existing_exp, "updated_at") and "updated_at" not in experiment_data : # Assuming an updated_at field
                    setattr(existing_exp, "updated_at", datetime.now(timezone.utc))
                session.add(existing_exp)
                exp_to_return = existing_exp
            else:
                # Create new
                # Ensure all required fields for Experiment model are present in experiment_data
                exp_to_return = Experiment(**experiment_data)
                session.add(exp_to_return)
            
            await session.commit()
            await session.refresh(exp_to_return)
            return exp_to_return


    async def get_experiment(self, experiment_id: uuid.UUID) -> Experiment | None:
        """Get experiment by its UUID."""
        return await self.get_by_id(experiment_id)

    async def get_active_experiments(self) -> Sequence[Experiment]:
        """Get all active experiments (status 'created' or 'running' and not ended)."""
        async with self.session_maker() as session:
            stmt = (
                select(Experiment)
                .where(
                    Experiment.status.in_(["created", "running"]),
                    (Experiment.end_time == None) | (Experiment.end_time > datetime.now(timezone.utc)),
                )
                .order_by(Experiment.start_time.desc())
            )
            result = await session.execute(stmt)
            return result.scalars().all()

    async def record_assignment(
        self, assignment_data: dict[str, Any]
    ) -> ExperimentAssignment | None:
        """Records a variant assignment. Uses INSERT ... ON CONFLICT DO NOTHING."""
        # Ensure experiment_id and event_id are UUIDs
        for key in ["experiment_id", "event_id"]:
            if isinstance(assignment_data.get(key), str):
                assignment_data[key] = uuid.UUID(assignment_data[key])
        
        if isinstance(assignment_data.get("assigned_at"), str):
            dt_obj = datetime.fromisoformat(assignment_data["assigned_at"])
            assignment_data["assigned_at"] = dt_obj.replace(tzinfo=timezone.utc) if dt_obj.tzinfo is None else dt_obj
        elif isinstance(assignment_data.get("assigned_at"), datetime) and assignment_data["assigned_at"].tzinfo is None:
            assignment_data["assigned_at"] = assignment_data["assigned_at"].replace(tzinfo=timezone.utc)


        stmt = pg_insert(ExperimentAssignment).values(**assignment_data)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=[ExperimentAssignment.experiment_id, ExperimentAssignment.event_id]
        )
        async with self.session_maker() as session:
            await session.execute(stmt)
            await session.commit()
            # To return the assignment (even if it already existed and did nothing), we'd need a select after.
            # For now, let's just return the input or fetch it.
            # A simple way if insert happened is to return the object, but ON CONFLICT DO NOTHING makes it tricky.
            # Let's try to fetch it:
            return await session.get(ExperimentAssignment, (assignment_data["experiment_id"], assignment_data["event_id"]))


    async def get_assignment(
        self, experiment_id: uuid.UUID, event_id: uuid.UUID
    ) -> ExperimentAssignment | None:
        """Get variant assignment for an event."""
        async with self.session_maker() as session:
            return await session.get(ExperimentAssignment, (experiment_id, event_id))

    async def save_outcome(self, outcome_data: dict[str, Any]) -> ExperimentOutcome:
        """Saves a prediction outcome for experiment analysis."""
        for key in ["experiment_id", "event_id"]: # outcome_id is auto-gen by default
            if isinstance(outcome_data.get(key), str):
                outcome_data[key] = uuid.UUID(outcome_data[key])

        if isinstance(outcome_data.get("recorded_at"), str):
             dt_obj = datetime.fromisoformat(outcome_data["recorded_at"])
             outcome_data["recorded_at"] = dt_obj.replace(tzinfo=timezone.utc) if dt_obj.tzinfo is None else dt_obj
        elif isinstance(outcome_data.get("recorded_at"), datetime) and outcome_data["recorded_at"].tzinfo is None:
            outcome_data["recorded_at"] = outcome_data["recorded_at"].replace(tzinfo=timezone.utc)


        # trade_return should be Decimal
        if "trade_return" in outcome_data and not isinstance(outcome_data["trade_return"], Decimal):
            outcome_data["trade_return"] = Decimal(str(outcome_data["trade_return"]))

        async with self.session_maker() as session:
            outcome = ExperimentOutcome(**outcome_data)
            session.add(outcome)
            await session.commit()
            await session.refresh(outcome)
            return outcome

    async def get_experiment_performance(
        self, experiment_id: uuid.UUID
    ) -> dict[str, dict[str, Any]]:
        """Get aggregated performance metrics for experiment variants."""
        stmt = (
            select(
                ExperimentOutcome.variant,
                func.count().label("sample_count"),
                func.sum(cast(ExperimentOutcome.correct_prediction, Integer)).label("correct_predictions"),
                func.sum(cast(ExperimentOutcome.signal_generated, Integer)).label("signals_generated"),
                func.sum(ExperimentOutcome.trade_return).label("total_return"),
                func.avg(cast(ExperimentOutcome.correct_prediction, Numeric)).label("accuracy"),
            )
            .where(ExperimentOutcome.experiment_id == experiment_id)
            .group_by(ExperimentOutcome.variant)
        )
        performance_summary: dict[str, dict[str, Any]] = {}
        async with self.session_maker() as session:
            result = await session.execute(stmt)
            for row in result.mappings(): # Use mappings() to get dict-like rows
                performance_summary[row["variant"]] = {
                    "sample_count": row["sample_count"],
                    "correct_predictions": row["correct_predictions"] or 0,
                    "signals_generated": row["signals_generated"] or 0,
                    "total_return": row["total_return"] or Decimal("0"),
                    "accuracy": row["accuracy"] or Decimal("0.0"),
                }
        return performance_summary

    async def save_results(self, experiment_id: uuid.UUID, results_data: dict[str, Any]) -> Experiment | None:
        """Save final experiment results by updating the Experiment model."""
        # Ensure 'completed_at' is a datetime object if provided, otherwise set to now
        if "completed_at" not in results_data:
            results_data["completed_at"] = datetime.now(timezone.utc)
        elif isinstance(results_data["completed_at"], str):
             dt_obj = datetime.fromisoformat(results_data["completed_at"])
             results_data["completed_at"] = dt_obj.replace(tzinfo=timezone.utc) if dt_obj.tzinfo is None else dt_obj
        elif isinstance(results_data["completed_at"], datetime) and results_data["completed_at"].tzinfo is None:
            results_data["completed_at"] = results_data["completed_at"].replace(tzinfo=timezone.utc)
            
        # status and completion_reason are part of results_data
        return await self.update(experiment_id, results_data)


    async def get_experiment_history(self, days: int = 30) -> Sequence[dict[str, Any]]:
        """Get experiment history for analysis, including total_assignments."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Subquery to count assignments
        subquery = (
            select(
                ExperimentAssignment.experiment_id,
                func.count(ExperimentAssignment.event_id.distinct()).label("total_assignments"),
            )
            .group_by(ExperimentAssignment.experiment_id)
            .subquery()
        )

        stmt = (
            select(Experiment, subquery.c.total_assignments)
            .outerjoin(subquery, Experiment.experiment_id == subquery.c.experiment_id)
            .where(Experiment.created_at > cutoff_date) # Assuming Experiment has created_at
            .order_by(Experiment.created_at.desc())
        )

        history = []
        async with self.session_maker() as session:
            result = await session.execute(stmt)
            for row in result: # row is a Row object
                exp = row[0] # The Experiment object
                total_assignments = row[1] or 0 # Value from subquery
                exp_dict = {column.name: getattr(exp, column.name) for column in exp.__table__.columns}
                exp_dict["total_assignments"] = total_assignments
                history.append(exp_dict)
        return history
