"""Enforce non-nullable reconciliation_id in position_adjustments.

Revision ID: enforce_non_nullable_reconciliation_id
Revises: add_position_id_to_orders
Create Date: 2025-06-11 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "enforce_non_nullable_reconciliation_id"
down_revision = "add_position_id_to_orders"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Enforce NOT NULL constraint on reconciliation_id."""
    op.alter_column(
        "position_adjustments",
        "reconciliation_id",
        existing_type=sa.dialects.postgresql.UUID(as_uuid=True),
        nullable=False)


def downgrade() -> None:
    """Revert reconciliation_id to nullable."""
    op.alter_column(
        "position_adjustments",
        "reconciliation_id",
        existing_type=sa.dialects.postgresql.UUID(as_uuid=True),
        nullable=True)