"""initial_schema_from_models_manual_v2.

Revision ID: 6bdebe2daca2
Revises: 1d03e90d0571
Create Date: 2025-05-28 21:10:50.098744

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "6bdebe2daca2"
down_revision: str | None = "1d03e90d0571"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
