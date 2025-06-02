"""initial_schema_from_models_manual.

Revision ID: 1d03e90d0571
Revises:
Create Date: 2025-05-28 21:10:06.655207

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "1d03e90d0571"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
