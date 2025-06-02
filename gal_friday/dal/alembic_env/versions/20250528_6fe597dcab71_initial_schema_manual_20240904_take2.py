"""initial_schema_manual_20240904_take2.

Revision ID: 6fe597dcab71
Revises: c84200ac0af7
Create Date: 2025-05-28 21:13:06.059358

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "6fe597dcab71"
down_revision: str | None = "c84200ac0af7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
