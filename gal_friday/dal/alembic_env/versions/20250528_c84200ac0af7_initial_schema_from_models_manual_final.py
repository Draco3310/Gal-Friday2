"""initial_schema_from_models_manual_final.

Revision ID: c84200ac0af7
Revises: d1749c89baa0
Create Date: 2025-05-28 21:12:23.833165

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "c84200ac0af7"
down_revision: str | None = "d1749c89baa0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
