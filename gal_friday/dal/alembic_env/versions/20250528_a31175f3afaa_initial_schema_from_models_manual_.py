"""initial_schema_from_models_manual_20240905.

Revision ID: a31175f3afaa
Revises: 6fe597dcab71
Create Date: 2025-05-28 21:13:51.070520

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "a31175f3afaa"
down_revision: str | None = "6fe597dcab71"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
