"""initial_schema_from_models_manual_v3.

Revision ID: d1749c89baa0
Revises: 6bdebe2daca2
Create Date: 2025-05-28 21:11:37.893894

"""
from collections.abc import Sequence
from typing import Any

# revision identifiers, used by Alembic.
revision: str = "d1749c89baa0"
down_revision: str | None = "6bdebe2daca2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[Any] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
