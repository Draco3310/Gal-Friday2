"""initial_schema_ddl_manual_v7.

Revision ID: 6c1cbd03238e
Revises: 5f3f435c87b5
Create Date: 2025-05-28 21:18:22.118325

"""
from collections.abc import Sequence
from typing import Any

# revision identifiers, used by Alembic.
revision: str = "6c1cbd03238e"
down_revision: str | None = "5f3f435c87b5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[Any] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
