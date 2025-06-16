"""initial_schema_ddl_manual_v9

Revision ID: ca9c62eed06d
Revises: 3d5abeaa9f0e
Create Date: 2025-05-28 21:20:36.352566

"""
from collections.abc import Sequence
from typing import Any

# revision identifiers, used by Alembic.
revision: str = "ca9c62eed06d"
down_revision: str | None = "3d5abeaa9f0e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[Any] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""