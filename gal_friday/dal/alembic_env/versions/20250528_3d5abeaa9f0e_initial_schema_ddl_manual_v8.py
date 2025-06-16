"""initial_schema_ddl_manual_v8

Revision ID: 3d5abeaa9f0e
Revises: 6c1cbd03238e
Create Date: 2025-05-28 21:19:08.967180

"""
from collections.abc import Sequence
from typing import Any

# revision identifiers, used by Alembic.
revision: str = "3d5abeaa9f0e"
down_revision: str | None = "6c1cbd03238e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[Any] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""