"""initial_schema_ddl_manual_v6

Revision ID: dd043229e679
Revises: 465b5ba8a9a2
Create Date: 2025-05-28 21:16:14.970591

"""
from collections.abc import Sequence
from typing import Any

# revision identifiers, used by Alembic.
revision: str = "dd043229e679"
down_revision: str | None = "465b5ba8a9a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[Any] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""