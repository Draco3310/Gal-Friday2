"""initial_schema_ddl_manual_v4

Revision ID: 465b5ba8a9a2
Revises: a31175f3afaa
Create Date: 2025-05-28 21:14:57.229832

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "465b5ba8a9a2"
down_revision: str | None = "a31175f3afaa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
