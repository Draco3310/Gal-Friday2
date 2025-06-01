"""initial_schema_from_models_ddl_final_v2

Revision ID: 5f3f435c87b5
Revises: dd043229e679
Create Date: 2025-05-28 21:16:59.578706

"""
from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "5f3f435c87b5"
down_revision: str | None = "dd043229e679"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
