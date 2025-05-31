"""initial_schema_from_models_manual_v3

Revision ID: d1749c89baa0
Revises: 6bdebe2daca2
Create Date: 2025-05-28 21:11:37.893894

"""
from typing import Sequence, Union

from alembic import op # type: ignore[import-not-found]
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd1749c89baa0'
down_revision: Union[str, None] = '6bdebe2daca2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
