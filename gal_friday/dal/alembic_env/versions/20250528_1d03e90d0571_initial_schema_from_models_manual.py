"""initial_schema_from_models_manual

Revision ID: 1d03e90d0571
Revises: 
Create Date: 2025-05-28 21:10:06.655207

"""
from typing import Sequence, Union

from alembic import op # type: ignore[import-not-found]
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1d03e90d0571'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
