"""initial_schema_ddl_manual_v9

Revision ID: ca9c62eed06d
Revises: 3d5abeaa9f0e
Create Date: 2025-05-28 21:20:36.352566

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ca9c62eed06d'
down_revision: Union[str, None] = '3d5abeaa9f0e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
