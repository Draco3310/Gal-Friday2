"""initial_schema_ddl_manual_v4

Revision ID: 465b5ba8a9a2
Revises: a31175f3afaa
Create Date: 2025-05-28 21:14:57.229832

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '465b5ba8a9a2'
down_revision: Union[str, None] = 'a31175f3afaa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
