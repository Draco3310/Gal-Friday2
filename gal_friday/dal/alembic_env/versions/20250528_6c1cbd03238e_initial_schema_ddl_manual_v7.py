"""initial_schema_ddl_manual_v7

Revision ID: 6c1cbd03238e
Revises: 5f3f435c87b5
Create Date: 2025-05-28 21:18:22.118325

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6c1cbd03238e'
down_revision: Union[str, None] = '5f3f435c87b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
