"""initial_schema_ddl_manual_v6

Revision ID: dd043229e679
Revises: 465b5ba8a9a2
Create Date: 2025-05-28 21:16:14.970591

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dd043229e679'
down_revision: Union[str, None] = '465b5ba8a9a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
