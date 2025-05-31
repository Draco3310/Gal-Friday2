"""initial_schema_ddl_manual_v8

Revision ID: 3d5abeaa9f0e
Revises: 6c1cbd03238e
Create Date: 2025-05-28 21:19:08.967180

"""
from typing import Sequence, Union

from alembic import op # type: ignore[import-not-found]
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3d5abeaa9f0e'
down_revision: Union[str, None] = '6c1cbd03238e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
