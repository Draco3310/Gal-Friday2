"""initial_schema_from_models_ddl_final_v2

Revision ID: 5f3f435c87b5
Revises: dd043229e679
Create Date: 2025-05-28 21:16:59.578706

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5f3f435c87b5'
down_revision: Union[str, None] = 'dd043229e679'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
