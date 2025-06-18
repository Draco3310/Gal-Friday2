"""add_position_id_to_orders.

Revision ID: add_position_id_to_orders
Revises: c84200ac0af7
Create Date: 2025-01-27 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "add_position_id_to_orders"
down_revision = "c84200ac0af7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add position_id foreign key to orders table."""
    # Add position_id column to orders table
    op.add_column("orders", sa.Column("position_id", postgresql.UUID(as_uuid=True), nullable=True))

    # Add foreign key constraint
    op.create_foreign_key(
        "fk_orders_position_id",
        "orders",
        "positions",
        ["position_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Add index for performance
    op.create_index("idx_orders_position_id", "orders", ["position_id"])


def downgrade() -> None:
    """Remove position_id foreign key from orders table."""
    # Drop index
    op.drop_index("idx_orders_position_id", table_name="orders")

    # Drop foreign key constraint
    op.drop_constraint("fk_orders_position_id", "orders", type_="foreignkey")

    # Drop column
    op.drop_column("orders", "position_id")
