# Position-Order Relationship Architecture

## Overview

This document describes the architectural decision and implementation of the relationship between Position and Order models in the Gal-Friday trading system.

## Decision Summary

**Decision**: Implement a one-to-many relationship from Position to Orders with foreign key on the Order side.

**Rationale**: Financial trading systems require comprehensive audit trails, debugging capabilities, and regulatory compliance that necessitate direct traceability from orders to position changes.

## Architecture Design

### Database Schema

```sql
-- Orders table with position_id foreign key
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES trade_signals(id),
    position_id UUID REFERENCES positions(id) ON DELETE SET NULL,
    trading_pair VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity NUMERIC(20,8) NOT NULL,
    limit_price NUMERIC(20,8),
    status VARCHAR(20) NOT NULL,
    exchange_order_id VARCHAR(100),
    filled_quantity NUMERIC(20,8) DEFAULT 0,
    average_fill_price NUMERIC(20,8),
    commission NUMERIC(20,8),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE INDEX idx_orders_position_id ON orders(position_id);
```

### SQLAlchemy Models

```python
# Position Model
class Position(Base):
    __tablename__ = "positions"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    trading_pair: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    # ... other fields ...
    
    # One-to-many relationship with orders
    orders = relationship("Order", back_populates="position", cascade="all, delete-orphan")

# Order Model  
class Order(Base):
    __tablename__ = "orders"
    
    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    signal_id: Mapped[UUID] = mapped_column(ForeignKey("trade_signals.id"))
    position_id: Mapped[UUID | None] = mapped_column(ForeignKey("positions.id"), nullable=True)
    trading_pair: Mapped[str] = mapped_column(String(20), nullable=False)
    # ... other fields ...
    
    # Many-to-one relationship with position
    position = relationship("Position", back_populates="orders")
```

## Business Logic Integration

### When to Set position_id

The `position_id` should be set when:

1. **Order Fills**: When an order is filled and affects a position
2. **Position Creation**: When an order creates a new position
3. **Position Modification**: When an order modifies an existing position

### Business Logic Flow

```python
# Example: PositionManager integration
class PositionManager:
    async def update_position_for_trade(
        self, 
        trading_pair: str,
        side: str, 
        quantity: Decimal,
        price: Decimal,
        order_id: UUID,  # New parameter
        **kwargs
    ) -> tuple[Decimal, Position | None]:
        """Update position and link the contributing order."""
        
        # Get or create position
        position = await self._get_or_create_db_position(trading_pair, side)
        
        # Update position with trade details
        # ... existing position update logic ...
        
        # Link the order to this position
        await self._link_order_to_position(order_id, position.id)
        
        return realized_pnl, position
    
    async def _link_order_to_position(self, order_id: UUID, position_id: UUID) -> None:
        """Link an order to a position."""
        await self.order_repository.update(
            order_id, 
            {"position_id": position_id}
        )
```

## Usage Patterns

### 1. Query Position with Orders

```python
# Get position with all contributing orders
async def get_position_with_orders(session: AsyncSession, position_id: UUID) -> Position:
    stmt = select(Position).options(selectinload(Position.orders)).where(Position.id == position_id)
    result = await session.execute(stmt)
    return result.scalar_one()

# Usage
position = await get_position_with_orders(session, position_id)
print(f"Position has {len(position.orders)} contributing orders")
for order in position.orders:
    print(f"Order {order.id}: {order.side} {order.quantity} @ {order.average_fill_price}")
```

### 2. Query Orders for Position

```python
# Get all orders that contributed to a specific position
async def get_orders_for_position(session: AsyncSession, position_id: UUID) -> list[Order]:
    stmt = select(Order).where(Order.position_id == position_id).order_by(Order.created_at)
    result = await session.execute(stmt)
    return result.scalars().all()
```

### 3. Audit Trail Query

```python
# Generate audit trail for position
async def generate_position_audit_trail(session: AsyncSession, position_id: UUID) -> dict:
    position = await get_position_with_orders(session, position_id)
    
    audit_trail = {
        "position_id": position.id,
        "trading_pair": position.trading_pair,
        "current_quantity": position.quantity,
        "current_entry_price": position.entry_price,
        "orders": []
    }
    
    for order in sorted(position.orders, key=lambda x: x.created_at):
        audit_trail["orders"].append({
            "order_id": order.id,
            "timestamp": order.created_at,
            "side": order.side,
            "quantity": order.quantity,
            "price": order.average_fill_price,
            "status": order.status
        })
    
    return audit_trail
```

## Performance Considerations

### Indexing Strategy

- **Primary Index**: `idx_orders_position_id` on `orders(position_id)`
- **Composite Indexes**: Consider adding based on query patterns:
  - `orders(position_id, created_at)` for chronological order queries
  - `orders(position_id, status)` for filtering by order status

### Query Optimization

```python
# Efficient loading with selectinload to avoid N+1 queries
stmt = select(Position).options(selectinload(Position.orders)).where(Position.is_active == True)

# For large result sets, use pagination
stmt = select(Position).options(selectinload(Position.orders)).limit(100).offset(page * 100)
```

## Data Integrity

### Foreign Key Constraints

- **ON DELETE SET NULL**: When a position is deleted, orders keep historical record but position_id becomes NULL
- **Nullable position_id**: Orders can exist without positions (e.g., pending orders)

### Cascade Behavior

- **cascade="all, delete-orphan"**: SQLAlchemy relationship cascade for ORM operations
- **Database-level**: Foreign key constraint with SET NULL for referential integrity

## Migration Strategy

### Database Migration

```sql
-- Add column
ALTER TABLE orders ADD COLUMN position_id UUID;

-- Add foreign key constraint  
ALTER TABLE orders ADD CONSTRAINT fk_orders_position_id 
    FOREIGN KEY (position_id) REFERENCES positions(id) ON DELETE SET NULL;

-- Add index
CREATE INDEX idx_orders_position_id ON orders(position_id);
```

### Data Backfill (Optional)

If needed, existing orders can be linked to positions through business logic:

```python
async def backfill_order_position_links():
    """Backfill position_id for existing orders based on business logic."""
    # This would analyze existing orders and fills to determine 
    # which orders contributed to which positions
    pass
```

## Testing Strategy

### Unit Tests

- Relationship creation and access
- Foreign key constraint validation
- Cascade delete behavior
- Query performance with relationships

### Integration Tests

- End-to-end order-to-position workflows
- Performance testing with large datasets
- Migration testing

## Best Practices

### For Developers

1. **Always use transactions** when creating orders and positions together
2. **Use selectinload** to avoid N+1 queries when accessing relationships
3. **Set position_id** when orders are filled and affect positions
4. **Consider nullable position_id** - not all orders immediately affect positions

### For Database Operations

1. **Use proper indexing** for position_id queries
2. **Monitor query performance** as data grows
3. **Consider partitioning** for very large order tables
4. **Regular maintenance** of foreign key indexes

## Monitoring and Alerts

### Key Metrics

- Orders without position_id when they should have one
- Positions with no contributing orders (potential data inconsistency)
- Query performance on position-order joins
- Foreign key violation attempts

### Alerting

```sql
-- Alert on orders that should have position_id but don't
SELECT COUNT(*) 
FROM orders 
WHERE status IN ('FILLED', 'PARTIALLY_FILLED') 
  AND position_id IS NULL 
  AND created_at > NOW() - INTERVAL '1 hour';
```

## Future Considerations

### Potential Enhancements

1. **Order Contributions**: Track percentage contribution of each order to position
2. **Historical Positions**: Archive closed positions with order history
3. **Multi-leg Orders**: Support for complex order strategies affecting multiple positions
4. **Performance Optimization**: Consider read replicas for audit queries

### Scalability

- Monitor table growth and query performance
- Consider archiving old order-position relationships
- Implement proper caching strategies for frequently accessed relationships 