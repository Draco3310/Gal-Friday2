import uuid
from datetime import datetime

from sqlalchemy import Column, BigInteger, String, Text, DateTime, JSON, Integer # Added Integer
from sqlalchemy.dialects.postgresql import UUID as PG_UUID # For signal_id
from sqlalchemy.sql import func

from .base import Base


class SystemLog(Base):
    __tablename__ = "system_logs"

    log_pk = Column(BigInteger, primary_key=True, autoincrement=True)
    log_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    source_module = Column(String(64), nullable=False, index=True)
    log_level = Column(String(10), nullable=False, index=True)  # CHECK constraint handled by application/DB
    message = Column(Text, nullable=False)
    
    # Contextual fields
    trading_pair = Column(String(16), nullable=True)
    signal_id = Column(PG_UUID(as_uuid=True), nullable=True) # Assuming this might link to signals.signal_id
    order_pk = Column(Integer, nullable=True) # Assuming this might link to orders.order_pk
    
    exception_type = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    context = Column(JSON, nullable=True) # For arbitrary additional context

    def __repr__(self):
        return (
            f"<SystemLog(log_pk={self.log_pk}, source_module='{self.source_module}', "
            f"log_level='{self.log_level}', message='{self.message[:50]}...')>"
        )
