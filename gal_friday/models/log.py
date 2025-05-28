import datetime

from sqlalchemy import Column, BigInteger, String, Integer, Text, DateTime
from sqlalchemy.dialects.postgresql import TIMESTAMPTZ
from sqlalchemy.types import JSON  # Generic JSON type

from .base import Base


class Log(Base):
    __tablename__ = "logs"

    id = Column(BigInteger, primary_key=True, index=True)
    timestamp = Column(TIMESTAMPTZ, nullable=False, default=datetime.datetime.utcnow)
    logger_name = Column(String(255), nullable=False)
    level_name = Column(String(50), nullable=False)
    level_no = Column(Integer, nullable=False)
    message = Column(Text, nullable=False)
    pathname = Column(Text, nullable=True)
    filename = Column(String(255), nullable=True)
    lineno = Column(Integer, nullable=True)
    func_name = Column(String(255), nullable=True)
    context_json = Column(JSON, nullable=True)
    exception_text = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Log(id={self.id}, name='{self.logger_name}', level='{self.level_name}')>"
