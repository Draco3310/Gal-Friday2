from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker

# Placeholder function for getting the database connection string
def get_database_connection_string() -> str:
    # In a real application, this would fetch from config/config.yaml
    # For now, using a placeholder value.
    # Ensure this matches the expected format, e.g., "postgresql+asyncpg://user:password@host/dbname"
    return "postgresql+asyncpg://user:password@host/dbname_placeholder"

DATABASE_URL = get_database_connection_string()

engine = create_async_engine(DATABASE_URL, echo=True, pool_size=5, max_overflow=10)

AsyncSessionFactory = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        yield session
