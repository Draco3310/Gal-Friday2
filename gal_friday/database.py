"""Database connection and session management with enterprise configuration."""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gal_friday.config_manager import ConfigManager


def get_database_connection_string() -> str:
    """Get database connection string from configuration.
    
    Returns:
        Database connection string in format: postgresql+asyncpg://user:password@host:port/dbname
        
    Raises:
        ValueError: If database configuration is invalid or missing
    """
    config_manager = ConfigManager()

    # Get database configuration with validation
    db_config = config_manager.get("database", {})

    if not db_config:
        raise ValueError("Database configuration not found in config")

    # Extract required parameters
    host = db_config.get("host")
    port = db_config.get("port", 5432)
    database = db_config.get("name")
    username = db_config.get("username")
    password = db_config.get("password")

    # Validate required parameters
    missing = []
    if not host:
        missing.append("host")
    if not database:
        missing.append("database name")
    if not username:
        missing.append("username")
    if not password:
        missing.append("password")

    if missing:
        raise ValueError(f"Missing required database configuration: {', '.join(missing)}")

    # Build connection string
    connection_string = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"

    # Add optional parameters
    options = []
    if db_config.get("ssl_mode"):
        options.append(f"sslmode={db_config['ssl_mode']}")
    if db_config.get("server_version"):
        options.append(f"server_version={db_config['server_version']}")

    if options:
        connection_string += "?" + "&".join(options)

    return connection_string


# Get configuration-based connection string
DATABASE_URL = get_database_connection_string()

# Create engine with production-grade settings
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Disable SQL echo in production
    pool_size=20,  # Increased pool size for production
    max_overflow=10,
    pool_pre_ping=True,  # Test connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
    connect_args={
        "server_settings": {"application_name": "gal_friday"},
        "command_timeout": 60,
        "connection_timeout": 10,
    })

# Create session factory
AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session.
    
    Yields:
        AsyncSession: Database session for async operations
        
    Note:
        Session is automatically closed when context exits
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_database() -> None:
    """Initialize database connection and verify connectivity.
    
    Raises:
        Exception: If database connection fails
    """
    try:
        # Test the connection
        async with engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: sync_conn.execute(text("SELECT 1")))
        print("✅ Database connection verified")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise


async def close_database() -> None:
    """Close database connections and dispose of connection pool."""
    await engine.dispose()
    print("✅ Database connections closed")