# Database Module Documentation

## Module Overview

The `gal_friday.database.py` module is responsible for establishing and managing asynchronous database connections for the Gal-Friday trading system. It utilizes SQLAlchemy's asynchronous features to interact with a relational database (typically PostgreSQL). The module provides a global asynchronous engine and a session factory to ensure consistent and efficient database access throughout the application.

## Key Features

-   **Asynchronous SQLAlchemy Engine:** Establishes a single `sqlalchemy.ext.asyncio.AsyncEngine` instance (`async_engine`) for the application, which manages a pool of database connections.
-   **Asynchronous Session Factory:** Provides an `AsyncSessionFactory` (a configured `sqlalchemy.orm.sessionmaker`) for creating new `AsyncSession` instances.
-   **Managed Database Sessions:** Offers a convenient asynchronous generator function, `get_db_session()`, to acquire and release `AsyncSession` instances, intended for use as a dependency in other parts of the application.
-   **Configuration-Driven Connection String (Intended):** Includes a placeholder function for retrieving the database connection string, with the intention that this string will be fetched from a configuration file (e.g., `config/config.yaml`) in a production setup.
-   **Configurable Connection Pool:** The SQLAlchemy engine is initialized with configurable connection pool parameters (`pool_size` and `max_overflow`) to optimize database connection management under load.

## Core Components

-   **`DATABASE_URL (str)`**:
    -   A module-level variable that holds the database connection string.
    -   **Note:** In the current implementation, this is initialized by calling `get_database_connection_string()`, which returns a placeholder. For a functional system, this must resolve to a valid SQLAlchemy asynchronous database URL.

-   **`engine (sqlalchemy.ext.asyncio.AsyncEngine)`**:
    -   The global SQLAlchemy asynchronous engine instance created using `create_async_engine`.
    -   It uses the `DATABASE_URL` and is configured with connection pooling options (`pool_size=20`, `max_overflow=10`, `echo=False` by default). `echo=True` can be useful for debugging SQL queries.

-   **`AsyncSessionFactory (sqlalchemy.orm.sessionmaker)`**:
    -   A session factory configured to create `AsyncSession` instances bound to the global `engine`.
    -   It is created using `sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)`. `expire_on_commit=False` is a common setting for asynchronous sessions to prevent objects from being expired prematurely when using detached objects.

## Functions

### `get_database_connection_string() -> str`

-   **Purpose:** This function is designed to retrieve the database connection string required by SQLAlchemy to connect to the target database.
-   **Current Implementation:**
    ```python
    # TODO: Implement fetching from config/config.yaml
    return "postgresql+asyncpg://user:password@host/dbname" # Placeholder
    ```
    Currently, it returns a hardcoded placeholder string for a PostgreSQL database.
-   **Intended Behavior:** In a production or properly configured development environment, this function should:
    1.  Access the application's configuration (e.g., via a `ConfigManager` instance or by directly reading `config/config.yaml`).
    2.  Fetch the database URL from a specific configuration key (e.g., `database.url`).
    3.  Return the fetched connection string.
    -   An error should be raised or logged if the connection string cannot be found in the configuration.

### `get_db_session() -> AsyncIterator[AsyncSession]`

-   **Purpose:** Provides an asynchronous context manager pattern (via an async generator) to yield an `AsyncSession` for database operations. This is typically used for dependency injection into services or repository methods that need to interact with the database.
-   **Usage:**
    ```python
    async def my_database_operation():
        async for session in get_db_session():
            # Use the session object here
            # e.g., result = await session.execute(...)
            # await session.commit() or await session.rollback()
            pass
    ```
    Or, more commonly when a single session is needed for a block of operations:
    ```python
    async def my_database_operation():
        session_generator = get_db_session()
        session: AsyncSession = await session_generator.__anext__()
        try:
            # Use the session object here
            # e.g., result = await session.execute(...)
            # await session.commit() # or rollback
        finally:
            await session.close() # Ensure session is closed
            # To exhaust the generator if it had more yields or cleanup in a finally block
            try:
                await session_generator.__anext__() # Should raise StopAsyncIteration
            except StopAsyncIteration:
                pass
    ```
    A more idiomatic way to use such a generator for a single session per operation, especially with FastAPI-like dependency injection, would involve the consumer handling the `try...finally` block to ensure `session.close()`. If `get_db_session` were designed as `asynccontextmanager`, `async with` would be more direct.
-   **Implementation Details:**
    ```python
    async def get_db_session() -> AsyncIterator[AsyncSession]:
        session: AsyncSession = AsyncSessionFactory()
        try:
            yield session
        finally:
            await session.close()
    ```
    -   It creates a new `AsyncSession` using the `AsyncSessionFactory`.
    -   It `yield`s this session to the caller.
    -   Crucially, it includes a `finally` block to ensure `await session.close()` is called, which releases the connection associated with the session back to the engine's connection pool, regardless of whether errors occurred in the calling code.

## Dependencies

-   **`sqlalchemy.ext.asyncio.create_async_engine`**: Used to create the asynchronous database engine.
-   **`sqlalchemy.ext.asyncio.AsyncSession`**: The class representing an asynchronous database session.
-   **`sqlalchemy.orm.sessionmaker`**: Used to create the session factory.
-   An asynchronous database driver compatible with SQLAlchemy, such as `asyncpg` for PostgreSQL (as implied by the placeholder URL).

## Configuration Notes

-   **`DATABASE_URL`**: This is the most critical configuration. It *must* be correctly set up in the application's configuration file (e.g., `config/config.yaml`) and fetched by `get_database_connection_string()` for the system to connect to a database.
    -   Example for PostgreSQL: `"postgresql+asyncpg://your_user:your_password@your_host:your_port/your_database_name"`
-   **Connection Pool Parameters**:
    -   `pool_size` (default in code: `20`): The number of database connections to keep open in the connection pool.
    -   `max_overflow` (default in code: `10`): The maximum number of additional connections that can be opened beyond `pool_size` under heavy load.
    -   These parameters can be tuned based on the expected concurrent database access needs of the application and the capacity of the database server. They would ideally also be configurable via `config.yaml` and passed to `create_async_engine`.
-   **SQLAlchemy Echo**: The `echo=False` parameter for `create_async_engine` means SQLAlchemy will not log all generated SQL statements. Setting this to `True` can be very useful for debugging database interactions but is generally too verbose for production.

## Usage Example

```python
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
# Assuming you have SQLAlchemy 1.4+ style for select
from sqlalchemy import select
# from sqlalchemy.future import select # For older SQLAlchemy 1.4.x versions

# Import your database models (SQLAlchemy ORM classes)
# Example: from gal_friday.models.your_model import YourTableModel

from gal_friday.database import get_db_session

# --- Define a dummy model for the example to be self-contained ---
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class YourTableModel(Base):
    __tablename__ = "example_table"
    id = Column(Integer, primary_key=True)
    name = Column(String)
# --- End of dummy model ---

async def fetch_some_data():
    # This is how you'd typically use the generator for a single operation block
    session_generator = get_db_session()
    session: AsyncSession = await session_generator.__anext__() # Get the session
    try:
        # Example: Create table if it doesn't exist (for demo purposes)
        # async with engine.begin() as conn: # engine would need to be accessible
        #     await conn.run_sync(Base.metadata.create_all)

        # Example query (actual query depends on your models)
        # For this example, let's assume the table might be empty or not exist without setup
        # result = await session.execute(select(YourTableModel).limit(5))
        # data = result.scalars().all()
        # print(f"Fetched data: {[item.name for item in data]}")

        # If you were making changes:
        # new_item = YourTableModel(name="Test Item")
        # session.add(new_item)
        # await session.commit()
        # print("Added new item.")

        print("Session obtained. Replace with actual database operations.")
        pass # Replace with actual database operations

    except Exception as e:
        print(f"An error occurred: {e}")
        # await session.rollback() # If operations were attempted
    finally:
        await session.close() # Ensure session is closed
        # Exhaust the generator to ensure its finally block runs if it had more yields
        try:
            await session_generator.__anext__()
        except StopAsyncIteration:
            pass


async def main():
    # Note: For this example to *actually* run against a DB,
    # 1. DATABASE_URL needs to point to a real PostgreSQL DB.
    # 2. The asyncpg driver must be installed (pip install asyncpg).
    # 3. The table 'example_table' would need to exist or be created.
    # The following line would typically be run once at application startup to create tables:
    # from gal_friday.database import engine # Assuming engine is exposed
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)

    await fetch_some_data()

# To run this example (assuming a valid DATABASE_URL and setup):
# if __name__ == "__main__":
#     asyncio.run(main())
```

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the database connectivity module.
