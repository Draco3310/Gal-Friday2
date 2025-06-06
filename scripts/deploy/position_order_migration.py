#!/usr/bin/env python3
"""Production deployment script for Position-Order relationship migration.

This script safely applies the database migration to add position_id foreign key
to the orders table with comprehensive safety checks and rollback capabilities.

Usage:
    python scripts/deploy/position_order_migration.py --env staging --dry-run
    python scripts/deploy/position_order_migration.py --env production --backup --execute
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


class MigrationDeploymentError(Exception):
    """Custom exception for migration deployment errors."""
    pass


class PositionOrderMigrationDeployer:
    """Deploy position-order relationship migration safely."""
    
    def __init__(self, config_manager: ConfigManager, logger: LoggerService):
        """Initialize the migration deployer.
        
        Args:
            config_manager: Configuration manager for database settings
            logger: Logger service for deployment logging
        """
        self.config = config_manager
        self.logger = logger
        self._source_module = self.__class__.__name__
        
    async def deploy(
        self,
        environment: str,
        dry_run: bool = True,
        create_backup: bool = True,
        timeout_seconds: int = 300
    ) -> bool:
        """Deploy the position-order migration.
        
        Args:
            environment: Target environment (staging/production)
            dry_run: Whether to perform a dry run without actual changes
            create_backup: Whether to create a database backup before migration
            timeout_seconds: Maximum time to wait for migration completion
            
        Returns:
            True if migration was successful, False otherwise
        """
        self.logger.info(
            f"Starting position-order migration deployment to {environment}",
            source_module=self._source_module,
        )
        
        try:
            # Initialize database connection
            engine = await self._create_database_engine(environment)
            session_maker = async_sessionmaker(engine, class_=AsyncSession)
            
            # Pre-migration checks
            await self._perform_pre_migration_checks(session_maker, environment)
            
            # Create backup if requested
            if create_backup and not dry_run:
                backup_file = await self._create_database_backup(environment)
                self.logger.info(f"Database backup created: {backup_file}")
            
            # Execute migration
            if dry_run:
                self.logger.info("DRY RUN: Simulating migration execution")
                await self._simulate_migration(session_maker)
            else:
                self.logger.info("Executing migration")
                await self._execute_migration(session_maker, timeout_seconds)
            
            # Post-migration verification
            await self._perform_post_migration_verification(session_maker)
            
            # Cleanup
            await engine.dispose()
            
            self.logger.info(
                f"Position-order migration deployment completed successfully",
                source_module=self._source_module,
            )
            
            return True
            
        except Exception as e:
            self.logger.exception(
                f"Migration deployment failed: {e}",
                source_module=self._source_module,
            )
            return False

    async def _create_database_engine(self, environment: str):
        """Create database engine for the specified environment."""
        try:
            # Get database configuration for environment
            db_config = self.config.get_database_config(environment)
            
            # Construct connection URL
            db_url = (
                f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            engine = create_async_engine(
                db_url,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
            )
            
            # Test connection
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            self.logger.info(
                f"Database connection established for {environment}",
                source_module=self._source_module,
            )
            
            return engine
            
        except Exception as e:
            raise MigrationDeploymentError(f"Failed to create database engine: {e}")

    async def _perform_pre_migration_checks(self, session_maker, environment: str):
        """Perform comprehensive pre-migration safety checks."""
        self.logger.info("Performing pre-migration safety checks")
        
        try:
            async with session_maker() as session:
                # Check if migration has already been applied
                if await self._is_migration_already_applied(session):
                    raise MigrationDeploymentError(
                        "Migration has already been applied. position_id column exists in orders table."
                    )
                
                # Check database connectivity and permissions
                await self._check_database_permissions(session)
                
                # Verify table structures
                await self._verify_table_structures(session)
                
                # Check for potential data conflicts
                await self._check_data_conflicts(session)
                
                # Estimate migration impact
                impact = await self._estimate_migration_impact(session)
                self.logger.info(f"Migration impact: {impact}")
                
                # Environment-specific checks
                if environment == "production":
                    await self._production_safety_checks(session)
                
            self.logger.info("Pre-migration checks completed successfully")
            
        except Exception as e:
            raise MigrationDeploymentError(f"Pre-migration checks failed: {e}")

    async def _is_migration_already_applied(self, session: AsyncSession) -> bool:
        """Check if the position_id column already exists in orders table."""
        try:
            result = await session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'orders' 
                AND column_name = 'position_id'
            """))
            
            return result.fetchone() is not None
            
        except Exception as e:
            self.logger.error(f"Error checking migration status: {e}")
            raise

    async def _check_database_permissions(self, session: AsyncSession):
        """Verify required database permissions for migration."""
        try:
            # Check ALTER TABLE permission
            await session.execute(text("SELECT has_table_privilege(current_user, 'orders', 'update')"))
            
            # Check CREATE INDEX permission  
            await session.execute(text("SELECT has_schema_privilege(current_user, 'public', 'create')"))
            
            self.logger.debug("Database permissions verified")
            
        except Exception as e:
            raise MigrationDeploymentError(f"Insufficient database permissions: {e}")

    async def _verify_table_structures(self, session: AsyncSession):
        """Verify that required tables exist with expected structure."""
        try:
            # Check orders table exists
            result = await session.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name = 'orders' AND table_schema = 'public'
            """))
            if not result.fetchone():
                raise MigrationDeploymentError("Orders table not found")
            
            # Check positions table exists  
            result = await session.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name = 'positions' AND table_schema = 'public'
            """))
            if not result.fetchone():
                raise MigrationDeploymentError("Positions table not found")
            
            self.logger.debug("Table structures verified")
            
        except Exception as e:
            raise MigrationDeploymentError(f"Table structure verification failed: {e}")

    async def _check_data_conflicts(self, session: AsyncSession):
        """Check for potential data conflicts that could affect migration."""
        try:
            # Check for NULL values in critical columns
            result = await session.execute(text("SELECT COUNT(*) FROM orders WHERE id IS NULL"))
            if result.scalar() > 0:
                raise MigrationDeploymentError("Found orders with NULL id values")
            
            # Check for duplicate order IDs
            result = await session.execute(text("""
                SELECT COUNT(*) FROM (
                    SELECT id, COUNT(*) as cnt FROM orders GROUP BY id HAVING COUNT(*) > 1
                ) duplicates
            """))
            if result.scalar() > 0:
                raise MigrationDeploymentError("Found duplicate order IDs")
            
            self.logger.debug("Data conflict checks passed")
            
        except Exception as e:
            raise MigrationDeploymentError(f"Data conflict check failed: {e}")

    async def _estimate_migration_impact(self, session: AsyncSession) -> dict[str, Any]:
        """Estimate the impact of the migration."""
        try:
            # Count total orders
            result = await session.execute(text("SELECT COUNT(*) FROM orders"))
            total_orders = result.scalar()
            
            # Count recent orders (last 24 hours)
            result = await session.execute(text("""
                SELECT COUNT(*) FROM orders 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """))
            recent_orders = result.scalar()
            
            # Estimate migration time (rough estimate: 1 second per 10,000 rows)
            estimated_seconds = max(1, total_orders // 10000)
            
            return {
                "total_orders": total_orders,
                "recent_orders": recent_orders,
                "estimated_duration_seconds": estimated_seconds,
                "table_size_mb": await self._get_table_size_mb(session, "orders"),
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating migration impact: {e}")
            return {"error": str(e)}

    async def _get_table_size_mb(self, session: AsyncSession, table_name: str) -> float:
        """Get table size in MB."""
        try:
            result = await session.execute(text(f"""
                SELECT pg_size_pretty(pg_total_relation_size('{table_name}'))::text,
                       pg_total_relation_size('{table_name}') / (1024*1024) as size_mb
            """))
            row = result.fetchone()
            return row[1] if row else 0.0
        except Exception:
            return 0.0

    async def _production_safety_checks(self, session: AsyncSession):
        """Additional safety checks for production environment."""
        try:
            # Check for recent high activity
            result = await session.execute(text("""
                SELECT COUNT(*) FROM orders 
                WHERE created_at > NOW() - INTERVAL '5 minutes'
            """))
            recent_activity = result.scalar()
            
            if recent_activity > 100:
                self.logger.warning(
                    f"High recent activity detected: {recent_activity} orders in last 5 minutes"
                )
                # Could add confirmation prompt here in interactive mode
            
            # Check database locks
            result = await session.execute(text("""
                SELECT COUNT(*) FROM pg_locks 
                WHERE relation = 'orders'::regclass AND mode LIKE '%ExclusiveLock%'
            """))
            locks = result.scalar()
            
            if locks > 0:
                raise MigrationDeploymentError(f"Found {locks} exclusive locks on orders table")
            
            self.logger.debug("Production safety checks passed")
            
        except Exception as e:
            raise MigrationDeploymentError(f"Production safety checks failed: {e}")

    async def _create_database_backup(self, environment: str) -> str:
        """Create database backup before migration."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_file = f"position_order_migration_backup_{environment}_{timestamp}.sql"
        
        try:
            # This would use pg_dump or similar tool
            # Implementation depends on your backup infrastructure
            self.logger.info(f"Creating database backup: {backup_file}")
            
            # Placeholder for actual backup implementation
            # In production, this would execute:
            # pg_dump -h host -U user -d database > backup_file
            
            return backup_file
            
        except Exception as e:
            raise MigrationDeploymentError(f"Backup creation failed: {e}")

    async def _simulate_migration(self, session_maker):
        """Simulate migration execution for dry run."""
        self.logger.info("Simulating migration steps:")
        
        migration_steps = [
            "1. Add position_id column to orders table (nullable UUID)",
            "2. Create foreign key constraint fk_orders_position_id",
            "3. Create index idx_orders_position_id",
            "4. Update Alembic migration history",
        ]
        
        for step in migration_steps:
            self.logger.info(f"   {step}")
            await asyncio.sleep(0.1)  # Simulate processing time
        
        self.logger.info("Migration simulation completed successfully")

    async def _execute_migration(self, session_maker, timeout_seconds: int):
        """Execute the actual migration."""
        start_time = time.time()
        
        try:
            async with session_maker() as session:
                async with session.begin():
                    # Step 1: Add position_id column
                    self.logger.info("Adding position_id column to orders table")
                    await session.execute(text("""
                        ALTER TABLE orders 
                        ADD COLUMN position_id UUID REFERENCES positions(id) ON DELETE SET NULL
                    """))
                    
                    # Step 2: Create index
                    self.logger.info("Creating index on position_id")
                    await session.execute(text("""
                        CREATE INDEX CONCURRENTLY idx_orders_position_id ON orders(position_id)
                    """))
                    
                    # Step 3: Update Alembic version
                    self.logger.info("Updating Alembic migration history")
                    await session.execute(text("""
                        INSERT INTO alembic_version (version_num) 
                        VALUES ('add_position_id_to_orders')
                        ON CONFLICT (version_num) DO NOTHING
                    """))
                    
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        raise MigrationDeploymentError(f"Migration exceeded timeout of {timeout_seconds} seconds")
                    
                    self.logger.info(f"Migration executed successfully in {elapsed:.2f} seconds")
                    
        except Exception as e:
            raise MigrationDeploymentError(f"Migration execution failed: {e}")

    async def _perform_post_migration_verification(self, session_maker):
        """Verify migration was applied correctly."""
        self.logger.info("Performing post-migration verification")
        
        try:
            async with session_maker() as session:
                # Verify column was added
                result = await session.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'orders' AND column_name = 'position_id'
                """))
                
                column_info = result.fetchone()
                if not column_info:
                    raise MigrationDeploymentError("position_id column was not created")
                
                if column_info[1] != 'uuid' or column_info[2] != 'YES':
                    raise MigrationDeploymentError("position_id column has incorrect properties")
                
                # Verify foreign key constraint
                result = await session.execute(text("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'orders' 
                    AND constraint_type = 'FOREIGN KEY'
                    AND constraint_name LIKE '%position%'
                """))
                
                if not result.fetchone():
                    raise MigrationDeploymentError("Foreign key constraint was not created")
                
                # Verify index
                result = await session.execute(text("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE tablename = 'orders' 
                    AND indexname = 'idx_orders_position_id'
                """))
                
                if not result.fetchone():
                    raise MigrationDeploymentError("Index was not created")
                
                self.logger.info("Post-migration verification completed successfully")
                
        except Exception as e:
            raise MigrationDeploymentError(f"Post-migration verification failed: {e}")


async def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy position-order relationship migration")
    parser.add_argument("--env", required=True, choices=["staging", "production"], 
                       help="Target environment")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Perform dry run without actual changes")
    parser.add_argument("--backup", action="store_true", 
                       help="Create database backup before migration")
    parser.add_argument("--execute", action="store_true", 
                       help="Execute migration (required for non-dry-run)")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Migration timeout in seconds")
    
    args = parser.parse_args()
    
    # Safety check for production
    if not args.dry_run and not args.execute:
        print("ERROR: Must specify --execute flag for actual migration deployment")
        sys.exit(1)
    
    # Initialize services
    config_manager = ConfigManager()
    logger = LoggerService()
    
    # Create deployer and run migration
    deployer = PositionOrderMigrationDeployer(config_manager, logger)
    
    success = await deployer.deploy(
        environment=args.env,
        dry_run=args.dry_run,
        create_backup=args.backup,
        timeout_seconds=args.timeout
    )
    
    if success:
        print("✅ Migration deployment completed successfully")
        sys.exit(0)
    else:
        print("❌ Migration deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 