"""Database migration management system using Alembic."""

import os
from collections.abc import Sequence  # Added Sequence, Tuple

from alembic import command  # type: ignore[import-not-found]
from alembic.config import Config  # type: ignore[import-not-found]
from alembic.script import ScriptDirectory  # type: ignore[import-not-found]

from gal_friday.logger_service import LoggerService


class MigrationManager:
    """Manages database schema migrations using Alembic's Python API."""

    def __init__(self, logger: LoggerService, project_root_path: str | None = None) -> None:
        """Initialize the MigrationManager.

        Args:
            logger: Logger service for logging messages.
            project_root_path: Absolute path to the project root (e.g., /app).
                               If None, assumes CWD is project root.
        """
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Determine project root. If running inside a container where CWD is /app, this is fine.
        # Otherwise, this needs to be passed or discovered more robustly.
        _root = project_root_path if project_root_path else os.getcwd()
        self.alembic_cfg_path = os.path.join(_root, "gal_friday", "dal", "alembic.ini")

        # Ensure alembic.ini exists
        if not os.path.exists(self.alembic_cfg_path):
            self.logger.error(
                f"Alembic config file not found at: {self.alembic_cfg_path}",
                source_module=self._source_module,
            )
            # This is a critical error, consider raising an exception or handling appropriately
            # For now, subsequent calls will likely fail if alembic_cfg cannot be loaded.

    def _get_alembic_config(self) -> Config:
        """Loads Alembic configuration from the .ini file.
        
        Ensures that the `prepend_sys_path` from `alembic.ini` (set to '..')
        correctly adds the project root to sys.path if Alembic commands are
        run from `gal_friday/dal`. When calling Alembic API programmatically,
        we might need to handle path adjustments if CWD isn't `gal_friday/dal`.
        However, `Config(self.alembic_cfg_path)` should handle this based on `prepend_sys_path`.
        The `script_location` in `alembic.ini` is relative to `alembic.ini`'s location.
        """
        alembic_cfg = Config(self.alembic_cfg_path)

        # The script_location in alembic.ini is relative to the ini file's directory.
        # Alembic's Config object should resolve this correctly.
        # For example, if alembic.ini is in gal_friday/dal and script_location is alembic_env,
        # it resolves to gal_friday/dal/alembic_env.
        return alembic_cfg

    def upgrade_to_head(self) -> None:
        """Upgrade the database to the latest revision ('head')."""
        self.logger.info("Attempting to upgrade database to head...", source_module=self._source_module)
        try:
            alembic_cfg = self._get_alembic_config()
            command.upgrade(alembic_cfg, "head")
            self.logger.info("Database upgrade to head completed successfully.", source_module=self._source_module)
        except Exception as e:
            self.logger.exception(
                f"Error during database upgrade to head: {e}",
                source_module=self._source_module,
            )
            raise # Re-raise after logging

    def downgrade_to_version(self, version: str) -> None:
        """Downgrade the database to a specific version."""
        self.logger.info(f"Attempting to downgrade database to version: {version}", source_module=self._source_module)
        try:
            alembic_cfg = self._get_alembic_config()
            command.downgrade(alembic_cfg, version)
            self.logger.info(f"Database downgrade to version {version} completed successfully.", source_module=self._source_module)
        except Exception as e:
            self.logger.exception(
                f"Error during database downgrade to version {version}: {e}",
                source_module=self._source_module,
            )
            raise

    def get_current_revision(self) -> Sequence[str | None]: # Alembic can have multiple heads
        """Get the current revision(s) of the database."""
        self.logger.debug("Fetching current database revision(s)...", source_module=self._source_module)
        try:
            alembic_cfg = self._get_alembic_config()
            # script_location should be correctly interpreted by Config relative to alembic.ini
            script = ScriptDirectory.from_config(alembic_cfg)

            # To get current heads, we need an EnvironmentContext.
            # This part might require a database connection to check the alembic_version table.
            # If this method is intended to run without a live DB connection, it might not work as expected
            # or might only return script heads, not DB heads.
            # The subtask is about refactoring MigrationManager, assuming it might be used in an env
            # where DB is eventually available.

            # The most reliable way to get DB heads needs a connection.
            # For now, let's try to get script heads which doesn't require DB connection.
            # heads = script.get_heads()
            # return heads
            # However, command.current() is what's usually used. It prints.
            # To capture stdout:

            # Alternative: Using EnvironmentContext to get DB heads (requires DB access)
            # This is more accurate if we want the DB's actual state.
            # If the goal is "current revision in the script directory", script.get_revision("head") or similar.
            # For now, sticking to a method that can get DB state if possible.

            # Capture stdout from command.current
            # This is a common workaround if the API doesn't return directly.
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()

            command.current(alembic_cfg, verbose=False) # verbose=True adds more details

            sys.stdout = old_stdout # Restore stdout
            output = captured_output.getvalue().strip()
            # Output is like "abc123def456 (head)" or "(no migration detected)"
            # Or multiple lines if multiple heads.

            if not output or "no migration detected" in output:
                self.logger.info("No current revision detected.", source_module=self._source_module)
                return tuple() # Return empty tuple for no revision

            # Parse output, e.g. "abc123def456 (head)" -> "abc123def456"
            # If multiple heads, they are on separate lines.
            revisions = []
            for line in output.splitlines():
                match = line.split(" ")[0]
                if match:
                    revisions.append(match)

            self.logger.info(f"Current database revision(s): {revisions}", source_module=self._source_module)
            return tuple(revisions)

        except Exception as e:
            self.logger.exception(
                f"Error fetching current database revision: {e}",
                source_module=self._source_module,
            )
            # Depending on policy, either raise or return an empty tuple/None
            return tuple() # Or raise e

    def stamp_revision(self, revision: str) -> None:
        """Stamp the database with a specific revision without running migrations."""
        self.logger.info(f"Stamping database with revision: {revision}", source_module=self._source_module)
        try:
            alembic_cfg = self._get_alembic_config()
            command.stamp(alembic_cfg, revision)
            self.logger.info(f"Database stamped with revision {revision} successfully.", source_module=self._source_module)
        except Exception as e:
            self.logger.exception(
                f"Error stamping database with revision {revision}: {e}",
                source_module=self._source_module,
            )
            raise

    def generate_revision(self, message: str, autogenerate: bool = True, revision_id: str | None = None) -> None:
        """Generate a new revision file."""
        self.logger.info(f"Generating new revision: '{message}' (autogenerate={autogenerate})", source_module=self._source_module)
        try:
            alembic_cfg = self._get_alembic_config()
            # Note: autogenerate=True requires a database connection to compare metadata.
            # If this is run in an environment without DB access, it might fail or produce empty diffs
            # unless env.py is specifically set up for "offline autogeneration" (which is complex).
            # The current env.py setup (from previous subtasks) attempts this.
            command.revision(alembic_cfg, message=message, autogenerate=autogenerate, rev_id=revision_id)
            self.logger.info(f"New revision generated successfully: {message}", source_module=self._source_module)
        except Exception as e:
            self.logger.exception(
                f"Error generating new revision '{message}': {e}",
                source_module=self._source_module,
            )
            raise
