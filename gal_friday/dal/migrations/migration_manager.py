"""Database migration management system using Alembic."""

import io  # For capturing stdout
import re  # For parsing revision output
from collections.abc import Sequence
from contextlib import redirect_stdout  # For capturing stdout
from pathlib import Path  # For PTH compliance

from alembic import command  # type: ignore[import-not-found]
from alembic.config import Config  # type: ignore[import-not-found]

# ScriptDirectory was unused after F841 fix, so removing if not needed.
# from alembic.script import ScriptDirectory  # type: ignore[import-not-found]
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

        _root_path = Path(project_root_path) if project_root_path else Path.cwd()
        self.alembic_cfg_path = str(_root_path / "gal_friday" / "dal" / "alembic.ini")

        if not Path(self.alembic_cfg_path).exists():
            self.logger.error(
                f"Alembic config file not found at: {self.alembic_cfg_path}",
                source_module=self._source_module,
            )

    def _get_alembic_config(self) -> Config:
        """Loads Alembic configuration from the .ini file."""
        return Config(self.alembic_cfg_path)

    def upgrade_to_head(self) -> None:
        """Upgrade the database to the latest revision ('head')."""
        self.logger.info(
            "Attempting to upgrade database to head...",
            source_module=self._source_module,
        )
        try:
            alembic_cfg = self._get_alembic_config()
            command.upgrade(alembic_cfg, "head")
            self.logger.info(
                "Database upgrade to head completed successfully.",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.exception(
                f"Error during database upgrade to head: {e}",
                source_module=self._source_module,
            )
            raise

    def downgrade_to_version(self, version: str) -> None:
        """Downgrade the database to a specific version."""
        self.logger.info(
            f"Attempting to downgrade database to version: {version}",
            source_module=self._source_module,
        )
        try:
            alembic_cfg = self._get_alembic_config()
            command.downgrade(alembic_cfg, version)
            self.logger.info(
                f"Database downgrade to version {version} completed successfully.",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.exception(
                f"Error during database downgrade to version {version}: {e}",
                source_module=self._source_module,
            )
            raise

    def get_current_revision(self) -> Sequence[str | None]: # Alembic can have multiple heads
        """Get the current revision(s) of the database."""
        self.logger.debug(
            "Fetching current database revision(s)...",
            source_module=self._source_module,
        )
        try:
            alembic_cfg = self._get_alembic_config()

            s = io.StringIO()
            with redirect_stdout(s):
                command.current(alembic_cfg, verbose=False)
            output = s.getvalue().strip()

            if not output or "no migration detected" in output:
                self.logger.info(
                    "No current revision detected.", source_module=self._source_module,
                )
                return tuple()

            matches = re.findall(r"([0-9a-fA-F]+)", output)
            revisions = [match for match in matches if match] # Ensure not empty strings

            self.logger.info(
                f"Current database revision(s): {revisions}",
                source_module=self._source_module,
            )
            return tuple(revisions)

        except Exception as e:
            self.logger.exception(
                f"Error fetching current database revision: {e}",
                source_module=self._source_module,
            )
            return tuple()

    def stamp_revision(self, revision: str) -> None:
        """Stamp the database with a specific revision without running migrations."""
        self.logger.info(
            f"Stamping database with revision: {revision}",
            source_module=self._source_module,
        )
        try:
            alembic_cfg = self._get_alembic_config()
            command.stamp(alembic_cfg, revision)
            self.logger.info(
                f"Database stamped with revision {revision} successfully.",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.exception(
                f"Error stamping database with revision {revision}: {e}",
                source_module=self._source_module,
            )
            raise

    def generate_revision(
        self, message: str, autogenerate: bool = True, revision_id: str | None = None,
    ) -> None:
        """Generate a new revision file."""
        self.logger.info(
            f"Generating new revision: '{message}' (autogenerate={autogenerate})",
            source_module=self._source_module,
        )
        try:
            alembic_cfg = self._get_alembic_config()
            command.revision(
                alembic_cfg,
                message=message,
                autogenerate=autogenerate,
                rev_id=revision_id,
            )
            self.logger.info(
                f"New revision generated successfully: {message}",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.exception(
                f"Error generating new revision '{message}': {e}",
                source_module=self._source_module,
            )
            raise
