"""Database migration management system using Alembic."""

import os
from collections.abc import Sequence

from alembic import command  # type: ignore
from alembic.config import Config  # type: ignore
from alembic.script import ScriptDirectory  # type: ignore
from sqlalchemy.exc import SQLAlchemyError

from gal_friday.exceptions import DatabaseConnectionError
from gal_friday.logger_service import LoggerService


class MigrationManager:
    """Manages database schema migrations using Alembic's Python API."""

    def __init__(self, logger: LoggerService, project_root_path: str | None = None) -> None:
        """Initialize the MigrationManager.

        Args:
            logger: Logger service for logging messages.
            project_root_path: Absolute path to the project root (e.g., ``/app``).
                If ``None``, uses the current working directory.
        """
        self.logger = logger
        self._source_module = self.__class__.__name__

        root = project_root_path or os.getcwd()
        self.alembic_dir = os.path.join(root, "gal_friday", "dal")
        self.alembic_cfg_path = os.path.join(self.alembic_dir, "alembic.ini")
        self.script_location = os.path.join(self.alembic_dir, "alembic_env")

        if not os.path.isfile(self.alembic_cfg_path):
            raise FileNotFoundError(
                f"Alembic config file not found at: {self.alembic_cfg_path}"
            )

    def _get_alembic_config(self) -> Config:
        """Load Alembic configuration with absolute paths."""

        alembic_cfg = Config(self.alembic_cfg_path)
        alembic_cfg.set_main_option("script_location", self.script_location)
        return alembic_cfg

    def get_script_revisions(self) -> Sequence[str]:
        """Return revisions available on disk without querying the database."""

        script = ScriptDirectory.from_config(self._get_alembic_config())
        return tuple(rev.revision for rev in script.walk_revisions())

    def get_database_heads(self) -> Sequence[str]:
        """Return revision identifiers currently applied to the database."""

        self.logger.debug(
            "Fetching database head revision(s)...", source_module=self._source_module
        )
        try:
            alembic_cfg = self._get_alembic_config()

            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            command.current(alembic_cfg, verbose=False)
            sys.stdout = old_stdout
            output = captured_output.getvalue().strip()

            if not output or "no version" in output or "no migration" in output:
                return tuple()

            return tuple(line.split(" ")[0] for line in output.splitlines() if line)
        except SQLAlchemyError as exc:
            error_msg = str(exc).lower()
            if "alembic_version" in error_msg and "does not exist" in error_msg:
                return tuple()
            raise DatabaseConnectionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            self.logger.exception(
                "Unexpected error reading database heads: %s", exc, source_module=self._source_module
            )
            raise

    def upgrade_to_head(self) -> None:
        """Upgrade the database to the latest revision ('head')."""
        self.logger.info("Attempting to upgrade database to head...", source_module=self._source_module)
        try:
            alembic_cfg = self._get_alembic_config()
            command.upgrade(alembic_cfg, "head")
            self.logger.info(
                "Database upgrade to head completed successfully.",
                source_module=self._source_module)
        except SQLAlchemyError as exc:
            self.logger.exception(
                "Database error during upgrade to head: %s", exc, source_module=self._source_module
            )
            raise DatabaseConnectionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            self.logger.exception(
                "Error during database upgrade to head: %s", exc, source_module=self._source_module
            )
            raise

    def downgrade_to_version(self, version: str) -> None:
        """Downgrade the database to a specific version."""
        self.logger.info(f"Attempting to downgrade database to version: {version}", source_module=self._source_module)
        try:
            alembic_cfg = self._get_alembic_config()
            command.downgrade(alembic_cfg, version)
            self.logger.info(
                f"Database downgrade to version {version} completed successfully.",
                source_module=self._source_module)
        except SQLAlchemyError as exc:
            self.logger.exception(
                "Database error during downgrade to %s: %s", version, exc, source_module=self._source_module
            )
            raise DatabaseConnectionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            self.logger.exception(
                "Error during database downgrade to version %s: %s", version, exc, source_module=self._source_module
            )
            raise

    def get_current_revision(self) -> str | None:
        """Return the current database revision or ``None`` if unavailable."""

        heads = self.get_database_heads()
        if not heads:
            return None
        if len(heads) > 1:
            self.logger.warning(
                "Multiple database heads detected: %s", heads, source_module=self._source_module
            )
        return heads[0]

    def stamp_revision(self, revision: str) -> None:
        """Stamp the database with a specific revision without running migrations."""
        self.logger.info(f"Stamping database with revision: {revision}", source_module=self._source_module)
        try:
            alembic_cfg = self._get_alembic_config()
            command.stamp(alembic_cfg, revision)
            self.logger.info(
                f"Database stamped with revision {revision} successfully.",
                source_module=self._source_module)
        except SQLAlchemyError as exc:
            self.logger.exception(
                "Database error stamping revision %s: %s", revision, exc, source_module=self._source_module
            )
            raise DatabaseConnectionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            self.logger.exception(
                "Error stamping database with revision %s: %s", revision, exc, source_module=self._source_module
            )
            raise

    def generate_revision(self, message: str, autogenerate: bool = True, revision_id: str | None = None) -> None:
        """Generate a new revision file."""
        self.logger.info(f"Generating new revision: '{message}' (autogenerate={autogenerate})", source_module=self._source_module)
        try:
            alembic_cfg = self._get_alembic_config()
            command.revision(
                alembic_cfg,
                message=message,
                autogenerate=autogenerate,
                rev_id=revision_id)
            self.logger.info(
                f"New revision generated successfully: {message}",
                source_module=self._source_module)
        except SQLAlchemyError as exc:
            self.logger.exception(
                "Database error generating revision '%s': %s", message, exc, source_module=self._source_module
            )
            raise DatabaseConnectionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            self.logger.exception(
                "Error generating new revision '%s': %s", message, exc, source_module=self._source_module
            )
            raise