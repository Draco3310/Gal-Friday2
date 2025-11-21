import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    """
    Loads configuration from a YAML file.
    """

    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        """
        Reads the YAML config file and returns a dictionary.
        Raises FileNotFoundError or yaml.YAMLError if issues occur.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {path.absolute()}")

        with open(path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                return config or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML config: {e}")

    @staticmethod
    def validate_keys(config: Dict[str, Any], required_keys: list) -> bool:
        """
        Checks if required keys are present in the config.
        """
        missing = [key for key in required_keys if key not in config]
        if missing:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")
        return True
