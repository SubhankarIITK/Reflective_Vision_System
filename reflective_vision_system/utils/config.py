import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_nested(config: dict, *keys, default=None):
    for key in keys:
        if not isinstance(config, dict):
            return default
        config = config.get(key, default)
    return config
