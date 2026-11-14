"""
VoxGuard - Configuration Manager
================================
Fulfills TASK 2 Section 8: Configuration Management.
Loads and validates experiment settings from configs/config.yaml.
Supports transparent switching between dummy and real dataset paths via ONE value:
`data_source: 'dummy'` vs `data_source: 'real'`.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Lightweight, robust YAML reader that parses key-value pairs, nested sections,
    and numbers/booleans without requiring external third-party dependencies.
    """
    result = {}
    current_section = None

    for raw_line in text.splitlines():
        # Strip comments
        line = re.sub(r"#.*$", "", raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())

        if indent == 0 and line.strip().endswith(":"):
            # New root section
            current_section = line.strip()[:-1].strip()
            result[current_section] = {}
        elif indent > 0 and ":" in line:
            parts = line.strip().split(":", 1)
            key = parts[0].strip()
            val_str = parts[1].strip()

            # Parse value type
            val = parse_yaml_value(val_str)

            if current_section:
                result[current_section][key] = val
            else:
                result[key] = val
        elif indent == 0 and ":" in line:
            parts = line.strip().split(":", 1)
            key = parts[0].strip()
            val_str = parts[1].strip()
            val = parse_yaml_value(val_str)
            result[key] = val
            current_section = None

    return result


def parse_yaml_value(val_str: str) -> Any:
    """Helper to convert scalar strings into booleans, numbers, or strings."""
    if not val_str:
        return None
    # Strip quotes
    if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
        return val_str[1:-1]
    if val_str.lower() in ("true", "yes", "on"):
        return True
    if val_str.lower() in ("false", "no", "off"):
        return False
    if val_str.lower() in ("null", "none"):
        return None
    try:
        if "." in val_str or "e" in val_str.lower():
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val_str


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Tries PyYAML first if installed, otherwise uses built-in parse_simple_yaml.
    """
    if config_path is None:
        path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found at: {path.resolve()}")

    content = path.read_text(encoding="utf-8")

    try:
        import yaml
        config = yaml.safe_load(content)
    except (ImportError, Exception):
        config = parse_simple_yaml(content)

    # Post-process data_source toggle
    data_source = config.get("data_source", "real").lower()
    if data_source == "dummy":
        config["metadata_path"] = "data/dummy/metadata.csv"
    else:
        config["metadata_path"] = "data/processed/metadata.csv"

    return config

