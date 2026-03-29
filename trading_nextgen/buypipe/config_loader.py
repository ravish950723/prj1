from pathlib import Path
from functools import lru_cache
from typing import Any
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs'


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Missing config file: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'YAML root must be a dict: {path}')
    return data


@lru_cache(maxsize=8)
def load_config(name: str) -> dict[str, Any]:
    return _read_yaml(CONFIG_DIR / name)


def get_patterns_config() -> dict[str, Any]:
    return load_config('patterns.yml')


def get_quant_config() -> dict[str, Any]:
    return load_config('quant.yml')


def get_columns_config() -> dict[str, Any]:
    return load_config('col.yml')


def get_columns_meta() -> list[dict[str, Any]]:
    data = get_columns_config()
    cols = data.get('columns', [])
    out: list[dict[str, Any]] = []
    for item in cols:
        if isinstance(item, dict) and item.get('name'):
            out.append(item)
        elif isinstance(item, str):
            out.append({'name': item, 'key': item, 'type': 'string', 'layer': 'core', 'required': False, 'default': ''})
    return out


def get_substages_config() -> dict[str, Any]:
    return load_config('substages.yml')
