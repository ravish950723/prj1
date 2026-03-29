from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import copy
import math
import yaml


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Schema root must be a dict: {p}')
    return data


def _normalize_columns(raw_columns: list[Any]) -> list[dict[str, Any]]:
    cols: list[dict[str, Any]] = []
    for item in raw_columns or []:
        if isinstance(item, str):
            cols.append({
                'name': item,
                'key': item,
                'type': 'string',
                'layer': 'core',
                'required': False,
                'default': '',
            })
        elif isinstance(item, dict) and item.get('name'):
            col = {
                'name': str(item['name']),
                'key': str(item.get('key') or item['name']),
                'type': str(item.get('type') or 'string'),
                'layer': str(item.get('layer') or 'core'),
                'required': bool(item.get('required', False)),
                'default': copy.deepcopy(item.get('default', '')),
            }
            cols.append(col)
    return cols


def load_schema_config(path: str | Path) -> dict[str, Any]:
    data = _read_yaml(path)
    raw_columns = data.get('columns', [])
    columns = _normalize_columns(raw_columns)
    rename_map = data.get('rename_map', {})
    if not isinstance(rename_map, dict):
        rename_map = {}
    required_columns = data.get('required_columns', [])
    if not isinstance(required_columns, list):
        required_columns = []
    groups = data.get('groups', {})
    if not isinstance(groups, dict):
        groups = {}

    # ensure required flags honor top-level required_columns for backwards compatibility
    required_set = {str(x) for x in required_columns}
    for col in columns:
        if col['name'] in required_set:
            col['required'] = True

    return {
        'version': data.get('version', 1),
        'description': data.get('description', ''),
        'columns_meta': columns,
        'columns': [c['name'] for c in columns],
        'rename_map': rename_map,
        'required_columns': required_columns,
        'groups': groups,
    }


def load_schema_columns(path: str | Path) -> list[str]:
    return load_schema_config(path)['columns']


def load_schema_meta(path: str | Path) -> list[dict[str, Any]]:
    return load_schema_config(path)['columns_meta']


def build_default_row(columns_or_meta: list[Any]) -> dict[str, Any]:
    if not columns_or_meta:
        return {}
    if isinstance(columns_or_meta[0], str):
        return {str(c): None for c in columns_or_meta}
    row: dict[str, Any] = {}
    for col in columns_or_meta:
        row[str(col['name'])] = copy.deepcopy(col.get('default'))
    return row


def _coerce_value(value: Any, declared_type: str) -> Any:
    t = (declared_type or 'string').lower()
    if value is None:
        return None
    if t == 'float':
        try:
            v = float(value)
            return None if math.isnan(v) else v
        except Exception:
            return value
    if t == 'int':
        try:
            return int(float(value))
        except Exception:
            return value
    if t == 'bool':
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {'y', 'yes', 'true', '1'}:
            return True
        if s in {'n', 'no', 'false', '0', ''}:
            return False
        return bool(value)
    return value


def align_row_to_schema(row: dict, columns_meta: list[dict]) -> dict:
    out = {}

    for col_meta in columns_meta:
        col = col_meta["name"]
        default = col_meta.get("default", None)

        # 🔥 KEY FIX: preserve existing values
        if col in row and row[col] not in (None, "", "N/A"):
            out[col] = row[col]
        else:
            out[col] = default

    return out