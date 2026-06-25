from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | None, base: Path = DEPLOY_ROOT) -> Path | None:
    if path in (None, ""):
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(str(path))
    if config_path is None:
        raise ValueError("config path is required")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(config_path)
    cfg["_deploy_root"] = str(DEPLOY_ROOT)
    return cfg


def model_path(model_cfg: dict[str, Any], key: str = "path") -> Path | None:
    return resolve_path(model_cfg.get(key))


def ensure_output_dir(cfg: dict[str, Any]) -> Path:
    out_dir = resolve_path(cfg["output"]["directory"])
    if out_dir is None:
        raise ValueError("output.directory is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

