"""
Storage helpers for the Data Lake: per-layer directories and parquet files.

Config comes from env vars: LAKE_DATA_DIR (base) plus optional per-layer
overrides (RAW_DATA_DIR, REF_DATA_DIR, ...). Base defaults to /data.

Example:
    from etl.generic import storage
    from etl.generic.storage import Layer, Mode

    storage.write(Layer.RAW, df, "bank.parquet")        # /data/raw/2026-06-15/bank.parquet
    df = storage.read(Layer.RAW, "bank.parquet")

    storage.write(Layer.REF, df, "rates.parquet", mode=Mode.STATIC)  # /data/ref/static/rates.parquet

    files = storage.list_files(Layer.RAW)
    storage.archive_file(Layer.RAW, "bank.parquet")
    storage.cleanup(Layer.RAW, older_than_days=30)
"""

from __future__ import annotations
import os, shutil
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal

import polars as pl


class Layer(str, Enum):
    """Data Lake layer names."""
    RAW     = "raw"
    REF     = "ref"
    STG     = "stg"
    INT     = "int"
    FACT    = "fact"
    FAILED  = "failed"
    ARCHIVE = "archive"

    def __str__(self) -> str:
        return self.value


class Mode(str, Enum):
    """DATE - date partition (subject to cleanup). STATIC - no date (ignored by cleanup)."""
    DATE   = "date"
    STATIC = "static"

    def __str__(self) -> str:
        return self.value


LayerName = Layer | Literal["raw", "ref", "stg", "int", "fact", "failed", "archive"]
"""Accepts both Layer enum and string literals."""

_ENV_MAP = {
    "raw": "RAW_DATA_DIR",
    "ref": "REF_DATA_DIR",
    "stg": "STG_DATA_DIR",
    "int": "INT_DATA_DIR",
    "fact": "FACT_DATA_DIR",
    "failed": "FAILED_DATA_DIR",
    "archive": "ARCHIVE_DATA_DIR",
}


def layer_dir(layer: LayerName) -> Path:
    """Resolve a layer base dir from env (per-layer override or LAKE_DATA_DIR/<layer>)."""
    key = str(layer)
    if key not in _ENV_MAP:
        raise ValueError(f"Unknown layer: {layer}. Available: {list(_ENV_MAP)}")
    override = os.environ.get(_ENV_MAP[key])
    if override:
        return Path(override)
    return Path(os.environ.get("LAKE_DATA_DIR", "/data")) / key


def path(layer: LayerName, filename: str = "", date: str | None = None, mode: Mode = Mode.DATE) -> Path:
    """Path within a layer: STATIC to static subdir, DATE to dated subdir."""
    sub = "static" if mode == Mode.STATIC else (date or datetime.now().strftime("%Y-%m-%d"))
    base = layer_dir(layer) / sub
    return base / filename if filename else base


def read(
    layer: LayerName,
    filename: str,
    date: str | None = None,
    mode: Mode = Mode.DATE,
    as_arrow: bool = False,
) -> pl.DataFrame:
    """Read a parquet file from a layer. as_arrow=True returns a pyarrow.Table."""
    file_path = path(layer, filename, date, mode)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pl.read_parquet(file_path)
    return df.to_arrow() if as_arrow else df


def read_all(
    layer: LayerName,
    date: str | None = None,
    pattern: str = "*.parquet",
    mode: Mode = Mode.DATE,
) -> dict[str, pl.DataFrame]:
    """Read all parquet files in a layer into {filename: DataFrame}."""
    folder = path(layer, date=date, mode=mode)
    if not folder.exists():
        return {}
    return {p.stem: read(layer, p.name, date, mode) for p in sorted(folder.glob(pattern))}


def write(
    layer: LayerName,
    data: pl.DataFrame,
    filename: str,
    date: str | None = None,
    mode: Mode = Mode.DATE,
    overwrite: bool = False,
) -> Path:
    """Write a parquet file to a layer. Archives the previous file unless overwrite=True."""
    file_path = path(layer, filename, date, mode)
    if file_path.exists() and not overwrite:
        archive_file(layer, filename, date, mode)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(file_path)
    return file_path


def list_files(
    layer: LayerName,
    date: str | None = None,
    pattern: str = "*.parquet",
    mode: Mode = Mode.DATE,
) -> list[Path]:
    """List parquet files in a layer. date='*' lists across all date partitions."""
    if date == "*" and mode != Mode.STATIC:
        base = layer_dir(layer)
        if not base.exists():
            return []
        results: list[Path] = []
        for sub in sorted(base.iterdir()):
            if sub.is_dir() and sub.name != "static":
                results.extend(sub.glob(pattern))
        return sorted(results)

    folder = path(layer, date=date, mode=mode)
    return sorted(folder.glob(pattern)) if folder.exists() else []


def list_dates(layer: LayerName) -> list[str]:
    """All date partitions of a layer (excludes static)."""
    base = layer_dir(layer)
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir() and len(d.name) == 10)


def archive_file(
    layer: LayerName,
    filename: str,
    date: str | None = None,
    mode: Mode = Mode.DATE,
) -> Path | None:
    """Move a file into the archive layer. Returns the new path, or None if missing."""
    source = path(layer, filename, date, mode)
    if not source.exists():
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sub = "static" if mode == Mode.STATIC else datetime.now().strftime("%Y-%m-%d")
    dest = layer_dir(Layer.ARCHIVE) / sub / str(layer) / f"{source.stem}_{timestamp}{source.suffix}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))

    if source.parent.exists() and not any(source.parent.iterdir()):
        source.parent.rmdir()
    return dest


def archive_layer(layer: LayerName, date: str | None = None, mode: Mode = Mode.DATE) -> list[Path]:
    """Archive all files of a layer."""
    archived: list[Path] = []
    for f in list_files(layer, date=date, mode=mode):
        file_mode = Mode.STATIC if "static" in f.parts else Mode.DATE
        if result := archive_file(layer, f.name, date, file_mode):
            archived.append(result)
    return archived


def cleanup(layer: LayerName, older_than_days: int = 30, dry_run: bool = True) -> list[Path]:
    """Delete date partitions older than N days (never touches static)."""
    cutoff = (datetime.now() - timedelta(days=older_than_days)).strftime("%Y-%m-%d")
    deleted: list[Path] = []
    for date_str in list_dates(layer):
        if date_str < cutoff:
            folder = layer_dir(layer) / date_str
            if not dry_run:
                shutil.rmtree(folder)
            print(f"{'Would delete' if dry_run else 'Deleted'}: {folder}")
            deleted.append(folder)
    return deleted