"""
    Storage module for managing Data Lake directories and parquet files.

    Example:

        from etl.storage import Storage, Layer, Mode

        storage = Storage(data="/data")

        # Mode.DATE (default) date-partitioned 
        storage.write(Layer.RAW, df, "bank.parquet")
        # /data/raw/2026-03-05/bank.parquet

        # Mode.STATIC no date, cleanup ignores
        storage.write(Layer.REF, df, "accounts.parquet", mode=Mode.STATIC)
        # /data/ref/static/accounts.parquet

        # Mode.BOTH write to both static and date 
        storage.write(Layer.REF, df, "accounts.parquet", mode=Mode.BOTH)
        # /data/ref/static/accounts.parquet
        # /data/ref/2026-03-05/accounts.parquet

        # Reading
        df = storage.read(Layer.RAW, "bank.parquet")
        df = storage.read(Layer.REF, "accounts.parquet", mode=Mode.STATIC)

        # Listing
        files = storage.list(Layer.RAW)
        files = storage.list(Layer.REF, mode=Mode.STATIC)
        files = storage.list(Layer.REF, mode=Mode.BOTH)  # static + date

        # Archiving (one method for all modes)
        storage.archive_file(Layer.RAW, "bank.parquet")
        storage.archive_file(Layer.REF, "accounts.parquet", mode=Mode.STATIC)
        storage.archive_file(Layer.REF, "accounts.parquet", mode=Mode.BOTH)

        # Cleanup (static is never deleted)
        storage.cleanup(Layer.RAW, older_than_days=30)
"""

from __future__ import annotations
import os, shutil
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
    """
    Write/read mode for storage operations.

    DATE   - date-partitioned directory (default). Affected by cleanup().
    STATIC - 'static' subdirectory, no date. Ignored by cleanup().
    BOTH   - write/archive to both static and date directories.
             Read returns from static.
    """
    DATE   = "date"
    STATIC = "static"
    BOTH   = "both"

    def __str__(self) -> str:
        return self.value


LayerName = Layer | Literal["raw", "ref", "stg", "int", "fact", "failed", "archive"]
"""Accepts both Layer enum and string literals."""

_ENV_MAP = {
    "data": "LAKE_DATA_DIR",
    "raw": "RAW_DATA_DIR",
    "ref": "REF_DATA_DIR",
    "stg": "STG_DATA_DIR",
    "int": "INT_DATA_DIR",
    "fact": "FACT_DATA_DIR",
    "failed": "FAILED_DATA_DIR",
    "archive": "ARCHIVE_DATA_DIR",
}


class Storage:
    """
    Unified Data Lake storage manager.

    Organizes data into layers with date-partitioned and/or static directories
    and provides methods for reading, writing, listing, archiving, and cleaning up parquet files.
    """

    def __init__(
        self,
        data: str | Path | None = None,
        raw: str | Path = "raw",
        ref: str | Path = "ref",
        stg: str | Path = "stg",
        int: str | Path = "int",
        fact: str | Path = "fact",
        failed: str | Path = "failed",
        archive: str | Path = "archive",
    ):
        self._base = Path(
            data or os.environ.get("LAKE_DATA_DIR", "/data")
        )

        self._layers: dict[str, Path] = {}
        layers_args = {
            "raw": raw,
            "ref": ref,
            "stg": stg,
            "int": int,
            "fact": fact,
            "failed": failed,
            "archive": archive,
        }

        for layer_name, default_path in layers_args.items():
            env_val = os.environ.get(_ENV_MAP[layer_name])

            if env_val:
                self._layers[layer_name] = Path(env_val)
            elif Path(default_path).is_absolute():
                self._layers[layer_name] = Path(default_path)
            else:
                self._layers[layer_name] = self._base / default_path
    
    @property
    def raw_dir(self) -> Path:
        return self._layers["raw"]
    
    @property
    def ref_dir(self) -> Path:
        return self._layers["ref"]
    
    @property
    def stg_dir(self) -> Path:
        return self._layers["stg"]
    
    @property
    def int_dir(self) -> Path:
        return self._layers["int"]
    
    @property
    def fact_dir(self) -> Path:
        return self._layers["fact"]
    
    @property
    def failed_dir(self) -> Path:   
        return self._layers["failed"]
    
    @property
    def archive_dir(self) -> Path:
        return self._layers["archive"]
    
    def layer_dir(self, layer: LayerName) -> Path:
        key = str(layer)
        if key not in self._layers:
            raise ValueError(f"Unknown layer: {layer}. Available: {list(self._layers.keys())}")
        return self._layers[key]
    
    @staticmethod
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def path(
        self,
        layer: LayerName,
        filename: str = "",
        date: str | None = None,
        mode: Mode = Mode.DATE,
    ) -> Path:
        """
        Build a full path for a layer.

        Args:
            layer: Layer name.
            filename: Optional filename.
            date: Date partition string (YYYY-MM-DD). Default: today.
            mode: DATE -> date subdir, STATIC/BOTH -> static subdir.

        Returns:
            Full path.
        """
        if mode in (Mode.STATIC, Mode.BOTH):
            base = self.layer_dir(layer) / "static"
        else:
            base = self.layer_dir(layer) / (date or self._today())
        return base / filename if filename else base
    
    def read(
        self,
        layer: LayerName,
        filename: str,
        date: str | None = None,
        mode: Mode = Mode.DATE,
        as_arrow: bool = False,
    ) -> pd.DataFrame | pa.Table:
        """
        Read a parquet file from the specified layer.

        For raw layer, reads via pandas (no schema enforcement).
        For other layers, reads via PyArrow (preserves exact types and schema).

        Args:
            layer: Layer name.
            filename: Parquet file name to read.
            date: Date partition string (YYYY-MM-DD). Default: today.
            mode: DATE -> from date dir, STATIC/BOTH -> from static dir.
            as_arrow: If True, always return pyarrow.Table.

        Returns:
            DataFrame or pyarrow.Table.
        """
        file_path = self.path(layer, filename, date, mode=mode)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if str(layer) == "raw" and not as_arrow:
            return pd.read_parquet(file_path)

        table = pq.read_table(file_path)
        return table if as_arrow else table.to_pandas(types_mapper=pd.ArrowDtype)
    
    def read_all(
        self,
        layer: LayerName,
        date: str | None = None,
        pattern: str = "*.parquet",
        mode: Mode = Mode.DATE,
        as_arrow: bool = False,
    ) -> dict[str, pd.DataFrame | pa.Table]:
        """
        Read all parquet files matching a pattern from the specified layer.

        Args:
            layer: Layer name.
            date: Date partition string (YYYY-MM-DD). Default: today.
            pattern: Glob pattern to match files (default: "*.parquet").
            mode: DATE -> from date dir, STATIC/BOTH -> from static dir.
            as_arrow: If True, return pyarrow.Table instead of DataFrame.

        Returns:
            Dict of {filename: DataFrame or Table}.
        """
        folder = self.path(layer, date=date, mode=mode)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        results = {}
        for file_path in sorted(folder.glob(pattern)):
            results[file_path.stem] = self.read(
                layer, file_path.name, date, mode=mode, as_arrow=as_arrow,
            )
        return results
    
    def write(
        self,
        layer: LayerName,
        data: pd.DataFrame | pa.Table,
        filename: str,
        date: str | None = None,
        mode: Mode = Mode.DATE,
        overwrite: bool = False,
        schema: pa.Schema | None = None,
    ) -> Path | list[Path]:
        """
        Write data as parquet to a layer.

        For raw layer: writes via pandas (fast, no schema enforcement).
        For other layers: writes via PyArrow (preserves types, supports schema).

        Args:
            layer: Layer name.
            data: pandas DataFrame or pyarrow Table to save.
            filename: Output file name.
            date: Date partition. Default: today.
            mode: DATE -> date dir only, STATIC -> static dir only,
                  BOTH -> writes to both static and date dirs.
            overwrite: If True, overwrite existing. Otherwise archive old file first.
            schema: Optional pyarrow Schema to enforce on write (non-raw layers only).

        Returns:
            Path (for DATE/STATIC) or list[Path] (for BOTH).
        """
        if mode == Mode.BOTH:
            path_static = self._write_single(layer, data, filename, date, Mode.STATIC, overwrite, schema)
            path_date = self._write_single(layer, data, filename, date, Mode.DATE, overwrite, schema)
            return [path_static, path_date]

        return self._write_single(layer, data, filename, date, mode, overwrite, schema)

    def _write_single(
        self,
        layer: LayerName,
        data: pd.DataFrame | pa.Table,
        filename: str,
        date: str | None,
        mode: Mode,
        overwrite: bool,
        schema: pa.Schema | None,
    ) -> Path:
        """Write to a single target path."""
        if mode == Mode.STATIC:
            file_path = self.layer_dir(layer) / "static" / filename
        else:
            file_path = self.layer_dir(layer) / (date or self._today()) / filename

        if file_path.exists() and not overwrite:
            self._archive_single(layer, filename, date, mode)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if str(layer) == "raw":
            if isinstance(data, pa.Table):
                data = data.to_pandas()
            data.to_parquet(file_path, index=False)
        else:
            if isinstance(data, pd.DataFrame):
                table = pa.Table.from_pandas(data, preserve_index=False)
            else:
                table = data

            if schema:
                table = table.cast(schema)

            pq.write_table(table, file_path)

        return file_path
    
    def list(
        self,
        layer: LayerName,
        date: str | None = None,
        pattern: str = "*.parquet",
        mode: Mode = Mode.DATE,
    ) -> list[Path]:
        """
        List files in a layer.

        Args:
            layer: Layer name.
            date: Date partition. Default: today. Use "*" for all dates.
            pattern: Glob pattern for filenames.
            mode: DATE -> date dirs only, STATIC -> static dir only,
                  BOTH -> all files (date + static combined).

        Returns:
            Sorted list of file paths.
        """
        if mode == Mode.BOTH:
            static_files = self._list_single(layer, None, pattern, Mode.STATIC)
            date_files = self._list_single(layer, date, pattern, Mode.DATE)
            return sorted(set(static_files + date_files))

        return self._list_single(layer, date, pattern, mode)

    def _list_single(
        self,
        layer: LayerName,
        date: str | None,
        pattern: str,
        mode: Mode,
    ) -> list[Path]:
        """List files from a single mode."""
        if mode == Mode.STATIC:
            folder = self.layer_dir(layer) / "static"
            if not folder.exists():
                return []
            return sorted(folder.glob(pattern))

        if date == "*":
            folder = self.layer_dir(layer)
            # Exclude static/ from wildcard listing
            return sorted(
                p for p in folder.rglob(pattern)
                if "static" not in p.parts
            )

        folder = self.layer_dir(layer) / (date or self._today())
        if not folder.exists():
            return []
        return sorted(folder.glob(pattern))
    
    def list_dates(self, layer: LayerName) -> list[str]:
        """
        List all date partitions in a layer (excludes static).

        Returns:
            Sorted list of date strings (e.g. ["2026-03-01", "2026-03-02"]).
        """
        layer_path = self.layer_dir(layer)
        if not layer_path.exists():
            return []
        return sorted(
            d.name for d in layer_path.iterdir()
            if d.is_dir() and len(d.name) == 10  # YYYY-MM-DD
        )

    # -------------------------------------------------------------------------
    # Archive
    # -------------------------------------------------------------------------

    def archive_file(
        self,
        layer: LayerName,
        filename: str,
        date: str | None = None,
        mode: Mode = Mode.DATE,
    ) -> Path | list[Path] | None:
        """
        Move a file to the archive layer.

        Args:
            layer: Original layer of the file.
            filename: Name of the file to archive.
            date: Date partition. Default: today.
            mode: DATE -> archive date file, STATIC -> archive static file,
                  BOTH -> archive from both locations.

        Returns:
            Path(s) in archive, or None if source file(s) not found.
        """
        if mode == Mode.BOTH:
            results = []
            r1 = self._archive_single(layer, filename, date, Mode.STATIC)
            r2 = self._archive_single(layer, filename, date, Mode.DATE)
            if r1:
                results.append(r1)
            if r2:
                results.append(r2)
            return results if results else None

        return self._archive_single(layer, filename, date, mode)

    def _archive_single(
        self,
        layer: LayerName,
        filename: str,
        date: str | None,
        mode: Mode,
    ) -> Path | None:
        """Archive a single file."""
        if mode == Mode.STATIC:
            source = self.layer_dir(layer) / "static" / filename
        else:
            source = self.layer_dir(layer) / (date or self._today()) / filename

        if not source.exists():
            return None

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stem = source.stem
        suffix = source.suffix

        if mode == Mode.STATIC:
            archive_path = self.archive_dir / "static" / f"{stem}_{timestamp}{suffix}"
        else:
            date_str = date or self._today()
            archive_path = self.archive_dir / date_str / f"{stem}_{timestamp}{suffix}"

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(archive_path))
        return archive_path

    def archive_layer(
        self,
        layer: LayerName,
        date: str | None = None,
        mode: Mode = Mode.DATE,
    ) -> list[Path]:
        """
        Archive all files from a layer.

        Args:
            layer: Source layer.
            date: Date partition. Default: today.
            mode: DATE -> archive date files, STATIC -> archive static files,
                  BOTH -> archive all.

        Returns:
            List of paths in archive.
        """
        files = self.list(layer, date=date, mode=mode)
        archived = []
        for f in files:
            file_mode = Mode.STATIC if "static" in f.parts else Mode.DATE
            result = self._archive_single(layer, f.name, date, file_mode)
            if result:
                archived.append(result)
        return archived
    
    def cleanup(
        self,
        layer: LayerName,
        older_than_days: int = 30,
        dry_run: bool = True,
    ) -> list[Path]:
        """
        Delete date partitions older than N days.
        Static directories are never affected by cleanup.

        Args:
            layer: Layer to clean.
            older_than_days: Delete partitions older than this.
            dry_run: If True, only return what would be deleted.

        Returns:
            List of deleted (or would-be-deleted) directories.
        """
        cutoff = datetime.now() - timedelta(days=older_than_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")

        deleted = []
        for date_str in self.list_dates(layer):
            if date_str < cutoff_str:
                folder = self.layer_dir(layer) / date_str
                if dry_run:
                    print(f"Would delete: {folder}")
                else:
                    shutil.rmtree(folder)
                    print(f"Deleted: {folder}")
                deleted.append(folder)
        return deleted
    
    def info(self) -> str:
        """Print storage configuration and stats."""
        lines = [f"Storage(base={self._base})", ""]
        for layer_name, layer_path in self._layers.items():
            dates = self.list_dates(layer_name) if layer_path.exists() else []
            date_files = len(self.list(layer_name, date="*")) if dates else 0
            static_path = layer_path / "static"
            static_files = len(list(static_path.glob("*.parquet"))) if static_path.exists() else 0
            lines.append(
                f"  {layer_name:12s} -> {layer_path}  "
                f"({len(dates)} dates, {date_files} date files, {static_files} static files)"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        layers = ", ".join(f"{k}={v}" for k, v in self._layers.items())
        return f"Storage({layers})"