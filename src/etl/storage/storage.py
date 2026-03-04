"""
    Storage module for managing Data Lake directories and parquet files.

    Example:

        from etl.storage import Storage

        # Variant 1: Using environment variables 
        storage = Storage()

        # Variant 2: Custom base directory with default subdirs
        storage = Storage(
            data="/data",
            raw="raw",
            ref="ref",
            stg="stg",
            int="int",
            fact="fact",
            failed="failed",
            archive="archive",
        )

        # Variant 3: Fully custom paths
        storage = Storage(
            raw="/mnt/lake/raw",
            ref="/mnt/lake/ref",
            stg="/mnt/lake/stg",
            int="/mnt/lake/int",
            fact="/mnt/lake/fact",
            failed="/mnt/lake/failed",
            archive="/mnt/lake/archive",
        )

        # Reading
        df = storage.read("raw", "bank_BerekeBank.parquet")
        df = storage.read("raw", "bank_BerekeBank.parquet", date="2026-03-01")

        # Writing
        storage.write("stg", df, "bank_clean.parquet")

        # Listing files
        files = storage.list("raw")
        files = storage.list("raw", date="2026-03-01", pattern="bank_*")

        # Moving raw → archive after processing
        storage.archive("raw", "bank_BerekeBank.parquet")

        # Cleanup
        storage.cleanup("raw", older_than_days=30)
"""

from __future__ import annotations
from ast import pattern
import os, shutil, glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd

LayerName = Literal["raw", "ref", "stg", "int", "fact", "failed", "archive"]

_DEFAULT_LAYERS = {
    "raw": "raw",
    "ref": "ref",
    "stg": "stg",
    "int": "int",
    "fact": "fact",
    "failed": "failed",
    "archive": "archive",
}

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

        Organizes data into layers with date-partitioned directories 
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
        if layer not in self._layers:
            raise ValueError(f"Unknown layer: {layer}")
        return self._layers[layer]
    
    @staticmethod
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")
    
    def path(self, layer: LayerName, filename: str = "", date: str | None = None) -> Path:
        """
        Build a full path for a layer, date partition and optional filename.

        Args:
            layer: Layer name (raw, stage, processed, archive).
            filename: Optional filename.
            date: Date partition string (YYYY-MM-DD). Default: today.

        Returns:
            Full path.
        """
        base = self.layer_dir(layer) / (date or self._today())
        return base / filename if filename else base
    
    def read(self, layer: LayerName, filename: str, date: str | None = None) -> pd.DataFrame:
        """
        Read a parquet file from the specified layer and date partition.

        Args:
            layer: Layer name (raw, stage, processed, archive).
            filename: Parquet file name to read.
            date: Date partition string (YYYY-MM-DD). Default: today.

        Returns:
            DataFrame read from the parquet file.
        """
        file_path = self.path(layer, filename, date)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_parquet(file_path)
    
    def read_all(self, layer: LayerName, date: str | None = None, pattern: str = "*.parquet") -> dict[str, pd.DataFrame]:
        """
        Read all parquet files from the specified layer and date partition into a single DataFrame.

        Args:
            layer: Layer name (raw, stage, processed, archive).
            date: Date partition string (YYYY-MM-DD). Default: today.
            pattern: Glob pattern to match files (default: "*.parquet").
        
        Returns:
            Dict of {filename: DataFrame}.
        """

        folder = self.path(layer, date=date)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        
        results = {}
        for file_path in sorted(folder.glob(pattern)):
            results[file_path.stem] = pd.read_parquet(file_path)
        return results
    
    def write(self, layer: LayerName, df: pd.DataFrame, filename: str, date: str | None = None, overwrite: bool = False) -> Path:
        """
        Write a DataFrame as parquet to a layer.

        Args:
            layer: Layer name.
            df: DataFrame to save.
            filename: Output file name.
            date: Date partition. Default: today.
            overwrite: If True, overwrite existing file. If False and file exists,
                       the old file is archived first.

        Returns:
            Path to the saved file.
        """

        file_path = self.path(layer, filename, date)

        if file_path.exists() and not overwrite:
            self.archive_file(layer, filename, date)

        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, index=False)
        return file_path
    
    def list(self, layer: LayerName, date: str | None = None, pattern: str = "*.parquet") -> list[Path]:
        """
        List files in a layer.

        Args:
            layer: Layer name.
            date: Date partition. Default: today. Use "*" for all dates.
            pattern: Glob pattern for filenames.

        Returns:
            Sorted list of file paths.
        """
        if date == "*":
            folder = self.layer_dir(layer)
            return sorted(folder.rglob(pattern))

        folder = self.path(layer, date=date)
        if not folder.exists():
            return []
        return sorted(folder.glob(pattern))
    
    def list_dates(self, layer: LayerName) -> list[str]:
        """
        List all date partitions in a layer.

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
    
    def archive_file(self, layer: LayerName, filename: str, date: str | None = None) -> Path | None:
        """
        Move a file from its current location to the archive layer.

        Args:
            layer: Original layer of the file.
            filename: Name of the file to archive.
            date: Date partition. Default: today.
        
        Returns:
            Path to the archived file, or None if original file did not exist.
        """

        source = self.path(layer, filename, date)
        if not source.exists():
            return None
        
        date_str  = date or self._today()
        timestamp = datetime.now().strftime("%H-%M-%S")
        stem = source.stem
        suffix = source.suffix

        archive_path = self.archive_dir / date_str / f"{stem}_{timestamp}{suffix}"
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(source), str(archive_path))
        return archive_path
    
    def archive_layer(self, layer: LayerName, date: str | None = None) -> list[Path]:
        """
        Archive all files from a layer's date partition.

        Args:
            layer: Source layer.
            date: Date partition. Default: today.

        Returns:
            List of paths in archive.
        """
        files = self.list(layer, date=date)
        archived = []
        for f in files:
            result = self.archive_file(layer, f.name, date)
            if result:
                archived.append(result)
        return archived
    
    def cleanup(self, layer: LayerName, older_than_days: int = 30, dry_run: bool = True) -> list[Path]:
        """
        Delete date partitions older than N days.

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
            total_files = len(self.list(layer_name, date="*")) if dates else 0
            lines.append(f"  {layer_name:12s} → {layer_path}  ({len(dates)} dates, {total_files} files)")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        layers = ", ".join(f"{k}={v}" for k, v in self._layers.items())
        return f"Storage({layers})"