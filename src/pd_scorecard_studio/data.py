"""Reproducible download and schema validation of the documented dataset(s).

`fetch_dataset` downloads from the source recorded in app_config.json, validates the schema,
and writes `data/raw/<id>.parquet` plus a manifest with SHA-256 hashes. `load_dataset` only
reads that validated file; it never falls back to invented data.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pd_scorecard_studio.config import DATA_DIR, DatasetConfig

USER_AGENT = "portfolio-app/0.1 (reproducible dataset fetch)"
Progress = Callable[[str], None]
BOOLEANS = {"true": True, "false": False, "1": True, "0": False}


class DataError(Exception):
    """Base class for data problems shown to the user."""


class DataNotAvailable(DataError):
    pass


class SchemaError(DataError):
    def __init__(self, dataset_id: str, problems: list[str]):
        super().__init__(f"{dataset_id}: schema validation failed: " + "; ".join(problems))
        self.problems = problems


def _download(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            content = resp.read(max_bytes + 1)
    except OSError as exc:  # URLError, HTTPError and timeouts are OSErrors
        raise DataNotAvailable(f"download failed for {url}: {exc}") from exc
    if len(content) > max_bytes:
        raise DataNotAvailable(f"{url} is larger than the {max_bytes:,}-byte limit")
    return content


def validate_schema(df: pd.DataFrame, ds: DatasetConfig) -> pd.DataFrame:
    """Coerce the expected columns to their types; raise SchemaError listing every problem."""
    problems: list[str] = []
    missing = [c for c in ds.expected_schema if c not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    out = df.copy()
    for column, kind in ds.expected_schema.items():
        if column in missing:
            continue
        raw = out[column]
        if kind in ("integer", "float"):
            values = pd.to_numeric(raw, errors="coerce")
            bad = values.isna() & raw.notna()
            if kind == "integer" and (values.dropna() % 1 != 0).any():
                problems.append(f"{column!r} expected integer but has fractional values")
        elif kind in ("date", "datetime"):
            values = pd.to_datetime(raw, errors="coerce")
            bad = values.isna() & raw.notna()
        elif kind == "boolean":
            values = raw.map(lambda v: BOOLEANS.get(str(v).strip().lower()) if pd.notna(v) else v)
            bad = values.isna() & raw.notna()
        else:
            values, bad = raw.astype("string"), pd.Series(False, index=raw.index)
        if bad.mean() > 0.01:
            problems.append(f"{column!r} expected {kind}: {bad.mean():.1%} of values do not parse")
        out[column] = values
    if len(out) < ds.expected_min_rows:
        problems.append(f"only {len(out)} rows (expected at least {ds.expected_min_rows})")
    if problems:
        raise SchemaError(ds.id, problems)
    return out


def _hf_parquet_urls(ds: DatasetConfig) -> list[str]:
    name = urllib.parse.quote(ds.source.dataset or "", safe="/")
    api = f"https://huggingface.co/api/datasets/{name}/parquet/{ds.source.config}/{ds.source.split}"
    urls = json.loads(_download(api, 1_000_000))
    if not urls:
        raise DataNotAvailable(f"no parquet files listed at {api}")
    return urls


def fetch_dataset(
    ds: DatasetConfig, data_dir: Path = DATA_DIR, progress: Progress = print
) -> dict[str, Any]:
    """Download, validate and save one dataset. Returns the manifest."""
    if ds.synthetic:
        raise DataNotAvailable(f"{ds.id} is SYNTHETIC and has no remote source")
    max_bytes = ds.source.max_mb * 1_000_000
    parts: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    if ds.source.kind == "huggingface_parquet":
        urls = _hf_parquet_urls(ds)
        for i, url in enumerate(urls, 1):
            progress(f"Downloading part {i}/{len(urls)} from Hugging Face...")
            content = _download(url, max_bytes)
            parts.append({"url": url, "bytes": len(content), "sha256": _sha(content)})
            frames.append(pd.read_parquet(io.BytesIO(content)))
    else:
        progress(f"Downloading {ds.source.url}...")
        content = _download(ds.source.url or "", max_bytes)
        parts.append({"url": ds.source.url, "bytes": len(content), "sha256": _sha(content)})
        frames.append(pd.read_csv(io.BytesIO(content)))
    progress("Validating schema...")
    df = validate_schema(pd.concat(frames, ignore_index=True), ds)

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_dir / f"{ds.id}.parquet", index=False)
    manifest = {
        "dataset_id": ds.id,
        "fetched_at": datetime.now(UTC).isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
        "sources": parts,
        "license": ds.license,
        "attribution": ds.attribution,
    }
    (data_dir / f"{ds.id}.manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    progress(f"Saved {len(df):,} validated rows.")
    return manifest


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_manifest(ds: DatasetConfig, data_dir: Path = DATA_DIR) -> dict[str, Any] | None:
    path = data_dir / f"{ds.id}.manifest.json"
    return json.loads(path.read_text("utf-8")) if path.is_file() else None


def load_dataset(ds: DatasetConfig, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{ds.id}.parquet"
    if not path.is_file():
        raise DataNotAvailable(
            f"{ds.name} has not been downloaded yet. Run `python scripts/fetch_data.py` "
            "or use the download button in the app."
        )
    return validate_schema(pd.read_parquet(path), ds)
