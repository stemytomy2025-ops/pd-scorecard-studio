"""Schema validation and the fetch pipeline, with downloads replaced by SYNTHETIC bytes."""

import io
import json

import pytest
from conftest import synthetic_frame

from pd_scorecard_studio import data


def test_valid_frame_passes(config):
    for ds in config.datasets:
        out = data.validate_schema(synthetic_frame(ds), ds)
        assert list(ds.expected_schema) == [c for c in out.columns if c in ds.expected_schema]


def test_missing_column_is_reported(config):
    ds = config.datasets[0]
    frame = synthetic_frame(ds).drop(columns=[next(iter(ds.expected_schema))])
    with pytest.raises(data.SchemaError, match="missing columns"):
        data.validate_schema(frame, ds)


def test_unparseable_values_are_reported(config):
    ds = config.datasets[0]
    typed = [c for c, t in ds.expected_schema.items() if t in ("integer", "float", "date")]
    if not typed:
        pytest.skip("no typed columns")
    frame = synthetic_frame(ds)
    frame[typed[0]] = "garbage"
    with pytest.raises(data.SchemaError, match="do not parse"):
        data.validate_schema(frame, ds)


def test_too_few_rows_is_reported(config):
    ds = config.datasets[0].model_copy(update={"expected_min_rows": 10**9})
    with pytest.raises(data.SchemaError, match="rows"):
        data.validate_schema(synthetic_frame(ds), ds)


def test_fetch_writes_validated_parquet_and_manifest(config, tmp_path, monkeypatch):
    ds = config.datasets[0]
    buffer = io.BytesIO()
    if ds.source.kind == "huggingface_parquet":
        synthetic_frame(ds).to_parquet(buffer)
    else:
        synthetic_frame(ds).to_csv(buffer, index=False)

    def fake_download(url, max_bytes):
        if "/api/datasets/" in url:
            return json.dumps(["https://example.invalid/part-0.parquet"]).encode()
        return buffer.getvalue()

    monkeypatch.setattr(data, "_download", fake_download)
    messages = []
    manifest = data.fetch_dataset(ds, tmp_path, progress=messages.append)
    assert manifest["rows"] == len(synthetic_frame(ds))
    assert len(manifest["sources"][0]["sha256"]) == 64
    assert any("Validating schema" in m for m in messages)
    loaded = data.load_dataset(ds, tmp_path)
    assert len(loaded) == manifest["rows"]


def test_load_without_download_explains_what_to_do(config, tmp_path):
    with pytest.raises(data.DataNotAvailable, match="fetch_data.py"):
        data.load_dataset(config.datasets[0], tmp_path)


def test_download_errors_become_data_errors(config, monkeypatch, tmp_path):
    def boom(url, max_bytes):
        raise data.DataNotAvailable("download failed for x: HTTP 404")

    monkeypatch.setattr(data, "_download", boom)
    with pytest.raises(data.DataError, match="404"):
        data.fetch_dataset(config.datasets[0], tmp_path)
    assert not (tmp_path / f"{config.datasets[0].id}.parquet").exists()
