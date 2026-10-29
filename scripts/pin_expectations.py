"""Pin expected eval values from the downloaded data, independently of the tool code.

Reads data/raw/<id>.parquet with pyarrow and computes each expectation in plain Python
(no pandas, no operations module), so the tool-level evals compare two separate
implementations.

Usage: python scripts/pin_expectations.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pyarrow.parquet as pq

from pd_scorecard_studio.config import DATA_DIR, load_config
from pd_scorecard_studio.data import load_manifest
from pd_scorecard_studio.evaluation import CASES_PATH

AGGS = {
    "sum": sum,
    "mean": statistics.fmean,
    "median": statistics.median,
    "min": min,
    "max": max,
    "count": len,
}


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return datetime.fromisoformat(str(value))


def _bucket(ts: datetime, freq: str) -> str:
    if freq == "hour":
        return ts.strftime("%Y-%m-%dT%H")
    if freq == "day":
        return ts.date().isoformat()
    if freq == "week":
        return (ts.date() - timedelta(days=ts.weekday())).isoformat()
    return ts.strftime("%Y-%m-01")


def _matches(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(str(row[c]) == str(v) or row[c] == v for c, v in filters.items())


def _in_range(ts: datetime, args: dict[str, Any]) -> bool:
    start, end = args.get("start"), args.get("end")
    if start and ts < datetime.fromisoformat(start):
        return False
    return not (end and ts >= datetime.fromisoformat(end) + timedelta(days=1))


def pin_case(case: dict[str, Any], params: dict[str, Any], rows: list[dict]) -> Any:
    args, kind = case["args"], case["expect"]["kind"]
    if kind in ("top_group", "group_value"):
        groups: dict[Any, list[float]] = defaultdict(list)
        for row in rows:
            if _matches(row, args.get("filters", {})) and row[args["metric"]] is not None:
                groups[row[params["group"]]].append(row[args["metric"]])
        agg = AGGS[args.get("agg", "mean")]
        values = {g: agg(v) for g, v in groups.items()}
        if kind == "top_group":
            return max(values, key=values.get)
        return round(values[case["expect"]["group"]], 4)
    if kind == "peak_period":
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            ts = _as_datetime(row[params["time"]])
            if params.get("hour"):
                ts = ts.replace(hour=0) + timedelta(hours=int(row[params["hour"]]))
            if _in_range(ts, args) and _matches(row, args.get("filters", {})):
                buckets[_bucket(ts, args.get("freq", "day"))].append(row[params["metric"]])
        agg = AGGS[args.get("agg", "sum")]
        values = {b: agg(v) for b, v in buckets.items()}
        return max(values, key=values.get)
    if kind == "matched_count":
        return sum(1 for row in rows if _matches(row, args.get("filters", {})))
    if kind == "overall_bad_rate":
        return round(statistics.fmean(int(row[params["target"]]) for row in rows), 4)
    return None  # property checks need no pinning


def main() -> int:
    config = load_config()
    cases_doc = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    tools = {t.name: t for t in config.tools}
    tables: dict[str, list[dict]] = {}
    hashes = {}
    for case in cases_doc["cases"]:
        tool = tools[case["tool"]]
        ds = config.dataset(tool.dataset_id)
        if ds.id not in tables:
            path = DATA_DIR / f"{ds.id}.parquet"
            if not path.is_file():
                print(f"ERROR: {path} missing; run scripts/fetch_data.py first")
                return 1
            tables[ds.id] = pq.read_table(path).to_pylist()
            manifest = load_manifest(ds) or {}
            hashes[ds.id] = [s["sha256"] for s in manifest.get("sources", [])]
        value = pin_case(case, tool.params, tables[ds.id])
        if value is not None:
            case["expect"]["value"] = value
            print(f"pinned {case['id']}: {case['expect']['kind']} = {value!r}")
    cases_doc["pinned"] = {
        "at": datetime.now(UTC).isoformat(),
        "by": "scripts/pin_expectations.py (pyarrow + plain Python, independent of the tools)",
        "source_sha256": hashes,
    }
    CASES_PATH.write_text(json.dumps(cases_doc, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
