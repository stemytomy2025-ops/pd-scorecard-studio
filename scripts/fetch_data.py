"""Download and validate every dataset listed in app_config.json.

Usage: python scripts/fetch_data.py
"""

from __future__ import annotations

import sys

from pd_scorecard_studio.config import load_config
from pd_scorecard_studio.data import DataError, fetch_dataset


def main() -> int:
    config = load_config()
    failures = 0
    for ds in config.datasets:
        print(f"== {ds.name} ({ds.license}) from {ds.landing_page}")
        try:
            manifest = fetch_dataset(ds)
        except DataError as exc:
            failures += 1
            print(f"ERROR: {exc}")
            continue
        for part in manifest["sources"]:
            print(f"   {part['url']}  sha256={part['sha256'][:16]}...  {part['bytes']:,} bytes")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
