"""Download jakarta_house.csv via Kaggle API.

Requires ~/.kaggle/kaggle.json (Kaggle -> Account -> Create New API Token).
No credentials are bundled. Alternatively place the CSV manually at
data/jakarta_house.csv — see README.
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

DATASET = "abiyyurasyiq/jakarta-house-price-dataset"
ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data" / "jakarta_house.csv"


def main() -> int:
    if TARGET.exists():
        print(f"Already present: {TARGET} ({TARGET.stat().st_size:,} bytes)")
        return 0

    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        print("ERROR: ~/.kaggle/kaggle.json missing.", file=sys.stderr)
        print("Get token from https://www.kaggle.com/settings/account then:", file=sys.stderr)
        print("  mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/", file=sys.stderr)
        print("  chmod 600 ~/.kaggle/kaggle.json", file=sys.stderr)
        print("Or download CSV manually into data/jakarta_house.csv.", file=sys.stderr)
        return 1

    os.environ.setdefault("KAGGLE_USERNAME", "")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package missing. Run: .venv/bin/pip install kaggle", file=sys.stderr)
        return 1

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=ROOT / "data", unzip=True)
    found = list((ROOT / "data").glob("*.csv"))
    if not found:
        print("ERROR: no CSV found after download.", file=sys.stderr)
        return 1
    src = max(found, key=lambda p: p.stat().st_size)
    if src != TARGET:
        shutil.move(str(src), str(TARGET))
    for extra in (ROOT / "data").glob("*.zip"):
        extra.unlink()
    print(f"Downloaded -> {TARGET} ({TARGET.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
