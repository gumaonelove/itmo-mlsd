"""Скачивает UCI Online Retail II и кэширует оба листа в parquet.

Датасет: https://archive.ics.uci.edu/dataset/502/online+retail+ii (Daqing Chen, 2019),
лицензия CC BY 4.0. Используется как реальный прокси «грязной» выгрузки инвойсов
для EDA качества биллинговых данных (notebooks/02_eda_billing_proxy.ipynb).

Запуск из корня репозитория:
    uv run python scripts/fetch_online_retail.py [--force]

Результат (data/external/ в .gitignore, в git не попадает):
    data/external/online_retail_ii/online_retail_ii.zip
    data/external/online_retail_ii/online_retail_II.xlsx
    data/external/online_retail_ii/online_retail_ii_2009_2010.parquet
    data/external/online_retail_ii/online_retail_ii_2010_2011.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "online_retail_ii"
ZIP_PATH = OUT_DIR / "online_retail_ii.zip"
XLSX_NAME = "online_retail_II.xlsx"
SHEETS = {
    "Year 2009-2010": "online_retail_ii_2009_2010.parquet",
    "Year 2010-2011": "online_retail_ii_2010_2011.parquet",
}
DTYPES = {"Invoice": "string", "StockCode": "string", "Description": "string", "Country": "string"}


def download(force: bool = False) -> Path:
    """Скачивает zip-архив, если его ещё нет в кэше."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        print(f"zip уже есть: {ZIP_PATH}")
        return ZIP_PATH
    print(f"скачиваю {URL}")
    with requests.get(URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = ZIP_PATH.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(ZIP_PATH)
    return ZIP_PATH


def extract(zip_path: Path) -> Path:
    """Извлекает xlsx из архива (только ожидаемый файл, без произвольных путей)."""
    xlsx_path = OUT_DIR / XLSX_NAME
    if xlsx_path.exists():
        return xlsx_path
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if Path(n).name == XLSX_NAME)
        xlsx_path.write_bytes(zf.read(member))
    return xlsx_path


def convert(xlsx_path: Path, force: bool = False) -> list[Path]:
    """Конвертирует оба листа в parquet с единообразными типами."""
    out = []
    for sheet, fname in SHEETS.items():
        target = OUT_DIR / fname
        if target.exists() and not force:
            print(f"parquet уже есть: {target}")
            out.append(target)
            continue
        print(f"читаю лист {sheet!r} (openpyxl, ~1–2 мин)")
        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=DTYPES, engine="openpyxl")
        df["Customer ID"] = df["Customer ID"].astype("Int64")
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df.to_parquet(target, index=False)
        print(f"  {len(df):,} строк → {target}")
        out.append(target)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="перекачать и пересоздать parquet")
    args = parser.parse_args()
    zip_path = download(force=args.force)
    print(f"sha256(zip) = {sha256(zip_path)}")
    xlsx_path = extract(zip_path)
    convert(xlsx_path, force=args.force)


if __name__ == "__main__":
    main()
