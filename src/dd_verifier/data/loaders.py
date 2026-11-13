"""Чтение data room одной компании с диска (формат data/synthetic/{split}/{company_id}/).

- `read_tables` — сырые таблицы с приведением типов (даты, bool, числа), без очистки;
- `load_data_room` — `DataRoom` для библиотеки метрик; по умолчанию удаляет точные дубли
  строк инвойсов;
- документы: `load_documents`, `split_slides`, `split_blocks`, `split_pages`;
- эталон: `load_claims`, `load_truth`; когорты компании: `load_cohorts`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from dd_verifier.metrics.core import DataRoom

CUSTOMER_COLUMNS = [
    "customer_id",
    "customer_name",
    "segment",
    "country",
    "channel",
    "signed_date",
    "go_live_date",
    "churn_date",
    "is_pilot",
]
INVOICE_COLUMNS = [
    "invoice_id",
    "customer_id",
    "issue_date",
    "service_period_start",
    "service_period_end",
    "amount",
    "currency",
    "payment_type",
    "status",
    "reseller_fee",
]
FX_COLUMNS = ["month", "currency", "usd_rate"]
CUSTOMER_DATES = ["signed_date", "go_live_date", "churn_date"]
INVOICE_DATES = ["issue_date", "service_period_start", "service_period_end"]

SLIDE_RE = re.compile(r"<!--\s*slide:\s*(\d+)\s*-->")
PAGE_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
IMAGE_ONLY = "<!-- image-only -->"


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    """Сырые таблицы data room: customers, invoices, fx_rates (типы приведены, строки как есть)."""
    path = Path(path)
    customers = pd.read_csv(path / "customers.csv", dtype={"customer_id": str})
    for col in CUSTOMER_DATES:
        customers[col] = pd.to_datetime(customers[col], errors="coerce")
    customers["is_pilot"] = _to_bool(customers["is_pilot"])

    invoices = pd.read_csv(path / "invoices.csv", dtype={"invoice_id": str, "customer_id": str})
    for col in INVOICE_DATES:
        invoices[col] = pd.to_datetime(invoices[col], errors="coerce")
    invoices["amount"] = pd.to_numeric(invoices["amount"], errors="coerce")
    invoices["reseller_fee"] = pd.to_numeric(invoices["reseller_fee"], errors="coerce").fillna(0.0)

    fx = pd.read_csv(path / "fx_rates.csv", dtype={"month": str, "currency": str})
    return {"customers": customers, "invoices": invoices, "fx_rates": fx}


def drop_exact_duplicates(invoices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Удалить полностью совпадающие строки инвойсов (дубли выгрузки); вернуть число удалённых."""
    dup = invoices.duplicated(keep="first")
    return invoices.loc[~dup].reset_index(drop=True), int(dup.sum())


def load_data_room(path: str | Path, clean: bool = True) -> DataRoom:
    """`DataRoom` компании. clean=True — удалить точные дубли строк инвойсов.

    Остальные особенности (void, кредит-ноты, валюты, годовая предоплата) обрабатывает
    библиотека метрик; строки без customer_id остаются в итогах выручки, но не попадают
    в клиентские метрики.
    """
    t = read_tables(path)
    invoices = drop_exact_duplicates(t["invoices"])[0] if clean else t["invoices"]
    return DataRoom(t["customers"], invoices, t["fx_rates"])


def load_documents(path: str | Path) -> dict[str, str]:
    """Тексты документов компании: {'deck.md': ..., 'press_release_01.md': ..., ...}."""
    docs = Path(path) / "documents"
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(docs.glob("*.md"))}


def split_slides(deck: str) -> list[dict]:
    """Слайды дека: [{'slide': N, 'text': ..., 'image_only': bool}]."""
    parts = SLIDE_RE.split(deck)
    return [
        {"slide": int(num), "text": body.strip(), "image_only": IMAGE_ONLY in body}
        for num, body in zip(parts[1::2], parts[2::2], strict=True)
    ]


def split_pages(memo: str) -> list[dict]:
    """Страницы меморандума: [{'page': N, 'text': ...}]."""
    parts = PAGE_RE.split(memo)
    return [
        {"page": int(n), "text": body.strip()}
        for n, body in zip(parts[1::2], parts[2::2], strict=True)
    ]


def split_blocks(text: str) -> list[str]:
    """Абзацы (блоки через пустую строку), нумерация с 1 включая заголовок: location.paragraph."""
    return [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]


def load_claims(path: str | Path) -> pd.DataFrame:
    """Эталонные утверждения gold/claims.jsonl одной компании."""
    lines = (Path(path) / "gold" / "claims.jsonl").read_text(encoding="utf-8").splitlines()
    return pd.DataFrame([json.loads(x) for x in lines if x.strip()])


def load_truth(path: str | Path) -> pd.DataFrame:
    """Помесячные истинные метрики (стандартные определения)."""
    return pd.read_csv(Path(path) / "gold" / "truth_monthly.csv", dtype={"month": str})


def load_cohorts(path: str | Path) -> dict[str, pd.DataFrame]:
    """Листы cohorts.xlsx (когортная таблица компании и лист notes с определением)."""
    return pd.read_excel(Path(path) / "cohorts.xlsx", sheet_name=None)


def company_dirs(root: str | Path, split: str = "dev") -> list[Path]:
    """Каталоги компаний сплита (dev по умолчанию; holdout — только для финальной оценки)."""
    return sorted(p for p in (Path(root) / split).iterdir() if p.is_dir())
