"""Поиск клиента по упоминанию в документах (логотипы, кейсы) — случай D10.

Имена в документах и в справочнике пишутся по-разному: регистр, «Inc.» / «LLC», пунктуация.
Сравниваем нормализованные имена.
"""

from __future__ import annotations

import re

import pandas as pd

from dd_verifier.metrics.core import DataRoom, month_end

LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "llc",
        "ltd",
        "limited",
        "gmbh",
        "co",
        "corp",
        "corporation",
        "plc",
        "bv",
        "ag",
        "sa",
        "sas",
    }
)


def normalize_name(name: str) -> str:
    """'Blue Fork Group, LLC' → 'blue fork group'."""
    text = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower().replace(".", "").replace("&", " and "))
    tokens = text.split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def find_customers(customers: pd.DataFrame, name: str) -> pd.DataFrame:
    """Записи справочника, чьё нормализованное имя совпадает с упомянутым."""
    key = normalize_name(name)
    return customers.loc[customers["customer_name"].map(normalize_name) == key]


def customer_status(dr: DataRoom, name: str, month: str) -> str:
    """Статус упомянутого клиента на конец месяца: active | churned | not_live | not_found."""
    rows = find_customers(dr.customers, name)
    if rows.empty:
        return "not_found"
    active = set(dr.active_ids(month))
    if rows["customer_id"].isin(active).any():
        return "active"
    me = month_end(month)
    if (rows["churn_date"].notna() & (rows["churn_date"] <= me)).any():
        return "churned"
    return "not_live"
