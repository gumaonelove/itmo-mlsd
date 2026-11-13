"""Явные именованные проверки качества данных data room.

`run_quality_checks(tables)` возвращает tidy-таблицу: одна строка — одна проверка
(check, table, severity, n_flagged, n_total, share, passed, description).
severity: error — расчёт метрик некорректен без исправления; warning — влияет на часть метрик;
info — описание состава данных (passed всегда True). EDA-ноутбуки переиспользуют эту функцию.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from dd_verifier.data.loaders import CUSTOMER_COLUMNS, FX_COLUMNS, INVOICE_COLUMNS
from dd_verifier.metrics.customers import normalize_name

Tables = dict[str, pd.DataFrame]
CheckFn = Callable[[Tables], tuple[int, int]]
CHECKS: list[tuple[str, str, str, str, CheckFn]] = []


def check(name: str, table: str, severity: str, description: str):
    """Декоратор: зарегистрировать проверку; функция возвращает (n_flagged, n_total)."""

    def wrap(fn: CheckFn) -> CheckFn:
        CHECKS.append((name, table, severity, description, fn))
        return fn

    return wrap


def _schema(df: pd.DataFrame, expected: list[str]) -> tuple[int, int]:
    return len(set(expected) - set(df.columns)), len(expected)


@check("schema_customers", "customers", "error", "нет обязательных колонок customers.csv")
def _schema_customers(t: Tables):
    return _schema(t["customers"], CUSTOMER_COLUMNS)


@check("schema_invoices", "invoices", "error", "нет обязательных колонок invoices.csv")
def _schema_invoices(t: Tables):
    return _schema(t["invoices"], INVOICE_COLUMNS)


@check("schema_fx_rates", "fx_rates", "error", "нет обязательных колонок fx_rates.csv")
def _schema_fx(t: Tables):
    return _schema(t["fx_rates"], FX_COLUMNS)


@check("unparsed_invoice_fields", "invoices", "error", "даты или сумма инвойса не распознаны")
def _unparsed_inv(t: Tables):
    inv = t["invoices"]
    cols = ["issue_date", "service_period_start", "service_period_end", "amount"]
    return int(inv[cols].isna().any(axis=1).sum()), len(inv)


@check("unparsed_signed_date", "customers", "error", "у клиента нет даты подписания")
def _unparsed_cust(t: Tables):
    return int(t["customers"]["signed_date"].isna().sum()), len(t["customers"])


@check("duplicate_customer_ids", "customers", "error", "повторяющиеся customer_id в справочнике")
def _dup_cust_ids(t: Tables):
    c = t["customers"]
    return int(c["customer_id"].duplicated().sum()), len(c)


@check("duplicate_invoice_rows", "invoices", "warning", "полностью совпадающие строки инвойсов")
def _dup_rows(t: Tables):
    inv = t["invoices"]
    return int(inv.duplicated().sum()), len(inv)


@check(
    "duplicate_invoice_ids",
    "invoices",
    "error",
    "один invoice_id с разным содержимым (после удаления точных дублей)",
)
def _dup_ids(t: Tables):
    inv = t["invoices"].drop_duplicates()
    return int(inv["invoice_id"].duplicated().sum()), len(inv)


@check("missing_customer_id", "invoices", "warning", "инвойс без customer_id")
def _missing_cust(t: Tables):
    inv = t["invoices"]
    return int(inv["customer_id"].isna().sum()), len(inv)


@check("unknown_customer_id", "invoices", "warning", "customer_id инвойса нет в справочнике")
def _unknown_cust(t: Tables):
    inv = t["invoices"]
    known = set(t["customers"]["customer_id"])
    has = inv["customer_id"].notna()
    return int((has & ~inv["customer_id"].isin(known)).sum()), len(inv)


@check("unknown_currency", "invoices", "error", "валюты инвойса нет в fx_rates")
def _unknown_ccy(t: Tables):
    inv = t["invoices"]
    return int((~inv["currency"].isin(set(t["fx_rates"]["currency"]))).sum()), len(inv)


@check("fx_rate_missing", "invoices", "error", "нет курса на месяц начала периода услуги")
def _fx_missing(t: Tables):
    inv = t["invoices"]
    keys = set(zip(t["fx_rates"]["month"], t["fx_rates"]["currency"], strict=True))
    months = inv["service_period_start"].dt.strftime("%Y-%m")
    miss = [(m, c) not in keys for m, c in zip(months, inv["currency"], strict=True)]
    return int(sum(miss)), len(inv)


@check("void_share", "invoices", "info", "доля аннулированных инвойсов (исключаются из расчётов)")
def _void(t: Tables):
    inv = t["invoices"]
    return int((inv["status"] == "void").sum()), len(inv)


@check("credit_note_share", "invoices", "info", "доля кредит-нот (вычитаются из выручки)")
def _credit(t: Tables):
    inv = t["invoices"]
    return int((inv["status"] == "credit_note").sum()), len(inv)


@check("credit_note_positive_amount", "invoices", "warning", "кредит-нота с положительной суммой")
def _credit_sign(t: Tables):
    inv = t["invoices"]
    return int(((inv["status"] == "credit_note") & (inv["amount"] > 0)).sum()), len(inv)


@check("negative_amount_not_credit", "invoices", "warning", "отрицательная сумма не у кредит-ноты")
def _neg(t: Tables):
    inv = t["invoices"]
    return int(((inv["status"] != "credit_note") & (inv["amount"] < 0)).sum()), len(inv)


@check("service_period_inverted", "invoices", "error", "конец периода услуги раньше начала")
def _inverted(t: Tables):
    inv = t["invoices"]
    return int((inv["service_period_end"] < inv["service_period_start"]).sum()), len(inv)


@check(
    "multi_month_service",
    "invoices",
    "info",
    "период услуги длиннее месяца (годовая предоплата, онбординг): выручка делится по месяцам",
)
def _multi(t: Tables):
    inv = t["invoices"]
    s, e = inv["service_period_start"], inv["service_period_end"]
    months = (e.dt.year - s.dt.year) * 12 + (e.dt.month - s.dt.month)
    return int((months > 0).sum()), len(inv)


@check(
    "issued_outside_service_month",
    "invoices",
    "info",
    "инвойс выставлен не в месяц начала услуги (поздний инвойс, предоплата)",
)
def _late(t: Tables):
    inv = t["invoices"]
    diff = inv["issue_date"].dt.to_period("M") != inv["service_period_start"].dt.to_period("M")
    return int(diff.sum()), len(inv)


@check(
    "reseller_fee_invalid",
    "invoices",
    "warning",
    "комиссия реселлера у прямого клиента или больше суммы",
)
def _fee(t: Tables):
    inv = t["invoices"].merge(
        t["customers"][["customer_id", "channel"]], on="customer_id", how="left"
    )
    fee = inv["reseller_fee"].fillna(0)
    bad = (fee.abs() > inv["amount"].abs()) | ((inv["channel"] == "direct") & (fee != 0))
    return int(bad.sum()), len(inv)


@check(
    "customer_name_variants",
    "customers",
    "warning",
    "разные customer_id с одинаковым нормализованным именем (регистр, Inc./LLC)",
)
def _names(t: Tables):
    c = t["customers"]
    norm = c["customer_name"].map(normalize_name)
    return int(norm.duplicated(keep=False).sum()), len(c)


@check(
    "customer_dates_order",
    "customers",
    "warning",
    "go_live раньше подписания или отток раньше go_live",
)
def _order(t: Tables):
    c = t["customers"]
    bad = (c["go_live_date"] < c["signed_date"]) | (c["churn_date"] < c["go_live_date"])
    return int(bad.sum()), len(c)


@check(
    "live_without_invoices", "customers", "warning", "запущенный клиент без подписочных инвойсов"
)
def _live_no_inv(t: Tables):
    c, inv = t["customers"], t["invoices"]
    billed = set(inv.loc[inv["payment_type"] == "subscription", "customer_id"].dropna())
    live = c["go_live_date"].notna()
    return int((live & ~c["customer_id"].isin(billed)).sum()), int(live.sum())


@check("pilots_with_invoices", "customers", "warning", "пилот, по которому есть инвойсы")
def _pilots(t: Tables):
    c, inv = t["customers"], t["invoices"]
    billed = set(inv["customer_id"].dropna())
    pilots = c["is_pilot"]
    return int((pilots & c["customer_id"].isin(billed)).sum()), int(pilots.sum())


@check("subscription_after_churn", "invoices", "warning", "подписка оказывается после даты оттока")
def _after_churn(t: Tables):
    inv = t["invoices"].merge(
        t["customers"][["customer_id", "churn_date"]], on="customer_id", how="left"
    )
    sub = (inv["payment_type"] == "subscription") & (inv["status"] != "void")
    bad = sub & inv["churn_date"].notna() & (inv["service_period_end"] >= inv["churn_date"])
    return int(bad.sum()), int(sub.sum())


def run_quality_checks(tables: Tables, company_id: str | None = None) -> pd.DataFrame:
    """Запустить все проверки; tables — результат `read_tables` (сырые, без очистки)."""
    rows = []
    for name, table, severity, description, fn in CHECKS:
        n_flagged, n_total = fn(tables)
        rows.append(
            {
                "company_id": company_id,
                "check": name,
                "table": table,
                "severity": severity,
                "n_flagged": n_flagged,
                "n_total": n_total,
                "share": n_flagged / n_total if n_total else 0.0,
                "passed": severity == "info" or n_flagged == 0,
                "description": description,
            }
        )
    return pd.DataFrame(rows)
