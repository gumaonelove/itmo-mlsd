"""Общие фикстуры: крошечный data room с ответами, посчитанными вручную.

Клиенты:
- C1 Enterprise, go-live 2024-02, годовая предоплата 12 000 EUR (курс февраля 1.2 → 1 200 $/мес),
  внедрение 2 000 $ на 2 месяца (02–03), обучение 300 $ в 03;
- C2 SMB, go-live 2023-06, 100 $/мес с 2023-06 по 2024-03, ушёл 2024-04-01; есть void-дубль за 03;
- C3 MidMarket через реселлера: gross 500/500/600 $ (03/04/05), комиссия 100/100/120,
  кредит-нота −50 (комиссия −10) за 04 → net 400/360/480;
- C4 — бесплатный пилот (подписан 2024-03, без go-live), C5 — подписан 2024-04, ещё не запущен;
- инвойс без customer_id: 50 $ подписки за 2024-03.
Покрытие данных: 2023-06..2024-05 (последний выставленный инвойс — май 2024).
"""

from __future__ import annotations

import pandas as pd
import pytest

from dd_verifier.metrics import DataRoom


def _customers() -> pd.DataFrame:
    rows = [
        (
            "C1",
            "Blue Fork Group LLC",
            "Enterprise",
            "direct",
            "2024-01-10",
            "2024-02-01",
            None,
            False,
        ),
        (
            "C2",
            "Golden Ladle Inc.",
            "SMB",
            "direct",
            "2023-05-20",
            "2023-06-01",
            "2024-04-01",
            False,
        ),
        (
            "C3",
            "Harbor Kitchen B.V.",
            "MidMarket",
            "reseller",
            "2024-02-01",
            "2024-03-01",
            None,
            False,
        ),
        ("C4", "Oak Table", "SMB", "direct", "2024-03-01", None, None, True),
        ("C5", "River Spoon Ltd", "MidMarket", "direct", "2024-04-01", None, None, False),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "customer_id",
            "customer_name",
            "segment",
            "channel",
            "signed_date",
            "go_live_date",
            "churn_date",
            "is_pilot",
        ],
    )
    df.insert(3, "country", "US")
    for col in ("signed_date", "go_live_date", "churn_date"):
        df[col] = pd.to_datetime(df[col])
    return df


def _invoices() -> pd.DataFrame:
    def inv(iid, cid, start, end, amount, ccy="USD", ptype="subscription", status="paid", fee=0.0):
        return (iid, cid, start, start, end, amount, ccy, ptype, status, fee)

    months = pd.period_range("2023-06", "2024-03", freq="M")
    rows = [inv(f"C2-{p}", "C2", f"{p}-01", str(p.end_time.date()), 100.0) for p in months]
    rows += [
        inv("I1", "C1", "2024-02-01", "2025-01-31", 12000.0, ccy="EUR"),
        inv("I2", "C1", "2024-02-01", "2024-03-31", 2000.0, ptype="implementation"),
        inv("I6", "C2", "2024-03-01", "2024-03-31", 100.0, status="void"),
        inv("I7", "C3", "2024-03-01", "2024-03-31", 500.0, fee=100.0),
        inv("I8", "C3", "2024-04-01", "2024-04-30", 500.0, fee=100.0),
        inv("I9", "C3", "2024-05-01", "2024-05-31", 600.0, fee=120.0),
        inv("I10", "C3", "2024-04-01", "2024-04-30", -50.0, status="credit_note", fee=-10.0),
        inv("I11", "C1", "2024-03-01", "2024-03-31", 300.0, ptype="one_time"),
        inv("I12", None, "2024-03-01", "2024-03-31", 50.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
    for col in ("issue_date", "service_period_start", "service_period_end"):
        df[col] = pd.to_datetime(df[col])
    return df


def _fx() -> pd.DataFrame:
    months = [str(p) for p in pd.period_range("2023-06", "2024-12", freq="M")]
    rows = [(m, "USD", 1.0) for m in months]
    rows += [(m, "EUR", 1.2 if m == "2024-02" else 1.1) for m in months]
    return pd.DataFrame(rows, columns=["month", "currency", "usd_rate"])


@pytest.fixture
def tiny() -> DataRoom:
    """Крошечный data room (см. docstring модуля)."""
    return DataRoom(_customers(), _invoices(), _fx())


@pytest.fixture
def tiny_tables() -> dict[str, pd.DataFrame]:
    """Те же таблицы словарём (для проверок качества)."""
    return {"customers": _customers(), "invoices": _invoices(), "fx_rates": _fx()}
