"""Загрузчики data room и проверки качества данных."""

from __future__ import annotations

import pandas as pd
import pytest

from dd_verifier.data import (
    load_data_room,
    read_tables,
    run_quality_checks,
    split_blocks,
    split_pages,
    split_slides,
)
from dd_verifier.metrics import mrr


def _write(tables: dict[str, pd.DataFrame], path) -> None:
    for name, df in tables.items():
        df.to_csv(path / f"{name}.csv", index=False, date_format="%Y-%m-%d")


def test_roundtrip_and_dedup(tiny_tables, tmp_path):
    inv = tiny_tables["invoices"]
    tiny_tables["invoices"] = pd.concat([inv, inv.iloc[[0, 1]]], ignore_index=True)
    _write(tiny_tables, tmp_path)
    raw = read_tables(tmp_path)
    assert raw["customers"]["is_pilot"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(raw["invoices"]["service_period_start"])
    assert len(raw["invoices"]) == len(inv) + 2
    dr = load_data_room(tmp_path)
    assert len(dr.invoices) == len(inv)
    assert mrr(dr, "2024-03").value == pytest.approx(1750.0)
    assert mrr(load_data_room(tmp_path, clean=False), "2023-06").value == pytest.approx(200.0)


def _flags(report: pd.DataFrame) -> dict[str, int]:
    return dict(zip(report["check"], report["n_flagged"], strict=True))


def test_quality_clean_tiny(tiny_tables):
    report = run_quality_checks(tiny_tables, "tiny")
    assert {"check", "table", "severity", "n_flagged", "n_total", "share", "passed"} <= set(
        report.columns
    )
    flags = _flags(report)
    assert flags["missing_customer_id"] == 1
    assert flags["void_share"] == 1
    assert flags["credit_note_share"] == 1
    assert flags["multi_month_service"] == 2  # годовая предоплата и внедрение
    assert flags["pilots_with_invoices"] == 0
    assert flags["duplicate_invoice_rows"] == 0
    assert report.loc[report["severity"] == "error", "passed"].all()


def test_quality_detects_planted_problems(tiny_tables):
    inv = tiny_tables["invoices"].copy()
    bad = inv.iloc[[0]].copy()
    dup = inv.iloc[[1]].copy()
    unknown = inv.iloc[[2]].assign(invoice_id="X1", customer_id="C99", currency="JPY")
    inverted = inv.iloc[[3]].assign(
        invoice_id="X2", service_period_end=pd.Timestamp("2023-01-01"), amount=-5.0
    )
    positive_credit = inv.iloc[[4]].assign(invoice_id="X3", status="credit_note", reseller_fee=1.0)
    bad["invoice_id"] = "C2-2023-06"  # тот же id, другое содержимое
    bad["amount"] = 999.0
    tables = dict(tiny_tables)
    tables["invoices"] = pd.concat(
        [inv, bad, dup, unknown, inverted, positive_credit], ignore_index=True
    )
    cust = tiny_tables["customers"]
    variant = cust.iloc[[0]].assign(customer_id="C6", customer_name="BLUE FORK GROUP")
    tables["customers"] = pd.concat([cust, variant], ignore_index=True)

    flags = _flags(run_quality_checks(tables))
    assert flags["duplicate_invoice_rows"] == 1
    assert flags["duplicate_invoice_ids"] == 1
    assert flags["unknown_customer_id"] == 1
    assert flags["unknown_currency"] == 1
    assert flags["fx_rate_missing"] == 1
    assert flags["service_period_inverted"] == 1
    assert flags["negative_amount_not_credit"] == 1
    assert flags["credit_note_positive_amount"] == 1
    assert flags["reseller_fee_invalid"] == 1  # комиссия у прямого клиента C2
    assert flags["customer_name_variants"] == 2
    assert flags["live_without_invoices"] == 1  # дубль C6 запущен, но без инвойсов


def test_document_splitters():
    deck = "<!-- slide: 1 -->\n# A\n\n- x\n<!-- slide: 2 -->\n<!-- image-only -->\n![c](i.png)\n"
    slides = split_slides(deck)
    assert [s["slide"] for s in slides] == [1, 2]
    assert slides[1]["image_only"] and not slides[0]["image_only"]
    assert split_blocks("# T\n\nfirst\n\nsecond line\nwrapped\n") == [
        "# T",
        "first",
        "second line\nwrapped",
    ]
    assert split_pages("<!-- page: 1 -->\nA\n<!-- page: 2 -->\nB")[1] == {"page": 2, "text": "B"}
