"""Библиотека метрик на крошечном data room: ответы посчитаны вручную (см. conftest.py)."""

from __future__ import annotations

import pytest

from dd_verifier.metrics import (
    NotEnoughData,
    active_customers,
    acv,
    arr,
    arr_growth,
    build_ledger,
    compute,
    customer_status,
    evaluate_claim,
    growth,
    grr,
    logo_churn,
    metric_sql,
    mrr,
    normalize_name,
    nrr,
    revenue,
    run_sql,
    top10_concentration,
)

approx = pytest.approx


def test_ledger_excludes_void_and_spreads_annual(tiny):
    led = build_ledger(tiny.invoices, tiny.fx_rates)
    assert "I6" not in set(led["invoice_id"])
    annual = led.loc[led["invoice_id"] == "I1"]
    assert len(annual) == 12  # 2024-02..2025-01
    assert annual["net_usd"].tolist() == approx([1200.0] * 12)  # 12 000 EUR × 1.2 / 12
    assert tiny.coverage == ("2023-06", "2024-05")


@pytest.mark.parametrize(
    ("month", "variant", "expected"),
    [
        ("2024-01", "standard", 100.0),
        ("2024-02", "standard", 1300.0),
        ("2024-03", "standard", 1750.0),  # C1 1200 + C2 100 + C3 400 + без клиента 50
        ("2024-04", "standard", 1560.0),  # кредит-нота C3: 400 − 40
        ("2024-05", "standard", 1680.0),
        ("2024-02", "incl_one_time", 2300.0),  # + внедрение 1000
        ("2024-03", "incl_one_time", 3050.0),  # + внедрение 1000 + обучение 300
        ("2024-03", "gross_channel", 1850.0),  # C3 по gross 500
        ("2024-04", "gross_channel", 1650.0),
    ],
)
def test_mrr(tiny, month, variant, expected):
    assert mrr(tiny, month, variant).value == approx(expected)


def test_arr_variants(tiny):
    assert arr(tiny, "2024-05").value == approx(20160.0)
    assert arr(tiny, "2024-05", "incl_one_time").value == approx(20160.0)
    assert arr(tiny, "2024-05", "gross_channel").value == approx(21600.0)
    assert arr(tiny, "2024-05", "peak_month").value == approx(21000.0)  # пик — март, 1750 × 12


def test_arr_growth_variants(tiny):
    assert arr_growth(tiny, "2024-02", "2024-05").value == approx(20160 / 15600)
    assert arr_growth(tiny, "2024-02", "2024-05", "incl_one_time").value == approx(20160 / 27600)
    assert arr_growth(tiny, "2024-02", "2024-05", "gross_channel").value == approx(21600 / 15600)
    # минимум окна 2024-01..2024-04 — январь (ARR 1200)
    assert arr_growth(tiny, "2024-01", "2024-05", "trough_base").value == approx(20160 / 1200)


def test_revenue(tiny):
    assert revenue(tiny, "2024-03", "2024-04").value == approx(3050 + 1560)
    assert revenue(tiny, "2024-03", "2024-04", "gross_channel").value == approx(3150 + 1650)


def test_active_customers(tiny):
    assert active_customers(tiny, "2024-03").value == 3
    assert active_customers(tiny, "2024-04").value == 2  # C2 ушёл 2024-04-01
    assert (
        active_customers(tiny, "2024-04", "signed_incl_pilots").value == 4
    )  # C1, C3, пилот C4, C5
    assert active_customers(tiny, "2024-03", "signed_incl_pilots").value == 4


def test_logo_churn(tiny):
    # апрель: ушёл 1 из 3; май: 0 из 2
    assert logo_churn(tiny, "2024-04", "2024-05").value == approx((1 / 3 + 0) / 2)
    assert logo_churn(tiny, "2024-04", "2024-05", "excl_pilots_and_small").value == approx(0.0)


def test_retention(tiny):
    assert nrr(tiny, "2024-05", cohort_month="2024-03").value == approx((1200 + 0 + 480) / 1700)
    survivors = nrr(tiny, "2024-05", "survivors_only", cohort_month="2024-03").value
    assert survivors == approx((1200 + 480) / 1600)
    assert grr(tiny, "2024-05", cohort_month="2024-03").value == approx((1200 + 0 + 400) / 1700)


def test_acv_and_concentration(tiny):
    assert acv(tiny, "2024-05").value == approx(10080.0)
    assert acv(tiny, "2024-05", "gross_channel").value == approx(10800.0)
    # 12 мес. выручки: C1 7100, C3 1240, C2 1000, без клиента 50
    assert top10_concentration(tiny, "2024-05").value == approx(9340 / 9390)


def test_growth_of_level_metric(tiny):
    assert growth(tiny, "active_customers", "2024-03", "2024-05").value == approx(2 / 3)


def test_not_enough_data(tiny):
    with pytest.raises(NotEnoughData):
        mrr(tiny, "2024-06")  # после последнего выставленного инвойса
    with pytest.raises(NotEnoughData):
        nrr(tiny, "2024-05")  # когорта 2023-05 вне покрытия
    with pytest.raises(NotEnoughData):
        revenue(tiny, "2023-01", "2023-12")


def test_unknown_variant(tiny):
    with pytest.raises(ValueError, match="неизвестный вариант"):
        mrr(tiny, "2024-05", "peak_month")


def test_trace_is_human_readable(tiny):
    res = compute(tiny, "arr", "2024-05")
    assert "ARR = MRR × 12" in res.explain()
    assert res.sql and "generate_series" in res.sql


SQL_CASES = [
    ("mrr", "2024-03", None, "standard"),
    ("mrr", "2024-03", None, "incl_one_time"),
    ("mrr", "2024-04", None, "gross_channel"),
    ("arr", "2024-05", None, "standard"),
    ("arr", "2024-05", None, "peak_month"),
    ("arr", "2024-05", None, "gross_channel"),
    ("arr", "2024-05", None, "incl_one_time"),
    ("arr_growth", "2024-05", "2024-02", "standard"),
    ("arr_growth", "2024-05", "2024-02", "incl_one_time"),
    ("arr_growth", "2024-05", "2024-01", "trough_base"),
    # peak_month требует 12 месяцев до начала окна; в 12-месячной фикстуре — только start = end
    ("arr_growth", "2024-05", "2024-05", "peak_month"),
    ("arr_growth", "2024-05", "2024-02", "gross_channel"),
    ("revenue", "2024-04", "2024-03", "standard"),
    ("revenue", "2024-04", "2024-03", "gross_channel"),
    ("active_customers", "2024-04", None, "standard"),
    ("active_customers", "2024-04", None, "signed_incl_pilots"),
    ("logo_churn", "2024-05", "2024-04", "standard"),
    ("logo_churn", "2024-05", "2024-04", "excl_pilots_and_small"),
    ("nrr", "2024-05", "2024-03", "standard"),
    ("nrr", "2024-05", "2024-03", "survivors_only"),
    ("grr", "2024-05", "2024-03", "standard"),
    ("acv", "2024-05", None, "standard"),
    ("acv", "2024-05", None, "gross_channel"),
    ("acv", "2024-05", None, "incl_one_time"),
    ("acv", "2024-05", None, "peak_month"),
    ("top10_concentration", "2024-05", None, "standard"),
]


@pytest.mark.parametrize(("metric", "end", "start", "variant"), SQL_CASES)
def test_sql_matches_pandas(tiny, metric, end, start, variant):
    res = compute(tiny, metric, end, start, variant)
    assert res.sql is not None
    assert run_sql(tiny, res.sql) == approx(res.value)


def test_sql_for_growth_of_customers(tiny):
    sql = metric_sql("active_customers", "2024-05", "2024-03", as_growth=True)
    assert run_sql(tiny, sql) == approx(2 / 3)


def test_name_normalization_and_status(tiny):
    assert normalize_name("BLUE FORK GROUP") == normalize_name("Blue Fork Group LLC")
    assert normalize_name("Harbor Kitchen") == normalize_name("Harbor Kitchen B.V.")
    assert customer_status(tiny, "Harbor Kitchen", "2024-05") == "active"
    assert customer_status(tiny, "golden ladle", "2024-05") == "churned"
    assert customer_status(tiny, "River Spoon", "2024-05") == "not_live"
    assert customer_status(tiny, "Silver Plate Inc.", "2024-05") == "not_found"


def test_evaluate_claim_dispatch(tiny):
    base = {"qualifier": None, "period_start": None, "period_end": "2024-05"}
    level = {**base, "claim_type": "level", "metric_id": "arr", "claimed_value": 21000.0}
    assert evaluate_claim(tiny, level).verdict == "SUPPORTED"  # +4.2%
    level["claimed_value"] = 24000.0
    assert evaluate_claim(tiny, level).verdict == "EXAGGERATED"  # +19%
    assert evaluate_claim(tiny, level, variant="peak_month").actual == approx(21000.0)
    count = {**base, "claim_type": "count", "metric_id": "active_customers", "claimed_value": 4.0}
    assert evaluate_claim(tiny, count).verdict == "CONTRADICTED"  # +100%
    missing = {**base, "claim_type": "share", "metric_id": "nrr", "claimed_value": 1.1}
    assert evaluate_claim(tiny, missing).verdict == "NOT_ENOUGH_DATA"
    mention = {
        **base,
        "claim_type": "customer_mention",
        "metric_id": None,
        "claimed_value": "GOLDEN LADLE",
    }
    assert evaluate_claim(tiny, mention).verdict == "CONTRADICTED"
    for ctype, verdict in [
        ("forecast", "UNVERIFIABLE"),
        ("out_of_ontology", "NOT_ENOUGH_DATA"),
        ("factual", "FACTUAL_OUT_OF_SCOPE"),
    ]:
        assert evaluate_claim(tiny, {**base, "claim_type": ctype}).verdict == verdict
    grow = {
        **base,
        "claim_type": "growth",
        "metric_id": "active_customers",
        "claimed_value": 1.2,
        "period_start": "2024-03",
    }
    assert evaluate_claim(tiny, grow).verdict == "CONTRADICTED"  # рост заявлен, факт ×0.67
