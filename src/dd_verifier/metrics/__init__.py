"""Библиотека метрик DD Verifier: онтология → протестированные функции + SQL + допуски P5."""

from dd_verifier.metrics.core import (
    DataRoom,
    MetricResult,
    NotEnoughData,
    add_months,
    build_ledger,
    month_range,
)
from dd_verifier.metrics.customers import customer_status, normalize_name
from dd_verifier.metrics.definitions import (
    active_customers,
    acv,
    arr,
    arr_growth,
    compute,
    growth,
    grr,
    logo_churn,
    mrr,
    nrr,
    revenue,
    top10_concentration,
)
from dd_verifier.metrics.sql import metric_sql, run_sql
from dd_verifier.metrics.tolerance import (
    Tolerance,
    VerdictResult,
    load_ontology,
    tolerance_for,
    verdict_for_metric,
    verdict_from_delta,
)
from dd_verifier.metrics.verify import ClaimCheck, evaluate_claim

__all__ = [
    "ClaimCheck",
    "DataRoom",
    "MetricResult",
    "NotEnoughData",
    "Tolerance",
    "VerdictResult",
    "active_customers",
    "acv",
    "add_months",
    "arr",
    "arr_growth",
    "build_ledger",
    "compute",
    "customer_status",
    "evaluate_claim",
    "growth",
    "grr",
    "load_ontology",
    "logo_churn",
    "metric_sql",
    "month_range",
    "mrr",
    "normalize_name",
    "nrr",
    "revenue",
    "run_sql",
    "tolerance_for",
    "top10_concentration",
    "verdict_for_metric",
    "verdict_from_delta",
]
