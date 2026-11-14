"""Проверка одного структурированного утверждения по данным (процесс B без LLM).

Этой же функцией генератор синтетики считает эталонные вердикты на чистых данных,
поэтому эталон не назначается вручную.
"""

from __future__ import annotations

from dataclasses import dataclass

from dd_verifier.metrics.core import DataRoom, MetricResult, NotEnoughData
from dd_verifier.metrics.customers import customer_status
from dd_verifier.metrics.definitions import compute
from dd_verifier.metrics.tolerance import (
    CONTRADICTED,
    FACTUAL_OUT_OF_SCOPE,
    NOT_ENOUGH_DATA,
    SUPPORTED,
    UNVERIFIABLE,
    verdict_for_metric,
)

NUMERIC_TYPES = frozenset({"level", "growth", "share", "count"})


@dataclass(frozen=True)
class ClaimCheck:
    """Итог проверки утверждения."""

    verdict: str
    actual: float | None
    delta: float | None
    reason: str
    result: MetricResult | None = None


def evaluate_claim(dr: DataRoom, claim: dict, variant: str = "standard") -> ClaimCheck:
    """Вердикт для утверждения в формате gold/claims.jsonl.

    Нужные поля: claim_type, metric_id, claimed_value, qualifier, period_start, period_end.
    `variant` — какое определение метрики применить (эталон — всегда standard).
    """
    ctype = claim["claim_type"]
    if ctype == "factual":
        return ClaimCheck(FACTUAL_OUT_OF_SCOPE, None, None, "фактологическое: ручная проверка")
    if ctype == "forecast":
        return ClaimCheck(UNVERIFIABLE, None, None, "прогноз или мнение")
    if ctype == "out_of_ontology":
        return ClaimCheck(NOT_ENOUGH_DATA, None, None, "метрика вне онтологии")
    if ctype == "customer_mention":
        status = customer_status(dr, str(claim["claimed_value"]), claim["period_end"])
        verdict = SUPPORTED if status == "active" else CONTRADICTED
        return ClaimCheck(verdict, None, None, f"клиент: {status}")
    if ctype not in NUMERIC_TYPES:
        raise ValueError(f"неизвестный claim_type {ctype!r}")

    metric_id = claim["metric_id"]
    as_growth = ctype == "growth" and metric_id != "arr_growth"
    try:
        res = compute(
            dr,
            metric_id,
            claim["period_end"],
            claim.get("period_start"),
            variant,
            as_growth=as_growth,
            with_sql=False,
        )
    except NotEnoughData as exc:
        return ClaimCheck(NOT_ENOUGH_DATA, None, None, str(exc))
    v = verdict_for_metric(
        metric_id,
        float(claim["claimed_value"]),
        res.value,
        claim.get("qualifier"),
        as_growth=as_growth,
    )
    return ClaimCheck(v.verdict, res.value, v.delta, v.reason, res)
