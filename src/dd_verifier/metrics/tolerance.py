"""Правила допусков P5: заявленное значение против рассчитанного → вердикт.

Пороги читаются из configs/ontology.yaml (`tolerance_defaults` и `tolerance` у метрики).

- уровни и число клиентов: δ = (заявлено − факт) / факт;
- рост: обе стороны — множители, δ считается по множителю;
- доли: δ = (заявлено − факт) × 100, в процентных пунктах.

|δ| ≤ lower → SUPPORTED; отклонение в пользу компании между lower и upper → EXAGGERATED;
в пользу компании > upper, не в пользу компании > lower или противоположное направление роста →
CONTRADICTED. Квантификаторы: more_than выполнен при факте ≥ N, less_than — при факте ≤ N
(иначе δ считается от N), about — как точное значение с допуском lower.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

ONTOLOGY_PATH = Path(__file__).resolve().parents[3] / "configs" / "ontology.yaml"
EPS = 1e-9

SUPPORTED, EXAGGERATED, CONTRADICTED = "SUPPORTED", "EXAGGERATED", "CONTRADICTED"
NOT_ENOUGH_DATA, UNVERIFIABLE = "NOT_ENOUGH_DATA", "UNVERIFIABLE"
FACTUAL_OUT_OF_SCOPE = "FACTUAL_OUT_OF_SCOPE"
VERDICTS = (SUPPORTED, EXAGGERATED, CONTRADICTED, NOT_ENOUGH_DATA, UNVERIFIABLE)


@dataclass(frozen=True)
class Tolerance:
    """Пороги допуска: mode = relative (доля) или pp (процентные пункты)."""

    mode: str
    lower: float
    upper: float


@dataclass(frozen=True)
class VerdictResult:
    """Вердикт по числу, отклонение и короткое объяснение."""

    verdict: str
    delta: float
    delta_unit: str
    reason: str


@cache
def load_ontology(path: str | None = None) -> dict:
    """Онтология метрик (кэшируется)."""
    return yaml.safe_load(Path(path or ONTOLOGY_PATH).read_text(encoding="utf-8"))


def metric_meta(metric_id: str) -> dict:
    """Описание метрики из онтологии."""
    for m in load_ontology()["metrics"]:
        if m["id"] == metric_id:
            return m
    raise KeyError(f"метрика {metric_id!r} вне онтологии")


def tolerance_for(metric_id: str | None = None, kind: str | None = None) -> Tolerance:
    """Пороги метрики (если задана) или типа метрики по умолчанию."""
    if metric_id is not None:
        tol = metric_meta(metric_id).get("tolerance")
        if tol:
            return Tolerance(tol["mode"], float(tol["lower"]), float(tol["upper"]))
        kind = kind or metric_meta(metric_id)["kind"]
    tol = load_ontology()["tolerance_defaults"][kind]
    return Tolerance(tol["mode"], float(tol["lower"]), float(tol["upper"]))


def _delta(claimed: float, actual: float, mode: str) -> float:
    if mode == "pp":
        return (claimed - actual) * 100.0
    if abs(actual) < EPS:
        return float("inf") if claimed > actual else float("-inf") if claimed < actual else 0.0
    return (claimed - actual) / abs(actual)


def verdict_from_delta(
    metric_kind: str,
    claimed: float,
    actual: float,
    qualifier: str | None = None,
    direction: str = "up",
    tolerance: Tolerance | None = None,
) -> VerdictResult:
    """Вердикт P5 для числового утверждения.

    metric_kind: level | count | growth | share; direction: up (больше — лучше для компании)
    или down (меньше — лучше, например churn); qualifier: exact | about | more_than | less_than.
    """
    tol = tolerance or tolerance_for(kind=metric_kind)
    qualifier = qualifier or "exact"
    unit = "pp" if tol.mode == "pp" else "rel"
    delta = _delta(claimed, actual, tol.mode)

    if qualifier == "more_than" and actual >= claimed - EPS:
        return VerdictResult(
            SUPPORTED, delta, unit, f"«более {claimed:g}» выполнено: факт {actual:.4g}"
        )
    if qualifier == "less_than" and actual <= claimed + EPS:
        return VerdictResult(
            SUPPORTED, delta, unit, f"«менее {claimed:g}» выполнено: факт {actual:.4g}"
        )
    if metric_kind == "growth" and (claimed - 1.0) * (actual - 1.0) < 0:
        return VerdictResult(
            CONTRADICTED,
            delta,
            unit,
            f"противоположное направление: заявлено ×{claimed:g}, факт ×{actual:.3f}",
        )

    size = abs(delta)
    favourable = delta > 0 if direction == "up" else delta < 0
    shown = f"δ = {delta:+.2f} п.п." if unit == "pp" else f"δ = {delta:+.1%}"
    if size <= tol.lower + EPS:
        return VerdictResult(
            SUPPORTED, delta, unit, f"{shown}: в пределах нижнего порога {tol.lower:g}"
        )
    if favourable and size <= tol.upper + EPS:
        return VerdictResult(
            EXAGGERATED,
            delta,
            unit,
            f"{shown}: завышение между порогами {tol.lower:g} и {tol.upper:g}",
        )
    if favourable:
        return VerdictResult(
            CONTRADICTED, delta, unit, f"{shown}: завышение выше верхнего порога {tol.upper:g}"
        )
    return VerdictResult(
        CONTRADICTED, delta, unit, f"{shown}: расхождение не в пользу компании выше нижнего порога"
    )


def verdict_for_metric(
    metric_id: str,
    claimed: float,
    actual: float,
    qualifier: str | None = None,
    *,
    as_growth: bool = False,
) -> VerdictResult:
    """Вердикт с порогами и направлением метрики из онтологии.

    `as_growth=True` — утверждение о росте уровня метрики (например, числа клиентов):
    используются пороги роста и направление метрики.
    """
    meta = metric_meta(metric_id)
    if as_growth and meta["kind"] != "growth":
        return verdict_from_delta("growth", claimed, actual, qualifier, meta["direction"])
    tol = tolerance_for(metric_id)
    return verdict_from_delta(meta["kind"], claimed, actual, qualifier, meta["direction"], tol)
