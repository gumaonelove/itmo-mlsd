"""Правила допусков P5: пороги из ontology.yaml, квантификаторы, направление, граничные случаи."""

from __future__ import annotations

import pytest

from dd_verifier.metrics import Tolerance, tolerance_for, verdict_for_metric, verdict_from_delta


def test_thresholds_read_from_ontology():
    assert tolerance_for("arr") == Tolerance("relative", 0.05, 0.25)
    assert tolerance_for("logo_churn") == Tolerance("pp", 0.3, 1.0)
    assert tolerance_for("nrr") == Tolerance("pp", 2.0, 5.0)
    assert tolerance_for("top10_concentration") == Tolerance("pp", 2.0, 5.0)
    assert tolerance_for(kind="growth") == Tolerance("relative", 0.05, 0.25)


@pytest.mark.parametrize(
    ("claimed", "verdict"),
    [
        (105.0, "SUPPORTED"),  # ровно нижний порог
        (104.0, "SUPPORTED"),
        (96.0, "SUPPORTED"),
        (106.0, "EXAGGERATED"),
        (125.0, "EXAGGERATED"),  # ровно верхний порог
        (126.0, "CONTRADICTED"),
        (94.0, "CONTRADICTED"),  # занижение не в пользу компании больше нижнего порога
    ],
)
def test_level_relative(claimed, verdict):
    assert verdict_from_delta("level", claimed, 100.0).verdict == verdict


def test_growth_examples_from_design_doc():
    # «×3» и «+200%» против факта ×2.6 → δ = +15% → EXAGGERATED
    res = verdict_from_delta("growth", 3.0, 2.6)
    assert res.verdict == "EXAGGERATED"
    assert res.delta == pytest.approx(0.1538, abs=1e-4)
    # противоположное направление: заявлен рост, по данным падение
    assert verdict_from_delta("growth", 1.15, 0.92).verdict == "CONTRADICTED"
    # «more than doubled» при ×2.6 выполнено
    assert verdict_from_delta("growth", 2.0, 2.6, "more_than").verdict == "SUPPORTED"


def test_qualifiers_on_counts():
    assert verdict_from_delta("count", 120, 127, "more_than").verdict == "SUPPORTED"  # «120+»
    assert (
        verdict_from_delta("count", 130, 127, "more_than").verdict == "SUPPORTED"
    )  # δ от N = 2.4%
    assert verdict_from_delta("count", 140, 127, "more_than").verdict == "EXAGGERATED"  # δ = 10%
    assert verdict_from_delta("level", 1.9e6, 1.82e6, "about").verdict == "SUPPORTED"  # +4.4%
    assert verdict_from_delta("level", 2.0e6, 1.82e6, "about").verdict == "EXAGGERATED"


@pytest.mark.parametrize(
    ("claimed", "qualifier", "verdict"),
    [
        (0.02, "less_than", "CONTRADICTED"),  # «< 2%» при факте 4.5%: −2.5 п.п. в пользу компании
        (0.04, None, "EXAGGERATED"),  # −0.5 п.п.: между 0.3 и 1.0
        (0.047, None, "SUPPORTED"),  # +0.2 п.п.
        (0.05, None, "CONTRADICTED"),  # +0.5 п.п. не в пользу компании
        (0.05, "less_than", "SUPPORTED"),  # «below 5%» выполнено
    ],
)
def test_churn_direction_down(claimed, qualifier, verdict):
    res = verdict_for_metric("logo_churn", claimed, 0.045, qualifier)
    assert res.verdict == verdict
    assert res.delta_unit == "pp"


@pytest.mark.parametrize(
    ("claimed", "verdict"), [(1.13, "SUPPORTED"), (1.15, "EXAGGERATED"), (1.20, "CONTRADICTED")]
)
def test_nrr_pp(claimed, verdict):
    assert verdict_for_metric("nrr", claimed, 1.12).verdict == verdict


def test_growth_of_level_metric_uses_growth_thresholds():
    res = verdict_for_metric("active_customers", 2.0, 2.49, "more_than", as_growth=True)
    assert res.verdict == "SUPPORTED"
    assert (
        verdict_for_metric("active_customers", 1.3, 0.95, as_growth=True).verdict == "CONTRADICTED"
    )
