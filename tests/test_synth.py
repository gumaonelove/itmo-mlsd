"""Инварианты генератора: детерминизм, цитаты, эталон = библиотека, покрытие, Pantrio, D09.

Holdout-компании генерируются, но тесты читают только holdout_summary.json (агрегаты).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from dd_verifier.data import (
    load_claims,
    load_data_room,
    load_documents,
    split_blocks,
    split_pages,
    split_slides,
)
from dd_verifier.metrics import compute, evaluate_claim
from dd_verifier.synth.claims import favourable_to_company
from dd_verifier.synth.generator import (
    CASE_IDS,
    EXCLUSIVE,
    assign_cases,
    company_configs,
    generate,
    load_yaml,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synth.yaml"
FIELDS = {
    "claim_id",
    "company_id",
    "family_id",
    "doc",
    "location",
    "quote",
    "claim_type",
    "metric_id",
    "claimed_value",
    "unit",
    "period_start",
    "period_end",
    "qualifier",
    "case_id",
    "true_value_standard",
    "true_value_company_def",
    "delta",
    "delta_p5",
    "gold_verdict",
    "true_value_observed",
    "noise_sensitive",
}


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    generate(CONFIG, out)
    return out


def dev_dirs(root: Path) -> list[Path]:
    return sorted(p for p in (root / "dev").iterdir() if p.is_dir())


def records(company_dir: Path) -> list[dict]:
    """Строки claims.jsonl как словари (None сохраняются, в отличие от DataFrame)."""
    lines = (company_dir / "gold" / "claims.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines]


def test_layout_and_manifest(synth_root):
    cfg = load_yaml(CONFIG)
    assert len(cfg["companies"]) == 15
    splits = Counter(c["split"] for c in cfg["companies"])
    assert splits == {"dev": 5, "holdout": 10}
    for d in dev_dirs(synth_root):
        manifest = json.loads((d / "manifest.json").read_text())
        for rel, digest in manifest["files"].items():
            assert hashlib.sha256((d / rel).read_bytes()).hexdigest() == digest
        names = set(manifest["files"])
        assert {
            "customers.csv",
            "invoices.csv",
            "fx_rates.csv",
            "cohorts.xlsx",
            "documents/deck.md",
            "documents/cfo_memo.md",
            "gold/claims.jsonl",
            "gold/truth_monthly.csv",
        } <= names
        assert 1 <= sum(n.startswith("documents/press_release_") for n in names) <= 3
        flagged = [c["claim_id"] for c in records(d) if c["noise_sensitive"]]
        assert manifest["dirty_vs_clean"]["verdict_mismatches"] == flagged


def test_quotes_are_exact_substrings_at_location(synth_root):
    for d in dev_dirs(synth_root):
        docs = load_documents(d)
        claims = load_claims(d)
        assert set(claims.columns) >= FIELDS
        assert 15 <= len(claims) <= 45
        for c in claims.itertuples():
            text = docs[c.doc]
            assert c.quote in text
            if c.doc == "deck.md":
                slide = {s["slide"]: s for s in split_slides(text)}[c.location["slide"]]
                assert c.quote in slide["text"] and not slide["image_only"]
            else:
                assert c.quote in split_blocks(text)[c.location["paragraph"] - 1]


def test_gold_reproduced_by_library_on_disk_data(synth_root):
    """Эталонная библиотека на файлах «как пришли» воспроизводит true_value_observed (G1 = 100%).

    Вердикт совпадает с эталоном у всех утверждений, кроме помеченных noise_sensitive:
    у них шум в данных сам по себе меняет вердикт, их отчитываем отдельно.
    """
    for d in dev_dirs(synth_root):
        dr = load_data_room(d)
        for c in records(d):
            check = evaluate_claim(dr, c)
            if c["true_value_observed"] is not None:
                assert check.actual == pytest.approx(c["true_value_observed"], rel=1e-6, abs=1e-6)
            if not c["noise_sensitive"]:
                assert check.verdict == c["gold_verdict"], c["claim_id"]


def test_delta_p5_is_zero_when_qualifier_met(synth_root):
    for d in dev_dirs(synth_root):
        for c in records(d):
            if c["delta"] is None:
                assert c["delta_p5"] is None
                continue
            actual, claimed = c["true_value_standard"], c["claimed_value"]
            met = (c["qualifier"] == "more_than" and actual >= claimed) or (
                c["qualifier"] == "less_than" and actual <= claimed
            )
            assert c["delta_p5"] == (0.0 if met else c["delta"]), c["claim_id"]


def test_d09_stale_value_favours_company(synth_root):
    """D09: устаревшее значение было лучше текущего — иначе это не «удобная» дата."""
    seen = 0
    for d in dev_dirs(synth_root):
        for c in records(d):
            if c["case_id"] == "D09":
                seen += 1
                assert favourable_to_company(c["metric_id"], c["delta"]), c["claim_id"]
    assert seen > 0


def test_case_verdicts_follow_catalog(synth_root):
    catalog = {c["id"]: c for c in load_yaml(CONFIG.parent / "discrepancy_catalog.yaml")["cases"]}
    for d in dev_dirs(synth_root):
        for c in records(d):
            allowed = catalog[c["case_id"]]["allowed_verdicts"] if c["case_id"] else ["SUPPORTED"]
            assert c["gold_verdict"] in allowed


def test_holdout_coverage_from_summary(synth_root):
    summary = json.loads((synth_root / "holdout_summary.json").read_text())
    assert summary["n_companies"] == 10
    assert all(summary["case_company_counts"].get(case, 0) >= 2 for case in CASE_IDS)
    fam = summary["family_verdict_counts"]
    assert 0 < fam["red_flag_families"] <= fam["non_supported_families_5cls"]
    assert summary["dirty_vs_clean_verdict_mismatches"] >= 0  # отчитываем, не гейтим


def test_assignment_constraints():
    cfg = load_yaml(CONFIG)
    companies = company_configs(cfg)
    first, second = assign_cases(cfg, companies), assign_cases(cfg, companies)
    assert first == second
    for c in companies:
        cases = set(first[c["id"]])
        assert all(len(cases & group) <= 1 for group in EXCLUSIVE)
        if "D05" in cases:
            assert c["reseller_share"] >= 0.35
    dev = Counter(x for c in companies if c["split"] == "dev" for x in set(first[c["id"]]))
    assert all(dev[case] >= 1 for case in CASE_IDS)


def test_determinism_same_seed(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(CONFIG, a, only=["pantrio", "routelane"])
    generate(CONFIG, b, only=["pantrio", "routelane"])
    for cid in ("pantrio", "routelane"):
        ma = json.loads((a / "dev" / cid / "manifest.json").read_text())
        mb = json.loads((b / "dev" / cid / "manifest.json").read_text())
        assert ma["files"] == mb["files"]


def test_pantrio_reproduces_design_doc_example(synth_root):
    d = synth_root / "dev" / "pantrio"
    dr = load_data_room(d)

    def val(metric, end, start=None, variant="standard"):
        return compute(dr, metric, end, start, variant, with_sql=False).value

    assert val("arr", "2024-12") == pytest.approx(0.70e6, rel=0.03)
    assert val("arr", "2025-12") == pytest.approx(1.82e6, rel=0.03)
    assert val("arr", "2024-12", variant="incl_one_time") == pytest.approx(0.75e6, rel=0.03)
    assert val("arr", "2025-12", variant="incl_one_time") == pytest.approx(2.25e6, rel=0.03)
    assert val("arr_growth", "2025-12", "2024-12") == pytest.approx(2.6, rel=0.03)
    assert val("arr_growth", "2025-12", "2024-12", "incl_one_time") == pytest.approx(3.0, rel=0.03)
    assert val("active_customers", "2025-12") == 127
    assert val("logo_churn", "2025-12", "2025-01") == pytest.approx(0.045, abs=0.003)
    assert val("logo_churn", "2025-12", "2025-01", "excl_pilots_and_small") < 0.02

    docs = load_documents(d)
    slide7 = {s["slide"]: s["text"] for s in split_slides(docs["deck.md"])}[7]
    assert "ARR grew 3x in 2025" in slide7
    page3 = {p["page"]: p["text"] for p in split_pages(docs["cfo_memo.md"])}[3]
    assert "ARR includes annualized implementation fees" in page3

    claims = load_claims(d).set_index("quote")
    growth = claims.loc["ARR grew 3x in 2025"]
    assert (growth["case_id"], growth["gold_verdict"]) == ("D01", "EXAGGERATED")
    assert growth["delta"] == pytest.approx(0.15, abs=0.01)
    customers = claims.loc["120+ restaurant groups use Pantrio every day"]
    assert (customers["case_id"], customers["gold_verdict"]) == ("H01", "SUPPORTED")
    churn = claims.loc["Logo churn < 2% monthly"]
    assert (churn["case_id"], churn["qualifier"], churn["gold_verdict"]) == (
        "D07",
        "less_than",
        "CONTRADICTED",
    )
