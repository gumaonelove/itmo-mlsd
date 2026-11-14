"""Оркестрация генератора синтетического data room (15 компаний, dev + holdout).

Шаги для компании: симуляция → калибровка масштаба → посадка расхождений в данные →
утверждения и эталон (библиотека метрик на чистых данных) → документы → шум → запись файлов
и manifest.json. После всех компаний — проверка покрытия holdout и holdout_summary.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook

from dd_verifier.data.loaders import drop_exact_duplicates, read_tables
from dd_verifier.data.quality import run_quality_checks
from dd_verifier.metrics import DataRoom, compute, evaluate_claim
from dd_verifier.metrics.core import NotEnoughData, add_months, month_end, month_range
from dd_verifier.metrics.customers import normalize_name
from dd_verifier.synth import claims as cl
from dd_verifier.synth import plant
from dd_verifier.synth.documents import (
    SLIDE_OF,
    render_deck,
    render_memo,
    render_press_release,
    render_sentence,
)
from dd_verifier.synth.names import legal_name, name_pool
from dd_verifier.synth.noise import add_noise
from dd_verifier.synth.simulate import Simulator, renumber_invoices

GENERATOR_VERSION = "0.2.0"
REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_IDS = [f"D{i:02d}" for i in range(1, 12)] + [
    "H01",
    "H02",
    "H03",
    "H04",
    "N01",
    "N02",
    "U01",
    "F01",
]
REPEATABLE = {"H01", "H02", "H03", "N01", "U01", "F01"}
EXCLUSIVE = [{"D01", "D02", "D05"}, {"D04", "D11"}]
RESELLER_MIN_SHARE = 0.35
FIXED_TS = (2026, 1, 1, 0, 0, 0)
CHECKED_TYPES = {"level", "growth", "share", "count", "customer_mention"}


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_catalog(path: str | Path | None = None) -> dict[str, dict]:
    """Каталог расхождений: {case_id: запись}."""
    data = load_yaml(path or REPO_ROOT / "configs" / "discrepancy_catalog.yaml")
    return {c["id"]: c for c in data["cases"]}


def company_configs(cfg: dict) -> list[dict]:
    """Конфиги компаний с подставленными значениями по умолчанию."""
    out = []
    for c in cfg["companies"]:
        merged = {**cfg["defaults"], **c}
        merged["expansion"] = {**cfg["defaults"]["expansion"], **c.get("expansion", {})}
        out.append(merged)
    return out


# --- распределение случаев каталога по компаниям ---------------------------------------------
def feasible(case: str, company: dict, assigned: list[str]) -> bool:
    """Можно ли посадить случай в компанию с уже назначенными случаями."""
    if case == "D05" and company.get("reseller_share", 0.0) < RESELLER_MIN_SHARE:
        return False
    if case not in REPEATABLE and case in assigned:
        return False
    return not any(case in g and set(assigned) & (g - {case}) for g in EXCLUSIVE)


def assign_cases(cfg: dict, companies: list[dict]) -> dict[str, list[str]]:
    """Детерминированное распределение случаев: покрытие id ≥ coverage_min, затем добивка."""
    rng = np.random.default_rng(cfg["assignment_seed"])
    result = {c["id"]: list(c.get("cases", [])) for c in companies}
    for split, k in cfg["coverage_min"].items():
        comps = [c for c in companies if c["split"] == split]
        free = [c for c in comps if not c.get("cases")]
        have = Counter(case for c in comps for case in set(result[c["id"]]))
        slots = [case for case in CASE_IDS for _ in range(max(0, k - have[case]))]
        # сначала самые ограниченные случаи (D05 — только компании с реселлерами)
        n_ok = {case: sum(feasible(case, c, []) for c in free) for case in CASE_IDS}
        noise = rng.random(len(slots))
        for j in sorted(range(len(slots)), key=lambda i: (n_ok[slots[i]], noise[i])):
            case = slots[j]
            options = [c for c in free if feasible(case, c, result[c["id"]])]
            if not options:
                raise ValueError(f"не удалось назначить {case} в сплите {split}")
            ties = rng.random(len(options))
            best = min(range(len(options)), key=lambda i: (len(result[options[i]["id"]]), ties[i]))
            result[options[best]["id"]].append(case)
        lo, hi = cfg["cases_per_company"][split]
        for c in free:
            target = int(rng.integers(lo, hi + 1))
            while len(result[c["id"]]) < target:
                options = [x for x in CASE_IDS if feasible(x, c, result[c["id"]])]
                result[c["id"]].append(options[int(rng.integers(len(options)))])
    return result


# --- одна компания ------------------------------------------------------------------------------
@dataclass
class CompanyResult:
    """Всё, что генератор пишет для компании."""

    cfg: dict
    clean: dict
    dirty: dict
    docs: dict[str, str]
    claims: list[dict]
    truth: pd.DataFrame
    cohorts: dict[str, pd.DataFrame]
    meta: dict = field(default_factory=dict)


def _plant_data(
    t: dict, ccfg: dict, cases: list[str], rng: np.random.Generator, end: str, names
) -> tuple[dict, dict]:
    """Калибровка масштаба и посадка расхождений в данные; возвращает таблицы и журнал."""
    log: dict = {}
    if "calibration" in ccfg:
        t, log["calibration"] = plant.calibrate_two_point(t, ccfg["calibration"], end)
    else:
        t, log["scale"] = plant.scale_to_arr(t, end, ccfg["arr_end"])
    y0 = add_months(end, -12)
    if "D05" in cases:
        metric = "arr" if rng.random() < 0.6 else "revenue"
        t, log["D05"] = plant.calibrate_reseller_fee(t, rng.uniform(0.08, 0.14), metric, end)
    if "D09" in cases:  # до D11/D02/D04: их размер считается от уже сниженного MRR
        # отдельный поток случайных чисел: посадка D09 не сдвигает параметры остальных случаев
        rng09 = np.random.default_rng([ccfg["seed"], 9])
        t, log["D09"] = plant.plant_recent_downgrade(t, rng09, rng09.uniform(0.03, 0.045), end)
    avoid: set[str] = set()
    if "D11" in cases:
        t, log["D11"] = plant.plant_seasonal_dip(t, rng, rng.uniform(0.05, 0.10))
        w0 = log["D11"]["window"][0]
        avoid = {add_months(w0, k) for k in (1, 2, 3)}
    if "D02" in cases:
        t, log["D02"] = plant.plant_peak_month(t, rng, rng.uniform(0.10, 0.18), end, avoid)
    if "D04" in cases:
        t, log["D04"] = plant.plant_trough(t, rng, rng.uniform(0.10, 0.18), y0)
    if "D01" in cases and "calibration" not in ccfg:
        t, log["D01"] = plant.calibrate_one_time(t, rng.uniform(0.10, 0.18), end)
    if "D03" in cases:
        t, log["D03"] = plant.plant_pilots(t, rng, rng.uniform(0.10, 0.18), end, names)
    t["invoices"] = renumber_invoices(t["invoices"], ccfg["invoice_prefix"])
    return t, log


BLOCKS = {
    "D01": {"arr", "mrr", "arr_growth", "acv"},
    "D02": {"arr", "mrr", "arr_growth", "acv"},
    "D05": {"arr", "mrr", "arr_growth", "acv", "revenue"},
    "D03": {"active_customers"},
    "D06": {"nrr"},
    "D07": {"logo_churn"},
}


def _make_specs(f: cl.ClaimFactory, cases: list[str], log: dict, fixed: bool) -> list[cl.ClaimSpec]:
    """Утверждения: сначала случаи каталога, затем плановые корректные."""
    for case in cases:
        f.blocked |= BLOCKS.get(case, set())
    builders = {
        "D01": lambda: cl.case_d01(f, fixed),
        "D02": lambda: cl.case_d02(f),
        "D03": lambda: cl.case_d03(f),
        "D04": lambda: cl.case_d04(f),
        "D05": lambda: cl.case_d05(f, log.get("D05", {}).get("metric", "arr")),
        "D06": lambda: cl.case_d06(f),
        "D07": lambda: cl.case_d07(f, fixed),
        "D08": lambda: cl.case_d08(f),
        "D09": lambda: cl.case_d09(f),
        "D10": lambda: cl.case_d10(f),
        "D11": lambda: cl.case_d11(f, log.get("D11", {}).get("window", [])),
        "H01": lambda: cl.case_h01(f, fixed),
        "H02": lambda: cl.case_h02(f),
        "H03": lambda: cl.case_h03(f),
        "H04": lambda: cl.case_h04(f),
        "N01": lambda: cl.case_n01(f),
        "N02": lambda: cl.case_n02(f),
        "U01": lambda: cl.case_u01(f),
        "F01": lambda: cl.case_f01(f),
    }
    # D-случаи раньше, чтобы занять свои метрики; затем H04 (нужна свободная метрика)
    order = sorted(cases, key=lambda c: (c[0] != "D", c != "H04", c))
    specs, seen = [], set()
    for case in order:
        spec = builders[case]()
        key = (spec.metric_id, spec.claim_type, spec.period_start, spec.period_end)
        if case in seen and spec.metric_id and key in f.used:
            continue  # повтор того же случая на той же метрике не добавляет разнообразия
        seen.add(case)
        specs.append(f.take(spec))
    lo, hi = f.cfg["plain_claims"]
    specs += cl.plain_claims(f, int(f.rng.integers(lo, hi + 1)))
    for s in specs:  # где окажется утверждение: дек, пресс-релиз или оба (семейство)
        if s.key in {"mention", "factual", "opinion"}:
            continue
        u = f.rng.random()
        s.venues = (
            ("deck", "press") if u < f.cfg["family_rate"] else ("press",) if u > 0.9 else ("deck",)
        )
        if s.template and "deck" not in s.venues:  # зафиксированные цитаты Pantrio — в деке
            s.venues = ("deck", *s.venues)
    return specs


def _render(
    ccfg: dict,
    specs: list[cl.ClaimSpec],
    gold: list[dict],
    rng: np.random.Generator,
    first: str,
    end: str,
    memo: dict,
) -> tuple[dict[str, str], list[dict]]:
    """Тексты документов и записи claims.jsonl с точными цитатами и местом в документе."""
    company = {k: ccfg[k] for k in ("name", "noun", "category", "city", "smb_label")}
    company["has_resellers"] = ccfg.get("reseller_share", 0) > 0
    company["end"] = end
    slides: dict[int, list[str]] = defaultdict(list)
    occurrences = []  # (spec_index, doc, location, quote)
    press: list[tuple[int, str]] = []
    for i, s in enumerate(specs):
        for venue in s.venues:
            text = render_sentence(s, venue, company, rng)
            if venue == "deck":
                slide = SLIDE_OF[s.key]
                slides[slide].append(text)
                occurrences.append((i, "deck.md", {"slide": slide}, text))
            else:
                press.append((i, text))
    deck = render_deck(company, slides)
    docs = {"deck.md": deck}
    n_pr = max(1, min(ccfg["press_releases"], len(press)))
    for k in range(1, n_pr + 1):
        mine = press[k - 1 :: n_pr]
        paragraphs, where = [], []
        j = 0
        while j < len(mine):
            size = 1 if rng.random() < 0.6 or j + 1 == len(mine) else 2
            group = mine[j : j + size]
            paragraphs.append(" ".join(t for _, t in group))
            where.append(group)
            j += size
        date = f"January {12 + 3 * k}, 2026"
        blocks = render_press_release(company, k, paragraphs, date)
        name = f"press_release_{k:02d}.md"
        docs[name] = "\n\n".join(blocks) + "\n"
        for p_idx, group in enumerate(where):
            for i, text in group:
                occurrences.append((i, name, {"paragraph": p_idx + 3}, text))
    docs["cfo_memo.md"] = render_memo(company, memo, first, end)

    records = []
    order = {"deck.md": 0}
    for n, (i, doc, loc, quote) in enumerate(
        sorted(occurrences, key=lambda o: (order.get(o[1], 1), o[1], next(iter(o[2].values())))),
        start=1,
    ):
        s = specs[i]
        if quote not in docs[doc]:
            raise AssertionError(f"цитата не найдена в {doc}: {quote}")
        records.append(
            {
                "claim_id": f"{ccfg['id']}-{n:03d}",
                "company_id": ccfg["id"],
                "family_id": f"{ccfg['id']}-F{i + 1:02d}",
                "doc": doc,
                "location": loc,
                "quote": quote,
                "claim_type": s.claim_type,
                "metric_id": s.metric_id,
                "claimed_value": s.claimed_value,
                "unit": s.unit,
                "period_start": s.period_start,
                "period_end": s.period_end,
                "qualifier": s.qualifier,
                "case_id": s.case_id,
                "company_variant": s.variant or ("stale_period" if s.case_id == "D09" else None),
                **gold[i],
            }
        )
    return docs, records


def truth_monthly(dr: DataRoom, months: list[str]) -> pd.DataFrame:
    """Истинные помесячные метрики по стандартным определениям."""
    first = months[0]

    def safe(metric: str, m: str, start: str | None = None) -> float:
        try:
            return compute(dr, metric, m, start, with_sql=False).value
        except NotEnoughData:
            return float("nan")

    c = dr.customers
    rows = []
    for m in months:
        live = c["go_live_date"].dt.strftime("%Y-%m") == m
        churned = c["churn_date"].dt.strftime("%Y-%m") == m
        prev_active = set(dr.active_ids(add_months(m, -1))) if m > first else set()
        rows.append(
            {
                "month": m,
                "mrr": safe("mrr", m),
                "arr": safe("arr", m),
                "revenue": safe("revenue", m, m),
                "active_customers": safe("active_customers", m),
                "new_customers": int((live & ~c["is_pilot"]).sum()),
                "churned_customers": int((churned & c["customer_id"].isin(prev_active)).sum()),
                "logo_churn": safe("logo_churn", m, m) if m > first else float("nan"),
                "nrr_12m": safe("nrr", m),
                "grr_12m": safe("grr", m),
                "acv": safe("acv", m),
                "top10_concentration_12m": safe("top10_concentration", m),
            }
        )
    return pd.DataFrame(rows).round(6)


def cohort_table(dr: DataRoom, survivors_only: bool, end: str) -> dict[str, pd.DataFrame]:
    """Когортная таблица «как готовит компания»: MRR по месяцу go-live."""
    first, last = dr.coverage
    mrr = dr.customer_mrr.loc[:, lambda df: (df.columns >= first) & (df.columns <= last)]
    c = dr.customers.set_index("customer_id")
    if survivors_only:
        churned = c.index[c["churn_date"].notna() & (c["churn_date"] <= month_end(end))]
        mrr = mrr.drop(index=mrr.index.intersection(churned))
    cohort = c["go_live_date"].reindex(mrr.index).dt.strftime("%Y-%m")
    table = mrr.groupby(cohort).sum().round(2)
    table.index.name = "cohort_go_live_month"
    note = (
        "Cohorts include only customers retained at the end of the period "
        "(churned customers removed)."
        if survivors_only
        else "All customers that went live in the cohort month, including those who later churned."
    )
    notes = pd.DataFrame(
        {"note": ["Company-prepared cohort table: subscription MRR in USD by go-live month.", note]}
    )
    return {"cohort_mrr": table.reset_index(), "notes": notes}


def build_company(ccfg: dict, gcfg: dict, catalog: dict, cases: list[str]) -> CompanyResult:
    """Сгенерировать одну компанию (без записи на диск)."""
    end = gcfg["end_month"]
    rng = np.random.default_rng(ccfg["seed"])
    sim = Simulator(ccfg, end, rng)
    customers, invoices, fx = sim.run()
    taken = set(customers["customer_name"].map(normalize_name))
    extra_names = (
        legal_name(n, "US", rng)
        for n in name_pool(ccfg["industry_key"], 2000, rng)
        if normalize_name(n) not in taken
    )
    try:
        t, log = _plant_data(
            {"customers": customers, "invoices": invoices, "fx_rates": fx},
            ccfg,
            cases,
            rng,
            end,
            extra_names,
        )
    except plant.PlantingError as exc:
        raise plant.PlantingError(f"{ccfg['id']}: {exc}") from exc
    dr = DataRoom(t["customers"], t["invoices"], t["fx_rates"])

    factory = cl.ClaimFactory(ccfg, dr, end, rng, catalog)
    specs = _make_specs(factory, cases, log, fixed=bool(ccfg.get("cases")))
    gold = [cl.gold_fields(factory, s) for s in specs]
    first = dr.coverage[0]
    docs, records = _render(ccfg, specs, gold, rng, first, end, factory.memo)
    if not gcfg["claims_per_company"][0] <= len(records) <= gcfg["claims_per_company"][1]:
        raise AssertionError(f"{ccfg['id']}: {len(records)} утверждений вне диапазона")

    dirty_c, dirty_i, noise_log = add_noise(
        t["customers"], t["invoices"], gcfg["noise"], end, ccfg["invoice_prefix"], rng
    )
    dirty = {"customers": dirty_c, "invoices": dirty_i, "fx_rates": fx}
    # Тот же путь, что у системы: файлы «как пришли» минус точные дубли (load_data_room).
    dr_dirty = DataRoom(dirty_c, drop_exact_duplicates(dirty_i)[0], fx)
    mismatches = []
    for r in records:
        observed = evaluate_claim(dr_dirty, r) if r["claim_type"] in CHECKED_TYPES else None
        r["true_value_observed"] = cl.round6(observed.actual) if observed else None
        r["noise_sensitive"] = bool(observed and observed.verdict != r["gold_verdict"])
        if r["noise_sensitive"]:
            mismatches.append(r["claim_id"])
    months = month_range(first, end)
    meta = {
        "plantings": log,
        "noise": noise_log,
        "cases": cases,
        "dirty_vs_clean": {
            "claims_checked": sum(r["claim_type"] in CHECKED_TYPES for r in records),
            "verdict_mismatches": mismatches,
        },
    }
    return CompanyResult(
        cfg=ccfg,
        clean=t,
        dirty=dirty,
        docs=docs,
        claims=records,
        truth=truth_monthly(dr, months),
        cohorts=cohort_table(dr, "D06" in cases, end),
        meta=meta,
    )


# --- запись на диск --------------------------------------------------------------------------
def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, date_format="%Y-%m-%d", lineterminator="\n")


def write_xlsx(sheets: dict[str, pd.DataFrame], path: Path) -> None:
    """Детерминированный xlsx: фиксированные метаданные и время в zip-архиве."""
    wb = Workbook()
    wb.remove(wb.active)
    for name, df in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(list(map(str, df.columns)))
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and np.isnan(v)) else v for v in row])
    wb.properties.creator = "dd-verifier synth"
    wb.properties.created = wb.properties.modified = pd.Timestamp("2026-01-01").to_pydatetime()
    buf = io.BytesIO()
    wb.save(buf)
    with zipfile.ZipFile(buf) as src, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = _fix_core_xml(data)
            dst.writestr(zipfile.ZipInfo(item.filename, date_time=FIXED_TS), data)


def _fix_core_xml(data: bytes) -> bytes:
    import re

    return re.sub(rb"(<dcterms:modified[^>]*>)[^<]*", rb"\g<1>2026-01-01T00:00:00Z", data)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_company(res: CompanyResult, out_root: Path, config_hash: str) -> dict:
    """Записать файлы компании и manifest.json; вернуть manifest."""
    cdir = out_root / res.cfg["split"] / res.cfg["id"]
    (cdir / "documents").mkdir(parents=True, exist_ok=True)
    (cdir / "gold").mkdir(parents=True, exist_ok=True)
    _write_csv(res.dirty["customers"], cdir / "customers.csv")
    _write_csv(res.dirty["invoices"], cdir / "invoices.csv")
    _write_csv(res.dirty["fx_rates"], cdir / "fx_rates.csv")
    write_xlsx(res.cohorts, cdir / "cohorts.xlsx")
    for name, text in res.docs.items():
        (cdir / "documents" / name).write_text(text, encoding="utf-8")
    lines = [json.dumps(r, ensure_ascii=False) for r in res.claims]
    (cdir / "gold" / "claims.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(res.truth, cdir / "gold" / "truth_monthly.csv")
    rows = {
        "customers.csv": len(res.dirty["customers"]),
        "invoices.csv": len(res.dirty["invoices"]),
        "fx_rates.csv": len(res.dirty["fx_rates"]),
        "gold/claims.jsonl": len(res.claims),
        "gold/truth_monthly.csv": len(res.truth),
        "cohorts.xlsx": len(res.cohorts["cohort_mrr"]),
    }
    files = sorted(p for p in cdir.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "company_id": res.cfg["id"],
        "split": res.cfg["split"],
        "seed": res.cfg["seed"],
        "config_hash": config_hash,
        "files": {p.relative_to(cdir).as_posix(): sha256(p) for p in files},
        "row_counts": rows,
        "clean_row_counts": {k: len(v) for k, v in res.clean.items()},
        "params": {
            "archetype": res.cfg["archetype"],
            "months": res.cfg["months"],
            "coverage": [res.truth["month"].iloc[0], res.truth["month"].iloc[-1]],
            "cases_assigned": res.meta["cases"],
            "cases_planted": sorted({r["case_id"] for r in res.claims if r["case_id"]}),
        },
        "plantings": res.meta["plantings"],
        "noise": res.meta["noise"],
        "dirty_vs_clean": res.meta["dirty_vs_clean"],
    }
    (cdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8"
    )
    return manifest


def holdout_summary(results: list[CompanyResult], quality: pd.DataFrame) -> dict:
    """Агрегаты по holdout без данных отдельных компаний."""
    claims = pd.DataFrame([r for res in results for r in res.claims])
    per_company = claims.groupby("company_id").size()
    case_companies = claims.dropna(subset=["case_id"]).groupby("case_id")["company_id"].nunique()
    rows = defaultdict(list)
    for res in results:
        for k, df in res.dirty.items():
            rows[k].append(len(df))
    q = quality.groupby("check").agg(
        severity=("severity", "first"),
        companies_flagged=("n_flagged", lambda s: int((s > 0).sum())),
        total_flagged=("n_flagged", "sum"),
        total_rows=("n_total", "sum"),
    )
    return {
        "generator_version": GENERATOR_VERSION,
        "n_companies": len(results),
        "claims_total": len(claims),
        "claims_per_company": {
            "min": int(per_company.min()),
            "max": int(per_company.max()),
            "mean": round(float(per_company.mean()), 1),
        },
        "verdict_counts": claims["gold_verdict"].value_counts().sort_index().to_dict(),
        "claim_type_counts": claims["claim_type"].value_counts().sort_index().to_dict(),
        "case_claim_counts": claims["case_id"]
        .fillna("plain")
        .value_counts()
        .sort_index()
        .to_dict(),
        "case_company_counts": case_companies.sort_index().to_dict(),
        "case_verdicts": {
            case: g["gold_verdict"].value_counts().sort_index().to_dict()
            for case, g in claims.assign(case_id=claims["case_id"].fillna("plain")).groupby(
                "case_id"
            )
        },
        "row_counts": {k: {"total": sum(v), "min": min(v), "max": max(v)} for k, v in rows.items()},
        "quality_checks": {
            k: {kk: (int(vv) if not isinstance(vv, str) else vv) for kk, vv in v.items()}
            for k, v in q.to_dict(orient="index").items()
        },
        "family_verdict_counts": family_verdicts(claims),
        "dirty_vs_clean_verdict_mismatches": sum(
            len(r.meta["dirty_vs_clean"]["verdict_mismatches"]) for r in results
        ),
    }


def family_verdicts(claims: pd.DataFrame) -> dict[str, int]:
    """Число семейств утверждений по эталонному вердикту (семейство = одно наблюдение).

    Семейство — одно утверждение, повторённое в нескольких документах; вердикт у повторов общий.
    """
    fam = claims.groupby("family_id")["gold_verdict"].agg(lambda v: v.mode().iloc[0])
    out = fam.value_counts().sort_index().to_dict()
    flags = fam.isin(["EXAGGERATED", "CONTRADICTED"])
    out["red_flag_families"] = int(flags.sum())
    out["non_supported_families_5cls"] = int(
        (~fam.isin(["SUPPORTED", "FACTUAL_OUT_OF_SCOPE"])).sum()
    )
    return {k: int(v) for k, v in out.items()}


def config_hash(ccfg: dict, gcfg: dict) -> str:
    payload = {
        "company": ccfg,
        "noise": gcfg["noise"],
        "end_month": gcfg["end_month"],
        "version": GENERATOR_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def generate(config_path: str | Path, out_dir: str | Path, only: list[str] | None = None) -> dict:
    """Сгенерировать все компании (или подмножество `only`) и вернуть сводку."""
    gcfg = load_yaml(config_path)
    catalog = load_catalog()
    companies = company_configs(gcfg)
    assignment = assign_cases(gcfg, companies)
    out = Path(out_dir)
    holdout, quality, manifests = [], [], {}
    for ccfg in companies:
        if only and ccfg["id"] not in only:
            continue
        res = build_company(ccfg, gcfg, catalog, assignment[ccfg["id"]])
        manifests[ccfg["id"]] = write_company(res, out, config_hash(ccfg, gcfg))
        if ccfg["split"] == "holdout":
            holdout.append(res)
            cdir = out / ccfg["split"] / ccfg["id"]
            quality.append(run_quality_checks(read_tables(cdir), ccfg["id"]))
    if holdout and not only:
        cover = Counter(
            case for r in holdout for case in {c["case_id"] for c in r.claims if c["case_id"]}
        )
        missing = {c: cover[c] for c in CASE_IDS if cover[c] < gcfg["coverage_min"]["holdout"]}
        if missing:
            raise AssertionError(f"покрытие holdout недостаточно: {missing}")
        summary = holdout_summary(holdout, pd.concat(quality, ignore_index=True))
        (out / "holdout_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return manifests
