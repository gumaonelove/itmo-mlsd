"""Утверждения компании: корректные «плановые» и посаженные случаи каталога.

Каждый случай превращается в `ClaimSpec`: метрика, период, заявленное значение (ровно то число,
которое написано в тексте), квантификатор и форма записи. Эталонный вердикт считается
библиотекой метрик на чистых данных (`gold_fields`), а не назначается вручную.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dd_verifier.metrics import DataRoom, compute, evaluate_claim
from dd_verifier.metrics.core import NotEnoughData, add_months, month_end
from dd_verifier.metrics.tolerance import metric_meta
from dd_verifier.synth.names import AWARDS, PARTNERS, name_pool, name_variant

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
ARR_DEF_CASES = {"D01", "D02", "D05"}


@dataclass
class ClaimSpec:
    """Одно утверждение до рендеринга в текст."""

    key: str
    claim_type: str
    metric_id: str | None
    claimed_value: float | str | None
    unit: str | None
    period_start: str | None
    period_end: str | None
    qualifier: str | None
    value_text: str = ""
    form: str = "num"
    q_text: str = ""
    period_text: str = ""
    case_id: str | None = None
    variant: str | None = None
    company_value: float | None = None
    venues: tuple[str, ...] = ("deck",)
    template: str | None = None
    extra: dict = field(default_factory=dict)

    def as_claim(self) -> dict:
        """Поля, которые нужны `evaluate_claim`."""
        return {
            "claim_type": self.claim_type,
            "metric_id": self.metric_id,
            "claimed_value": self.claimed_value,
            "qualifier": self.qualifier,
            "period_start": self.period_start,
            "period_end": self.period_end,
        }


# --- запись чисел ---------------------------------------------------------------------------
def _steps(v: float) -> list[float]:
    if v >= 1e6:
        return [5e5, 2.5e5, 1e5, 5e4, 1e4, 1e3]
    if v >= 1e3:
        return [5e4, 2.5e4, 1e4, 5e3, 1e3, 100.0, 10.0]
    return [100.0, 50.0, 10.0, 5.0, 1.0]


def nice_number(v: float, mode: str, steps: list[float] | None = None) -> float:
    """«Красивое» число рядом с v.

    round — ближайшее с ошибкой ≤ 2%; up — ближайшее сверху в пределах (0,5%; 4,5%];
    down — снизу не меньше 80% от v; about — ближайшее с ошибкой ≤ 4%.
    """
    for step in steps or _steps(abs(v)):
        if mode == "round" or mode == "about":
            x = round(v / step) * step
            if x > 0 and abs(x / v - 1) <= (0.02 if mode == "round" else 0.04):
                return x
        elif mode == "up":
            x = math.ceil(v / step) * step
            if 0.005 < x / v - 1 <= 0.045:
                return x
        elif mode == "down":
            x = math.floor(v / step) * step
            if x > 0 and x / v >= 0.8:
                return x
    finest = (steps or _steps(abs(v)))[-1]  # запасной вариант: самый мелкий шаг (точная запись)
    return round(v / finest) * finest


def fmt_money(x: float, rng: np.random.Generator) -> str:
    """$1.8M / $1.8 million / $640K / $950."""
    if x >= 1e6:
        num = f"{x / 1e6:.3f}".rstrip("0").rstrip(".")
        return f"${num}M" if rng.random() < 0.7 else f"${num} million"
    if x >= 1e3:
        return f"${x / 1e3:.2f}".rstrip("0").rstrip(".") + "K"
    return f"${x:,.0f}"


def fmt_pct(share: float, decimals: int) -> str:
    """0.045 → '4.5%'."""
    return f"{share * 100:.{decimals}f}%"


QUALIFIER_WORDS = {
    "more_than": ["more than ", "over "],
    "about": ["about ", "approximately ", "~"],
    "less_than": ["below ", "under ", "less than ", "< "],
}


def q_word(qualifier: str | None, rng: np.random.Generator) -> str:
    words = QUALIFIER_WORDS.get(qualifier or "", [""])
    return words[int(rng.integers(len(words)))]


# --- контекст компании ----------------------------------------------------------------------
class ClaimFactory:
    """Строит утверждения для одной компании по чистому data room."""

    def __init__(self, cfg: dict, dr: DataRoom, end: str, rng: np.random.Generator, catalog: dict):
        self.cfg, self.dr, self.end, self.rng, self.catalog = cfg, dr, end, rng, catalog
        self.y0 = add_months(end, -12)
        self.year = end[:4]
        self.year_start = f"{self.year}-01"
        self.used: set[tuple] = set()
        self.blocked: set[str] = set()
        self.memo: dict[str, str] = {}  # тема меморандума → ключ определения

    # общие помощники
    def choice(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    def value(
        self,
        metric_id: str,
        end: str,
        start: str | None = None,
        variant: str = "standard",
        as_growth: bool = False,
    ) -> float:
        return compute(
            self.dr, metric_id, end, start, variant, as_growth=as_growth, with_sql=False
        ).value

    def level_period(self) -> str:
        month = f"{MONTH_NAMES[int(self.end[5:]) - 1]} {self.year}"
        return self.choice([f"as of {month}", f"at the end of {self.year}", f"in {month}"])

    def growth_period(self, start: str, end: str) -> str:
        if start == add_months(end, -12) and end.endswith("-12"):
            return self.choice([f"in {end[:4]}", "year over year", "over the last 12 months"])
        if start == add_months(end, -3) and int(end[5:]) % 3 == 0:
            return f"in Q{int(end[5:]) // 3} {end[:4]}"
        s, e = int(start[5:]) - 1, int(end[5:]) - 1
        return f"from {MONTH_NAMES[s]} to {MONTH_NAMES[e]} {end[:4]}"

    # конструкторы утверждений по типу метрики
    def money(
        self,
        key: str,
        metric_id: str,
        true: float,
        mode: str = "round",
        qualifier=None,
        end: str | None = None,
        start: str | None = None,
        **kw,
    ) -> ClaimSpec:
        x = nice_number(true, mode)
        return ClaimSpec(
            key,
            "level",
            metric_id,
            x,
            "USD",
            start,
            end or self.end,
            qualifier,
            value_text=fmt_money(x, self.rng),
            q_text=q_word(qualifier, self.rng),
            period_text=kw.pop("period_text", None)
            or (f"in {(end or self.end)[:4]}" if metric_id == "revenue" else self.level_period()),
            **kw,
        )

    def count(
        self,
        true: float,
        mode: str = "exact",
        end: str | None = None,
        period_text: str | None = None,
        **kw,
    ) -> ClaimSpec:
        n = round(true)
        qualifier, text = None, f"{n:,}"
        if mode == "plus":
            step = 10 if n >= 50 else 5
            x = (n // step) * step
            qualifier, text = "more_than", f"{x:,}+"
            n = x
        elif mode == "more_than":
            n = int(nice_number(true, "down", [100, 50, 25, 10, 5]))
            qualifier, text = "more_than", f"{n:,}"
        elif mode == "about":
            n = int(nice_number(true, "about", [50, 25, 10, 5]))
            qualifier, text = "about", f"{n:,}"
        return ClaimSpec(
            "customers",
            "count",
            "active_customers",
            float(n),
            "customers",
            None,
            end or self.end,
            qualifier,
            value_text=text,
            q_text=q_word(qualifier, self.rng) if mode != "plus" else "",
            period_text=period_text or self.level_period(),
            **kw,
        )

    def share(
        self,
        key: str,
        metric_id: str,
        true: float,
        decimals: int,
        qualifier=None,
        start: str | None = None,
        end: str | None = None,
        x: float | None = None,
        **kw,
    ) -> ClaimSpec:
        x = round(true, decimals + 2) if x is None else x
        period = kw.pop("period_text", None)
        if period is None:
            period = f"in {self.year}" if metric_id != "top10_concentration" else ""
        return ClaimSpec(
            key,
            "share",
            metric_id,
            x,
            "share",
            start,
            end or self.end,
            qualifier,
            value_text=fmt_pct(x, decimals),
            q_text=q_word(qualifier, self.rng),
            period_text=period,
            **kw,
        )

    def growth(
        self, metric_id: str, true: float, start: str, end: str, form: str | None = None, **kw
    ) -> ClaimSpec:
        """Рост как множитель; форма записи: mult (2.6x) | pct (160%) | word (tripled)."""
        words = []
        for w, target in (("doubled", 2.0), ("tripled", 3.0)):
            if abs(true / target - 1) <= 0.03:
                words.append((w, target, None))
        if true >= 2.0:
            words.append(("more than doubled", 2.0, "more_than"))
        forms = ["mult", "pct"] + (["word"] if words else [])
        form = form or self.choice(forms)
        qualifier = None
        if form == "word":
            text, x, qualifier = self.choice(words)
        elif form == "mult":
            x = round(true, 1) if abs(round(true, 1) / true - 1) <= 0.025 else round(true, 2)
            if abs(round(true) / true - 1) <= 0.015 and true >= 2:
                x = float(round(true))
            text = f"{x:g}x"
        else:
            pct = round((true - 1) * 100)
            x, text = 1 + pct / 100, f"{pct}%"
        key = "arr_growth" if metric_id == "arr_growth" else "customers_growth"
        return ClaimSpec(
            key,
            "growth",
            metric_id,
            float(x),
            "x",
            start,
            end,
            qualifier,
            value_text=text,
            form=form,
            period_text=self.growth_period(start, end),
            **kw,
        )

    # --- проверка уникальности ---
    def take(self, spec: ClaimSpec) -> ClaimSpec:
        self.used.add((spec.metric_id, spec.claim_type, spec.period_start, spec.period_end))
        return spec

    def free(self, metric_id: str, claim_type: str, start, end) -> bool:
        key = (metric_id, claim_type, start, end)
        return key not in self.used and metric_id not in self.blocked


# --- случаи каталога ------------------------------------------------------------------------
def case_d01(f: ClaimFactory, fixed: bool) -> ClaimSpec:
    f.memo["arr"] = "d01_disclosed" if fixed or f.rng.random() < 0.6 else "d01_hint"
    if fixed:  # Pantrio: «ARR grew 3x in 2025»
        true = f.value("arr_growth", f.end, f.y0, "incl_one_time")
        spec = f.growth(
            "arr_growth",
            true,
            f.y0,
            f.end,
            form="mult",
            case_id="D01",
            variant="incl_one_time",
            template="ARR grew {v} {p}",
        )
        spec.period_text = "in 2025"
        return spec
    metric = f.choice(["arr", "arr", "mrr", "acv"])
    true = f.value(metric, f.end, variant="incl_one_time")
    return f.money(metric, metric, true, "round", case_id="D01", variant="incl_one_time")


def case_d02(f: ClaimFactory) -> ClaimSpec:
    f.memo.setdefault("arr", "vague")
    true = f.value("arr", f.end, variant="peak_month")
    return f.money("arr", "arr", true, "round", case_id="D02", variant="peak_month")


def case_d03(f: ClaimFactory) -> ClaimSpec:
    f.memo["customers"] = f.choice(["vague", "d03_hint"])
    true = f.value("active_customers", f.end, variant="signed_incl_pilots")
    return f.count(true, "exact", case_id="D03", variant="signed_incl_pilots")


def case_d04(f: ClaimFactory) -> ClaimSpec:
    true = f.value("arr_growth", f.end, f.y0, "trough_base")
    form = f.choice(["mult", "pct"])  # словесные формы («more than doubled») могли бы стать верными
    return f.growth(
        "arr_growth", true, f.y0, f.end, form=form, case_id="D04", variant="trough_base"
    )


def case_d05(f: ClaimFactory, metric: str) -> ClaimSpec:
    f.memo["revenue"] = "d05_disclosed"
    true = f.value(metric, f.end, f.year_start if metric == "revenue" else None, "gross_channel")
    start = f.year_start if metric == "revenue" else None
    return f.money(
        metric, metric, true, "round", start=start, case_id="D05", variant="gross_channel"
    )


def case_d06(f: ClaimFactory) -> ClaimSpec:
    f.memo["nrr"] = f.choice(["d06_disclosed", "d06_hint"])
    true = f.value("nrr", f.end, f.y0, "survivors_only")
    qualifier = f.choice([None, None, "more_than"])
    x = round(true, 2) if qualifier is None else math.floor(true * 20) / 20
    return f.share(
        "nrr",
        "nrr",
        true,
        0,
        qualifier,
        f.y0,
        x=x,
        case_id="D06",
        variant="survivors_only",
        period_text=f.choice([f"in {f.year}", f.level_period()]),
    )


def case_d07(f: ClaimFactory, fixed: bool) -> ClaimSpec:
    f.memo["churn"] = "d07_disclosed"
    true = f.value("logo_churn", f.end, f.year_start, "excl_pilots_and_small")
    kw = {"start": f.year_start, "case_id": "D07", "variant": "excl_pilots_and_small"}
    if fixed:  # Pantrio: «Logo churn < 2% monthly»
        spec = f.share("churn", "logo_churn", true, 0, "less_than", x=0.02, **kw)
        spec.q_text, spec.template, spec.period_text = "< ", "Logo churn {q}{v} monthly", ""
        return spec
    if f.rng.random() < 0.5:
        return f.share("churn", "logo_churn", true, 1, x=round(true, 3), **kw)
    return f.share("churn", "logo_churn", true, 1, "less_than", x=math.ceil(true * 200) / 200, **kw)


def case_d08(f: ClaimFactory) -> ClaimSpec:
    options = [
        m
        for m in ("arr", "mrr", "revenue", "active_customers", "acv")
        if f.free(m, "level", None, f.end)
    ]
    metric = f.choice(options or ["revenue"])
    factor = f.rng.uniform(1.32, 1.5) if f.rng.random() < 0.6 else f.rng.uniform(0.75, 0.86)
    if metric == "active_customers":
        return f.count(f.value(metric, f.end) * factor, "exact", case_id="D08")
    start = f.year_start if metric == "revenue" else None
    true = f.value(metric, f.end, start)
    return f.money(metric, metric, true * factor, "round", start=start, case_id="D08")


def case_d09(f: ClaimFactory) -> ClaimSpec:
    """Ищем метрику, которая была лучше 6–12 месяцев назад, и подаём старое значение как текущее.

    Берём только устаревшие значения в пользу компании (с учётом направления метрики): ухудшение
    за последние месяцы гарантирует `plant.plant_recent_downgrade`.
    """
    e, y0, ys = f.end, f.y0, f.year_start
    candidates = []
    for lag in (6, 9, 12):
        old_e = add_months(e, -lag)
        candidates += [
            ("nrr", "share", y0, e, add_months(old_e, -12), old_e),
            ("grr", "share", y0, e, add_months(old_e, -12), old_e),
            ("logo_churn", "share", ys, e, add_months(old_e, -11), old_e),
            ("arr_growth", "growth", y0, e, add_months(old_e, -12), old_e),
            ("acv", "level", None, e, None, old_e),
            ("active_customers", "count", None, e, None, old_e),
            ("top10_concentration", "share", None, e, None, old_e),
        ]
    order = f.rng.permutation(len(candidates))
    fallback = None
    for i in order:
        metric, ctype, s, en, old_s, old_e = candidates[i]
        if not f.free(metric, ctype, s, en):
            continue
        try:
            stale = f.value(metric, old_e, old_s)
        except NotEnoughData:
            continue
        kw = {
            "case_id": "D09",
            "variant": None,
            "company_value": stale,
            "extra": {"stale_period": [old_s, old_e]},
        }
        if metric in {"nrr", "grr"}:
            spec = f.share(
                metric,
                metric,
                stale,
                0,
                None,
                s,
                x=round(stale, 2),
                period_text=f.level_period(),
                **kw,
            )
        elif metric == "logo_churn":
            spec = f.share("churn", metric, stale, 1, None, s, x=round(stale, 3), **kw)
        elif metric == "arr_growth":
            spec = f.growth(metric, stale, s, en, form=f.choice(["mult", "pct"]), **kw)
        elif metric == "top10_concentration":
            spec = f.share("top10", metric, stale, 0, None, x=round(stale, 2), **kw)
        elif metric == "acv":
            spec = f.money("acv", "acv", stale, "round", **kw)
        else:
            spec = f.count(stale, "exact", **kw)
        check = evaluate_claim(f.dr, spec.as_claim())
        if not favourable_to_company(metric, check.delta):
            continue  # устаревшее значение хуже текущего — это не «удобная» дата
        if check.verdict == "EXAGGERATED":
            return spec
        if check.verdict == "CONTRADICTED" and fallback is None:
            fallback = spec
    if fallback is None:
        raise RuntimeError("D09: нет метрики, которая 6–12 месяцев назад была заметно лучше")
    return fallback


def favourable_to_company(metric_id: str, delta: float | None) -> bool:
    """Отклонение заявленного от факта в пользу компании (с учётом направления метрики)."""
    if delta is None or delta == 0:
        return False
    return delta > 0 if metric_meta(metric_id)["direction"] == "up" else delta < 0


def case_d10(f: ClaimFactory) -> ClaimSpec:
    c = f.dr.customers
    churned = c.loc[
        c["churn_date"].notna()
        & c["go_live_date"].notna()
        & (c["churn_date"] <= month_end(f.end))
        & (c["churn_date"] >= f"{f.year}-01-01")
    ]
    if f.rng.random() < 0.5 and len(churned):
        name = churned.iloc[int(f.rng.integers(len(churned)))]["customer_name"]
        kind = "churned"
    else:
        known = set(c["customer_name"].map(lambda s: s.lower()))
        pool = name_pool(f.cfg["industry_key"], len(known) + 40, f.rng)
        name = next(
            n for n in pool if n.lower() not in known and not any(n.lower() in k for k in known)
        )
        kind = "not_found"
    shown = name_variant(name, f.rng) if f.rng.random() < 0.5 else name
    return mention(f, shown, "D10", kind)


def mention(f: ClaimFactory, name: str, case_id: str | None, kind: str) -> ClaimSpec:
    return ClaimSpec(
        "mention",
        "customer_mention",
        None,
        name,
        None,
        None,
        f.end,
        None,
        value_text=name,
        case_id=case_id,
        venues=(f.choice(["deck", "press"]),),
        extra={"mention_kind": kind},
    )


def case_d11(f: ClaimFactory, window: list[str]) -> ClaimSpec:
    s, e = window
    claimed = 1 + round(f.rng.uniform(0.05, 0.2), 2)
    spec = f.growth("arr_growth", claimed, s, e, form="pct", case_id="D11")
    return spec


def case_h01(f: ClaimFactory, fixed: bool) -> ClaimSpec:
    if fixed or (f.free("active_customers", "count", None, f.end) and f.rng.random() < 0.5):
        spec = f.count(f.value("active_customers", f.end), "plus", case_id="H01")
        if fixed:
            spec.template = "{q}{v} {noun} use {company} every day"
        return spec
    options = [m for m in ("arr", "mrr", "acv") if f.free(m, "level", None, f.end)]
    if options:
        metric = f.choice(options)
        return f.money(metric, metric, f.value(metric, f.end), "up", case_id="H01")
    if f.free("revenue", "level", f.year_start, f.end):
        true = f.value("revenue", f.end, f.year_start)
        return f.money("revenue", "revenue", true, "up", start=f.year_start, case_id="H01")
    return f.count(f.value("active_customers", f.end), "plus", case_id="H01")


def case_h02(f: ClaimFactory) -> ClaimSpec:
    opts = []
    if f.free("active_customers", "count", None, f.end):
        opts.append("customers")
    if f.free("arr", "level", None, f.end):
        opts += ["arr_more", "arr_about"]
    if f.free("nrr", "share", f.y0, f.end):
        opts.append("nrr")
    if f.free("logo_churn", "share", f.year_start, f.end):
        opts.append("churn")
    if f.free("grr", "share", f.y0, f.end):
        opts.append("grr")
    if f.free("revenue", "level", f.year_start, f.end):
        opts.append("revenue")
    if not opts:
        raise RuntimeError(f"{f.cfg['id']}: H02 — все метрики заняты")
    kind = f.choice(opts)
    if kind == "customers":
        return f.count(
            f.value("active_customers", f.end), f.choice(["more_than", "about"]), case_id="H02"
        )
    if kind.startswith("arr"):
        mode, q = ("down", "more_than") if kind == "arr_more" else ("about", "about")
        return f.money("arr", "arr", f.value("arr", f.end), mode, q, case_id="H02")
    if kind == "nrr":
        true = f.value("nrr", f.end, f.y0)
        return f.share(
            "nrr", "nrr", true, 0, "more_than", f.y0, x=math.floor(true * 20) / 20, case_id="H02"
        )
    if kind == "churn":
        true = f.value("logo_churn", f.end, f.year_start)
        x = math.ceil(true * 200) / 200 + (
            0.005 if math.ceil(true * 200) / 200 - true < 0.001 else 0
        )
        return f.share(
            "churn", "logo_churn", true, 1, "less_than", f.year_start, x=x, case_id="H02"
        )
    if kind == "grr":
        true = f.value("grr", f.end, f.y0)
        return f.share(
            "grr", "grr", true, 0, "more_than", f.y0, x=math.floor(true * 20) / 20, case_id="H02"
        )
    true = f.value("revenue", f.end, f.year_start)
    return f.money("revenue", "revenue", true, "about", "about", start=f.year_start, case_id="H02")


def case_h03(f: ClaimFactory) -> ClaimSpec:
    metric = "arr_growth" if f.free("arr_growth", "growth", f.y0, f.end) else "active_customers"
    true = f.value(metric, f.end, f.y0, as_growth=metric != "arr_growth")
    spec = f.growth(metric, true, f.y0, f.end, case_id="H03")
    if spec.form == "num":
        spec.form = "pct"
    return spec


def case_h04(f: ClaimFactory) -> ClaimSpec:
    """Стандартное определение метрики в меморандуме + утверждение, совпадающее со стандартом."""
    e, y0, ys = f.end, f.y0, f.year_start
    # метрика → (тема меморандума, тип утверждения, начало периода)
    table = {
        "arr": ("arr", "level", None),
        "active_customers": ("customers", "count", None),
        "logo_churn": ("churn", "share", ys),
        "nrr": ("nrr", "share", y0),
        "grr": ("nrr", "share", y0),
        "revenue": ("revenue", "level", ys),
        "top10_concentration": ("concentration", "share", None),
    }
    options = [
        m for m, (topic, ctype, start) in table.items()
        if topic not in f.memo and f.free(m, ctype, start, e)
    ]  # fmt: skip
    if not options:
        raise RuntimeError(f"{f.cfg['id']}: H04 — нет свободной метрики для определения")
    metric = f.choice(options)
    topic, _, start = table[metric]
    f.memo[topic] = f"h04_{metric}"
    true = f.value(metric, e, start)
    if metric in {"arr", "revenue"}:
        return f.money(metric, metric, true, "round", start=start, case_id="H04")
    if metric == "active_customers":
        return f.count(true, "exact", case_id="H04")
    key = {"logo_churn": "churn", "top10_concentration": "top10"}.get(metric, metric)
    decimals = 1 if metric == "logo_churn" else 0
    x = round(true, decimals + 2)
    return f.share(key, metric, true, decimals, None, start, x=x, case_id="H04")


OOO = [
    ("cac_payback", "CAC payback of {v} months", lambda r: int(r.integers(6, 19)), "months"),
    ("ltv_to_cac", "LTV/CAC of {v}x", lambda r: round(float(r.uniform(2.5, 6.0)), 1), "x"),
    ("gross_margin", "Gross margin of {v}%", lambda r: int(r.integers(68, 86)), "share"),
    ("burn_multiple", "Burn multiple of {v}x", lambda r: round(float(r.uniform(0.8, 2.2)), 1), "x"),
]


def case_n01(f: ClaimFactory) -> ClaimSpec:
    mid, tmpl, gen, unit = OOO[int(f.rng.integers(len(OOO)))]
    v = gen(f.rng)
    value = v / 100 if unit == "share" else float(v)
    return ClaimSpec(
        "ooo",
        "out_of_ontology",
        None,
        value,
        unit,
        None,
        f.end,
        None,
        value_text=str(v),
        case_id="N01",
        template=tmpl,
        extra={"ooo_metric": mid},
    )


def case_n02(f: ClaimFactory) -> ClaimSpec:
    first = f.dr.coverage[0]
    if f.rng.random() < 0.5:
        year = int(first[:4]) - (1 if first.endswith("-01") else 0)
        true = f.value("revenue", add_months(first, 11), first) * f.rng.uniform(0.6, 0.9)
        return f.money(
            "revenue",
            "revenue",
            true,
            "round",
            start=f"{year}-01",
            end=f"{year}-12",
            period_text=f"in {year}",
            case_id="N02",
        )
    end = add_months(first, 11)
    x = round(float(f.rng.uniform(1.02, 1.25)), 2)
    return f.share(
        "nrr",
        "nrr",
        x,
        0,
        None,
        add_months(end, -12),
        end,
        x=x,
        period_text=f"as of {MONTH_NAMES[int(end[5:]) - 1]} {end[:4]}",
        case_id="N02",
    )


def case_u01(f: ClaimFactory) -> ClaimSpec:
    if f.rng.random() < 0.5:
        target = nice_number(f.value("arr", f.end) * f.rng.uniform(2.0, 3.2), "round")
        text = fmt_money(target, f.rng)
        return ClaimSpec(
            "forecast",
            "forecast",
            "arr",
            target,
            "USD",
            None,
            "2026-12",
            None,
            value_text=text,
            case_id="U01",
        )
    return ClaimSpec(
        "opinion",
        "forecast",
        None,
        None,
        None,
        None,
        None,
        None,
        case_id="U01",
        venues=(f.choice(["deck", "press"]),),
    )


def case_f01(f: ClaimFactory) -> ClaimSpec:
    if f.rng.random() < 0.5:
        extra = {"partner": f.choice(PARTNERS), "kind": "partner"}
    else:
        extra = {"award": f.choice(AWARDS), "kind": "award"}
    return ClaimSpec(
        "factual",
        "factual",
        None,
        None,
        None,
        None,
        None,
        None,
        case_id="F01",
        venues=(f.choice(["deck", "press"]),),
        extra=extra,
    )


# --- плановые (корректные) утверждения ------------------------------------------------------
def plain_claims(f: ClaimFactory, n: int) -> list[ClaimSpec]:
    """n корректных утверждений по метрикам, не занятым случаями каталога."""
    e, y0, ys = f.end, f.y0, f.year_start
    py, py_start = y0[:4], add_months(y0, -11)
    makers = {
        "arr_prev": lambda: f.money(
            "arr", "arr", f.value("arr", y0), end=y0, period_text=f"at the end of {py}"
        ),
        "revenue_prev": lambda: f.money(
            "revenue",
            "revenue",
            f.value("revenue", y0, py_start),
            start=py_start,
            end=y0,
            period_text=f"in {py}",
        ),
        "customers_prev": lambda: f.count(
            f.value("active_customers", y0), "exact", end=y0, period_text=f"at the end of {py}"
        ),
        "arr": lambda: f.money("arr", "arr", f.value("arr", e)),
        "mrr": lambda: f.money("mrr", "mrr", f.value("mrr", e)),
        "revenue": lambda: f.money("revenue", "revenue", f.value("revenue", e, ys), start=ys),
        "acv": lambda: f.money("acv", "acv", f.value("acv", e)),
        "arr_growth": lambda: f.growth("arr_growth", f.value("arr_growth", e, y0), y0, e),
        "customers": lambda: f.count(f.value("active_customers", e), "exact"),
        "customers_growth": lambda: f.growth(
            "active_customers", f.value("active_customers", e, y0, as_growth=True), y0, e
        ),
        "nrr": lambda: f.share(
            "nrr", "nrr", (t := f.value("nrr", e, y0)), 0, None, y0, x=round(t, 2)
        ),
        "grr": lambda: f.share(
            "grr", "grr", (t := f.value("grr", e, y0)), 0, None, y0, x=round(t, 2)
        ),
        "churn": lambda: f.share(
            "churn", "logo_churn", (t := f.value("logo_churn", e, ys)), 1, None, ys, x=round(t, 3)
        ),
        "top10": lambda: f.share(
            "top10",
            "top10_concentration",
            (t := f.value("top10_concentration", e)),
            0,
            x=round(t, 2),
        ),
    }
    keys = {
        "arr_prev": ("arr", "level", None, y0),
        "revenue_prev": ("revenue", "level", py_start, y0),
        "customers_prev": ("active_customers", "count", None, y0),
        "arr": ("arr", "level", None, e),
        "mrr": ("mrr", "level", None, e),
        "revenue": ("revenue", "level", ys, e),
        "acv": ("acv", "level", None, e),
        "arr_growth": ("arr_growth", "growth", y0, e),
        "customers": ("active_customers", "count", None, e),
        "customers_growth": ("active_customers", "growth", y0, e),
        "nrr": ("nrr", "share", y0, e),
        "grr": ("grr", "share", y0, e),
        "churn": ("logo_churn", "share", ys, e),
        "top10": ("top10_concentration", "share", None, e),
    }
    out = []
    for k in f.rng.permutation(sorted(makers)):
        if len(out) >= n:
            break
        if f.free(*keys[k]):
            try:
                out.append(f.take(makers[k]()))
            except NotEnoughData:
                continue
    active = f.dr.active_ids(e)
    c = f.dr.customers.set_index("customer_id").loc[active]
    big = c.assign(mrr=f.dr.mrr_of(e).reindex(active)).sort_values("mrr", ascending=False)
    n_mentions = int(f.rng.integers(2, 4))
    for name in (
        big["customer_name"].head(8).sample(n_mentions, random_state=int(f.rng.integers(2**31)))
    ):
        shown = (
            name_variant(name, f.rng) if f.rng.random() < f.cfg["doc_name_variant_rate"] else name
        )
        out.append(mention(f, shown, None, "active"))
    return out


def round6(v: float | None) -> float | None:
    """Округление значений эталона до 6 знаков (None сохраняется)."""
    return None if v is None else round(float(v), 6)


_rnd = round6


# --- эталон ---------------------------------------------------------------------------------
def gold_fields(f: ClaimFactory, spec: ClaimSpec) -> dict:
    """Эталонный вердикт и истинные значения (стандарт и определение компании)."""
    check = evaluate_claim(f.dr, spec.as_claim())
    company = spec.company_value
    if company is None and spec.variant and check.actual is not None:
        as_growth = spec.claim_type == "growth" and spec.metric_id != "arr_growth"
        company = f.value(
            spec.metric_id, spec.period_end, spec.period_start, spec.variant, as_growth
        )
    elif company is None:
        company = check.actual
    allowed = ["SUPPORTED"] if spec.case_id is None else f.catalog[spec.case_id]["allowed_verdicts"]
    if check.verdict not in allowed:
        raise AssertionError(
            f"{f.cfg['id']}: случай {spec.case_id or 'plain'} ({spec.metric_id}, заявлено "
            f"{spec.claimed_value}) дал {check.verdict}, ожидалось {allowed}; {check.reason}"
        )
    return {
        "true_value_standard": _rnd(check.actual),
        "true_value_company_def": _rnd(company),
        "delta": _rnd(check.delta),
        "delta_p5": _rnd(p5_delta(spec.qualifier, spec.claimed_value, check.actual, check.delta)),
        "gold_verdict": check.verdict,
    }


def p5_delta(qualifier, claimed, actual, delta: float | None) -> float | None:
    """δ по правилу P5: 0, если квантификатор «более / менее N» выполнен; иначе сырое δ от N."""
    if delta is None or actual is None:
        return delta
    if qualifier == "more_than" and actual >= float(claimed) - 1e-9:
        return 0.0
    if qualifier == "less_than" and actual <= float(claimed) + 1e-9:
        return 0.0
    return delta
