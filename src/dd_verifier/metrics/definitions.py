"""Функции метрик онтологии (configs/ontology.yaml) и их вариантов «по версии компании».

Каждая функция чистая: на входе `DataRoom` (таблицы) и параметры, на выходе `MetricResult`
со значением и следом расчёта. Арифметику делает только этот код (P6).
"""

from __future__ import annotations

from collections.abc import Callable

from dd_verifier.metrics.core import (
    PAYING_EPS,
    DataRoom,
    MetricResult,
    NotEnoughData,
    add_months,
    month_end,
    month_range,
    month_start,
    usd,
)

MRR_VARIANTS = ("standard", "incl_one_time", "gross_channel")
ARR_VARIANTS = ("standard", "incl_one_time", "peak_month", "gross_channel")
GROWTH_VARIANTS = (*ARR_VARIANTS, "trough_base")


def _check_variant(variant: str, allowed: tuple[str, ...], metric_id: str) -> None:
    if variant not in allowed:
        raise ValueError(f"{metric_id}: неизвестный вариант {variant!r}, допустимы {allowed}")


def _month_total(dr: DataRoom, month: str, variant: str) -> float:
    led = dr.ledger
    rows = led.loc[led["month"] == month]
    if variant != "incl_one_time":
        rows = rows.loc[rows["payment_type"] == "subscription"]
    col = "gross_usd" if variant == "gross_channel" else "net_usd"
    return float(rows[col].sum())


def mrr(dr: DataRoom, month: str, variant: str = "standard") -> MetricResult:
    """MRR на месяц: подписка (net) / + разовые платежи / реселлеры по gross."""
    _check_variant(variant, MRR_VARIANTS, "mrr")
    dr.require(month)
    value = _month_total(dr, month, variant)
    what = {
        "standard": "подписочные инвойсы, net (amount − reseller_fee)",
        "incl_one_time": "все типы платежей (подписка + one_time + implementation), net",
        "gross_channel": "подписочные инвойсы, gross (reseller_fee не вычитается)",
    }[variant]
    trace = (
        "Леджер: инвойсы без void, курс месяца начала услуги; признание по месяцам периода услуги.",
        f"MRR {month}: {what} = {usd(value)}",
    )
    return MetricResult("mrr", variant, value, "USD", None, month, trace)


def arr(dr: DataRoom, month: str, variant: str = "standard") -> MetricResult:
    """ARR на месяц = MRR × 12; peak_month — максимум MRR за 12 месяцев × 12."""
    _check_variant(variant, ARR_VARIANTS, "arr")
    if variant == "peak_month":
        months = month_range(add_months(month, -11), month)
        dr.require(*months)
        series = {m: _month_total(dr, m, "standard") for m in months}
        peak = max(series, key=series.__getitem__)
        value = 12 * series[peak]
        trace = (
            f"Подписочный MRR за {months[0]}..{month}; максимум в {peak} = {usd(series[peak])}",
            f"ARR (peak_month) = {usd(series[peak])} × 12 = {usd(value)}",
        )
    else:
        base = mrr(dr, month, variant)
        value = 12 * base.value
        trace = (*base.trace, f"ARR = MRR × 12 = {usd(value)}")
    return MetricResult("arr", variant, value, "USD", None, month, trace)


def arr_growth(dr: DataRoom, start: str, end: str, variant: str = "standard") -> MetricResult:
    """Рост ARR как множитель ARR(end)/ARR(start); trough_base — от минимума внутри окна."""
    _check_variant(variant, GROWTH_VARIANTS, "arr_growth")
    if variant == "trough_base":
        window = month_range(start, add_months(end, -1))
        dr.require(*window, end)
        series = {m: 12 * _month_total(dr, m, "standard") for m in window}
        base_month = min(series, key=series.__getitem__)
        a0, a1 = series[base_month], arr(dr, end).value
        note = f"база = минимальный ARR в окне {start}..{window[-1]}: {base_month}"
    else:
        a0, a1 = arr(dr, start, variant).value, arr(dr, end, variant).value
        base_month, note = start, f"база = ARR на {start}"
    if a0 <= 0:
        raise NotEnoughData(f"ARR на базе {base_month} не положителен")
    value = a1 / a0
    trace = (note, f"ARR {base_month} = {usd(a0)}, ARR {end} = {usd(a1)}", f"рост = ×{value:.3f}")
    return MetricResult("arr_growth", variant, value, "x", start, end, trace)


def revenue(dr: DataRoom, start: str, end: str, variant: str = "standard") -> MetricResult:
    """Признанная выручка за [start, end], все типы платежей, net (или gross по реселлерам)."""
    _check_variant(variant, ("standard", "gross_channel"), "revenue")
    months = month_range(start, end)
    dr.require(*months)
    led = dr.ledger
    col = "gross_usd" if variant == "gross_channel" else "net_usd"
    value = float(led.loc[led["month"].between(start, end), col].sum())
    trace = (
        f"Выручка {start}..{end} ({col}, все типы платежей, без void, с кредит-нотами)"
        f" = {usd(value)}",
    )
    return MetricResult("revenue", variant, value, "USD", start, end, trace)


def active_customers(dr: DataRoom, month: str, variant: str = "standard") -> MetricResult:
    """Активные платящие клиенты на конец месяца; signed_incl_pilots — все подписанные записи."""
    _check_variant(variant, ("standard", "signed_incl_pilots"), "active_customers")
    dr.require(month)
    c, me = dr.customers, month_end(month)
    if variant == "standard":
        value = len(dr.active_ids(month))
        rule = "go_live ≤ конца месяца, не ушли, подписочный MRR > 0"
    else:
        alive = (c["signed_date"] <= me) & (c["churn_date"].isna() | (c["churn_date"] > me))
        value = int(alive.sum())
        rule = "signed_date ≤ конца месяца, не ушли (включая пилоты и незапущенных)"
    trace = (f"Клиенты на {me:%Y-%m-%d}: {rule} → {value}",)
    return MetricResult("active_customers", variant, float(value), "customers", None, month, trace)


def logo_churn(dr: DataRoom, start: str, end: str, variant: str = "standard") -> MetricResult:
    """Средний месячный logo churn за [start, end]."""
    _check_variant(variant, ("standard", "excl_pilots_and_small"), "logo_churn")
    months = month_range(start, end)
    dr.require(add_months(start, -1), *months)
    c = dr.customers.set_index("customer_id")
    rates, trace = [], []
    for m in months:
        base = dr.active_ids(add_months(m, -1))
        if variant == "excl_pilots_and_small":
            sub = c.loc[base]
            base = list(sub.index[~sub["is_pilot"].astype(bool) & (sub["segment"] != "SMB")])
        if not base:
            continue
        cd = c.loc[base, "churn_date"]
        churned = int(((cd >= month_start(m)) & (cd <= month_end(m))).sum())
        rates.append(churned / len(base))
        trace.append(f"{m}: ушли {churned} из {len(base)} → {churned / len(base):.4f}")
    if not rates:
        raise NotEnoughData("нет активных клиентов в периоде")
    value = sum(rates) / len(rates)
    trace.append(f"среднее по {len(rates)} мес. = {value:.4f}")
    return MetricResult("logo_churn", variant, value, "share", start, end, tuple(trace))


def _retention(dr: DataRoom, month: str, cohort_month: str | None) -> tuple:
    cohort_month = cohort_month or add_months(month, -12)
    dr.require(cohort_month, month)
    cohort = dr.active_ids(cohort_month)
    if not cohort:
        raise NotEnoughData(f"нет активных клиентов в когорте {cohort_month}")
    start = dr.mrr_of(cohort_month).reindex(cohort, fill_value=0.0)
    end = dr.mrr_of(month).reindex(cohort, fill_value=0.0).clip(lower=0.0)
    return cohort_month, start, end


def nrr(
    dr: DataRoom, month: str, variant: str = "standard", cohort_month: str | None = None
) -> MetricResult:
    """NRR: выручка когорты, активной 12 месяцев назад, сейчас / тогда (ушедшие = 0)."""
    _check_variant(variant, ("standard", "survivors_only"), "nrr")
    cohort_month, start, end = _retention(dr, month, cohort_month)
    note = f"когорта {cohort_month}: {len(start)} клиентов, MRR {usd(start.sum())}"
    if variant == "survivors_only":
        cd = dr.customers.set_index("customer_id").loc[start.index, "churn_date"]
        keep = (cd.isna() | (cd > month_end(month))).to_numpy()
        start, end = start[keep], end[keep]
        note += f"; без ушедших осталось {len(start)}"
    value = float(end.sum() / start.sum())
    trace = (note, f"MRR этих клиентов в {month} = {usd(end.sum())}", f"NRR = {value:.4f}")
    return MetricResult("nrr", variant, value, "share", cohort_month, month, trace)


def grr(dr: DataRoom, month: str, variant: str = "standard", cohort_month: str | None = None):
    """GRR: как NRR, но выручка клиента не выше его стартовой (без допродаж)."""
    _check_variant(variant, ("standard",), "grr")
    cohort_month, start, end = _retention(dr, month, cohort_month)
    kept = end.where(end < start, start)
    value = float(kept.sum() / start.sum())
    trace = (
        f"когорта {cohort_month}: {len(start)} клиентов, MRR {usd(start.sum())}",
        f"Σ min(MRR {month}, MRR {cohort_month}) = {usd(kept.sum())}",
        f"GRR = {value:.4f}",
    )
    return MetricResult("grr", variant, value, "share", cohort_month, month, trace)


def acv(dr: DataRoom, month: str, variant: str = "standard") -> MetricResult:
    """ACV = ARR (в том же варианте) / активные платящие клиенты."""
    _check_variant(variant, ARR_VARIANTS, "acv")
    a = arr(dr, month, variant)
    n = active_customers(dr, month).value
    if n == 0:
        raise NotEnoughData("нет активных клиентов")
    value = a.value / n
    trace = (*a.trace, f"активных клиентов {int(n)}", f"ACV = {usd(value)}")
    return MetricResult("acv", variant, value, "USD", None, month, trace)


def top10_concentration(dr: DataRoom, month: str, variant: str = "standard") -> MetricResult:
    """Доля выручки 10 крупнейших клиентов за последние 12 месяцев."""
    _check_variant(variant, ("standard",), "top10_concentration")
    start = add_months(month, -11)
    dr.require(*month_range(start, month))
    led = dr.ledger.loc[dr.ledger["month"].between(start, month)]
    total = float(led["net_usd"].sum())
    by_cust = led.groupby("customer_id")["net_usd"].sum().sort_values(ascending=False)
    top = float(by_cust.head(10).sum())
    value = top / total if total > 0 else float("nan")
    trace = (
        f"Выручка {start}..{month}: всего {usd(total)}, топ-10 клиентов {usd(top)}",
        f"доля = {value:.4f}",
    )
    return MetricResult("top10_concentration", variant, value, "share", start, month, trace)


def growth(
    dr: DataRoom, metric_id: str, start: str, end: str, variant: str = "standard"
) -> MetricResult:
    """Рост уровня любой метрики как множитель value(end) / value(start)."""
    if metric_id == "arr_growth":
        return arr_growth(dr, start, end, variant)
    fn = LEVEL_FUNCTIONS[metric_id]
    a0, a1 = fn(dr, start, variant), fn(dr, end, variant)
    if a0.value <= 0:
        raise NotEnoughData(f"{metric_id} на {start} не положителен")
    value = a1.value / a0.value
    trace = (*a0.trace, *a1.trace, f"рост {metric_id} = ×{value:.3f}")
    return MetricResult(metric_id, variant, value, "x", start, end, trace)


LEVEL_FUNCTIONS: dict[str, Callable[..., MetricResult]] = {
    "mrr": mrr,
    "arr": arr,
    "active_customers": active_customers,
    "acv": acv,
}

PERIOD_FUNCTIONS = {"revenue": revenue, "logo_churn": logo_churn}
POINT_FUNCTIONS = {
    **LEVEL_FUNCTIONS,
    "top10_concentration": top10_concentration,
}


def default_start(metric_id: str, end: str) -> str:
    """Начало периода по умолчанию (P4): год до end для роста и ретенции, 12 мес. для потоков."""
    if metric_id in {"arr_growth", "nrr", "grr"}:
        return add_months(end, -12)
    return add_months(end, -11)


def compute(
    dr: DataRoom,
    metric_id: str,
    period_end: str,
    period_start: str | None = None,
    variant: str = "standard",
    *,
    as_growth: bool = False,
    with_sql: bool = True,
) -> MetricResult:
    """Единая точка входа: метрика онтологии (или рост уровня метрики) за период.

    Для роста (`arr_growth` или `as_growth=True`) period_start — месяц базы; для revenue и
    logo_churn — первый месяц периода; для nrr/grr — месяц когорты; для уровней игнорируется.
    """
    from dd_verifier.metrics.sql import metric_sql  # локальный импорт: избегаем цикла

    if metric_id == "arr_growth" or as_growth:
        start = period_start or default_start("arr_growth", period_end)
        res = growth(dr, metric_id, start, period_end, variant)
    elif metric_id in PERIOD_FUNCTIONS:
        start = period_start or default_start(metric_id, period_end)
        res = PERIOD_FUNCTIONS[metric_id](dr, start, period_end, variant)
    elif metric_id in {"nrr", "grr"}:
        fn = nrr if metric_id == "nrr" else grr
        res = fn(dr, period_end, variant, cohort_month=period_start)
    elif metric_id in POINT_FUNCTIONS:
        res = POINT_FUNCTIONS[metric_id](dr, period_end, variant)
    else:
        raise KeyError(f"метрика {metric_id!r} вне онтологии")
    if not with_sql:
        return res
    sql = metric_sql(metric_id, res.period_end, res.period_start, variant, as_growth=as_growth)
    return MetricResult(
        res.metric_id,
        res.variant,
        res.value,
        res.unit,
        res.period_start,
        res.period_end,
        res.trace,
        sql,
    )


__all__ = [
    "PAYING_EPS",
    "active_customers",
    "acv",
    "arr",
    "arr_growth",
    "compute",
    "growth",
    "grr",
    "logo_churn",
    "mrr",
    "nrr",
    "revenue",
    "top10_concentration",
]
