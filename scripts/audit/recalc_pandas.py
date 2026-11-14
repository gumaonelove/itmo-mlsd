"""Независимый слепой пересчёт метрик и вердиктов для утверждений стартапов.

Скрипт самодостаточен: читает только invoices.csv, customers.csv, fx_rates.csv пяти компаний
и список утверждений без эталона (claims_blind.jsonl), считает метрики по стандартным
определениям онтологии (configs/ontology.yaml) и выносит вердикт по правилам допусков P5.

Запуск (из корня репозитория):
    uv run python scripts/audit/recalc_pandas.py --claims claims_blind.jsonl --out result.csv

Обычно его запускает scripts/audit/compare_gold.py: он готовит список утверждений без эталона.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATA = Path("data/synthetic/dev")

# Допуски и направление метрик — переписаны из configs/ontology.yaml.
TOLERANCE = {
    "mrr": ("relative", 0.05, 0.25),
    "arr": ("relative", 0.05, 0.25),
    "arr_growth": ("relative", 0.05, 0.25),
    "revenue": ("relative", 0.05, 0.25),
    "active_customers": ("relative", 0.05, 0.25),
    "logo_churn": ("pp", 0.3, 1.0),
    "grr": ("pp", 2.0, 5.0),
    "nrr": ("pp", 2.0, 5.0),
    "acv": ("relative", 0.05, 0.25),
    "top10_concentration": ("pp", 2.0, 5.0),
}
DIRECTION = {
    "mrr": "up",
    "arr": "up",
    "arr_growth": "up",
    "revenue": "up",
    "active_customers": "up",
    "logo_churn": "down",
    "grr": "up",
    "nrr": "up",
    "acv": "up",
    "top10_concentration": "down",
}
EPS = 1e-9  # защита от шума плавающей точки на границах порогов


class NotEnoughDataError(Exception):
    """Для расчёта нужен месяц вне доступной истории данных."""


def month_end(m: pd.Period) -> pd.Timestamp:
    """Последний день месяца m (момент «на дату»)."""
    return m.to_timestamp(how="end").normalize()


# ---------------------------------------------------------------------------
# Загрузка и признание выручки
# ---------------------------------------------------------------------------


@dataclass
class Company:
    """Данные одной компании: помесячно признанная выручка, справочник клиентов, границы истории."""

    rec: pd.DataFrame  # строки: invoice × месяц признания, колонка usd (net, в USD)
    customers: pd.DataFrame
    first: pd.Period  # первый месяц истории
    last: pd.Period  # последний месяц, по которому есть курсы (конец «факта»)

    def need(self, *months: pd.Period) -> None:
        """Проверить, что все месяцы лежат внутри истории, иначе — NotEnoughDataError."""
        for m in months:
            if m < self.first or m > self.last:
                raise NotEnoughDataError(f"месяц {m} вне истории [{self.first}, {self.last}]")


def load_company(path: Path) -> Company:
    """Загрузить таблицы компании и разложить инвойсы по месяцам периода услуги.

    Правила: точные дубли строк удаляются; void исключаются; кредит-ноты (отрицательные суммы)
    остаются и вычитаются; net = amount − reseller_fee; перевод в USD по курсу месяца начала
    периода услуги; сумма делится поровну на число календарных месяцев периода.
    """
    inv = pd.read_csv(path / "invoices.csv", dtype={"customer_id": "string"})
    cust = pd.read_csv(path / "customers.csv")
    fx = pd.read_csv(path / "fx_rates.csv")

    inv = inv.drop_duplicates()
    inv = inv[inv["status"] != "void"].copy()

    start = pd.to_datetime(inv["service_period_start"]).dt.to_period("M")
    end = pd.to_datetime(inv["service_period_end"]).dt.to_period("M")
    inv["start_m"] = start
    inv["n_months"] = (end - start).apply(lambda x: x.n) + 1

    fx["month"] = pd.PeriodIndex(fx["month"], freq="M")
    inv = inv.merge(
        fx.rename(columns={"month": "start_m"}),
        on=["start_m", "currency"],
        how="left",
        validate="m:1",
    )
    if inv["usd_rate"].isna().any():
        raise ValueError(f"{path.name}: нет курса для части инвойсов")

    inv["net_usd_total"] = (inv["amount"] - inv["reseller_fee"]) * inv["usd_rate"]
    inv["usd"] = inv["net_usd_total"] / inv["n_months"]

    # Разворачиваем каждый инвойс в n_months строк (по одной на месяц признания).
    rep = inv.loc[inv.index.repeat(inv["n_months"])].copy()
    rep["k"] = rep.groupby(level=0).cumcount()
    rep["month"] = rep["start_m"] + rep["k"]
    rec = rep[["invoice_id", "customer_id", "payment_type", "status", "month", "usd"]].reset_index(
        drop=True
    )

    for col in ("signed_date", "go_live_date", "churn_date"):
        cust[col] = pd.to_datetime(cust[col])

    return Company(rec=rec, customers=cust, first=fx["month"].min(), last=fx["month"].max())


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------


def mrr(co: Company, m: pd.Period, attributed_only: bool = False) -> float:
    """MRR(m): подписочная net-выручка, признанная в месяце m (включая кредит-ноты)."""
    r = co.rec[(co.rec["month"] == m) & (co.rec["payment_type"] == "subscription")]
    if attributed_only:
        r = r[r["customer_id"].notna()]
    return float(r["usd"].sum())


def arr(co: Company, m: pd.Period, attributed_only: bool = False) -> float:
    """ARR(m) = 12 × MRR(m)."""
    return 12.0 * mrr(co, m, attributed_only)


def revenue(co: Company, start: pd.Period, end: pd.Period, attributed_only: bool = False) -> float:
    """Признанная выручка всех типов платежей за месяцы [start, end]."""
    r = co.rec[(co.rec["month"] >= start) & (co.rec["month"] <= end)]
    if attributed_only:
        r = r[r["customer_id"].notna()]
    return float(r["usd"].sum())


def customer_mrr(co: Company, m: pd.Period) -> pd.Series:
    """Подписочный MRR по клиентам в месяце m (инвойсы без customer_id отбрасываются)."""
    r = co.rec[
        (co.rec["month"] == m)
        & (co.rec["payment_type"] == "subscription")
        & co.rec["customer_id"].notna()
    ]
    return r.groupby("customer_id")["usd"].sum()


def directory_active(co: Company, m: pd.Period) -> set[str]:
    """Клиенты справочника, запущенные к концу m и не ушедшие к концу m (без условия оплаты)."""
    c = co.customers
    me = month_end(m)
    ok = (
        c["go_live_date"].notna()
        & (c["go_live_date"] <= me)
        & (c["churn_date"].isna() | (c["churn_date"] > me))
    )
    return set(c.loc[ok, "customer_id"])


def active_set(co: Company, m: pd.Period) -> set[str]:
    """Активные платящие на конец m: go-live ≤ конец m, не ушли, подписочный MRR_c(m) > 0."""
    cm = customer_mrr(co, m)
    paying = set(cm[cm > 0].index)
    return directory_active(co, m) & paying


def active_customers(co: Company, m: pd.Period) -> int:
    """Число активных платящих клиентов на конец m."""
    return len(active_set(co, m))


def logo_churn(co: Company, start: pd.Period, end: pd.Period, directory: bool = False) -> float:
    """Средняя по месяцам [start, end] доля ушедших в месяце m среди активных на конец m−1.

    «Ушёл в месяце m» — churn_date (первый день без услуги) приходится на месяц m.
    directory=True — альтернатива: активность только по справочнику, без условия MRR > 0.
    """
    churn_m = co.customers.set_index("customer_id")["churn_date"].dt.to_period("M")
    rates = []
    m = start
    while m <= end:
        base = directory_active(co, m - 1) if directory else active_set(co, m - 1)
        churned = [c for c in base if churn_m.get(c) == m]
        rates.append(len(churned) / len(base))
        m += 1
    return sum(rates) / len(rates)


def retention(co: Company, m: pd.Period, kind: str, zero_churned: bool = True) -> float:
    """GRR / NRR за 12 месяцев на месяц m по когорте активных платящих клиентов в m−12.

    NRR = Σ MRR_c(m) / Σ MRR_c(m−12); GRR = Σ min(MRR_c(m), MRR_c(m−12)) / Σ MRR_c(m−12).
    zero_churned=True — у ушедших к концу m клиентов MRR_c(m) принудительно 0
    (даже если у них ещё признаётся предоплата за период после churn_date).
    """
    base_m = m - 12
    cohort = sorted(active_set(co, base_m))
    before = customer_mrr(co, base_m).reindex(cohort).fillna(0.0)
    now = customer_mrr(co, m).reindex(cohort).fillna(0.0)
    if zero_churned:
        ch = co.customers.set_index("customer_id")["churn_date"].reindex(cohort)
        gone = ch.notna() & (ch <= month_end(m))
        now[gone.values] = 0.0
    now = now.clip(lower=0.0)  # отрицательный MRR (переизбыток кредит-нот) считаем нулём
    capped = pd.concat([now, before], axis=1).min(axis=1)
    num = now.sum() if kind == "nrr" else capped.sum()
    return float(num / before.sum())


def top10_concentration(co: Company, m: pd.Period, attributed_denominator: bool = False) -> float:
    """Доля 10 крупнейших клиентов в признанной выручке (все типы, net) за [m−11, m]."""
    r = co.rec[(co.rec["month"] >= m - 11) & (co.rec["month"] <= m)]
    by_c = r[r["customer_id"].notna()].groupby("customer_id")["usd"].sum()
    top = by_c.sort_values(ascending=False).head(10).sum()
    denom = by_c.sum() if attributed_denominator else r["usd"].sum()
    return float(top / denom)


# ---------------------------------------------------------------------------
# Вердикт
# ---------------------------------------------------------------------------


def verdict(
    metric: str, claimed: float, fact: float, qualifier: str | None, is_growth: bool
) -> tuple[float, str, str]:
    """Отклонение δ и вердикт по правилам P5 с учётом квантификатора.

    Возвращает (δ, вердикт, пометка). δ всегда считается как (заявлено − факт) [/факт | ×100],
    где «заявлено» — число N из утверждения.
    """
    mode, lower, upper = TOLERANCE[metric]
    up = DIRECTION[metric] == "up"
    delta = (claimed - fact) / fact if mode == "relative" else (claimed - fact) * 100.0

    if qualifier == "more_than" and fact >= claimed:
        return delta, "SUPPORTED", "квантификатор more_than выполнен (факт ≥ N)"
    if qualifier == "less_than" and fact <= claimed:
        return delta, "SUPPORTED", "квантификатор less_than выполнен (факт ≤ N)"

    if is_growth and ((claimed > 1 and fact < 1) or (claimed < 1 and fact > 1)):
        return delta, "CONTRADICTED", "противоположное направление роста"

    if abs(delta) <= lower + EPS:
        return delta, "SUPPORTED", ""
    favourable = delta > 0 if up else delta < 0
    if favourable and abs(delta) <= upper + EPS:
        return delta, "EXAGGERATED", ""
    return delta, "CONTRADICTED", "" if favourable else "отклонение не в пользу компании"


# ---------------------------------------------------------------------------
# Расчёт по утверждению
# ---------------------------------------------------------------------------


def compute(co: Company, cl: dict) -> tuple[float, float | None, str]:
    """Значение метрики, альтернативное значение и пояснение для одного утверждения."""
    metric = cl["metric_id"]
    end = pd.Period(cl["period_end"], freq="M")
    start = pd.Period(cl["period_start"], freq="M") if cl.get("period_start") else None
    growth = cl["claim_type"] == "growth"

    if metric == "mrr":
        co.need(end)
        return mrr(co, end), mrr(co, end, True), "alt: без инвойсов с пустым customer_id"
    if metric == "arr":
        co.need(end)
        return arr(co, end), arr(co, end, True), "alt: без инвойсов с пустым customer_id"
    if metric == "arr_growth":
        # period_start — базовый месяц (для «роста в 2025» это 2024-12), как в онтологии.
        co.need(start, end)
        return (
            arr(co, end) / arr(co, start),
            arr(co, end) / arr(co, start + 1),
            f"ARR({end})/ARR({start}); alt: база — первый месяц внутри периода ARR({start + 1})",
        )
    if metric == "revenue":
        co.need(start, end)
        return (
            revenue(co, start, end),
            revenue(co, start, end, True),
            "alt: без инвойсов с пустым customer_id",
        )
    if metric == "active_customers":
        if growth:
            co.need(start, end)
            v = active_customers(co, end) / active_customers(co, start)
            alt = len(directory_active(co, end)) / len(directory_active(co, start))
            return v, alt, f"рост = AC({end})/AC({start}); alt: активность только по справочнику"
        co.need(end)
        return (
            float(active_customers(co, end)),
            float(len(directory_active(co, end))),
            "alt: активность только по справочнику",
        )
    if metric == "acv":
        co.need(end)
        act = active_set(co, end)
        v = arr(co, end) / len(act)
        alt = 12.0 * customer_mrr(co, end).reindex(sorted(act)).sum() / len(act)
        return v, alt, "alt: ARR только по MRR активных клиентов"
    if metric == "logo_churn":
        co.need(start - 1, end)
        return (
            logo_churn(co, start, end),
            logo_churn(co, start, end, True),
            "alt: активность только по справочнику",
        )
    if metric in ("nrr", "grr"):
        # period_start в утверждениях — это m−12; само значение считается на period_end.
        co.need(end - 12, end)
        return (
            retention(co, end, metric),
            retention(co, end, metric, zero_churned=False),
            f"когорта активных в {end - 12}; alt: MRR ушедших не обнуляется",
        )
    if metric == "top10_concentration":
        co.need(end - 11, end)
        return (
            top10_concentration(co, end),
            top10_concentration(co, end, True),
            "alt: знаменатель без инвойсов с пустым customer_id",
        )
    raise ValueError(f"неизвестная метрика {metric}")


def fmt(x: float | None) -> str:
    """Число с полной точностью или пустая строка."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return repr(float(x))


def main() -> None:
    """Точка входа: пересчитать все утверждения и записать result.csv."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--claims", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    claims = [json.loads(line) for line in args.claims.read_text().splitlines() if line.strip()]
    companies = {
        cid: load_company(args.data_dir / cid) for cid in sorted({c["company_id"] for c in claims})
    }

    rows = []
    for cl in claims:
        base = {"claim_id": cl["claim_id"], "metric_id": cl["metric_id"]}
        if cl["claim_type"] == "forecast":
            rows.append(
                {
                    **base,
                    "value": "",
                    "delta": "",
                    "verdict": "UNVERIFIABLE",
                    "alt_value": "",
                    "note": "прогноз",
                }
            )
            continue
        if cl["metric_id"] not in TOLERANCE:
            rows.append(
                {
                    **base,
                    "value": "",
                    "delta": "",
                    "verdict": "NOT_ENOUGH_DATA",
                    "alt_value": "",
                    "note": "метрика вне онтологии",
                }
            )
            continue
        co = companies[cl["company_id"]]
        try:
            value, alt, note = compute(co, cl)
        except NotEnoughDataError as e:
            rows.append(
                {
                    **base,
                    "value": "",
                    "delta": "",
                    "verdict": "NOT_ENOUGH_DATA",
                    "alt_value": "",
                    "note": str(e),
                }
            )
            continue
        delta, v, vnote = verdict(
            cl["metric_id"],
            float(cl["claimed_value"]),
            value,
            cl.get("qualifier"),
            cl["claim_type"] == "growth",
        )
        rows.append(
            {
                **base,
                "value": fmt(value),
                "delta": fmt(delta),
                "verdict": v,
                "alt_value": fmt(alt),
                "note": "; ".join(x for x in (vnote, note) if x),
            }
        )

    out = pd.DataFrame(
        rows, columns=["claim_id", "metric_id", "value", "delta", "verdict", "alt_value", "note"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"записано {len(out)} строк в {args.out}")
    print(out["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
