"""Независимый слепой пересчёт метрик и вердиктов по утверждениям (вторая реализация, SQL).

Вся логика признания выручки и расчёта метрик написана вручную на SQL (DuckDB, in-memory)
по формулам и стандартным определениям configs/ontology.yaml. Pandas используется только
для чтения claims_blind.jsonl и записи итогового CSV.

Запуск (из каталога репозитория):
    uv run python scripts/audit/recalc_sql.py --claims claims_blind.jsonl --out result.csv

Обычно его запускает scripts/audit/compare_gold.py: он готовит список утверждений без эталона.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DATA = Path("data/synthetic/dev")
COMPANIES = ["brewbase", "clinicore", "pantrio", "routelane", "tallybird"]

# Допуски и направление метрик — переписаны вручную из configs/ontology.yaml.
TOLERANCE = {
    "mrr": ("relative", 0.05, 0.25, "up"),
    "arr": ("relative", 0.05, 0.25, "up"),
    "arr_growth": ("relative", 0.05, 0.25, "up"),
    "revenue": ("relative", 0.05, 0.25, "up"),
    "active_customers": ("relative", 0.05, 0.25, "up"),
    "logo_churn": ("pp", 0.3, 1.0, "down"),
    "grr": ("pp", 2.0, 5.0, "up"),
    "nrr": ("pp", 2.0, 5.0, "up"),
    "acv": ("relative", 0.05, 0.25, "up"),
    "top10_concentration": ("pp", 2.0, 5.0, "down"),
}


class NotEnoughDataError(Exception):
    """Для расчёта не хватает истории (месяц вне покрытия данных или пустая когорта)."""


# ---------------------------------------------------------------------------
# Загрузка и признание выручки
# ---------------------------------------------------------------------------


def load(con: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    """Читает CSV пяти компаний и строит базовые таблицы.

    inv   — инвойсы без точных дублей строк (DISTINCT по всем колонкам);
    rec   — помесячное признание: каждый не-void инвойс (кредит-ноты с отрицательной суммой
            остаются и вычитаются) делится равномерно на n_months месяцев периода услуги,
            сумма net = amount − reseller_fee, перевод в USD по курсу месяца начала периода услуги;
    cus   — справочник клиентов;
    cal   — календарь покрытия данных компании: от первого месяца начала услуги в инвойсах
            до последнего месяца курсов fx (конец истории);
    cmrr  — подписочный MRR клиента по месяцам;
    active — активные платящие клиенты на конец месяца (go-live ≤ конец m, churn пусто или
            > конец m, подписочный MRR_c(m) > 0).
    """
    inv_parts, cus_parts, fx_parts = [], [], []
    for c in COMPANIES:
        d = data_dir / c
        inv_parts.append(
            f"""select distinct '{c}' as company, *
                from read_csv('{d / "invoices.csv"}', header=true,
                columns={{'invoice_id':'VARCHAR','customer_id':'VARCHAR','issue_date':'DATE',
                          'service_period_start':'DATE','service_period_end':'DATE','amount':'DOUBLE',
                          'currency':'VARCHAR','payment_type':'VARCHAR','status':'VARCHAR',
                          'reseller_fee':'DOUBLE'}})"""
        )
        cus_parts.append(
            f"""select '{c}' as company, * from read_csv('{d / "customers.csv"}', header=true,
                columns={{'customer_id':'VARCHAR','customer_name':'VARCHAR','segment':'VARCHAR',
                          'country':'VARCHAR','channel':'VARCHAR','signed_date':'DATE',
                          'go_live_date':'DATE','churn_date':'DATE','is_pilot':'BOOLEAN'}})"""
        )
        fx_parts.append(
            f"""select '{c}' as company, * from read_csv('{d / "fx_rates.csv"}', header=true,
                columns={{'month':'VARCHAR','currency':'VARCHAR','usd_rate':'DOUBLE'}})"""
        )
    con.execute("create table inv as " + " union all ".join(inv_parts))
    con.execute("create table cus as " + " union all ".join(cus_parts))
    con.execute("create table fx as " + " union all ".join(fx_parts))

    con.execute(
        """
        create table rec as
        with x as (
            select i.company, i.invoice_id, i.customer_id, i.payment_type,
                   date_trunc('month', i.service_period_start)::date as m0,
                   date_trunc('month', i.service_period_end)::date as m1,
                   date_diff('month', date_trunc('month', i.service_period_start),
                                      date_trunc('month', i.service_period_end)) + 1 as n_months,
                   (i.amount - i.reseller_fee) * f.usd_rate as usd_total
            from inv i
            left join fx f
              on f.company = i.company and f.currency = i.currency
             and f.month = strftime(i.service_period_start, '%Y-%m')
            where i.status <> 'void'
        )
        select company, invoice_id, customer_id, payment_type,
               strftime(t.m, '%Y-%m') as month,
               usd_total / n_months as usd
        from x, unnest(generate_series(m0, m1, interval 1 month)) as t(m)
        """
    )
    missing_fx = con.execute("select count(*) from rec where usd is null").fetchone()[0]
    assert missing_fx == 0, f"нет курса для {missing_fx} строк признания"

    con.execute(
        """
        create table cal as
        with b as (
            select i.company,
                   min(date_trunc('month', service_period_start))::date as first_m,
                   (select max(strptime(month || '-01', '%Y-%m-%d'))::date from fx f
                     where f.company = i.company) as last_m
            from inv i group by i.company
        )
        select company, strftime(t.m, '%Y-%m') as month, last_day(t.m) as month_end
        from b, unnest(generate_series(first_m, last_m, interval 1 month)) as t(m)
        """
    )
    con.execute(
        """
        create table cmrr as
        select company, customer_id, month, sum(usd) as mrr
        from rec where payment_type = 'subscription' and customer_id is not null
        group by all
        """
    )
    con.execute(
        """
        create table active as
        select c.company, k.month, c.customer_id, m.mrr
        from cus c
        join cal k on k.company = c.company
        join cmrr m on m.company = c.company and m.customer_id = c.customer_id and m.month = k.month
        where c.go_live_date <= k.month_end
          and (c.churn_date is null or c.churn_date > k.month_end)
          and m.mrr > 0
        """
    )


# ---------------------------------------------------------------------------
# Метрики (каждая — один SQL-запрос по формуле онтологии)
# ---------------------------------------------------------------------------


def shift(month: str, k: int) -> str:
    """Сдвигает месяц 'YYYY-MM' на k месяцев."""
    y, m = map(int, month.split("-"))
    idx = y * 12 + (m - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def require(con, company: str, *months: str) -> None:
    """Проверяет, что все нужные месяцы лежат внутри покрытия данных компании."""
    lo, hi = con.execute(
        "select min(month), max(month) from cal where company = ?", [company]
    ).fetchone()
    for m in months:
        if m < lo or m > hi:
            raise NotEnoughDataError(f"месяц {m} вне истории [{lo}, {hi}]")


def mrr(con, company: str, month: str, with_unknown: bool = True) -> float:
    """MRR(m): подписочная net-выручка в USD, признанная в месяце m (весь бизнес).

    with_unknown=False — альтернатива без инвойсов с пустым customer_id.
    """
    require(con, company, month)
    return con.execute(
        """select coalesce(sum(usd), 0) from rec
           where company = ? and month = ? and payment_type = 'subscription'
             and (? or customer_id is not null)""",
        [company, month, with_unknown],
    ).fetchone()[0]


def arr(con, company: str, month: str, with_unknown: bool = True) -> float:
    """ARR(m) = 12 × MRR(m)."""
    return 12 * mrr(con, company, month, with_unknown)


def revenue(con, company: str, start: str, end: str, with_unknown: bool = True) -> float:
    """Вся признанная net-выручка (все типы платежей) за месяцы [start, end]."""
    require(con, company, start, end)
    return con.execute(
        """select coalesce(sum(usd), 0) from rec
           where company = ? and month between ? and ? and (? or customer_id is not null)""",
        [company, start, end, with_unknown],
    ).fetchone()[0]


def active_customers(con, company: str, month: str) -> int:
    """Число активных платящих клиентов на конец месяца m."""
    require(con, company, month)
    return con.execute(
        "select count(*) from active where company = ? and month = ?", [company, month]
    ).fetchone()[0]


def logo_churn(
    con, company: str, start: str, end: str, numerator_active_only: bool = True
) -> float:
    """Средняя по месяцам [start, end] доля: ушли в m / активны на конец m−1.

    «Ушёл в месяце m» = churn_date попадает в месяц m. Основное прочтение: числитель — те
    из активных на конец m−1, у кого churn_date в m; альтернатива (numerator_active_only=False) —
    все записи справочника с churn_date в m.
    """
    require(con, company, shift(start, -1), end)
    return con.execute(
        """
        with months as (
            select month,
                   strftime(strptime(month || '-01', '%Y-%m-%d') - interval 1 month, '%Y-%m')
                       as prev
            from cal where company = $c and month between $s and $e
        ),
        per_m as (
            select mo.month,
                   (select count(*) from active a
                     where a.company = $c and a.month = mo.prev) as den,
                   case when $act then
                       (select count(*) from active a join cus u
                          on u.company = a.company and u.customer_id = a.customer_id
                        where a.company = $c and a.month = mo.prev
                          and strftime(u.churn_date, '%Y-%m') = mo.month)
                   else
                       (select count(*) from cus u
                        where u.company = $c and strftime(u.churn_date, '%Y-%m') = mo.month)
                   end as num
            from months mo
        )
        select avg(num::double / den) from per_m
        """,
        {"c": company, "s": start, "e": end, "act": numerator_active_only},
    ).fetchone()[0]


def retention(con, company: str, month: str, kind: str) -> float:
    """GRR / NRR за 12 месяцев: когорта — активные на конец m−12.

    GRR = Σ min(MRR_c(m), MRR_c(m−12)) / Σ MRR_c(m−12); NRR = Σ MRR_c(m) / Σ MRR_c(m−12);
    ушедшие клиенты остаются в когорте с MRR_c(m) = 0 (нет инвойсов → 0).
    """
    base = shift(month, -12)
    require(con, company, base, month)
    num_expr = "least(coalesce(n.mrr, 0), b.mrr)" if kind == "grr" else "coalesce(n.mrr, 0)"
    row = con.execute(
        f"""
        select sum({num_expr}), sum(b.mrr), count(*)
        from active b
        left join cmrr n on n.company = b.company and n.customer_id = b.customer_id and n.month = ?
        where b.company = ? and b.month = ?
        """,
        [month, company, base],
    ).fetchone()
    if not row[2]:
        raise NotEnoughDataError(f"пустая когорта на {base}")
    return row[0] / row[1]


def top10(con, company: str, month: str, with_unknown_in_total: bool = True) -> float:
    """Доля выручки (все типы, net) 10 крупнейших клиентов за [m−11, m] во всей выручке.

    Инвойсы без customer_id не могут попасть в топ-10, но входят в знаменатель
    (альтернатива: исключить их и из знаменателя).
    """
    start = shift(month, -11)
    require(con, company, start, month)
    return con.execute(
        """
        with per_c as (
            select customer_id, sum(usd) as rev from rec
            where company = $c and month between $s and $e and customer_id is not null
            group by customer_id
        ),
        top as (select sum(rev) as t from (select rev from per_c order by rev desc limit 10))
        select (select t from top) /
               (select sum(usd) from rec where company = $c and month between $s and $e
                  and ($u or customer_id is not null))
        """,
        {"c": company, "s": start, "e": month, "u": with_unknown_in_total},
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Утверждение → значение → вердикт
# ---------------------------------------------------------------------------


def compute(con, cl: dict) -> tuple[float, float | None, str]:
    """Возвращает (значение, альтернативное значение или None, примечание) для утверждения."""
    c, mid, ctype = cl["company_id"], cl["metric_id"], cl["claim_type"]
    s, e = cl["period_start"], cl["period_end"]
    if mid == "mrr":
        return mrr(con, c, e), mrr(con, c, e, False), "alt: без инвойсов с пустым customer_id"
    if mid == "arr":
        return arr(con, c, e), arr(con, c, e, False), "alt: без инвойсов с пустым customer_id"
    if mid == "arr_growth":
        v = arr(con, c, e) / arr(con, c, s)
        a = arr(con, c, e, False) / arr(con, c, s, False)
        return v, a, f"ARR({e})/ARR({s}); alt: без инвойсов с пустым customer_id"
    if mid == "revenue":
        return (
            revenue(con, c, s, e),
            revenue(con, c, s, e, False),
            "alt: без инвойсов с пустым customer_id",
        )
    if mid == "active_customers":
        if ctype == "growth":
            return (
                active_customers(con, c, e) / active_customers(con, c, s),
                None,
                f"active({e})/active({s})",
            )
        return active_customers(con, c, e), None, ""
    if mid == "acv":
        n = active_customers(con, c, e)
        return (
            arr(con, c, e) / n,
            arr(con, c, e, False) / n,
            "ARR/active; alt: ARR без инвойсов с пустым customer_id",
        )
    if mid == "logo_churn":
        return (
            logo_churn(con, c, s, e),
            logo_churn(con, c, s, e, False),
            "alt: числитель — все записи справочника с churn_date в месяце",
        )
    if mid in ("grr", "nrr"):
        if s and shift(e, -12) != s:
            raise ValueError(f"period_start {s} ≠ period_end−12 для {cl['claim_id']}")
        return retention(con, c, e, mid), None, f"когорта активных на {shift(e, -12)}"
    if mid == "top10_concentration":
        return (
            top10(con, c, e),
            top10(con, c, e, False),
            "alt: знаменатель без инвойсов с пустым customer_id",
        )
    raise NotEnoughDataError(f"метрика {mid} вне онтологии")


def verdict(cl: dict, fact: float) -> tuple[float, str, str]:
    """Считает δ и вердикт по правилам P5 с учётом квантификатора и направления роста."""
    mode, lower, upper, direction = TOLERANCE[cl["metric_id"]]
    claimed = float(cl["claimed_value"])
    delta = (claimed - fact) / fact if mode == "relative" else (claimed - fact) * 100
    q = cl.get("qualifier")
    note = ""
    if q == "more_than" and fact >= claimed:
        return delta, "SUPPORTED", "more_than выполнен (факт ≥ N)"
    if q == "less_than" and fact <= claimed:
        return delta, "SUPPORTED", "less_than выполнен (факт ≤ N)"
    if cl["claim_type"] == "growth" and (claimed - 1) * (fact - 1) < 0:
        return delta, "CONTRADICTED", "противоположное направление роста"
    in_favor = delta > 0 if direction == "up" else delta < 0
    if abs(delta) <= lower + 1e-12:
        v = "SUPPORTED"
    elif in_favor and abs(delta) <= upper + 1e-12:
        v = "EXAGGERATED"
    else:
        v = "CONTRADICTED"
    return delta, v, note


def main() -> None:
    """Точка входа: пересчитывает все утверждения и пишет result.csv."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--claims", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    con = duckdb.connect()
    load(con, args.data)
    claims = pd.read_json(args.claims, lines=True, dtype=False).to_dict("records")

    rows = []
    for cl in claims:
        cl = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in cl.items()}
        row = {
            "claim_id": cl["claim_id"],
            "metric_id": cl["metric_id"],
            "value": None,
            "delta": None,
            "verdict": None,
            "alt_value": None,
            "note": "",
        }
        if cl["claim_type"] == "forecast":
            row.update(verdict="UNVERIFIABLE", note="прогноз")
        else:
            try:
                value, alt, note = compute(con, cl)
                delta, v, vnote = verdict(cl, value)
                row.update(
                    value=value,
                    delta=delta,
                    verdict=v,
                    alt_value=alt,
                    note="; ".join(x for x in (note, vnote) if x),
                )
            except NotEnoughDataError as exc:
                row.update(verdict="NOT_ENOUGH_DATA", note=str(exc))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.17g")
    print(out.to_string())
    print(out["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
