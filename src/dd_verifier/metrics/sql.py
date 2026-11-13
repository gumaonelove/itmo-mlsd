"""Эквивалентные DuckDB SQL-запросы для метрик онтологии.

SQL — доказательство для аналитика (дизайн-документ, 2.2.2): тот же расчёт, что и в pandas-функциях
`definitions.py`, его можно выполнить на таблицах data room (`run_sql`). Тесты проверяют,
что SQL и pandas дают одинаковые значения.
"""

from __future__ import annotations

import duckdb

from dd_verifier.metrics.core import (
    PAYING_EPS,
    DataRoom,
    add_months,
    month_end,
    month_range,
    month_start,
)

LEDGER_CTE = """WITH inv AS (
    SELECT i.invoice_id, i.customer_id, i.payment_type, i.amount,
           coalesce(i.reseller_fee, 0) AS reseller_fee, f.usd_rate,
           date_trunc('month', i.service_period_start) AS m0,
           date_trunc('month', i.service_period_end) AS m1
    FROM invoices i
    JOIN fx_rates f
      ON f.currency = i.currency AND f.month = strftime(i.service_period_start, '%Y-%m')
    WHERE i.status <> 'void'
),
ledger_raw AS (
    SELECT *, unnest(generate_series(m0, m1, INTERVAL 1 MONTH)) AS m FROM inv
),
ledger AS (
    SELECT invoice_id, customer_id, payment_type, strftime(m, '%Y-%m') AS month,
           (amount - reseller_fee) * usd_rate / (date_diff('month', m0, m1) + 1) AS net_usd,
           amount * usd_rate / (date_diff('month', m0, m1) + 1) AS gross_usd
    FROM ledger_raw
),
cust_mrr AS (
    SELECT customer_id, month, sum(net_usd) AS mrr
    FROM ledger
    WHERE payment_type = 'subscription' AND customer_id IS NOT NULL
    GROUP BY customer_id, month
)"""


def _d(ts) -> str:
    return f"DATE '{ts:%Y-%m-%d}'"


def _mrr_expr(month: str, variant: str) -> str:
    col = "gross_usd" if variant == "gross_channel" else "net_usd"
    where = f"month = '{month}'"
    if variant != "incl_one_time":
        where += " AND payment_type = 'subscription'"
    return f"(SELECT coalesce(sum({col}), 0) FROM ledger WHERE {where})"


def _arr_expr(month: str, variant: str) -> str:
    if variant == "peak_month":
        return (
            "(SELECT 12 * max(v) FROM (SELECT month, sum(net_usd) AS v FROM ledger "
            f"WHERE payment_type = 'subscription' AND month BETWEEN '{add_months(month, -11)}' "
            f"AND '{month}' GROUP BY month))"
        )
    return f"(12 * {_mrr_expr(month, variant)})"


def _active_expr(month: str, variant: str) -> str:
    me = _d(month_end(month))
    if variant == "signed_incl_pilots":
        return (
            f"(SELECT count(*) FROM customers WHERE signed_date <= {me} "
            f"AND (churn_date IS NULL OR churn_date > {me}))"
        )
    return (
        "(SELECT count(*) FROM customers c JOIN cust_mrr cm USING (customer_id) "
        f"WHERE cm.month = '{month}' AND cm.mrr > {PAYING_EPS} AND c.go_live_date <= {me} "
        f"AND (c.churn_date IS NULL OR c.churn_date > {me}))"
    )


def _active_cohort_cte(month: str) -> str:
    me = _d(month_end(month))
    return (
        f"SELECT c.customer_id, c.churn_date, cm.mrr FROM customers c "
        f"JOIN cust_mrr cm USING (customer_id) WHERE cm.month = '{month}' "
        f"AND cm.mrr > {PAYING_EPS} AND c.go_live_date <= {me} "
        f"AND (c.churn_date IS NULL OR c.churn_date > {me})"
    )


def _logo_churn_sql(start: str, end: str, variant: str) -> str:
    rows = ",\n        ".join(
        f"('{m}', '{add_months(m, -1)}', {_d(month_start(m))}, {_d(month_end(m))}, "
        f"{_d(month_end(add_months(m, -1)))})"
        for m in month_range(start, end)
    )
    extra = (
        " AND NOT c.is_pilot AND c.segment <> 'SMB'" if variant == "excl_pilots_and_small" else ""
    )
    return f"""{LEDGER_CTE},
months(month, prev, m_start, m_end, prev_end) AS (
    VALUES
        {rows}
),
base AS (
    SELECT mo.month, c.customer_id, c.churn_date, mo.m_start, mo.m_end
    FROM months mo
    JOIN cust_mrr cm ON cm.month = mo.prev AND cm.mrr > {PAYING_EPS}
    JOIN customers c ON c.customer_id = cm.customer_id
    WHERE c.go_live_date <= mo.prev_end
      AND (c.churn_date IS NULL OR c.churn_date > mo.prev_end){extra}
),
monthly AS (
    SELECT month,
           avg(CASE WHEN churn_date BETWEEN m_start AND m_end THEN 1.0 ELSE 0.0 END) AS rate
    FROM base GROUP BY month
)
SELECT avg(rate) FROM monthly"""


def _retention_sql(metric_id: str, month: str, cohort: str, variant: str) -> str:
    me = _d(month_end(month))
    survivors = (
        f"WHERE s.churn_date IS NULL OR s.churn_date > {me}" if variant == "survivors_only" else ""
    )
    end_expr = "greatest(coalesce(e.mrr, 0), 0)"
    num = end_expr if metric_id == "nrr" else f"least({end_expr}, s.mrr)"
    return f"""{LEDGER_CTE},
s AS ({_active_cohort_cte(cohort)}),
e AS (SELECT customer_id, mrr FROM cust_mrr WHERE month = '{month}')
SELECT sum({num}) / sum(s.mrr)
FROM s LEFT JOIN e USING (customer_id)
{survivors}"""


def metric_sql(
    metric_id: str,
    period_end: str,
    period_start: str | None = None,
    variant: str = "standard",
    *,
    as_growth: bool = False,
) -> str | None:
    """SQL-запрос, возвращающий одно число — значение метрики (None, если SQL не предусмотрен)."""
    if metric_id == "arr_growth" or (as_growth and metric_id == "arr"):
        start = period_start or add_months(period_end, -12)
        if variant == "trough_base":
            base = (
                "(SELECT 12 * min(v) FROM (SELECT month, sum(net_usd) AS v FROM ledger "
                f"WHERE payment_type = 'subscription' AND month BETWEEN '{start}' "
                f"AND '{add_months(period_end, -1)}' GROUP BY month))"
            )
            return f"{LEDGER_CTE}\nSELECT {_arr_expr(period_end, 'standard')} / {base}"
        expr = f"{_arr_expr(period_end, variant)} / {_arr_expr(start, variant)}"
        return f"{LEDGER_CTE}\nSELECT {expr}"
    if as_growth:
        start = period_start or add_months(period_end, -12)
        exprs = {"mrr": _mrr_expr, "active_customers": _active_expr}
        if metric_id not in exprs:
            return None
        fn = exprs[metric_id]
        return f"{LEDGER_CTE}\nSELECT {fn(period_end, variant)} / {fn(start, variant)}"
    if metric_id == "mrr":
        return f"{LEDGER_CTE}\nSELECT {_mrr_expr(period_end, variant)}"
    if metric_id == "arr":
        return f"{LEDGER_CTE}\nSELECT {_arr_expr(period_end, variant)}"
    if metric_id == "revenue":
        start = period_start or add_months(period_end, -11)
        col = "gross_usd" if variant == "gross_channel" else "net_usd"
        return (
            f"{LEDGER_CTE}\nSELECT sum({col}) FROM ledger "
            f"WHERE month BETWEEN '{start}' AND '{period_end}'"
        )
    if metric_id == "active_customers":
        return f"{LEDGER_CTE}\nSELECT {_active_expr(period_end, variant)}"
    if metric_id == "acv":
        return (
            f"{LEDGER_CTE}\nSELECT {_arr_expr(period_end, variant)} / "
            f"{_active_expr(period_end, 'standard')}"
        )
    if metric_id == "logo_churn":
        return _logo_churn_sql(period_start or add_months(period_end, -11), period_end, variant)
    if metric_id in {"nrr", "grr"}:
        cohort = period_start or add_months(period_end, -12)
        return _retention_sql(metric_id, period_end, cohort, variant)
    if metric_id == "top10_concentration":
        start = add_months(period_end, -11)
        return f"""{LEDGER_CTE},
rev AS (
    SELECT customer_id, sum(net_usd) AS r FROM ledger
    WHERE month BETWEEN '{start}' AND '{period_end}' GROUP BY customer_id
)
SELECT (SELECT sum(r) FROM (SELECT r FROM rev WHERE customer_id IS NOT NULL
                            ORDER BY r DESC LIMIT 10)) / (SELECT sum(r) FROM rev)"""
    return None


def run_sql(dr: DataRoom, sql: str) -> float:
    """Выполнить SQL метрики на таблицах data room в DuckDB и вернуть число."""
    con = duckdb.connect()
    try:
        con.register("customers", dr.customers)
        con.register("invoices", dr.invoices)
        con.register("fx_rates", dr.fx_rates)
        (value,) = con.execute(sql).fetchone()
    finally:
        con.close()
    return float(value) if value is not None else float("nan")
