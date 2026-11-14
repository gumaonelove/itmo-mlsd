"""Калибровка масштаба и «посадка» расхождений каталога в данные (до генерации утверждений).

Каждая функция получает чистые таблицы, меняет их минимально и возвращает новые таблицы
и словарь параметров (что сделано) для manifest.json. Размер эффекта задаётся целевым
разрывом между вариантом компании и стандартом и проверяется библиотекой метрик.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from dd_verifier.metrics import DataRoom, arr, compute, mrr
from dd_verifier.metrics.core import add_months, month_end, month_start
from dd_verifier.synth.simulate import scale_invoices


class PlantingError(RuntimeError):
    """Расхождение нельзя посадить с заданными параметрами — конфиг нужно поправить."""


def _dr(t: dict) -> DataRoom:
    return DataRoom(t["customers"], t["invoices"], t["fx_rates"])


def _month_mask(inv: pd.DataFrame, month: str) -> pd.Series:
    return (inv["service_period_start"] <= month_end(month)) & (
        inv["service_period_end"] >= month_start(month)
    )


def _go_live_before(t: dict, date: str) -> pd.Series:
    """Маска инвойсов клиентов, запущенных до даты (старая когорта)."""
    live = t["customers"].set_index("customer_id")["go_live_date"]
    return t["invoices"]["customer_id"].map(live) < pd.Timestamp(date)


def scale_to_arr(t: dict, end: str, arr_end: float) -> tuple[dict, dict]:
    """Масштабировать все суммы так, чтобы стандартный ARR на конец истории = arr_end."""
    factor = arr_end / arr(_dr(t), end).value
    inv = scale_invoices(t["invoices"], pd.Series(True, index=t["invoices"].index), factor)
    return {**t, "invoices": inv}, {"scale": round(factor, 6)}


def calibrate_two_point(t: dict, cal: dict, end: str) -> tuple[dict, dict]:
    """Калибровка Pantrio: ARR (стандарт и с внедрением) в двух точках.

    Подписку старой когорты (go-live до начала года роста) масштабируем на a, новой — на b;
    платежи за внедрение — на c и d. Система линейна, поэтому решается точно.
    """
    start, cut = cal["arr_start_month"], f"{add_months(cal['arr_start_month'], 1)}-01"
    old = _go_live_before(t, cut)
    inv = t["invoices"]
    sub = inv["payment_type"] == "subscription"
    impl = inv["payment_type"] == "implementation"
    one = inv["payment_type"] == "one_time"
    dr = _dr(t)
    led = dr.ledger.assign(
        old=dr.ledger["invoice_id"].map(dict(zip(inv["invoice_id"], old, strict=True)))
    )

    def total(month: str, ptype: str, is_old: bool) -> float:
        rows = led.loc[
            (led["month"] == month) & (led["payment_type"] == ptype) & (led["old"] == is_old)
        ]
        return float(rows["net_usd"].sum())

    a = cal["arr_start"] / 12 / total(start, "subscription", True)
    b = (cal["arr_end"] / 12 - a * total(end, "subscription", True)) / total(
        end, "subscription", False
    )
    extra_start = (cal["arr_company_start"] - cal["arr_start"]) / 12
    extra_end = (cal["arr_company_end"] - cal["arr_end"]) / 12
    c = (extra_start - a * total(start, "one_time", True)) / total(start, "implementation", True)
    d = (
        extra_end
        - a * total(end, "one_time", True)
        - b * total(end, "one_time", False)
        - c * total(end, "implementation", True)
    ) / total(end, "implementation", False)
    if min(a, b, c, d) <= 0:
        raise PlantingError(
            f"калибровка Pantrio невозможна: a={a:.3f} b={b:.3f} c={c:.3f} d={d:.3f}"
        )
    for mask, f in (
        ((sub | one) & old, a),
        ((sub | one) & ~old, b),
        (impl & old, c),
        (impl & ~old, d),
    ):
        inv = scale_invoices(inv, mask, f)
    return {**t, "invoices": inv}, {
        "a_sub_old": a,
        "b_sub_new": b,
        "c_impl_old": c,
        "d_impl_new": d,
    }


def _add_rows(t: dict, rows: list[dict]) -> dict:
    inv = pd.concat([t["invoices"], pd.DataFrame(rows)], ignore_index=True)
    return {**t, "invoices": inv}


def _usd_to_local(t: dict, month: str, currency: str) -> float:
    fx = t["fx_rates"]
    return float(fx.loc[(fx["month"] == month) & (fx["currency"] == currency), "usd_rate"].iloc[0])


def _row(t: dict, cust: dict, month: str, usd_net: float, status: str) -> dict:
    """Инвойс на месяц услуги с заданной net-суммой в USD (cust: id, валюта, ставка комиссии)."""
    fee_rate = cust["fee_rate"]
    gross_local = usd_net / (1 - fee_rate) / _usd_to_local(t, month, cust["currency"])
    return {
        "customer_id": cust["customer_id"],
        "issue_date": month_start(month),
        "service_period_start": month_start(month),
        "service_period_end": month_end(month),
        "amount": round(gross_local, 2),
        "currency": cust["currency"],
        "payment_type": "subscription",
        "status": status,
        "reseller_fee": round(gross_local * fee_rate, 2),
    }


def _billing_profile(t: dict) -> dict[str, dict]:
    """Валюта и ставка комиссии реселлера каждого клиента по его инвойсам."""
    inv = t["invoices"]
    ccy = inv.groupby("customer_id")["currency"].first()
    ratio = inv["reseller_fee"] / inv["amount"].where(inv["amount"] != 0)
    fee = ratio.groupby(inv["customer_id"]).max().fillna(0.0).clip(0, 0.9)
    return {
        cid: {"customer_id": cid, "currency": ccy[cid], "fee_rate": float(fee[cid])}
        for cid in ccy.index
    }


def plant_seasonal_dip(t: dict, rng: np.random.Generator, gap: float) -> tuple[dict, dict]:
    """D11: в окне квартала клиенты сезонно сокращают подписку — ARR конца окна ниже начала.

    Основной механизм — помесячные клиенты снижают подписку в 3 месяцах окна (масштаб factor);
    если помесячных мало (годовая предоплата), то сезонные скидки кредит-нотами в тех же месяцах.
    """
    inv = t["invoices"]
    windows = [("2025-06", "2025-09"), ("2025-03", "2025-06")]
    tried: list[float] = []
    monthly = (inv["payment_type"] == "subscription") & (
        inv["service_period_start"].dt.to_period("M") == inv["service_period_end"].dt.to_period("M")
    )
    mon_ids = set(inv.loc[monthly, "invoice_id"])
    dr = _dr(t)
    led = dr.ledger
    for k in rng.permutation(len(windows)):
        w0, w1 = windows[k]
        months = [add_months(w0, j) for j in (1, 2, 3)]
        base, end_total = mrr(dr, w0).value, mrr(dr, w1).value
        mo_end = float(
            led.loc[
                (led["month"] == w1)
                & led["invoice_id"].isin(mon_ids)
                & (led["payment_type"] == "subscription"),
                "net_usd",
            ].sum()
        )
        s = ((1 - gap) * base - (end_total - mo_end)) / mo_end if mo_end > 0 else -1
        tried.append(round(s, 3))
        if 0.3 <= s <= 0.97:
            mask = monthly & pd.concat([_month_mask(inv, m) for m in months], axis=1).any(axis=1)
            return {**t, "invoices": scale_invoices(inv, mask, s)}, {
                "window": [w0, w1],
                "factor": s,
            }
    w0, w1 = windows[int(rng.integers(len(windows)))]
    cut = 1 - (1 - gap) * mrr(dr, w0).value / mrr(dr, w1).value
    if not 0 < cut <= 0.5:
        raise PlantingError(
            f"D11: нет окна с допустимым сокращением подписки (factor={tried}, cut={cut:.2f})"
        )
    for m in (add_months(w0, j) for j in (1, 2, 3)):
        t = _add_credits(t, m, cut * mrr(_dr(t), m).value)
    return t, {"window": [w0, w1], "credit_share": cut}


def _retention_gap(dr: DataRoom, end: str) -> float:
    """На сколько нужно снизить NRR/GRR на конец истории, чтобы лучшее значение 6–12 месяцев
    назад было выше текущего (в долях; ≤ 0 — уже выше)."""
    need = []
    for metric in ("nrr", "grr"):
        now = compute(dr, metric, end, add_months(end, -12), with_sql=False).value
        best = max(
            compute(dr, metric, m, add_months(m, -12), with_sql=False).value
            for m in (add_months(end, -lag) for lag in (6, 9, 12))
        )
        need.append(now - best)
    return max(need)


def plant_recent_downgrade(
    t: dict, rng: np.random.Generator, gap: float, end: str
) -> tuple[dict, dict]:
    """D09: в последние 3 месяца часть клиентов старой когорты получает скидку (кредит-ноты).

    После посадки лучшее значение NRR и GRR за 6–12 месяцев до конца истории выше текущего
    примерно на gap (доли, 0,04 = 4 п.п.), поэтому у компании появляется «удобное» прошлое
    значение; ACV и рост ARR тоже снижаются. Кредит-ноты, а не правка инвойсов, — чтобы
    годовая предоплата не меняла прошлые месяцы.
    """
    dr = _dr(t)
    months = [add_months(end, -k) for k in (2, 1, 0)]
    cohort = dr.active_ids(add_months(end, -12))
    eligible = sorted(set(cohort) & set(dr.active_ids(end)))
    if not eligible:
        raise PlantingError("D09: нет клиентов старой когорты")
    base = float(dr.mrr_of(add_months(end, -12)).reindex(cohort).fillna(0).sum())
    mrr_end = dr.mrr_of(end)
    drop = max(_retention_gap(dr, end), 0.0) + gap
    prof = _billing_profile(t)
    order = list(rng.permutation(eligible))
    tried = []
    for share in (0.5, 0.3, 0.2, 0.1, 0.75, 1.0):  # доля клиентов со скидкой
        chosen = sorted(order[: max(1, round(len(eligible) * share))])
        cut = drop * base / float(mrr_end.reindex(chosen).sum())
        tried.append(round(cut, 2))
        if not 0.1 <= cut <= 0.6:
            continue
        rows = []
        for m in months:
            mrr_m = dr.mrr_of(m)
            for cid in chosen:
                row = _row(t, prof[cid], m, cut * float(mrr_m.get(cid, 0.0)), "credit_note")
                row["amount"], row["reseller_fee"] = -row["amount"], -row["reseller_fee"]
                rows.append(row)
        out = _add_rows(t, rows)
        if _retention_gap(_dr(out), end) < -gap / 2:  # GRR с потолком реагирует слабее
            return out, {"months": months, "customers": len(chosen), "cut": cut, "drop": drop}
    raise PlantingError(f"D09: скидка {tried} не дала ухудшения NRR/GRR")


def plant_peak_month(
    t: dict, rng: np.random.Generator, gap: float, end: str, avoid: set
) -> tuple[dict, dict]:
    """D02: разовая месячная «сезонная надстройка» создаёт пик MRR на gap выше последнего месяца."""
    candidates = [m for m in (add_months(end, -k) for k in (3, 4, 5, 6)) if m not in avoid]
    month = candidates[int(rng.integers(len(candidates)))]
    dr = _dr(t)
    add = (1 + gap) * mrr(dr, end).value - mrr(dr, month).value
    if add <= 0:
        raise PlantingError("D02: MRR месяца-пика уже выше целевого")
    ids = dr.active_ids(month)
    chosen = sorted(rng.choice(ids, size=max(1, len(ids) // 3), replace=False))
    weights = dr.mrr_of(month).reindex(chosen).to_numpy()
    prof = _billing_profile(t)
    rows = [
        _row(t, prof[cid], month, add * w / weights.sum(), "paid")
        for cid, w in zip(chosen, weights, strict=True)
    ]
    return _add_rows(t, rows), {"month": month, "add_usd": add, "customers": len(chosen)}


def plant_trough(t: dict, rng: np.random.Generator, gap: float, start: str) -> tuple[dict, dict]:
    """D04: SLA-кредит-ноты в одном из первых месяцев окна — MRR проседает, рост «от дна» выше."""
    month = add_months(start, int(rng.integers(1, 3)))
    dr = _dr(t)
    credit = mrr(dr, month).value - mrr(dr, start).value / (1 + gap)
    return _add_credits(t, month, credit), {"month": month, "credit_usd": credit}


def _add_credits(t: dict, month: str, credit: float) -> dict:
    """Кредит-ноты на сумму credit (USD, net) активным клиентам месяца пропорционально их MRR."""
    dr = _dr(t)
    ids = dr.active_ids(month)
    weights = dr.mrr_of(month).reindex(ids).to_numpy()
    if credit <= 0 or credit > 0.5 * weights.sum():
        raise PlantingError(f"кредит {credit:.0f} в {month} вне допустимого диапазона")
    prof = _billing_profile(t)
    rows = []
    for cid, w in zip(ids, weights, strict=True):
        row = _row(t, prof[cid], month, credit * w / weights.sum(), "credit_note")
        row["amount"], row["reseller_fee"] = -row["amount"], -row["reseller_fee"]
        rows.append(row)
    return _add_rows(t, rows)


def calibrate_reseller_fee(t: dict, gap: float, metric: str, end: str) -> tuple[dict, dict]:
    """D05: ставка комиссии реселлера так, чтобы gross-вариант был выше net на gap."""
    dr = _dr(t)
    inv = t["invoices"]
    resellers = set(t["customers"].loc[t["customers"]["channel"] == "reseller", "customer_id"])
    led = dr.ledger
    if metric == "arr":
        rows = led.loc[(led["month"] == end) & (led["payment_type"] == "subscription")]
    else:
        rows = led.loc[led["month"].between(add_months(end, -11), end)]
    total_gross = float(rows["gross_usd"].sum())
    reseller_gross = float(rows.loc[rows["customer_id"].isin(resellers), "gross_usd"].sum())
    if reseller_gross <= 0:
        raise PlantingError("D05: нет реселлерской выручки")
    f = gap * total_gross / (reseller_gross * (1 + gap))
    if not 0.05 <= f <= 0.45:
        raise PlantingError(f"D05: ставка комиссии {f:.2f} нереалистична")
    inv = inv.copy()
    mask = inv["customer_id"].isin(resellers)
    inv.loc[mask, "reseller_fee"] = (inv.loc[mask, "amount"] * f).round(2)
    return {**t, "invoices": inv}, {"fee_rate": f, "metric": metric}


def calibrate_one_time(t: dict, gap: float, end: str) -> tuple[dict, dict]:
    """D01: масштаб платежей за внедрение так, чтобы MRR с разовыми был выше стандарта на gap."""
    dr = _dr(t)
    led = dr.ledger.loc[dr.ledger["month"] == end]
    std = float(led.loc[led["payment_type"] == "subscription", "net_usd"].sum())
    impl = float(led.loc[led["payment_type"] == "implementation", "net_usd"].sum())
    one = float(led.loc[led["payment_type"] == "one_time", "net_usd"].sum())
    if impl <= 0:
        raise PlantingError("D01: нет платежей за внедрение в последнем месяце")
    k = (gap * std - one) / impl
    if k <= 0:
        raise PlantingError("D01: разовые платежи уже превышают целевой разрыв")
    mask = t["invoices"]["payment_type"] == "implementation"
    return {**t, "invoices": scale_invoices(t["invoices"], mask, k)}, {"impl_scale": k}


def plant_pilots(
    t: dict, rng: np.random.Generator, gap: float, end: str, names: Iterator[str]
) -> tuple[dict, dict]:
    """D03: добавить бесплатные пилоты, чтобы «подписанные» превышали активных на gap."""
    from dd_verifier.metrics import active_customers

    dr = _dr(t)
    active = active_customers(dr, end).value
    signed = active_customers(dr, end, "signed_incl_pilots").value
    need = round((1 + gap) * active - signed)
    if need < 0:
        raise PlantingError("D03: пилотов уже больше целевого")
    c = t["customers"]
    sample = c.loc[~c["is_pilot"] & c["go_live_date"].notna()].sample(
        n=need, replace=True, random_state=int(rng.integers(2**31))
    )
    start_id = int(c["customer_id"].str[1:].astype(int).max())
    rows = []
    for k, (_, src) in enumerate(sample.iterrows(), start=1):
        rows.append(
            {
                "customer_id": f"C{start_id + k:05d}",
                "customer_name": next(names),
                "segment": src["segment"],
                "country": src["country"],
                "channel": "direct",
                "signed_date": month_start(end) - pd.Timedelta(days=int(rng.integers(0, 85))),
                "go_live_date": pd.NaT,
                "churn_date": pd.NaT,
                "is_pilot": True,
            }
        )
    return {**t, "customers": pd.concat([c, pd.DataFrame(rows)], ignore_index=True)}, {
        "pilots_added": need
    }
