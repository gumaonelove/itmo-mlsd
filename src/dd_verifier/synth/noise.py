"""«Грязь» выгрузки поверх чистых таблиц (блок `noise:` в configs/synth.yaml).

Большая часть шума нейтральна для метрик при правильной обработке (void, пары
ошибочный инвойс + кредит-нота, поздние инвойсы); дубли строк удаляются загрузчиком;
пустой customer_id и записи-дубли клиентов немного влияют на клиентские метрики —
это измеряется в manifest.json (`dirty_vs_clean`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dd_verifier.metrics.core import add_months, month_start
from dd_verifier.synth.names import name_variant


def _pick(rng: np.random.Generator, idx: pd.Index, rate: float) -> pd.Index:
    n = int(rng.binomial(len(idx), rate)) if len(idx) else 0
    return pd.Index(sorted(rng.choice(idx, size=n, replace=False))) if n else pd.Index([])


def add_noise(
    customers: pd.DataFrame,
    invoices: pd.DataFrame,
    noise: dict,
    end: str,
    prefix: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Вернуть «грязные» customers, invoices и счётчики добавленного шума."""
    inv = invoices.copy()
    counts: dict[str, int] = {}
    next_id = len(inv) + 1

    def new_ids(n: int) -> list[str]:
        nonlocal next_id
        ids = [f"INV-{prefix}-{k:06d}" for k in range(next_id, next_id + n)]
        next_id += n
        return ids

    one_month = inv["service_period_start"].dt.to_period("M") == inv[
        "service_period_end"
    ].dt.to_period("M")
    is_sub = inv["payment_type"] == "subscription"
    regular = inv["status"].isin(["paid", "open"])

    # поздние инвойсы: выставлены через 1–2 месяца после начала услуги
    cutoff = month_start(add_months(end, -noise["late_issue_max_months"]))
    late = _pick(
        rng,
        inv.index[one_month & regular & (inv["service_period_start"] < cutoff)],
        noise["late_issue_rate"],
    )
    shift = rng.integers(1, noise["late_issue_max_months"] + 1, size=len(late))
    days = rng.integers(0, 20, size=len(late))
    inv.loc[late, "issue_date"] = [
        d + pd.DateOffset(months=int(s)) + pd.Timedelta(days=int(x))
        for d, s, x in zip(inv.loc[late, "issue_date"], shift, days, strict=True)
    ]
    counts["late_issued"] = len(late)

    # аннулированные и перевыставленные: исходный инвойс void, рядом корректный
    void_src = _pick(rng, inv.index[regular], noise["void_reissue_rate"])
    voids = inv.loc[void_src].copy()
    voids["invoice_id"], voids["status"] = new_ids(len(voids)), "void"
    voids["issue_date"] = voids["issue_date"] - pd.Timedelta(days=2)
    counts["void_reissued"] = len(voids)

    # ошибочный инвойс + полная кредит-нота на него
    err_src = _pick(rng, inv.index[regular], noise["erroneous_invoice_credit_rate"])
    wrong = inv.loc[err_src].copy()
    wrong["invoice_id"], wrong["status"] = new_ids(len(wrong)), "paid"
    credit = wrong.copy()
    credit["invoice_id"], credit["status"] = new_ids(len(credit)), "credit_note"
    credit["amount"], credit["reseller_fee"] = -credit["amount"], -credit["reseller_fee"]
    credit["issue_date"] = credit["issue_date"] + pd.Timedelta(days=10)
    counts["erroneous_with_credit_note"] = len(wrong)

    # пустой customer_id у части помесячных подписочных инвойсов
    blank = _pick(rng, inv.index[one_month & is_sub & regular], noise["missing_customer_id_rate"])
    inv.loc[blank, "customer_id"] = None
    counts["missing_customer_id"] = len(blank)

    inv = pd.concat([inv, voids, wrong, credit], ignore_index=True)
    dups = inv.loc[_pick(rng, inv.index, noise["duplicate_invoice_rate"])]
    counts["duplicate_rows"] = len(dups)
    inv = pd.concat([inv, dups], ignore_index=True).sort_values(
        ["issue_date", "invoice_id"], kind="stable"
    )

    # справочник: другое написание имён и записи-дубли клиентов без инвойсов
    cust = customers.copy()
    variant = _pick(rng, cust.index, noise["customer_name_case_rate"])
    cust.loc[variant, "customer_name"] = [
        name_variant(n, rng) for n in cust.loc[variant, "customer_name"]
    ]
    counts["customer_name_variants"] = len(variant)
    live = cust.index[cust["go_live_date"].notna() & ~cust["is_pilot"]]
    dup_src = _pick(rng, live, noise["customer_duplicate_record_rate"])
    dup = cust.loc[dup_src].copy()
    base = int(cust["customer_id"].str[1:].astype(int).max())
    dup["customer_id"] = [f"C{base + k:05d}" for k in range(1, len(dup) + 1)]
    dup["customer_name"] = [name_variant(n, rng) for n in dup["customer_name"]]
    counts["duplicate_customer_records"] = len(dup)
    cust = pd.concat([cust, dup], ignore_index=True)
    return cust, inv.reset_index(drop=True), counts
