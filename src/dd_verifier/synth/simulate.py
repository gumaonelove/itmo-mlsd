"""Симуляция жизненного цикла клиентов и «чистого» биллинга одной компании.

Модель (помесячно, конец истории — end_month):
- число активных платящих клиентов следует целевой траектории `active_path`
  (лог-линейная интерполяция);
- отток детерминирован по количеству: churn_total × активные (с переносом дробной части),
  из них churn_core — для «основных» клиентов (не SMB), остальное — SMB; так фактический
  logo churn воспроизводит заданный, а вариант «без SMB» — заданный core;
- клиенты с годовой предоплатой уходят только в конце оплаченного года;
- цены задаются в валюте клиента; помесячные клиенты иногда расширяются или сокращают подписку,
  годовые — меняют цену при продлении;
- новые клиенты платят за внедрение (implementation, распределено по онбордингу), иногда — разовые
  платежи за обучение (one_time); реселлерские клиенты несут комиссию партнёра в reseller_fee;
- бесплатные пилоты и подписанные, но не запущенные клиенты есть в справочнике без инвойсов.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dd_verifier.metrics.core import add_months, month_index, month_range, month_start
from dd_verifier.synth.names import legal_name, name_pool

CURRENCY_COUNTRIES = {"USD": ["US"], "EUR": ["DE", "FR", "NL", "ES", "IE"], "GBP": ["GB"]}
FX_START = {"USD": 1.0, "EUR": 1.08, "GBP": 1.26}


@dataclass
class Customer:
    """Состояние клиента в симуляции."""

    customer_id: str
    customer_name: str
    segment: str
    country: str
    currency: str
    channel: str
    signed_date: pd.Timestamp
    go_live_date: pd.Timestamp | None
    billing: str = "monthly"
    price: float = 0.0  # подписка в месяц, в валюте клиента
    next_term: int = -1  # индекс месяца начала следующего оплаченного года (для annual)
    churn_date: pd.Timestamp | None = None
    is_pilot: bool = False
    history: list = field(default_factory=list)


def simulate_fx(months: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """Курсы USD за единицу валюты: случайное блуждание вокруг типичных уровней."""
    rows = []
    for ccy, start in FX_START.items():
        rate = start * (1 + rng.normal(0, 0.02)) if ccy != "USD" else 1.0
        for m in months:
            rows.append({"month": m, "currency": ccy, "usd_rate": round(rate, 4)})
            if ccy != "USD":
                rate *= float(np.exp(rng.normal(0, 0.012)))
    return pd.DataFrame(rows)


def target_path(path: dict, months: list[str]) -> np.ndarray:
    """Целевое число активных клиентов по месяцам (лог-линейно между опорными точками)."""
    knots = {
        months[0] if k == "start" else months[-1] if k == "end" else k: v for k, v in path.items()
    }
    xs = np.array([month_index(k) for k in sorted(knots)], dtype=float)
    ys = np.log([knots[k] for k in sorted(knots)])
    grid = np.array([month_index(m) for m in months], dtype=float)
    return np.exp(np.interp(grid, xs, ys))


class Simulator:
    """Пошаговая симуляция одной компании; результат — таблицы customers / invoices / fx_rates."""

    def __init__(self, cfg: dict, end_month: str, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.months = month_range(add_months(end_month, -cfg["months"] + 1), end_month)
        self.fx = simulate_fx(self.months, rng)
        self.rates = self.fx.set_index(["month", "currency"])["usd_rate"].to_dict()
        n_names = int(max(cfg["active_path"].values()) * 3.0) + 60
        self.names = iter(name_pool(cfg["industry_key"], n_names, rng))
        self.customers: list[Customer] = []
        self.invoices: list[dict] = []
        self.carry = {"total": 0.0, "core": 0.0}

    # --- создание клиентов -------------------------------------------------------------------
    def _new_customer(self, i: int, *, pilot: bool = False, initial: bool = False) -> Customer:
        cfg, rng = self.cfg, self.rng
        segs = list(cfg["segment_mix"])
        segment = segs[rng.choice(len(segs), p=np.array(list(cfg["segment_mix"].values())))]
        ccys = list(cfg["currency_mix"])
        currency = ccys[rng.choice(len(ccys), p=np.array(list(cfg["currency_mix"].values())))]
        country = CURRENCY_COUNTRIES[currency][int(rng.integers(len(CURRENCY_COUNTRIES[currency])))]
        reseller = not pilot and rng.random() < cfg.get("reseller_share", 0.0)
        m0 = month_start(self.months[i])
        if initial:
            go_live = (
                m0
                - pd.DateOffset(months=int(rng.integers(1, 30)))
                + pd.Timedelta(days=int(rng.integers(0, 27)))
            )
        else:
            go_live = m0 + pd.Timedelta(days=int(rng.integers(0, 27)))
        signed = go_live - pd.Timedelta(days=int(rng.integers(7, 80)))
        c = Customer(
            customer_id=f"C{len(self.customers) + 1:05d}",
            customer_name=legal_name(next(self.names), country, rng),
            segment=segment,
            country=country,
            currency=currency,
            channel="reseller" if reseller else "direct",
            signed_date=signed.normalize(),
            go_live_date=None if pilot else go_live.normalize(),
            is_pilot=pilot,
        )
        if not pilot:
            usd_price = cfg["price_usd"][segment] * float(
                rng.lognormal(0, cfg.get("price_sigma", 0.35))
            )
            c.price = round(usd_price / self.rates[(self.months[i], currency)], 0)
            annual_p = cfg.get("annual_prepay_share", 0.0) * (1.6 if segment != "SMB" else 0.5)
            c.billing = "annual" if rng.random() < annual_p else "monthly"
        self.customers.append(c)
        return c

    # --- инвойсы ------------------------------------------------------------------------------
    def _invoice(self, c: Customer, i: int, n_months: int, amount: float, ptype: str) -> None:
        start = month_start(self.months[i])
        end = month_start(add_months(self.months[i], n_months - 1)) + pd.offsets.MonthEnd(0)
        fee_rate = self.cfg.get("reseller_fee_rate", 0.0) if c.channel == "reseller" else 0.0
        status = "paid"
        if i == len(self.months) - 1 and self.rng.random() < 0.35:
            status = "open"
        self.invoices.append(
            {
                "customer_id": c.customer_id,
                "issue_date": start,
                "service_period_start": start,
                "service_period_end": end,
                "amount": round(amount, 2),
                "currency": c.currency,
                "payment_type": ptype,
                "status": status,
                "reseller_fee": round(amount * fee_rate, 2),
            }
        )

    def _start_term(self, c: Customer, i: int, length: int) -> None:
        length = min(length, 12)
        self._invoice(c, i, length, c.price * length, "subscription")
        c.next_term = i + length

    # --- шаги симуляции -----------------------------------------------------------------------
    def _churn(self, i: int, active: list[Customer]) -> list[Customer]:
        cfg, rng = self.cfg, self.rng
        churn_cfg = cfg["churn"]
        eligible = [c for c in active if c.billing == "monthly" or c.next_term == i]
        core = [c for c in eligible if c.segment != "SMB"]
        smb = [c for c in eligible if c.segment == "SMB"]
        n_core_active = sum(c.segment != "SMB" for c in active)
        exp_core = churn_cfg["core"] * n_core_active + self.carry["core"]
        exp_total = churn_cfg["total"] * len(active) + self.carry["total"]
        n_core = min(int(exp_core), len(core))
        n_smb = min(max(int(exp_total) - n_core, 0), len(smb))
        self.carry["core"] = min(exp_core - n_core, 3.0)
        self.carry["total"] = min(exp_total - n_core - n_smb, 3.0)
        picked = [core[j] for j in rng.choice(len(core), n_core, replace=False)] if n_core else []
        picked += [smb[j] for j in rng.choice(len(smb), n_smb, replace=False)] if n_smb else []
        for c in picked:
            c.churn_date = month_start(self.months[i])
        ids = {c.customer_id for c in picked}
        return [c for c in active if c.customer_id not in ids]

    def _expand(self, i: int, active: list[Customer]) -> None:
        exp = self.cfg["expansion"]
        for c in active:
            if c.billing == "monthly":
                u = self.rng.random()
                if u < exp["monthly_prob"]:
                    c.price = round(c.price * (1 + self.rng.uniform(0.05, 2 * exp["size"])), 0)
                elif u < exp["monthly_prob"] + exp["contraction_prob"]:
                    c.price = round(c.price * (1 - self.rng.uniform(0.05, 0.3)), 0)
            elif c.next_term == i:
                c.price = round(
                    c.price * float(np.exp(self.rng.normal(exp["annual_uplift"], 0.08))), 0
                )
                self._start_term(c, i, 12)

    def _onboard(self, c: Customer, i: int) -> None:
        cfg = self.cfg
        if c.billing == "annual":
            self._start_term(c, i, 12)
        fee = c.price * cfg["impl_fee_months"] * float(self.rng.lognormal(0, 0.2))
        if fee > 0:
            n = min(cfg["onboarding_months"], len(self.months) - i)
            self._invoice(c, i, n, fee, "implementation")

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Выполнить симуляцию и вернуть (customers, invoices, fx_rates)."""
        cfg, rng, months = self.cfg, self.rng, self.months
        target = target_path(cfg["active_path"], months)
        active: list[Customer] = []
        for i in range(len(months)):
            if i == 0:
                for _ in range(round(target[0])):
                    c = self._new_customer(0, initial=True)
                    if c.billing == "annual":
                        self._start_term(c, 0, int(rng.integers(1, 13)))
                    active.append(c)
            else:
                active = self._churn(i, active)
                self._expand(i, active)
                for _ in range(max(0, round(target[i] - len(active)))):
                    c = self._new_customer(i)
                    self._onboard(c, i)
                    active.append(c)
            for c in active:
                if c.billing == "monthly":
                    self._invoice(c, i, 1, c.price, "subscription")
                if rng.random() < cfg.get("one_time_prob", 0.01):
                    self._invoice(c, i, 1, c.price * rng.uniform(0.2, 1.0), "one_time")
            if i > 0:
                for _ in range(rng.poisson(cfg.get("pilot_rate", 0.03) * len(active))):
                    p = self._new_customer(i, pilot=True)
                    end = i + int(rng.integers(2, 5))
                    p.churn_date = month_start(months[end]) if end < len(months) else None
        for _ in range(int(rng.integers(1, 2 + round(0.03 * len(active))))):
            p = self._new_customer(len(months) - 1)
            p.go_live_date = None
        return self._customers_df(), self._invoices_df(), self.fx

    def _customers_df(self) -> pd.DataFrame:
        rows = [
            {
                "customer_id": c.customer_id,
                "customer_name": c.customer_name,
                "segment": c.segment,
                "country": c.country,
                "channel": c.channel,
                "signed_date": c.signed_date,
                "go_live_date": c.go_live_date,
                "churn_date": c.churn_date,
                "is_pilot": c.is_pilot,
            }
            for c in self.customers
        ]
        df = pd.DataFrame(rows)
        for col in ("signed_date", "go_live_date", "churn_date"):
            df[col] = pd.to_datetime(df[col])
        return df

    def _invoices_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.invoices)
        return renumber_invoices(df, self.cfg["invoice_prefix"])


def renumber_invoices(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Отсортировать по дате и присвоить invoice_id вида INV-<prefix>-000001."""
    order = ["issue_date", "customer_id", "payment_type", "service_period_start", "amount"]
    df = df.drop(columns=["invoice_id"], errors="ignore").sort_values(order, kind="stable")
    df = df.reset_index(drop=True)
    df.insert(0, "invoice_id", [f"INV-{prefix}-{k:06d}" for k in range(1, len(df) + 1)])
    return df


def scale_invoices(invoices: pd.DataFrame, mask: pd.Series, factor: float) -> pd.DataFrame:
    """Умножить сумму и комиссию реселлера у выбранных инвойсов (линейная калибровка)."""
    out = invoices.copy()
    out.loc[mask, "amount"] = (out.loc[mask, "amount"] * factor).round(2)
    out.loc[mask, "reseller_fee"] = (out.loc[mask, "reseller_fee"] * factor).round(2)
    return out
