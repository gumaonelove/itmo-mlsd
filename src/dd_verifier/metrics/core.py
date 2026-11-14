"""Базовые структуры библиотеки метрик.

- `DataRoom` — три таблицы data room (customers, invoices, fx_rates) и кэш производных таблиц;
- `build_ledger` — помесячный леджер признанной выручки: каждый инвойс (кроме void)
  переводится в USD
  по курсу месяца начала периода услуги и распределяется равномерно по месяцам этого периода;
- `MetricResult` — значение метрики + человекочитаемый след расчёта + эквивалентный SQL;
- `NotEnoughData` — период утверждения не покрыт данными (→ вердикт NOT_ENOUGH_DATA).

Месяц везде — строка `YYYY-MM`; значение «на месяц» означает «на конец месяца».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd

PAYING_EPS = 0.005  # клиент «платит» в месяце, если подписочный MRR > полцента


class NotEnoughData(Exception):  # noqa: N818 — имя соответствует вердикту NOT_ENOUGH_DATA
    """Данных недостаточно для расчёта метрики за запрошенный период."""


def month_index(month: str) -> int:
    """'2025-12' → 2025 * 12 + 11 (сквозной номер месяца)."""
    year, mon = month.split("-")
    return int(year) * 12 + int(mon) - 1


def month_from_index(idx: int) -> str:
    """Обратное к `month_index`."""
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def add_months(month: str, k: int) -> str:
    """Сдвиг месяца на k (k может быть отрицательным)."""
    return month_from_index(month_index(month) + k)


def month_range(start: str, end: str) -> list[str]:
    """Все месяцы от start до end включительно."""
    return [month_from_index(i) for i in range(month_index(start), month_index(end) + 1)]


def month_start(month: str) -> pd.Timestamp:
    """Первый день месяца."""
    return pd.Timestamp(f"{month}-01")


def month_end(month: str) -> pd.Timestamp:
    """Последний день месяца (00:00)."""
    return month_start(month) + pd.offsets.MonthEnd(0)


def usd(x: float) -> str:
    """Форматирование суммы для следа расчёта."""
    return f"${x:,.0f}"


@dataclass(frozen=True)
class MetricResult:
    """Результат расчёта метрики с доказательствами."""

    metric_id: str
    variant: str
    value: float
    unit: str
    period_start: str | None
    period_end: str
    trace: tuple[str, ...] = ()
    sql: str | None = None

    def explain(self) -> str:
        """След расчёта одной строкой на шаг."""
        head = f"{self.metric_id} [{self.variant}] {self.period_start or ''}..{self.period_end}"
        return "\n".join([head, *self.trace, f"= {self.value:.6g} {self.unit}"])


def _dt_month_idx(s: pd.Series) -> np.ndarray:
    return (s.dt.year * 12 + s.dt.month - 1).to_numpy(dtype=np.int64)


def build_ledger(invoices: pd.DataFrame, fx_rates: pd.DataFrame) -> pd.DataFrame:
    """Помесячный леджер признанной выручки.

    Возвращает строки (invoice_id, customer_id, month, payment_type, net_usd, gross_usd):
    по одной на каждый месяц периода услуги каждого не-void инвойса. Кредит-ноты имеют
    отрицательную сумму и поэтому вычитаются автоматически. net = amount − reseller_fee.
    Строки без курса (неизвестная валюта / месяц без курса) отбрасываются, их число
    сохраняется в `ledger.attrs["dropped_no_fx"]`.
    """
    inv = invoices.loc[invoices["status"] != "void"].copy()
    m0 = _dt_month_idx(inv["service_period_start"])
    m1 = _dt_month_idx(inv["service_period_end"])
    n = np.maximum(m1 - m0 + 1, 1)
    inv["fx_month"] = [month_from_index(i) for i in m0]
    rates = fx_rates.rename(columns={"month": "fx_month"})[["fx_month", "currency", "usd_rate"]]
    inv = inv.merge(rates, on=["fx_month", "currency"], how="left")
    dropped = int(inv["usd_rate"].isna().sum())
    keep = inv["usd_rate"].notna().to_numpy()
    inv, m0, n = inv.loc[keep].reset_index(drop=True), m0[keep], n[keep]

    rep = np.repeat(np.arange(len(inv)), n)
    offset = np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n)
    month_idx = np.repeat(m0, n) + offset
    names = {i: month_from_index(i) for i in np.unique(month_idx)}
    per_month = 1.0 / np.repeat(n, n)
    rate = inv["usd_rate"].to_numpy()[rep]
    amount = inv["amount"].to_numpy(dtype=float)[rep]
    fee = inv["reseller_fee"].fillna(0).to_numpy(dtype=float)[rep]
    ledger = pd.DataFrame(
        {
            "invoice_id": inv["invoice_id"].to_numpy()[rep],
            "customer_id": inv["customer_id"].to_numpy()[rep],
            "month": [names[i] for i in month_idx],
            "payment_type": inv["payment_type"].to_numpy()[rep],
            "net_usd": (amount - fee) * rate * per_month,
            "gross_usd": amount * rate * per_month,
        }
    )
    ledger.attrs["dropped_no_fx"] = dropped
    return ledger


@dataclass
class DataRoom:
    """Таблицы data room одной компании и кэш производных таблиц.

    Ожидаемые типы: даты — datetime64, `is_pilot` — bool, суммы — float.
    Используйте `dd_verifier.data.load_data_room` для чтения с диска.
    """

    customers: pd.DataFrame
    invoices: pd.DataFrame
    fx_rates: pd.DataFrame
    _active_cache: dict = field(default_factory=dict, repr=False)

    @cached_property
    def ledger(self) -> pd.DataFrame:
        """Помесячный леджер признанной выручки (см. `build_ledger`)."""
        return build_ledger(self.invoices, self.fx_rates)

    @cached_property
    def coverage(self) -> tuple[str, str]:
        """Покрытие данными: от первого месяца услуги до месяца последнего выставленного инвойса."""
        valid = self.invoices.loc[self.invoices["status"] != "void"]
        first = valid["service_period_start"].min()
        last = valid["issue_date"].max()
        return f"{first:%Y-%m}", f"{last:%Y-%m}"

    @cached_property
    def customer_mrr(self) -> pd.DataFrame:
        """Подписочный MRR (net, USD) по клиентам × месяцам; пропуски = 0."""
        sub = self.ledger.loc[
            (self.ledger["payment_type"] == "subscription") & self.ledger["customer_id"].notna()
        ]
        return sub.pivot_table(
            index="customer_id", columns="month", values="net_usd", aggfunc="sum", fill_value=0.0
        )

    def mrr_of(self, month: str) -> pd.Series:
        """Подписочный MRR каждого клиента в месяце (Series по customer_id)."""
        if month in self.customer_mrr.columns:
            return self.customer_mrr[month]
        return pd.Series(0.0, index=self.customer_mrr.index)

    def require(self, *months: str) -> None:
        """Проверить, что все месяцы внутри покрытия, иначе `NotEnoughData`."""
        first, last = self.coverage
        bad = [m for m in months if not first <= m <= last]
        if bad:
            raise NotEnoughData(f"месяцы {bad} вне покрытия данными {first}..{last}")

    def active_ids(self, month: str) -> list[str]:
        """Активные платящие клиенты на конец месяца (стандартное определение)."""
        if month not in self._active_cache:
            c, me = self.customers, month_end(month)
            live = (
                c["go_live_date"].notna()
                & (c["go_live_date"] <= me)
                & (c["churn_date"].isna() | (c["churn_date"] > me))
            )
            paying = self.mrr_of(month)
            paying_ids = set(paying.index[paying > PAYING_EPS])
            self._active_cache[month] = sorted(set(c.loc[live, "customer_id"]) & paying_ids)
        return self._active_cache[month]
