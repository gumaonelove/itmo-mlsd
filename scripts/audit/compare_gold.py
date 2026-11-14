"""Сверка эталонной разметки dev с двумя независимыми пересчётами.

Эталон (`gold/claims.jsonl`) считает библиотека `src/dd_verifier`. Здесь он проверяется
двумя реализациями, которые библиотеку не импортируют и написаны только по
`configs/ontology.yaml` и схемам таблиц: `recalc_pandas.py` и `recalc_sql.py`.

Скрипт:
1. берёт числовые утверждения dev и убирает из них все поля эталона (значения, δ, вердикт,
   случай каталога) — получается список «вслепую»;
2. запускает оба пересчёта на этом списке;
3. сравнивает значения с `true_value_observed`, вердикты — с `gold_verdict`,
   и пишет таблицу в `reports/eda/audit_gold_recalc.csv`.

Запуск из корня репозитория:
    uv run python scripts/audit/compare_gold.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BLIND_FIELDS = [
    "claim_id",
    "company_id",
    "doc",
    "quote",
    "claim_type",
    "metric_id",
    "claimed_value",
    "unit",
    "period_start",
    "period_end",
    "qualifier",
]
# Эталонные значения в claims.jsonl округлены до 6 знаков после запятой.
ABS_TOL = 5e-7
REL_TOL = 1e-9


def load_gold(data_dir: Path) -> list[dict]:
    """Числовые утверждения dev (у них есть metric_id) со всеми полями эталона."""
    rows = []
    for path in sorted(data_dir.glob("*/gold/claims.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["metric_id"]:
                    rows.append(row)
    return rows


def run_recalc(script: str, data_arg: str, data_dir: Path, claims: Path, out: Path) -> pd.DataFrame:
    """Запускает один пересчёт отдельным процессом и читает его CSV."""
    cmd = [sys.executable, str(HERE / script), data_arg, str(data_dir)]
    cmd += ["--claims", str(claims), "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return pd.read_csv(out).set_index("claim_id")


def same_value(x: float, gold: float | None) -> bool:
    """Значения совпадают с точностью до округления эталона; пусто равно пусто."""
    if gold is None or pd.isna(gold):
        return pd.isna(x)
    if pd.isna(x):
        return False
    return abs(x - gold) <= max(ABS_TOL, REL_TOL * abs(gold))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/synthetic/dev"))
    ap.add_argument("--out", type=Path, default=Path("reports/eda/audit_gold_recalc.csv"))
    args = ap.parse_args()

    gold = load_gold(args.data_dir)
    with tempfile.TemporaryDirectory() as tmp:
        blind = Path(tmp) / "claims_blind.jsonl"
        blind.write_text(
            "\n".join(json.dumps({k: g[k] for k in BLIND_FIELDS}, ensure_ascii=False) for g in gold)
            + "\n",
            encoding="utf-8",
        )
        res_pd = run_recalc(
            "recalc_pandas.py", "--data-dir", args.data_dir, blind, Path(tmp) / "pandas.csv"
        )
        res_sql = run_recalc("recalc_sql.py", "--data", args.data_dir, blind, Path(tmp) / "sql.csv")

    rows = []
    for g in gold:
        cid = g["claim_id"]
        v_pd, v_sql = res_pd.loc[cid, "value"], res_sql.loc[cid, "value"]
        rows.append(
            {
                "claim_id": cid,
                "metric_id": g["metric_id"],
                "case_id": g["case_id"] or "",
                "claimed_value": g["claimed_value"],
                "gold_value_observed": g["true_value_observed"],
                "value_pandas": v_pd,
                "value_sql": v_sql,
                "gold_verdict": g["gold_verdict"],
                "verdict_pandas": res_pd.loc[cid, "verdict"],
                "verdict_sql": res_sql.loc[cid, "verdict"],
                "values_match": same_value(v_pd, g["true_value_observed"])
                and same_value(v_sql, g["true_value_observed"]),
                "verdicts_match": res_pd.loc[cid, "verdict"]
                == res_sql.loc[cid, "verdict"]
                == g["gold_verdict"],
            }
        )
    table = pd.DataFrame(rows)
    # В файл значения пишутся с точностью эталона: последние знаки float зависят от порядка
    # суммирования в DuckDB и менялись бы от запуска к запуску.
    value_cols = ["value_pandas", "value_sql"]
    table[value_cols] = table[value_cols].round(6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    with_value = table["gold_value_observed"].notna()
    print(f"утверждений: {len(table)}, из них с рассчитанным значением: {with_value.sum()}")
    print(f"значения совпали с эталоном: {table.loc[with_value, 'values_match'].sum()}")
    print(f"вердикты совпали с эталоном: {table['verdicts_match'].sum()}")
    by_metric = table.groupby("metric_id").agg(
        n=("claim_id", "size"), values=("values_match", "sum"), verdicts=("verdicts_match", "sum")
    )
    print(by_metric.to_string())
    bad = table[~(table["values_match"] & table["verdicts_match"])]
    if len(bad):
        print("\nРасхождения:")
        print(bad.to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
