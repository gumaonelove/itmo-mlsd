"""CLI генератора.

uv run python -m dd_verifier.synth --config configs/synth.yaml --out data/synthetic
"""

from __future__ import annotations

import argparse
import time

from dd_verifier.synth.generator import generate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Генератор синтетического data room DD Verifier")
    parser.add_argument("--config", default="configs/synth.yaml")
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--only", nargs="*", help="сгенерировать только эти company_id (отладка)")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    manifests = generate(args.config, args.out, args.only)
    n_holdout = 0
    for cid, m in manifests.items():
        if m["split"] != "dev":  # holdout не показываем по компаниям: только агрегаты
            n_holdout += 1
            continue
        rows = m["row_counts"]
        print(
            f"{m['split']:8s} {cid:11s} customers={rows['customers.csv']:4d} "
            f"invoices={rows['invoices.csv']:6d} claims={rows['gold/claims.jsonl']:3d}"
        )
    if n_holdout:
        print(f"holdout: {n_holdout} компаний (агрегаты — holdout_summary.json)")
    print(f"готово за {time.perf_counter() - t0:.1f} с → {args.out}")


if __name__ == "__main__":
    main()
