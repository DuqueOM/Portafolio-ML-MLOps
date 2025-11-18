from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pulp


def optimize_selection(csv_path: Path, n_select: int) -> dict:
    df = pd.read_csv(csv_path)
    if "product" not in df.columns:
        raise SystemExit("CSV must contain 'product' column as expected reserves")

    n = len(df)
    prob = pulp.LpProblem("well_selection", pulp.LpMaximize)
    x = [
        pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for i in range(n)
    ]

    # Objective: maximize total expected product
    prob += pulp.lpSum(df.loc[i, "product"] * x[i] for i in range(n))

    # Constraint: choose at most n_select wells
    prob += pulp.lpSum(x) <= n_select

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    chosen = [i for i in range(n) if x[i].value() > 0.5]
    total_product = float(df.loc[chosen, "product"].sum()) if chosen else 0.0

    return {
        "n_candidates": n,
        "n_selected": len(chosen),
        "total_product": total_product,
        "indices": chosen,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize well selection (demo ILP)")
    ap.add_argument("--csv", required=True, help="CSV path with 'product' column")
    ap.add_argument("--n-select", type=int, default=200)
    args = ap.parse_args()

    res = optimize_selection(Path(args.csv), args.n_select)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
