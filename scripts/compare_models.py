"""Combine saved results from all models into comparison tables.

    python scripts/compare_models.py                 # newest finished run of each model
    python scripts/compare_models.py --runs all      # every run ever recorded
    python scripts/compare_models.py --include-debug # keep *_DEBUG partial runs

Writes to results/summary/:
    FINAL_MODEL_PERFORMANCE_TABLE.csv     paper table (match % per variable + overall)
    ALL_MODELS_VARIABLE_COMPARISON.csv
    ALL_<table>.csv                       every tidy table concatenated (ready to plot)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crashllm.cli import base_parser  # noqa: E402
from crashllm.config import load_config  # noqa: E402
from crashllm.results import EVAL_TABLES, load_table  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--runs", default="latest", choices=["latest", "all"])
    p.add_argument("--include-debug", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg["paths"]["results_dir"]
    out_dir = results_dir / "summary"
    out_dir.mkdir(exist_ok=True)

    tables = {}
    for name in EVAL_TABLES + ("training_history",):
        df = load_table(results_dir, name, runs=args.runs)
        if not df.empty and not args.include_debug:
            df = df[~df["model"].astype(str).str.contains("_DEBUG")]
        tables[name] = df
        if not df.empty:
            df.to_csv(out_dir / f"ALL_{name}.csv", index=False)

    vm, ov = tables["variable_metrics"], tables["overall_metrics"]
    if vm.empty:
        sys.exit("No finished runs found in results/. Run a model script first.")

    vm.sort_values(["variable", "macro_f1"], ascending=[True, False]).to_csv(
        out_dir / "ALL_MODELS_VARIABLE_COMPARISON.csv", index=False
    )

    keys = ["model", "run_id"]
    paper = vm.pivot_table(index=keys, columns="variable", values="match_percentage").reset_index()
    paper.columns.name = None
    paper = paper.merge(
        ov[keys + ["overall_field_match_percentage", "exact_match_percentage", "mean_macro_f1",
                   "mean_invalid_rate"]],
        on=keys, how="left",
    ).sort_values("exact_match_percentage", ascending=False)
    paper.to_csv(out_dir / "FINAL_MODEL_PERFORMANCE_TABLE.csv", index=False)

    print(paper.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
