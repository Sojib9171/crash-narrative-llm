"""Make a standard set of figures from the saved CSVs (no GPU / model needed).

    python scripts/plot_results.py
    python scripts/plot_results.py --format pdf --runs all

Figures go to results/figures/. For custom plots, use notebooks/plot_results.ipynb
or `crashllm.results.load_table(...)` in your own script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crashllm import plotting  # noqa: E402
from crashllm.cli import base_parser  # noqa: E402
from crashllm.config import load_config  # noqa: E402
from crashllm.results import load_table  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--runs", default="latest", choices=["latest", "all"])
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--include-debug", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    rd = cfg["paths"]["results_dir"]
    fig_dir = rd / "figures"
    fig_dir.mkdir(exist_ok=True)

    def load(name):
        df = load_table(rd, name, runs=args.runs)
        if not df.empty and not args.include_debug:
            df = df[~df["model"].astype(str).str.contains("_DEBUG")]
        return df

    vm, ov = load("variable_metrics"), load("overall_metrics")
    cm, pc = load("confusion_matrices"), load("per_class_metrics")
    cr, hist = load("case_results"), load("training_history")
    if vm.empty:
        sys.exit("No results found. Run a model script first.")

    def save(fig, name):
        fig.savefig(fig_dir / f"{name}.{args.format}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved", name)

    for metric in ("match_percentage", "macro_f1", "weighted_f1"):
        save(plotting.metric_by_variable(vm, metric), f"variable_{metric}")
    save(plotting.overall_comparison(ov), "overall_comparison")
    save(plotting.fields_correct_distribution(cr), "fields_correct_distribution")

    for var in cfg["data"]["target_cols"]:
        save(plotting.per_class_f1(pc, var), f"per_class_f1_{var}")
        for model in cm["model"].unique():
            safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model)
            save(plotting.confusion_matrix(cm, model, var), f"cm_{safe}_{var}")

    if not hist.empty:
        step_hist = hist[hist["level"] == "step"] if "level" in hist.columns else hist
        save(plotting.training_curves(step_hist), "training_loss")

    print(f"\nFigures in {fig_dir}")


if __name__ == "__main__":
    main()
