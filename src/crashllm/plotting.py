"""Ready-made figures built ONLY from the saved CSVs (no model needed).

Every function takes a tidy DataFrame from `crashllm.results.load_table`
and returns the matplotlib Figure, so you can restyle or save it yourself.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metric_by_variable(variable_metrics: pd.DataFrame, metric: str = "macro_f1", ax=None):
    """Grouped bars: one group per target variable, one bar per model."""
    pivot = variable_metrics.pivot_table(index="variable", columns="model", values=metric)
    ax = ax or plt.subplots(figsize=(max(8, 1.6 * len(pivot)), 5))[1]
    pivot.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.set_title(f"{metric} by variable")
    ax.legend(title="model", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure


def overall_comparison(overall_metrics: pd.DataFrame,
                       metrics=("overall_field_match_percentage", "exact_match_percentage"),
                       ax=None):
    df = overall_metrics.set_index("model")[list(metrics)]
    df = df.sort_values(metrics[-1])
    ax = ax or plt.subplots(figsize=(8, 0.6 * len(df) + 2))[1]
    df.plot.barh(ax=ax, width=0.8)
    ax.set_xlabel("%")
    ax.set_ylabel("")
    ax.set_title("Overall reconstruction performance")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure


def confusion_matrix(confusion: pd.DataFrame, model: str, variable: str,
                     value: str = "count", ax=None, cmap="Blues"):
    """Heatmap of one model/variable from the long-format confusion table."""
    sub = confusion[(confusion["model"] == model) & (confusion["variable"] == variable)]
    if sub.empty:
        raise ValueError(f"No confusion data for {model} / {variable}")
    labels = list(dict.fromkeys(sub["true_label"]))
    mat = (sub.pivot(index="true_label", columns="pred_label", values=value)
              .reindex(index=labels, columns=labels).fillna(0))

    n = len(labels)
    ax = ax or plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.8)))[1]
    im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=100 if value == "row_percentage" else None)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fmt = "{:.0f}" if value == "row_percentage" else "{:d}"
    thresh = np.nanmax(mat.values) / 2 if mat.values.size else 0
    for i in range(n):
        for j in range(n):
            v = mat.values[i, j]
            ax.text(j, i, fmt.format(int(v) if value == "count" else v), ha="center", va="center",
                    color="white" if v > thresh else "black", fontsize=8)
    ax.set_title(f"{model}\n{variable} ({value})")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.figure.tight_layout()
    return ax.figure


def training_curves(history: pd.DataFrame, x: str = "step", y: str = "loss", ax=None):
    ax = ax or plt.subplots(figsize=(8, 5))[1]
    for model, g in history.groupby("model"):
        g = g.dropna(subset=[y])
        if x in g:
            ax.plot(g[x], g[y], label=model)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Training loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure


def per_class_f1(per_class: pd.DataFrame, variable: str, ax=None):
    sub = per_class[per_class["variable"] == variable]
    pivot = sub.pivot_table(index="label", columns="model", values="f1")
    ax = ax or plt.subplots(figsize=(max(8, 0.8 * len(pivot)), 5))[1]
    pivot.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("F1")
    ax.set_title(f"Per-class F1 — {variable}")
    ax.legend(title="model", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure


def fields_correct_distribution(case_results: pd.DataFrame, ax=None):
    """How many of the six fields each model gets right per crash."""
    counts = (case_results.groupby("model")["number_fields_correct"]
              .value_counts(normalize=True).mul(100).unstack(fill_value=0))
    ax = ax or plt.subplots(figsize=(9, 5))[1]
    counts.T.plot.bar(ax=ax, width=0.8)
    ax.set_xlabel("Number of fields correct")
    ax.set_ylabel("% of test crashes")
    ax.set_title("Distribution of correctly reconstructed fields")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure
