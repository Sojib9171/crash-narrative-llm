"""Model-agnostic evaluation. Returns tidy DataFrames; saving is done by RunRecorder."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from .labels import INVALID


def evaluate(test_df, pred_df, cfg, vocab, model_name: str) -> dict[str, pd.DataFrame]:
    """Compare predictions with ground truth.

    pred_df must contain `case_id` and `pred_<target>` for every target.
    Any extra columns (e.g. `conf_<target>`) are carried into case_results.

    Macro/weighted metrics are computed over the TRAINING label set, so an
    `__invalid__` output counts as a miss for the true class but is not added
    as an extra zero-F1 class. The invalid rate is reported separately.
    """
    case_col = cfg["data"]["case_col"]
    targets = cfg["data"]["target_cols"]
    pred_cols = [f"pred_{t}" for t in targets]

    missing = [c for c in [case_col] + pred_cols if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {missing}")

    truth = test_df[[case_col] + targets].copy()
    pred = pred_df.copy()
    pred[case_col] = pred[case_col].astype(str)
    results = truth.merge(pred, on=case_col, how="left")

    for c in pred_cols:
        results[c] = results[c].fillna(INVALID).astype(str)

    variable_rows, per_class_rows, cm_rows = [], [], []

    for t in targets:
        y_true = results[t].astype(str)
        y_pred = results[f"pred_{t}"].astype(str)
        labels = list(vocab[t])
        results[f"correct_{t}"] = (y_true == y_pred)

        variable_rows.append({
            "model": model_name,
            "variable": t,
            "n": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "match_percentage": accuracy_score(y_true, y_pred) * 100,
            "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
            "invalid_rate": float((y_pred == INVALID).mean()),
        })

        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        for lab, pi, ri, fi, si in zip(labels, p, r, f, s):
            per_class_rows.append({
                "model": model_name, "variable": t, "label": lab,
                "precision": pi, "recall": ri, "f1": fi, "support": int(si),
            })

        cm_labels = labels + ([INVALID] if (y_pred == INVALID).any() else [])
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0.0)
        for i, tl in enumerate(cm_labels):
            for j, pl in enumerate(cm_labels):
                cm_rows.append({
                    "model": model_name, "variable": t,
                    "true_label": tl, "pred_label": pl,
                    "count": int(cm[i, j]), "row_percentage": float(cm_pct[i, j]),
                })

    correct = results[[f"correct_{t}" for t in targets]].to_numpy()
    results["number_fields_correct"] = correct.sum(axis=1)
    results["exact_match"] = results["number_fields_correct"] == len(targets)
    results.insert(0, "model", model_name)

    variable_metrics = pd.DataFrame(variable_rows)
    overall = pd.DataFrame([{
        "model": model_name,
        "n_test_cases": len(results),
        "overall_field_match_percentage": correct.mean() * 100,
        "exact_match_percentage": results["exact_match"].mean() * 100,
        "mean_macro_f1": variable_metrics["macro_f1"].mean(),
        "mean_weighted_f1": variable_metrics["weighted_f1"].mean(),
        "mean_variable_accuracy": variable_metrics["accuracy"].mean(),
        "mean_invalid_rate": variable_metrics["invalid_rate"].mean(),
    }])

    return {
        "case_results": results,
        "variable_metrics": variable_metrics,
        "per_class_metrics": pd.DataFrame(per_class_rows),
        "confusion_matrices": pd.DataFrame(cm_rows),
        "overall_metrics": overall,
    }


def print_summary(eval_out: dict) -> None:
    vm = eval_out["variable_metrics"]
    ov = eval_out["overall_metrics"].iloc[0]
    print("\n" + "=" * 90)
    print("MODEL:", ov["model"])
    print("=" * 90)
    cols = ["variable", "match_percentage", "macro_f1", "weighted_f1", "invalid_rate"]
    print(vm[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 90)
    print(f"Overall field-level match : {ov['overall_field_match_percentage']:.2f}%")
    print(f"Exact match (all fields)  : {ov['exact_match_percentage']:.2f}%")
    print(f"Mean macro F1             : {ov['mean_macro_f1']:.4f}")
