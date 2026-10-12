"""Saving every run in a plot-friendly layout, and loading it back.

Layout
------
results/
  runs_index.csv                       one row per finished run (+ overall metrics)
  <run_name>/
    LATEST                             id of the most recent run
    <run_id>/                          e.g. 20261006-153012
      run_config.json                  hyper-parameters, model id, split seed, timings
      predictions.csv                  raw predictions (written incrementally -> resumable)
      raw_outputs.jsonl                LLM text outputs, token counts, latency (LLMs only)
      training_history.csv             loss per step/epoch (trained models only)
      case_results.csv                 truth + prediction + correct flags per case
      variable_metrics.csv             accuracy / F1 per target variable
      per_class_metrics.csv            precision / recall / F1 per class
      confusion_matrices.csv           long format: true_label, pred_label, count, row_percentage
      overall_metrics.csv              exact match, field match, mean F1 ...

All tables are "long"/tidy and carry `model` and `run_id` columns, so you can
concatenate runs and plot with pandas / seaborn / matplotlib directly.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime
from pathlib import Path

import hashlib

import pandas as pd

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT

EVAL_TABLES = (
    "case_results",
    "variable_metrics",
    "per_class_metrics",
    "confusion_matrices",
    "overall_metrics",
)


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class RunRecorder:
    """Handles one run's output folder."""

    def __init__(self, cfg, run_name: str, resume: str | None = None, params: dict | None = None):
        self.cfg = cfg
        self.run_name = run_name
        self.model_root = Path(cfg["paths"]["results_dir"]) / run_name
        self.model_root.mkdir(parents=True, exist_ok=True)

        if resume:
            run_id = self._resolve(resume)
            print(f"Resuming run {run_name}/{run_id}")
        else:
            run_id = _new_run_id()

        self.run_id = run_id
        self.run_dir = self.model_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.model_root / "LATEST").write_text(run_id)

        self._history: list[dict] = []
        hist_file = self.run_dir / "training_history.csv"
        if resume and hist_file.exists():
            self._history = pd.read_csv(hist_file).to_dict("records")

        self._t0 = time.time()
        self.params = self._load_params()
        meta = dict(params or {})
        meta.pop("run_name", None)   # config's base name; the recorder's name wins
        self.update_params(
            **meta,
            run_name=run_name,
            run_id=run_id,
            config_file=cfg.get("_config_path"),
            started_at=self.params.get("started_at", datetime.now().isoformat(timespec="seconds")),
            python=platform.python_version(),
            prompt_version=PROMPT_VERSION,
            system_prompt_sha1=hashlib.sha1(SYSTEM_PROMPT.encode()).hexdigest()[:12],
        )
        self._add_split_info()

    # ------------------------------------------------------------------ setup
    def _resolve(self, resume: str) -> str:
        if resume == "latest":
            latest = self.model_root / "LATEST"
            if not latest.exists():
                raise FileNotFoundError(f"No previous run found for {self.run_name}")
            return latest.read_text().strip()
        if not (self.model_root / resume).exists():
            raise FileNotFoundError(f"Run {self.model_root / resume} does not exist")
        return resume

    def _load_params(self) -> dict:
        f = self.run_dir / "run_config.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def _add_split_info(self):
        info = Path(self.cfg["paths"]["split_dir"]) / "split_info.json"
        if info.exists():
            self.update_params(split=json.loads(info.read_text()))

    def update_params(self, **kwargs) -> None:
        self.params.update(kwargs)
        (self.run_dir / "run_config.json").write_text(json.dumps(self.params, indent=2, default=str))

    # --------------------------------------------------------------- training
    def log_history(self, **row) -> None:
        """Log one training step/epoch, e.g. log_history(epoch=1, step=10, loss=0.42)."""
        row = {"model": self.run_name, "run_id": self.run_id, **row}
        self._history.append(row)
        pd.DataFrame(self._history).to_csv(self.run_dir / "training_history.csv", index=False)

    # ------------------------------------------------------------ predictions
    @property
    def predictions_file(self) -> Path:
        return self.run_dir / "predictions.csv"

    def load_existing_predictions(self) -> dict[str, dict]:
        """Case-id -> record from an interrupted run (same run folder only)."""
        if not self.predictions_file.exists():
            return {}
        df = pd.read_csv(self.predictions_file, dtype=str, keep_default_na=False)
        case_col = self.cfg["data"]["case_col"]
        print(f"Resuming from {len(df)} saved predictions.")
        return {str(r[case_col]): r for r in df.to_dict("records")}

    def save_predictions(self, records: list[dict]) -> None:
        pd.DataFrame(records).to_csv(self.predictions_file, index=False)

    def append_raw_output(self, record: dict) -> None:
        with open(self.run_dir / "raw_outputs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # -------------------------------------------------------------- evaluation
    def save_evaluation(self, eval_out: dict) -> None:
        for name in EVAL_TABLES:
            df = eval_out[name].copy()
            df.insert(1 if "model" in df.columns else 0, "run_id", self.run_id)
            df.to_csv(self.run_dir / f"{name}.csv", index=False)

        self.update_params(
            finished_at=datetime.now().isoformat(timespec="seconds"),
            wall_time_sec=round(time.time() - self._t0, 1),
        )
        self._append_index(eval_out["overall_metrics"])
        print(f"\nResults saved to: {self.run_dir}")

    def _append_index(self, overall: pd.DataFrame) -> None:
        index_file = Path(self.cfg["paths"]["results_dir"]) / "runs_index.csv"
        row = overall.copy()
        row.insert(1, "run_id", self.run_id)
        row["run_dir"] = str(self.run_dir)
        row["finished_at"] = self.params.get("finished_at")
        if index_file.exists():
            old = pd.read_csv(index_file, dtype={"run_id": str})
            old = old[~((old["model"] == self.run_name) & (old["run_id"] == self.run_id))]
            row = pd.concat([old, row], ignore_index=True)
        row.to_csv(index_file, index=False)


# =====================================================================
# Loading results back (for comparison tables and plotting)
# =====================================================================

def list_runs(results_dir) -> pd.DataFrame:
    """All finished runs (from runs_index.csv)."""
    f = Path(results_dir) / "runs_index.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f, dtype={"run_id": str})


def latest_run_dirs(results_dir) -> dict[str, Path]:
    """run_name -> folder of its most recent *finished* run."""
    runs = list_runs(results_dir)
    if runs.empty:
        return {}
    runs = runs.sort_values("run_id").groupby("model").tail(1)
    return {r.model: Path(r.run_dir) for r in runs.itertuples()}


def load_table(results_dir, table: str, runs: str | list[str] = "latest") -> pd.DataFrame:
    """Concatenate one table across runs.

    table : one of EVAL_TABLES or "training_history"
    runs  : "latest" (newest finished run per model), "all", or a list of
            run folders / "<run_name>/<run_id>" strings.
    """
    results_dir = Path(results_dir)
    if runs == "latest":
        dirs = list(latest_run_dirs(results_dir).values())
    elif runs == "all":
        dirs = [Path(p) for p in list_runs(results_dir).get("run_dir", [])]
    else:
        dirs = [Path(p) if Path(p).is_absolute() else results_dir / p for p in runs]

    frames = []
    for d in dirs:
        f = d / f"{table}.csv"
        if f.exists():
            # Only empty cells are NaN, so labels like "n/a" or "null" stay strings.
            frames.append(pd.read_csv(f, dtype={"run_id": str}, keep_default_na=False, na_values=[""]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
