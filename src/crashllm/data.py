"""Loading, cleaning and the fixed 70/30 train/test split."""

from __future__ import annotations

import json
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def normalize_column_name(x) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"\s+", "_", x)
    return re.sub(r"[^a-z0-9_]", "", x)


def _read_any(path) -> pd.DataFrame:
    path = str(path)
    # dtype=str keeps labels such as "1" / "2" as strings (no int/str mismatch).
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return pd.read_excel(path, dtype=str)


def load_dataset(cfg) -> pd.DataFrame:
    d = cfg["data"]
    case_col, text_col, targets = d["case_col"], d["text_col"], d["target_cols"]

    df = _read_any(cfg["paths"]["data_path"])
    df.columns = [normalize_column_name(c) for c in df.columns]

    required = [case_col, text_col] + targets
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required columns not found: {missing}\n"
            f"Available columns: {df.columns.tolist()}\n"
            "Edit data.case_col / text_col / target_cols in configs/config.yaml."
        )

    df = df[required].copy()

    df = df[df[text_col].notna()].copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]

    # Missing output values become a real "unknown" class.
    for col in targets:
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
        df.loc[df[col].isin(["", "nan", "none"]), col] = "unknown"

    df[case_col] = df[case_col].astype(str).str.strip()
    return df.reset_index(drop=True)


def _all_test_classes_in_train(train_df, test_df, targets) -> bool:
    return all(set(test_df[c]).issubset(set(train_df[c])) for c in targets)


def create_fixed_split(df, cfg):
    """Try deterministic seeds until every test class also appears in train."""
    d = cfg["data"]
    case_col, targets = d["case_col"], d["target_cols"]
    test_size, strat_col = d["test_size"], d.get("stratify_col")
    start = cfg["seed"]

    has_dupes = df[case_col].duplicated().any()

    for split_seed in range(start, start + d.get("max_split_attempts", 3000)):
        if has_dupes:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
            tr_idx, te_idx = next(splitter.split(df, groups=df[case_col]))
            train_df, test_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        else:
            stratify = None
            if strat_col:
                counts = df[strat_col].value_counts()
                if counts.min() >= 2 and int(round(len(df) * test_size)) >= df[strat_col].nunique():
                    stratify = df[strat_col]
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=split_seed, stratify=stratify
            )

        if _all_test_classes_in_train(train_df, test_df, targets):
            return train_df.reset_index(drop=True), test_df.reset_index(drop=True), split_seed

    raise RuntimeError(
        "Could not create a split where every test class is represented in training. "
        "Some classes probably contain only one case."
    )


def split_paths(cfg):
    s = cfg["paths"]["split_dir"]
    return s / "train.csv", s / "test.csv", s / "split_info.json"


def prepare_split(cfg, reset: bool = False):
    """Create (or load) the fixed split and save it to data/splits/."""
    train_file, test_file, info_file = split_paths(cfg)
    df = load_dataset(cfg)

    if train_file.exists() and test_file.exists() and not reset:
        train_df, test_df = load_split(cfg)
        print("Loaded EXISTING fixed split (not re-randomized).")
        case_col = cfg["data"]["case_col"]
        if set(train_df[case_col]) | set(test_df[case_col]) != set(df[case_col]):
            print(
                "WARNING: the master dataset differs from the one used for the saved split.\n"
                "If you intentionally added/removed cases, re-run with --reset-split."
            )
        return train_df, test_df

    train_df, test_df, used_seed = create_fixed_split(df, cfg)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    info = {
        "seed": int(used_seed),
        "source_file": str(cfg["paths"]["data_path"]),
        "total_cases": int(len(df)),
        "train_cases": int(len(train_df)),
        "test_cases": int(len(test_df)),
        "train_percentage": float(len(train_df) / len(df) * 100),
        "test_percentage": float(len(test_df) / len(df) * 100),
    }
    info_file.write_text(json.dumps(info, indent=2))
    print(f"Created and SAVED new fixed split (seed {used_seed}).")
    return train_df, test_df


def load_split(cfg):
    """Load the saved split. Every model script uses this, never re-splitting."""
    train_file, test_file, _ = split_paths(cfg)
    if not (train_file.exists() and test_file.exists()):
        raise FileNotFoundError(
            "No saved split found. Run `python scripts/prepare_data.py` first."
        )
    kw = dict(dtype=str, keep_default_na=False)
    return pd.read_csv(train_file, **kw), pd.read_csv(test_file, **kw)


def class_distribution(train_df, test_df, cfg) -> pd.DataFrame:
    """Long-format class counts per split (handy for plotting)."""
    rows = []
    for col in cfg["data"]["target_cols"]:
        for split_name, frame in (("train", train_df), ("test", test_df)):
            for label, n in frame[col].value_counts().items():
                rows.append({"variable": col, "split": split_name, "label": label, "count": int(n)})
    return pd.DataFrame(rows)


def narrative_overlap(train_df, test_df, cfg) -> int:
    text_col = cfg["data"]["text_col"]

    def norm(s):
        return set(s.str.lower().str.replace(r"\s+", " ", regex=True))

    return len(norm(train_df[text_col]) & norm(test_df[text_col]))
