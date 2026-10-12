"""Small helpers shared by the scripts in scripts/."""

from __future__ import annotations

import argparse

from .config import load_config, set_seed
from .data import load_split
from .labels import build_label_vocab, save_label_vocab
from .prompts import check_prompt_labels


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="Path to a YAML config (default: configs/config.yaml)")
    return p


def setup(args):
    """Load config, seed everything, load the fixed split and the label vocabulary."""
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    train_df, test_df = load_split(cfg)
    vocab = build_label_vocab(train_df, cfg["data"]["target_cols"])
    save_label_vocab(vocab, cfg["paths"]["split_dir"] / "label_vocab.json")
    print(f"Train: {len(train_df)}   Test: {len(test_df)}")
    check_prompt_labels(vocab)
    return cfg, train_df, test_df, vocab


def debug_name(run_name: str, max_rows) -> str:
    """Partial runs get their own name so they never pollute the comparison."""
    return f"{run_name}_DEBUG" if max_rows else run_name
