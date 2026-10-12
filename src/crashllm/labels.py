"""Allowed label vocabulary and canonicalisation of free-text LLM outputs."""

from __future__ import annotations

import json
import re

INVALID = "__invalid__"

DIRECTION_ALIASES = {
    "northbound": "nb", "southbound": "sb", "eastbound": "eb", "westbound": "wb",
    "n/b": "nb", "s/b": "sb", "e/b": "eb", "w/b": "wb",
}
MOVEMENT_ALIASES = {
    "through": "th", "straight": "th", "going straight": "th",
    "left turn": "lt", "turning left": "lt",
    "right turn": "rt", "turning right": "rt",
}
FAULT_ALIASES = {
    "u1": "unit 1", "unit1": "unit 1", "1": "unit 1",
    "u2": "unit 2", "unit2": "unit 2", "2": "unit 2",
}


def build_label_vocab(train_df, target_cols) -> dict[str, list[str]]:
    """Allowed categories = exactly those seen in TRAINING data."""
    return {col: sorted(train_df[col].astype(str).unique().tolist()) for col in target_cols}


def save_label_vocab(vocab, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)


def _alias_table(target: str) -> dict:
    if target.endswith("_direction"):
        return DIRECTION_ALIASES
    if target.endswith("_movement"):
        return MOVEMENT_ALIASES
    if target == "unit_at_fault":
        return FAULT_ALIASES
    return {}


def canonicalize_prediction(value, target: str, vocab: dict) -> str:
    """Map a model output to an allowed label, or INVALID.

    The raw value is checked against the vocabulary FIRST, so a correct label
    (e.g. "1" or "northbound") is never rewritten by an alias into something
    that does not exist in the dataset.
    """
    allowed = vocab[target]
    v = re.sub(r"\s+", " ", str(value).strip().lower())

    if v in allowed:
        return v

    aliased = _alias_table(target).get(v)
    if aliased is not None and aliased in allowed:
        return aliased

    # Reverse lookup: model wrote the short form but dataset uses the long one.
    for long_form, short_form in _alias_table(target).items():
        if v == short_form and long_form in allowed:
            return long_form

    return INVALID
