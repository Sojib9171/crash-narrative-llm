"""Prompt construction shared by GPT-4o and the fine-tuned local LLMs."""

from __future__ import annotations


def trim_narrative(text, max_chars: int = 5000) -> str:
    """Keep the beginning (70%) and ending (30%) of long narratives."""
    text = str(text)
    if len(text) <= max_chars:
        return text
    first = int(max_chars * 0.70)
    last = max_chars - first
    return text[:first] + "\n...[middle truncated]...\n" + text[-last:]


def structured_answer_from_row(row, target_cols) -> dict:
    return {col: str(row[col]) for col in target_cols}


def build_system_prompt(target_cols) -> str:
    fields = "\n".join(f"{i}. {c}" for i, c in enumerate(target_cols, start=1))
    return f"""You are an expert traffic-crash narrative coding system.

Your ONLY input is an investigator/police crash narrative.

Reconstruct the crash and predict exactly these {len(target_cols)} fields:

{fields}

Use only information supported by the narrative.

Do not invent a direction or movement that is not supported.
If the correct dataset category is unknown, use the allowed
'unknown' value when it exists.

Directions represent the unit's approach direction before the crash.
Movements represent what the unit was doing at the conflict point.

Return ONLY the required structured JSON.
Do not provide explanations."""


def build_prediction_prompt(description, vocab, target_cols, max_chars: int = 5000) -> str:
    allowed = "\n".join(f"{col}: {vocab[col]}" for col in target_cols)
    return f"""Allowed labels:

{allowed}

Crash narrative:

{trim_narrative(description, max_chars)}

Predict all {len(target_cols)} fields."""
