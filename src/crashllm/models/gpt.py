"""Model 2 — GPT-4o few-shot with strict JSON-schema output."""

from __future__ import annotations

import json
import time

import pandas as pd
from tqdm.auto import tqdm

from ..labels import INVALID, canonicalize_prediction
from ..prompts import (
    build_prediction_prompt,
    build_system_prompt,
    structured_answer_from_row,
    trim_narrative,
)


def select_few_shot_examples(train_df, n_examples, seed, stratify_col="crash_type"):
    """One example per crash type first, then random fill (training set only)."""
    chosen = []
    for _, group in train_df.groupby(stratify_col):
        chosen.append(group.sample(1, random_state=seed).index[0])
        if len(chosen) >= n_examples:
            break
    if len(chosen) < n_examples:
        remaining = train_df.drop(index=chosen, errors="ignore")
        extra = remaining.sample(min(n_examples - len(chosen), len(remaining)), random_state=seed)
        chosen.extend(extra.index.tolist())
    return train_df.loc[chosen].copy()


def build_schema(vocab, targets):
    return {
        "type": "object",
        "properties": {t: {"type": "string", "enum": vocab[t]} for t in targets},
        "required": list(targets),
        "additionalProperties": False,
    }


def build_user_message(description, few_shot_df, vocab, cfg):
    targets, text_col = cfg["data"]["target_cols"], cfg["data"]["text_col"]
    g = cfg["gpt"]
    examples = []
    for i, (_, row) in enumerate(few_shot_df.iterrows(), start=1):
        examples.append(
            f"EXAMPLE {i}\n\nNarrative:\n{trim_narrative(row[text_col], g['few_shot_max_chars'])}\n\n"
            f"Correct output:\n{json.dumps(structured_answer_from_row(row, targets))}\n"
        )
    # System prompt is sent once, as the system message (not duplicated here).
    return (
        "These examples come only from the training set.\n\n"
        + "\n".join(examples)
        + "\nNOW CLASSIFY THIS TEST CRASH:\n\n"
        + build_prediction_prompt(description, vocab, targets, cfg["prompt"]["max_narrative_chars"])
    )


def predict_gpt(cfg, client, train_df, test_df, vocab, recorder, max_rows=None) -> pd.DataFrame:
    g = cfg["gpt"]
    targets, text_col, case_col = cfg["data"]["target_cols"], cfg["data"]["text_col"], cfg["data"]["case_col"]

    few_shot_df = select_few_shot_examples(
        train_df, g["n_few_shot"], cfg["seed"], cfg["data"].get("stratify_col", "crash_type")
    )
    recorder.update_params(few_shot_case_ids=few_shot_df[case_col].tolist())

    system_prompt = build_system_prompt(targets)
    schema = build_schema(vocab, targets)

    existing = recorder.load_existing_predictions()
    subset = test_df if max_rows is None else test_df.head(max_rows)
    records = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=g["model"]):
        case_id = str(row[case_col])
        if case_id in existing:
            records.append(existing[case_id])
            continue

        user_msg = build_user_message(row[text_col], few_shot_df, vocab, cfg)
        parsed, output_text, usage, error = None, None, None, None
        t0 = time.time()

        for attempt in range(g["max_retries"]):
            try:
                response = client.responses.create(
                    model=g["model"],
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    text={"format": {
                        "type": "json_schema", "name": "crash_reconstruction",
                        "strict": True, "schema": schema,
                    }},
                    max_output_tokens=g["max_output_tokens"],
                    temperature=0,
                )
                output_text = response.output_text
                parsed = json.loads(output_text)
                usage = getattr(response, "usage", None)
                break
            except Exception as e:  # noqa: BLE001
                error = repr(e)
                print(f"[{case_id}] attempt {attempt + 1} failed: {e}")
                time.sleep(min(60, 2 ** attempt))

        record = {case_col: case_id}
        for t in targets:
            record[f"pred_{t}"] = (
                INVALID if parsed is None else canonicalize_prediction(parsed.get(t, INVALID), t, vocab)
            )
        records.append(record)
        recorder.save_predictions(records)          # checkpoint after every call
        recorder.append_raw_output({
            case_col: case_id,
            "output_text": output_text,
            "parsed_ok": parsed is not None,
            "error": None if parsed is not None else error,
            "latency_sec": round(time.time() - t0, 3),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        })

    return pd.DataFrame(records)
