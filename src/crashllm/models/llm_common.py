"""Chat templating, length-safe prompt building, SFT dataset and JSON parsing."""

from __future__ import annotations

import json
import re

import torch
from torch.utils.data import Dataset

from ..prompts import build_prediction_prompt, build_system_prompt, structured_answer_from_row


def apply_template(tokenizer, messages, model_id, add_generation_prompt):
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False        # no reasoning mode for classification
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_fitted_prompt(tokenizer, model_id, description, vocab, targets, max_chars, token_budget):
    """Render the chat prompt, shrinking the NARRATIVE until it fits the budget.

    This replaces plain right-truncation, which used to cut off the end of the
    prompt (the generation cue) or, during training, the whole answer.
    Returns (prompt_text, prompt_ids, narrative_chars_used).
    """
    system_prompt = build_system_prompt(targets)
    chars = max_chars
    while True:
        user = build_prediction_prompt(description, vocab, targets, chars)
        text = apply_template(
            tokenizer,
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            model_id, add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= token_budget:
            return text, ids, chars
        if chars <= 200:
            # Instructions + label list alone exceed the budget (budget far too
            # small). Keep the END of the prompt so the generation cue and the
            # answer are never cut. Increase max_length / max_prompt_tokens.
            print(f"WARNING: prompt ({len(ids)} tokens) exceeds budget {token_budget}; "
                  "truncating from the left. Increase qlora.max_length / max_prompt_tokens.")
            ids = ids[-token_budget:]
            return tokenizer.decode(ids), ids, chars
        chars = int(chars * 0.8)


def answer_text(prompt_text, answer_json, tokenizer):
    """Assistant target appended directly after the generation prompt.

    Building `prompt + answer` (instead of rendering the template twice) keeps
    the prompt an exact prefix of the training sequence for every model.
    DeepSeek-R1-Distill's template opens a <think> block in the generation
    prompt, so we close it immediately and train the model to answer directly.
    """
    prefix = "</think>\n\n" if prompt_text.rstrip().endswith("<think>") else ""
    return prefix + answer_json + (tokenizer.eos_token or "")


class CrashSFTDataset(Dataset):
    """Supervised fine-tuning examples; loss only on the answer tokens."""

    def __init__(self, df, tokenizer, model_id, vocab, cfg, max_length=1536):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.vocab = vocab
        self.targets = cfg["data"]["target_cols"]
        self.text_col = cfg["data"]["text_col"]
        self.max_chars = cfg["prompt"]["max_narrative_chars"]
        self.max_length = max_length
        self.n_shortened = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        answer_json = json.dumps(structured_answer_from_row(row, self.targets))

        # Reserve room for the answer, then fit the prompt into the rest.
        probe = self.tokenizer(answer_json + "</think>\n\n" + (self.tokenizer.eos_token or ""),
                               add_special_tokens=False)["input_ids"]
        budget = self.max_length - len(probe) - 2

        prompt_text, prompt_ids, chars = build_fitted_prompt(
            self.tokenizer, self.model_id, row[self.text_col], self.vocab,
            self.targets, self.max_chars, budget,
        )
        if chars < self.max_chars and len(str(row[self.text_col])) > chars:
            self.n_shortened += 1

        ans_ids = self.tokenizer(answer_text(prompt_text, answer_json, self.tokenizer),
                                 add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + ans_ids
        labels = [-100] * len(prompt_ids) + ans_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def causal_lm_collator(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(t, value):
        return torch.cat([t, torch.full((max_len - len(t),), value, dtype=torch.long)])

    return {
        "input_ids": torch.stack([pad(x["input_ids"], pad_token_id) for x in batch]),
        "attention_mask": torch.stack([pad(x["attention_mask"], 0) for x in batch]),
        "labels": torch.stack([pad(x["labels"], -100) for x in batch]),
    }


def extract_json_object(text, targets):
    """Return the last flat JSON object containing all targets, else None."""
    text = str(text).strip().replace("```json", "").replace("```", "")
    for candidate in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)):
        try:
            obj = json.loads(candidate)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict) and all(k in obj for k in targets):
            return obj
    return None
