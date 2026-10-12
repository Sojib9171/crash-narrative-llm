"""Models 3–5 — QLoRA fine-tuning and inference for open chat LLMs."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..labels import INVALID, canonicalize_prediction
from .llm_common import CrashSFTDataset, build_fitted_prompt, causal_lm_collator, extract_json_object


def get_compute_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def adapter_dir(cfg, run_name) -> Path:
    return Path(cfg["paths"]["models_dir"]) / run_name


def load_base_model(cfg, model_id, hf_token=None):
    q = cfg["qlora"]
    dtype = get_compute_dtype()
    kwargs = {"token": hf_token}

    if q.get("load_in_4bit", True):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    if torch.cuda.is_available():
        # Whole model on the single GPU (standard for QLoRA); "auto" refuses
        # 8B models on 8 GB cards because of its safety margin.
        kwargs["device_map"] = {"": torch.cuda.current_device()}

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:          # older transformers
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    return model, tokenizer


def train_qlora(cfg, model, tokenizer, model_id, run_name, train_df, vocab, recorder):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    q = cfg["qlora"]
    use_gc = q.get("gradient_checkpointing", True)

    if q.get("load_in_4bit", True):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
    elif use_gc:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    lc = q["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        bias="none", task_type="CAUSAL_LM", target_modules=lc["target_modules"],
    ))
    model.print_trainable_parameters()

    dataset = CrashSFTDataset(train_df, tokenizer, model_id, vocab, cfg, max_length=q["max_length"])
    loader = DataLoader(
        dataset, batch_size=q["batch_size"], shuffle=True,
        collate_fn=partial(causal_lm_collator, pad_token_id=tokenizer.pad_token_id),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=q["learning_rate"])
    accum = q["grad_accum_steps"]
    device = next(model.parameters()).device

    model.train()
    optimizer.zero_grad()
    opt_step = 0
    for epoch in range(1, q["epochs"] + 1):
        running, window = 0.0, []
        progress = tqdm(loader, desc=f"{run_name} epoch {epoch}/{q['epochs']}")
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                print(f"Non-finite loss at step {step}; skipping batch.")
                continue
            (loss / accum).backward()
            running += loss.item()
            window.append(loss.item())

            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                opt_step += 1
                recorder.log_history(epoch=epoch, step=opt_step, loss=sum(window) / len(window), level="step")
                window = []
            progress.set_postfix(loss=running / (step + 1))

        recorder.log_history(epoch=epoch, step=opt_step, loss=running / len(loader), level="epoch")
        print(f"Epoch {epoch} mean loss = {running / len(loader):.4f}")

    out = adapter_dir(cfg, run_name)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "training_info.json").write_text(json.dumps({
        "base_model": model_id, "run_id": recorder.run_id,
        "examples_with_shortened_narrative": dataset.n_shortened,
        **{k: q[k] for k in ("epochs", "learning_rate", "batch_size", "grad_accum_steps", "max_length")},
        "lora": q["lora"],
    }, indent=2))
    recorder.update_params(adapter_dir=str(out), examples_with_shortened_narrative=dataset.n_shortened)
    print("LoRA adapter saved to:", out)

    model.config.use_cache = True
    return model


def load_adapter(model, path):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, path)


@torch.no_grad()
def predict_local_llm(cfg, model, tokenizer, model_id, test_df, vocab, recorder, max_rows=None):
    q = cfg["qlora"]
    targets, text_col, case_col = cfg["data"]["target_cols"], cfg["data"]["text_col"], cfg["data"]["case_col"]

    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device

    existing = recorder.load_existing_predictions()
    subset = test_df if max_rows is None else test_df.head(max_rows)
    records = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"{recorder.run_name} predict"):
        case_id = str(row[case_col])
        if case_id in existing:
            records.append(existing[case_id])
            continue

        _, prompt_ids, chars = build_fitted_prompt(
            tokenizer, model_id, row[text_col], vocab, targets,
            cfg["prompt"]["max_narrative_chars"], q["max_prompt_tokens"],
        )
        input_ids = torch.tensor([prompt_ids], device=device)
        t0 = time.time()
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=q["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[0, input_ids.shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = extract_json_object(output_text, targets)

        record = {case_col: case_id}
        for t in targets:
            record[f"pred_{t}"] = (
                INVALID if parsed is None else canonicalize_prediction(parsed.get(t, INVALID), t, vocab)
            )
        records.append(record)
        recorder.save_predictions(records)
        recorder.append_raw_output({
            case_col: case_id,
            "output_text": output_text,
            "parsed_ok": parsed is not None,
            "prompt_tokens": len(prompt_ids),
            "new_tokens": int(new_tokens.shape[0]),
            "narrative_chars_used": chars,
            "latency_sec": round(time.time() - t0, 3),
        })

    return pd.DataFrame(records)
