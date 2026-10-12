# Crash-Narrative Reconstruction — BERT vs GPT-4o vs QLoRA LLMs

Predicts six fields from a police crash narrative (`unit_1_direction`, `unit_1_movement`,
`unit_2_direction`, `unit_2_movement`, `unit_at_fault`, `crash_type`) with five models:

| # | Model | Script |
|---|-------|--------|
| 1 | BERT multi-task classifier | `scripts/run_bert.py` |
| 2 | GPT-4o, 8-shot, strict JSON schema | `scripts/run_gpt.py` |
| 3 | Llama 3.1 8B Instruct + QLoRA | `scripts/run_qlora.py --model llama` |
| 4 | Qwen3 8B + QLoRA | `scripts/run_qlora.py --model qwen` |
| 5 | DeepSeek-R1-Distill-Qwen 7B + QLoRA | `scripts/run_qlora.py --model deepseek` |

## Project layout

```
crash-narrative-llm/
├── configs/config.yaml        ← ALL settings: paths, columns, hyper-parameters
├── data/
│   ├── raw/                   ← put crashdata_clean_20260929.xlsx here
│   └── splits/                ← fixed train/test split (created once, reused)
├── src/crashllm/
│   ├── config.py              config loading, .env secrets, seeding
│   ├── data.py                cleaning + fixed 70/30 split
│   ├── labels.py              allowed label vocabulary + output canonicalisation
│   ├── prompts.py             shared system / user prompts
│   ├── evaluation.py          metrics → tidy DataFrames
│   ├── results.py             RunRecorder (saving) + load_table (loading)
│   ├── plotting.py            ready-made figures from saved CSVs
│   └── models/
│       ├── bert.py            Model 1
│       ├── gpt.py             Model 2
│       ├── llm_common.py      chat templates, SFT dataset, JSON parsing
│       └── qlora.py           Models 3–5
├── scripts/                   one entry point per step (also in VS Code Run menu)
├── notebooks/plot_results.ipynb   ← make your own plots here
├── results/                   created automatically
└── trained_models/            saved BERT weights + LoRA adapters
```

## Setup (VS Code)

1. Open the `crash-narrative-llm` folder in VS Code.
2. Create a virtual environment in the integrated terminal:
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
   ```
   Then pick it with **Ctrl+Shift+P → Python: Select Interpreter → .venv**.
3. Install PyTorch for your GPU first (pick your CUDA version at <https://pytorch.org>), e.g.
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
4. Install the rest, plus this project in editable mode:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
5. Copy `.env.example` to `.env` and add `OPENAI_API_KEY` and `HF_TOKEN`
   (Llama needs approved access on Hugging Face).
6. Put the dataset in `data/raw/`, or point `paths.data_path` in `configs/config.yaml` at it.

## Running

Run the steps from the terminal, or from **Run and Debug (Ctrl+Shift+D)**, which has a
ready-made entry for each one:

```bash
python scripts/prepare_data.py                 # 0. fixed split (run once)
python scripts/run_bert.py                     # 1.
python scripts/run_gpt.py --max-rows 10        # 2. quick check first
python scripts/run_gpt.py                      #    full run
python scripts/run_qlora.py --model llama      # 3.
python scripts/run_qlora.py --model qwen       # 4.
python scripts/run_qlora.py --model deepseek   # 5.
python scripts/compare_models.py               # 6. comparison tables
python scripts/plot_results.py                 # 7. standard figures
```

Useful flags:

- `--force-retrain`: retrain even if a saved model or adapter exists. By default it is loaded.
- `--max-rows N`: predict only N test cases. The run is saved as `<name>_DEBUG` and
  left out of the comparison tables.
- `--resume latest`: continue an interrupted GPT or LLM prediction run in the same folder.
- `--train-only` (QLoRA): train and save the adapter, then skip prediction.
- `--config other.yaml`: use another config, e.g. for an A100 with the 14B models.

The QLoRA models need an NVIDIA GPU with about 16 GB or more of memory (4-bit).
Run one model per command; GPU memory is freed when the script exits.

## Saved results (for plotting later)

Every run gets its own time-stamped folder, so nothing is overwritten:

```
results/
├── runs_index.csv                    one row per finished run with headline metrics
├── <MODEL_NAME>/
│   ├── LATEST                        newest run id
│   └── 20261006-153012/
│       ├── run_config.json           every hyper-parameter, model id, split seed, timing
│       ├── predictions.csv           raw predictions (+ BERT softmax confidences)
│       ├── raw_outputs.jsonl         LLM raw text, token counts, latency per case
│       ├── training_history.csv      loss per optimizer step and per epoch
│       ├── case_results.csv          truth, prediction and correct flags for each case
│       ├── variable_metrics.csv      accuracy, macro/weighted F1 and invalid rate per variable
│       ├── per_class_metrics.csv     precision, recall, F1 and support per class
│       ├── confusion_matrices.csv    long format: true_label, pred_label, count, row %
│       └── overall_metrics.csv       exact match, field-level match, mean F1
├── summary/                          written by compare_models.py
│   ├── FINAL_MODEL_PERFORMANCE_TABLE.csv
│   └── ALL_<table>.csv               the same tables for all models, concatenated
└── figures/                          written by plot_results.py
```

Every table is in long (tidy) format and has `model` and `run_id` columns. To plot your own way:

```python
from crashllm.config import load_config
from crashllm.results import load_table

cfg = load_config()
vm = load_table(cfg["paths"]["results_dir"], "variable_metrics")        # newest run per model
hist = load_table(cfg["paths"]["results_dir"], "training_history", runs="all")
```

`notebooks/plot_results.ipynb` has worked examples.

## Changes from the Colab notebook

- **Environment:** the numpy/pandas version pins that caused `numpy.dtype size changed` are gone.
- **Labels stay strings:** the split CSVs are read with `dtype=str`, so labels like `"1"` and `"2"`
  are never turned into integers.
- **Long prompts:** the narrative is shortened until the prompt fits the token budget. Before,
  the end of the prompt, or during training the whole answer, could be cut off, which gave NaN losses.
- **Training labels:** each training sequence is the prompt followed by the answer, so the prompt
  is always an exact prefix. This fixes the label misalignment for DeepSeek-R1, whose template
  opens a `<think>` block.
- **Aliases:** canonicalisation checks the raw value against the allowed labels before applying
  an alias, so correct answers are no longer marked `__invalid__`.
- **GPT-4o:** the system prompt is sent once (it used to be duplicated), and `temperature=0` makes
  runs reproducible.
- **Macro F1:** computed over the training label set. `__invalid__` is reported separately as
  `invalid_rate` instead of counting as an extra zero-F1 class.
- **Final cells:** the missing `glob` import and `model_comparison` are handled in `compare_models.py`.
