"""Models 3–5 — QLoRA fine-tune (or load adapter), predict, evaluate, save.

    python scripts/run_qlora.py --model llama
    python scripts/run_qlora.py --model qwen --force-retrain
    python scripts/run_qlora.py --model deepseek --max-rows 10
    python scripts/run_qlora.py --model llama --resume latest
    python scripts/run_qlora.py --model llama --train-only

Model keys are defined under qlora.models in configs/config.yaml.
Run one model per process — GPU memory is released when the script exits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crashllm.cli import base_parser, debug_name, setup  # noqa: E402
from crashllm.config import get_secret  # noqa: E402
from crashllm.evaluation import evaluate, print_summary  # noqa: E402
from crashllm.results import RunRecorder  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", required=True, help="Key under qlora.models (e.g. llama, qwen, deepseek)")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--train-only", action="store_true", help="Train/save the adapter, skip prediction")
    p.add_argument("--max-rows", type=int, default=None, help="Only predict the first N test cases")
    p.add_argument("--resume", default=None, help="'latest' or a run_id to continue predictions")
    args = p.parse_args()

    cfg, train_df, test_df, vocab = setup(args)
    q = cfg["qlora"]
    if args.model not in q["models"]:
        sys.exit(f"Unknown model '{args.model}'. Options: {list(q['models'])}")

    from crashllm.models import qlora

    spec = q["models"][args.model]
    model_id, base_name = spec["hf_id"], spec["run_name"]
    run_name = debug_name(base_name, args.max_rows)

    hyper = {k: v for k, v in q.items() if k != "models"}
    recorder = RunRecorder(cfg, run_name, resume=args.resume,
                           params={"model_type": "qlora", "hf_id": model_id, **hyper,
                                   "max_rows": args.max_rows})

    hf_token = get_secret("HF_TOKEN")
    model, tokenizer = qlora.load_base_model(cfg, model_id, hf_token)

    adir = qlora.adapter_dir(cfg, base_name)
    if (adir / "adapter_config.json").exists() and not args.force_retrain:
        print(f"Existing LoRA adapter found at {adir} — loading (use --force-retrain to retrain).")
        model = qlora.load_adapter(model, adir)
        recorder.update_params(trained_in_this_run=False, adapter_dir=str(adir))
    else:
        model = qlora.train_qlora(cfg, model, tokenizer, model_id, base_name, train_df, vocab, recorder)
        recorder.update_params(trained_in_this_run=True)

    if args.train_only:
        print("--train-only: done.")
        return

    pred_df = qlora.predict_local_llm(cfg, model, tokenizer, model_id, test_df, vocab, recorder, args.max_rows)

    eval_df = test_df if args.max_rows is None else test_df.head(args.max_rows)
    out = evaluate(eval_df, pred_df, cfg, vocab, run_name)
    print_summary(out)
    recorder.save_evaluation(out)


if __name__ == "__main__":
    main()
