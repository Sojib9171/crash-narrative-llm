"""Model 2 — GPT-4o few-shot. Needs OPENAI_API_KEY in .env.

    python scripts/run_gpt.py --max-rows 10     # quick check (saved separately as *_DEBUG)
    python scripts/run_gpt.py                   # full test set
    python scripts/run_gpt.py --resume latest   # continue an interrupted run
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
    p.add_argument("--max-rows", type=int, default=None, help="Only the first N test cases")
    p.add_argument("--resume", default=None, help="'latest' or a run_id to continue")
    args = p.parse_args()

    cfg, train_df, test_df, vocab = setup(args)

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not found. Add it to the .env file in the project root.")

    from openai import OpenAI

    from crashllm.models.gpt import predict_gpt

    g = cfg["gpt"]
    run_name = debug_name(g["run_name"], args.max_rows)
    recorder = RunRecorder(cfg, run_name, resume=args.resume,
                           params={"model_type": "openai", **g, "max_rows": args.max_rows})

    client = OpenAI(api_key=api_key)
    pred_df = predict_gpt(cfg, client, train_df, test_df, vocab, recorder, args.max_rows)

    eval_df = test_df if args.max_rows is None else test_df.head(args.max_rows)
    out = evaluate(eval_df, pred_df, cfg, vocab, run_name)
    print_summary(out)
    recorder.save_evaluation(out)


if __name__ == "__main__":
    main()
