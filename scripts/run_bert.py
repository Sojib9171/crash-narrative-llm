"""Model 1 — train (or load) the multi-task BERT classifier, predict, evaluate, save.

    python scripts/run_bert.py                  # trains if no saved model exists
    python scripts/run_bert.py --force-retrain
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crashllm.cli import base_parser, setup  # noqa: E402
from crashllm.config import get_device  # noqa: E402
from crashllm.evaluation import evaluate, print_summary  # noqa: E402
from crashllm.results import RunRecorder  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--force-retrain", action="store_true")
    args = p.parse_args()

    cfg, train_df, test_df, vocab = setup(args)
    from crashllm.models import bert  # heavy import after arg parsing

    device = get_device()
    print("Device:", device)
    b = cfg["bert"]
    recorder = RunRecorder(cfg, b["run_name"], params={"model_type": "bert", **b, "device": str(device)})

    if (bert.model_dir(cfg) / "model.pt").exists() and not args.force_retrain:
        print("Existing BERT model found — loading (use --force-retrain to retrain).")
        model, tokenizer, codec = bert.load_bert(cfg, device)
        recorder.update_params(trained_in_this_run=False)
    else:
        model, tokenizer, codec = bert.train_bert(cfg, train_df, recorder, device)
        recorder.update_params(trained_in_this_run=True)

    pred_df = bert.predict_bert(cfg, model, tokenizer, codec, test_df, device)
    recorder.save_predictions(pred_df.to_dict("records"))

    out = evaluate(test_df, pred_df, cfg, vocab, b["run_name"])
    print_summary(out)
    recorder.save_evaluation(out)


if __name__ == "__main__":
    main()
