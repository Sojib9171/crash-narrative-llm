"""Step 0 — clean the dataset and create / load the fixed 70/30 split.

    python scripts/prepare_data.py               # create once, then reuse
    python scripts/prepare_data.py --reset-split # deliberately re-split
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crashllm.cli import base_parser  # noqa: E402
from crashllm.config import load_config, set_seed  # noqa: E402
from crashllm.data import class_distribution, narrative_overlap, prepare_split  # noqa: E402
from crashllm.labels import build_label_vocab, save_label_vocab  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--reset-split", action="store_true", help="Create a brand-new split")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    train_df, test_df = prepare_split(cfg, reset=args.reset_split)
    total = len(train_df) + len(test_df)
    print(f"TRAIN: {len(train_df)} ({len(train_df) / total * 100:.2f}%)")
    print(f"TEST : {len(test_df)} ({len(test_df) / total * 100:.2f}%)")

    split_dir = cfg["paths"]["split_dir"]
    vocab = build_label_vocab(train_df, cfg["data"]["target_cols"])
    save_label_vocab(vocab, split_dir / "label_vocab.json")

    dist = class_distribution(train_df, test_df, cfg)
    dist.to_csv(split_dir / "class_distribution.csv", index=False)
    for var, g in dist.groupby("variable", sort=False):
        print(f"\n{var}")
        print(g.pivot_table(index="label", columns="split", values="count", fill_value=0).astype(int).to_string())

    overlap = narrative_overlap(train_df, test_df, cfg)
    print(f"\nExact narrative overlap between train and test: {overlap}")
    if overlap:
        print("WARNING: identical narrative text exists in both sets (leakage).")

    print(f"\nSaved to {split_dir}")


if __name__ == "__main__":
    main()
