"""Model 1 — multi-task BERT classifier (one linear head per target)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


class LabelCodec:
    """Per-target label <-> id mapping, fitted on TRAINING data only."""

    def __init__(self, classes: dict[str, list[str]]):
        self.classes = classes
        self.to_id = {t: {c: i for i, c in enumerate(cs)} for t, cs in classes.items()}

    @classmethod
    def fit(cls, train_df, targets):
        return cls({t: sorted(train_df[t].astype(str).unique().tolist()) for t in targets})

    def encode(self, target, values):
        return np.array([self.to_id[target][str(v)] for v in values], dtype=np.int64)

    def decode(self, target, ids):
        return [self.classes[target][int(i)] for i in ids]

    def num_classes(self):
        return {t: len(cs) for t, cs in self.classes.items()}

    def save(self, path):
        Path(path).write_text(json.dumps(self.classes, indent=2))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))


class CrashBERTDataset(Dataset):
    def __init__(self, df, tokenizer, codec, targets, text_col, max_length=384, with_labels=True):
        self.texts = df[text_col].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = targets
        self.labels = (
            {t: codec.encode(t, df[t]) for t in targets} if with_labels else None
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tok = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        item = {"input_ids": tok["input_ids"].squeeze(0),
                "attention_mask": tok["attention_mask"].squeeze(0)}
        if self.labels is not None:
            for t in self.targets:
                item[t] = torch.tensor(self.labels[t][idx], dtype=torch.long)
        return item


class MultiTaskBERT(nn.Module):
    def __init__(self, base_model_name, num_classes: dict[str, int], dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, n) for t, n in num_classes.items()})

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        rep = self.dropout(out.last_hidden_state[:, 0, :])   # [CLS]
        return {t: head(rep) for t, head in self.heads.items()}


def _class_weighted_losses(train_df, codec, targets, device, weighted=True):
    losses = {}
    for t in targets:
        if not weighted:
            losses[t] = nn.CrossEntropyLoss()
            continue
        ids = codec.encode(t, train_df[t])
        counts = np.bincount(ids, minlength=len(codec.classes[t]))
        w = len(ids) / (len(counts) * np.maximum(counts, 1))
        losses[t] = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    return losses


def model_dir(cfg) -> Path:
    return Path(cfg["paths"]["models_dir"]) / cfg["bert"]["run_name"]


def train_bert(cfg, train_df, recorder, device):
    b = cfg["bert"]
    targets, text_col = cfg["data"]["target_cols"], cfg["data"]["text_col"]

    tokenizer = AutoTokenizer.from_pretrained(b["model_name"])
    codec = LabelCodec.fit(train_df, targets)
    model = MultiTaskBERT(b["model_name"], codec.num_classes(), b["dropout"]).to(device)

    loader = DataLoader(
        CrashBERTDataset(train_df, tokenizer, codec, targets, text_col, b["max_length"]),
        batch_size=b["batch_size"], shuffle=True,
    )
    losses = _class_weighted_losses(train_df, codec, targets, device, b.get("class_weighted_loss", True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=b["learning_rate"], weight_decay=b["weight_decay"])

    global_step = 0
    for epoch in range(1, b["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        per_target = {t: 0.0 for t in targets}
        for batch in tqdm(loader, desc=f"BERT epoch {epoch}/{b['epochs']}"):
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = 0
            for t in targets:
                lt = losses[t](logits[t], batch[t].to(device))
                per_target[t] += lt.item()
                loss = loss + lt
            loss = loss / len(targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            epoch_loss += loss.item()
            recorder.log_history(epoch=epoch, step=global_step, loss=loss.item(), level="step")

        n = len(loader)
        recorder.log_history(
            epoch=epoch, step=global_step, loss=epoch_loss / n, level="epoch",
            **{f"loss_{t}": v / n for t, v in per_target.items()},
        )
        print(f"Epoch {epoch}/{b['epochs']}  loss = {epoch_loss / n:.4f}")

    out = model_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    codec.save(out / "label_classes.json")
    tokenizer.save_pretrained(out)
    print("BERT saved to", out)
    return model, tokenizer, codec


def load_bert(cfg, device):
    b = cfg["bert"]
    out = model_dir(cfg)
    codec = LabelCodec.load(out / "label_classes.json")
    tokenizer = AutoTokenizer.from_pretrained(out)
    model = MultiTaskBERT(b["model_name"], codec.num_classes(), b["dropout"])
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu"))
    return model.to(device), tokenizer, codec


@torch.no_grad()
def predict_bert(cfg, model, tokenizer, codec, test_df, device) -> pd.DataFrame:
    b = cfg["bert"]
    targets, text_col, case_col = cfg["data"]["target_cols"], cfg["data"]["text_col"], cfg["data"]["case_col"]

    loader = DataLoader(
        CrashBERTDataset(test_df, tokenizer, codec, targets, text_col, b["max_length"], with_labels=False),
        batch_size=b["eval_batch_size"], shuffle=False,
    )
    model.eval()
    preds = {t: [] for t in targets}
    confs = {t: [] for t in targets}
    for batch in tqdm(loader, desc="BERT predict"):
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        for t in targets:
            prob = torch.softmax(logits[t], dim=1)
            conf, ids = prob.max(dim=1)
            preds[t].extend(codec.decode(t, ids.cpu().numpy()))
            confs[t].extend(conf.cpu().numpy().tolist())

    out = pd.DataFrame({case_col: test_df[case_col].astype(str).values})
    for t in targets:
        out[f"pred_{t}"] = preds[t]
        out[f"conf_{t}"] = confs[t]        # softmax confidence — useful for calibration plots
    return out
