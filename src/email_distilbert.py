"""DistilBERT fine-tuned for email phishing classification.
Same leak-aware grouped split and same 20k head-to-head training set as the
LR baseline, so the comparison and McNemar are fair. CPU-only, so config is
deliberately modest (max_length=192, 2 epochs) per the task brief.

Saves per-example test predictions to results/email_distilbert.csv.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from email_baseline_lr import N_TRAIN_HEADTOHEAD
from email_data import get_split

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128   # modest, per the CPU brief; front-loaded email signal fits
EPOCHS = 1      # single epoch — CPU budget; documented as a limitation
BATCH = 16
LR = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def encode(tokenizer, texts):
    enc = tokenizer(list(texts), truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def main():
    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"threads={torch.get_num_threads()}  max_len={MAX_LEN}  epochs={EPOCHS}  batch={BATCH}")

    train, test = get_split(cap_train=N_TRAIN_HEADTOHEAD, cap_test=None)
    print(f"train={len(train)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.train()

    print("tokenising...")
    tr_ids, tr_mask = encode(tok, train["text"])
    tr_y = torch.tensor(train["label"].values)
    te_ids, te_mask = encode(tok, test["text"])

    dl = DataLoader(TensorDataset(tr_ids, tr_mask, tr_y), batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(dl) * EPOCHS
    print(f"steps/epoch={len(dl)}  total_steps={total_steps}")

    t0 = time.time()
    step = 0
    for ep in range(EPOCHS):
        for ids, mask, y in dl:
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=y)
            out.loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                eta = el / step * (total_steps - step)
                print(f"  ep{ep+1} step {step}/{total_steps} loss={out.loss.item():.4f} "
                      f"elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)
        print(f"epoch {ep+1} done at {(time.time()-t0)/60:.1f}m", flush=True)

    # Inference on full test
    print("evaluating on full test...", flush=True)
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(te_ids), 64):
            out = model(input_ids=te_ids[i:i+64], attention_mask=te_mask[i:i+64])
            probs.append(torch.softmax(out.logits, dim=1)[:, 1].numpy())
    proba = np.concatenate(probs)
    pred = (proba >= 0.5).astype(int)
    y = test["label"].values

    m = {"model": "DistilBERT",
         "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
         "recall": recall_score(y, pred), "f1": f1_score(y, pred),
         "auc_roc": roc_auc_score(y, proba)}
    print("\n=== DistilBERT (head-to-head 20k, full test) ===")
    for k, v in m.items():
        if k != "model":
            print(f"  {k:10s} {v:.4f}")

    out = test[["label"]].copy()
    out["bert_pred"] = pred
    out["bert_proba"] = proba
    out.to_csv(RESULTS / "email_distilbert.csv", index=False)
    pd.DataFrame([m]).to_csv(RESULTS / "email_distilbert_metrics.csv", index=False)
    # save the fine-tuned model for the off-corpus sanity test
    model.save_pretrained(RESULTS.parent / "models" / "email_distilbert")
    tok.save_pretrained(RESULTS.parent / "models" / "email_distilbert")
    print(f"total {(time.time()-t0)/60:.1f}m. Saved predictions + model.")


if __name__ == "__main__":
    main()
