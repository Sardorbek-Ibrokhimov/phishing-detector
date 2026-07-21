"""Fair comparison of LR-TFIDF vs DistilBERT on the SAME
leak-aware grouped split and SAME 8k head-to-head training set. Metrics
table + McNemar's exact test on the shared held-out test set.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def metrics(y, pred, proba):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "auc_roc": roc_auc_score(y, proba)}


def mcnemar(y, a, b):
    a_right = a == y
    b_right = b == y
    b_only = int((~a_right & b_right).sum())   # B right, A wrong
    a_only = int((a_right & ~b_right).sum())   # A right, B wrong
    n = a_only + b_only
    p = binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0
    return a_only, b_only, p


def main():
    lr = pd.read_csv(RESULTS / "email_lr_headtohead.csv")
    bert = pd.read_csv(RESULTS / "email_distilbert.csv")
    assert len(lr) == len(bert) and (lr["label"].values == bert["label"].values).all(), \
        "LR and DistilBERT were evaluated on different test rows — split mismatch"

    y = lr["label"].values
    m_lr = metrics(y, lr["lr_pred"].values, lr["lr_proba"].values)
    m_bert = metrics(y, bert["bert_pred"].values, bert["bert_proba"].values)

    table = pd.DataFrame([{"model": "LR-TFIDF (8k)", **m_lr},
                          {"model": "DistilBERT (8k)", **m_bert}]).round(4)
    print("\n=== Head-to-head (same split, same 8k train, full test) ===")
    print(table.to_string(index=False))

    a_only, b_only, p = mcnemar(y, lr["lr_pred"].values, bert["bert_pred"].values)
    sig = "significant (p<0.05)" if p < 0.05 else "NOT significant (p>=0.05)"
    print(f"\nMcNemar: LR-only-correct={a_only}, DistilBERT-only-correct={b_only}, "
          f"p={p:.4g} -> {sig}")

    table.to_csv(RESULTS / "email_comparison.csv", index=False)
    with open(RESULTS / "email_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Email classifier: LR-TFIDF vs DistilBERT (head-to-head)\n\n")
        f.write(table.to_markdown(index=False))
        f.write(f"\n\nMcNemar: LR-only-correct={a_only}, DistilBERT-only-correct={b_only}, "
                f"p={p:.4g} -> {sig}\n")
    print(f"\nSaved: {RESULTS/'email_comparison.csv'}, {RESULTS/'email_comparison.md'}")


if __name__ == "__main__":
    main()
