"""Part 5: broader accuracy sanity check.

Takes 20 phishing + 20 benign URLs that were HELD OUT from training (the
test partition of the domain-grouped split, random_state=42, that
train_final_model uses — ), classifies each
with the retrained model IN-PROCESS (not via the live API — the deployed
API still serves the pre-fix model this session; retraining/redeploying it
is explicitly deferred), and reports how many verdicts were correct.

Caveat (): the benign URLs are real Curlie-listed
URLs on domains a human directory editor added, and the phishing URLs are
real PhishTank entries. These are held out from the model's fit, on
domains not seen in training, so this is closer to genuine generalisation
than the old version — but it's still drawn from the same two source
datasets the model trains on, not a fully independent corpus. See
review_experiments.py's out-of-distribution check for that.
"""

import csv
import sys
from pathlib import Path

import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shap_explain import explain_prediction, train_final_model
from train_baseline import grouped_train_test_split, load_merged_url_dataset

RESULTS = Path(__file__).resolve().parent.parent / "results"
RANDOM_STATE = 42
N_PER_CLASS = 20


def held_out_sample():
    df = load_merged_url_dataset()
    # Same domain-grouped partition as train_final_model.
    _, test_idx = grouped_train_test_split(df)
    test_df = df.loc[test_idx]
    phish = test_df[test_df["label"] == 1].sample(N_PER_CLASS, random_state=RANDOM_STATE)
    benign = test_df[test_df["label"] == 0].sample(N_PER_CLASS, random_state=RANDOM_STATE)
    return phish, benign


def main():
    print("Training model for sanity check...")
    model, _X_test, cols = train_final_model()
    explainer = shap.TreeExplainer(model)

    def classify(url):
        r = explain_prediction(url, model, explainer, cols)
        return r["verdict"], r["confidence"]

    phish, benign = held_out_sample()
    rows = []
    counts = {"phishing": [0, 0], "benign": [0, 0]}  # [correct, total]

    for _, r in phish.iterrows():
        verdict, conf = classify(r["url"])
        correct = verdict == "phishing"
        counts["phishing"][0] += correct
        counts["phishing"][1] += 1
        rows.append(("phishing", r["url"], verdict, conf, correct))

    for _, r in benign.iterrows():
        verdict, conf = classify(r["url"])
        correct = verdict == "legitimate"
        counts["benign"][0] += correct
        counts["benign"][1] += 1
        rows.append(("legitimate", r["url"], verdict, conf, correct))

    total_correct = sum(c[0] for c in counts.values())
    total = sum(c[1] for c in counts.values())

    out_csv = RESULTS / "accuracy_sanity.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_label", "url", "predicted", "confidence", "correct"])
        w.writerows(rows)

    summary = [
        "Part 5 — accuracy sanity check (held-out, in-distribution)",
        "=" * 58,
        f"Phishing: {counts['phishing'][0]}/{counts['phishing'][1]} correct",
        f"Benign:   {counts['benign'][0]}/{counts['benign'][1]} correct",
        f"Overall:  {total_correct}/{total} correct ({total_correct/total:.1%})",
        "",
        "Misclassified:",
    ]
    for lbl, url, pred, conf, ok in rows:
        if not ok:
            summary.append(f"  [{lbl} -> {pred}@{conf}] {url}")
    if total_correct == total:
        summary.append("  (none)")

    out_txt = RESULTS / "accuracy_sanity.txt"
    out_txt.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print(f"\nSaved: {out_csv}\nSaved: {out_txt}")


if __name__ == "__main__":
    main()
