"""Off-corpus sanity test (the email analogue of the live URL
test). Hand-written realistic emails that belong to NEITHER training corpus.
The question: does the classifier generalise to genuine phishing/legit email,
or did it only learn to separate Enron/mailing-list ham from the phishing
campaigns it was trained on?

Both models see the SAME text, normalised to the corpus preprocessing style
(lowercase, punctuation stripped) so format is held constant and only content
generalisation is tested.
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from email_baseline_lr import N_TRAIN_HEADTOHEAD, make_lr
from email_data import get_split

MODELS = Path(__file__).resolve().parent.parent / "models"

# Genuine, hand-written emails from neither corpus. LEGIT: a real-style
# newsletter, an order confirmation, a colleague's note, a bank notice.
# PHISH: credential reset, prize scam, invoice fraud, account-locked.
LEGIT = [
    ("newsletter", "Hi Sam, this week in our engineering blog: we shipped the new "
        "dashboard, three tips for faster CI, and an interview with our SRE team. "
        "Read it all on the website. You can manage your subscription in account settings."),
    ("order confirmation", "Thanks for your order #48213. Your parcel of two books and "
        "a notebook has shipped and should arrive Tuesday. Track it from your orders page. "
        "Questions? Reply to this email and our support team will help."),
    ("colleague", "Sam, quick one before the standup: I pushed the fix for the auth bug to "
        "the feature branch, can you review when you get a sec? Also lunch at 12 if you're around. Cheers, Priya"),
    ("bank statement", "Your monthly statement for the account ending 4471 is now available. "
        "Log in to online banking to view it. We will never ask for your full password by email."),
    ("meeting invite", "You are invited to the quarterly planning meeting on Thursday at 2pm "
        "in the main conference room. The agenda and pre-read are attached. Let me know if you cannot make it."),
    ("receipt", "Here is your receipt for the annual software subscription, 89 dollars, "
        "charged to your card on file. Your licence is active until next year. Manage billing in your profile."),
]
PHISH = [
    ("credential reset", "Dear customer your account has been temporarily suspended due to "
        "unusual activity. You must verify your identity within 24 hours or your account will be "
        "permanently closed. Click the link below and confirm your password and card details immediately."),
    ("prize scam", "Congratulations you have been selected as the winner of our annual lottery "
        "worth five million dollars. To claim your prize send us your full name address and bank "
        "account number and a processing fee of 200 dollars via wire transfer today."),
    ("invoice fraud", "Please find attached the overdue invoice. Payment must be made today to "
        "avoid legal action. Update the beneficiary bank details to the new account provided below "
        "and confirm once the transfer is complete. Do not call, reply only by email."),
    ("account locked", "We detected a login to your account from a new device. If this was not you "
        "your account is at risk. Verify now by entering your username password and one time code on "
        "our secure page or your access will be revoked within one hour."),
    ("delivery scam", "Your package could not be delivered because of an unpaid customs fee of 3 "
        "dollars. Confirm your payment and shipping details at the link to reschedule delivery, "
        "otherwise your parcel will be returned to sender."),
    ("ceo fraud", "Are you at your desk. I need you to process an urgent wire transfer to a new "
        "supplier before end of day, this is confidential do not discuss with anyone. Send me the "
        "available balance and I will forward the account details. Sent from my iphone."),
]


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())
    # collapse handled by the vectorizer / tokenizer


def main():
    train, _ = get_split(cap_train=N_TRAIN_HEADTOHEAD, cap_test=None)
    lr = make_lr()
    lr.fit(train["text"], train["label"])

    tok = AutoTokenizer.from_pretrained(MODELS / "email_distilbert")
    bert = AutoModelForSequenceClassification.from_pretrained(MODELS / "email_distilbert")
    bert.eval()

    def bert_prob(text):
        enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            return float(torch.softmax(bert(**enc).logits, dim=1)[0, 1])

    print("=== OFF-CORPUS SANITY TEST (1 = phishing) ===")
    lr_ok = bert_ok = 0
    total = 0
    for want, group in [(0, LEGIT), (1, PHISH)]:
        print(f"\n--- {'LEGIT (want 0)' if want==0 else 'PHISH (want 1)'} ---")
        for name, raw in group:
            x = normalise(raw)
            p_lr = float(lr.predict_proba([x])[0, 1])
            p_bert = bert_prob(x)
            lr_v = int(p_lr >= 0.5); bert_v = int(p_bert >= 0.5)
            lr_ok += (lr_v == want); bert_ok += (bert_v == want); total += 1
            flag_lr = "ok " if lr_v == want else "XX "
            flag_b = "ok " if bert_v == want else "XX "
            print(f"  {name:18s} | LR {flag_lr}{p_lr:.2f} | BERT {flag_b}{p_bert:.2f}")

    print(f"\nLR-TFIDF   correct off-corpus: {lr_ok}/{total}")
    print(f"DistilBERT correct off-corpus: {bert_ok}/{total}")


if __name__ == "__main__":
    main()
