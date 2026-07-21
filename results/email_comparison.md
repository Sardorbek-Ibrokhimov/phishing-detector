# Email classifier: LR-TFIDF vs DistilBERT (head-to-head)

| model           |   accuracy |   precision |   recall |     f1 |   auc_roc |
|:----------------|-----------:|------------:|---------:|-------:|----------:|
| LR-TFIDF (8k)   |     0.9805 |      0.977  |   0.986  | 0.9815 |    0.9981 |
| DistilBERT (8k) |     0.9723 |      0.9863 |   0.9604 | 0.9732 |    0.9973 |

McNemar: LR-only-correct=341, DistilBERT-only-correct=207, p=1.136e-08 -> significant (p<0.05)
