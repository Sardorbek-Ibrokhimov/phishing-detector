# LR vs XGBoost, clean feature set (no URL-length, no uses_https)

Features used (11): num_hyphens, num_digits, num_special, has_at, has_ip_host, num_subdomains, has_port, host_entropy, num_suspicious_tokens, digit_ratio, tld_length

|                     |   accuracy |   precision |   recall |     f1 |   auc_roc |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| Logistic Regression |     0.6673 |      0.6233 |   0.4208 | 0.5024 |    0.615  |
| XGBoost             |     0.7213 |      0.7823 |   0.4182 | 0.545  |    0.6878 |

McNemar's exact test: LR-only-correct=808, XGB-only-correct=1729, p=3.282e-76 -> significant (p < 0.05)
