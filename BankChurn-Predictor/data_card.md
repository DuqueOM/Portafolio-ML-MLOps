# Data Card — BankChurn Predictor

## Dataset Overview
- **Name:** Beta Bank Churn Dataset (educational/simulated)
- **Records:** 10,000 customers
- **Features:** 10 tabular customer attributes (demographics + usage)
- **Target:** `Exited` (1 = churn, 0 = retained)

## Source & Licensing
- **Origin:** TripleTen / open educational dataset bundled with the project.
- **License:** MIT-compatible for portfolio/demo use (see `DATA_LICENSE`).
- **PII:** None. All identifiers are synthetic or removed.

## Schema Summary
| Column | Type | Notes |
|--------|------|-------|
| `CreditScore` | float | 350–850, standardized score |
| `Geography` | category | France, Spain, Germany |
| `Gender` | category | Male/Female |
| `Age` | int | 18–92 |
| `Tenure` | int | 0–10 years as customer |
| `Balance` | float | 0–250k |
| `NumOfProducts` | int | 1–4 |
| `HasCrCard` | int | 0/1 |
| `IsActiveMember` | int | 0/1 |
| `EstimatedSalary` | float | 11–200k |
| `Exited` | int | Target |

## Splits & Versioning
- Default split: 60% train / 20% validation / 20% test (stratified).
- Seeds defined in `configs/config.yaml` (42 by default).
- Processed datasets stored under `data/processed/` via `make preprocess`.

## Quality Considerations
- No missing values in raw CSV, but outliers exist (age > 80, zero balance).
- Strong class imbalance (79.6% retain vs 20.4% churn).
- Some features highly correlated (Age & Tenure) but below multicollinearity threshold.

## Bias & Ethical Notes
- Includes geography & age which can induce bias. Monitor fairness gaps by subgroup.
- Synthetic data may not reflect real customer diversity; treat metrics as illustrative only.

## Refresh Strategy
- Replace/augment with real bank churn data before production use.
- Recompute descriptive stats and fairness diagnostics per refresh.
- Track dataset version in MLflow/DVC when available.

## Contacts
- Maintainer: Daniel Duque (DuqueOM)
- Repository: `BankChurn-Predictor/`
