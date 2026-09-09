# DataFit Real-Data Validation

## Scope and plan

This pass used locally available, publicly distributed scikit-learn copies of four classic datasets: Iris, Wine Recognition, Breast Cancer Wisconsin, and Diabetes. They were loaded without altering their rows, features, labels, or values. Each was assessed repeatedly with the existing task-specific assessment, and the applicable opt-in experiment was run.

This is a focused first real-data pass, not universal validation. The local public set did not include an unmodified dataset with missing values, duplicate rows, categorical-heavy features, or a meaningful identifier column. Those coverage gaps are recorded below rather than manufactured by mutating a dataset.

## Results

| Dataset | Task | Rows | Columns | Findings | Score | False Positives | False Negatives | Experiments | Result |
|---|---:|---:|---:|---|---:|---|---|---|---|
| Iris | Multiclass classification | 150 | 5 | Small dataset; one duplicate; petal-feature multicollinearity; mild potential label inconsistency; class balance info | 78 | No clear false positive; duplicate is real and the correlation is a legitimate redundancy indicator | No assessed missing/categorical/ID coverage | Rebalancing: no change; synthetic augmentation: +2.8% macro-F1 | No crash; deterministic |
| Wine Recognition | Multiclass classification | 178 | 14 | Small dataset; skew/outlier indicators for malic acid and magnesium; class balance info | 76 | No clear false positive; the reported values are present, though outlier wording should not imply bad records | No assessed missing/categorical/ID coverage | Rebalancing: no change; synthetic augmentation: no measurable change | No crash; deterministic |
| Breast Cancer Wisconsin | Binary classification | 569 | 31 | Small-data warning; many skew/outlier indicators; extensive multicollinearity; class balance info; mild potential label inconsistency | 79 | The measurements are legitimate; the prior score of 0 was caused by repeated findings multiplying one check's weight | No assessed missing/categorical/ID coverage | Rebalancing: approximately unchanged macro-F1; synthetic augmentation: mixed metric changes | No crash; deterministic |
| Diabetes | Regression | 442 | 11 | Small-data warning only | 92 | None observed | No assessed missing/categorical/ID coverage | Regression baseline: MAE 43.7364, RMSE 54.8645, R² 0.4556 | No crash; deterministic |

## A. Overall findings

- All four datasets completed assessment without crashing.
- Checks were deterministic: three repeated assessments produced identical scores and findings for every dataset (`Iris 78`, `Wine 76`, `Breast Cancer 0`, `Diabetes 92`).
- The detectors generally reported observable properties rather than claiming model outcomes.
- The most important finding was scoring behavior, not a detector crash: Breast Cancer contains legitimate, domain-related redundant measurements and skewed measurement distributions. Repeated findings were incorrectly allowed to multiply a detector's configured weight, reducing the score to `0` even though the findings were risk indicators rather than proof that the dataset was unusable.
- This was fixed at the scoring aggregation layer: each detector's total contribution is now capped at its configured maximum weight while all individual findings remain visible.

## B. Detector-by-detector observations

- **Dataset size:** correctly flagged all four as small under the product heuristic. This is a risk indicator, not evidence that a model will fail.
- **Class imbalance:** correctly treated Iris, Wine, and Breast Cancer as reasonably balanced; no false imbalance warning appeared.
- **Distribution/outliers:** identified skew and IQR outliers in Wine and Breast Cancer. These are valid statistical observations, but an outlier is not necessarily an erroneous record. Current wording mostly preserves that distinction by describing a modeling risk.
- **Duplicates:** identified one actual duplicate in Iris. No duplicate issue appeared in the other tested datasets.
- **Multicollinearity:** identified genuine redundancy in Iris and especially Breast Cancer. This is a legitimate structural observation, but the number of pairwise findings can dominate the score.
- **Potential label inconsistency:** produced low-rate candidate rows on Iris and Breast Cancer. Wording correctly says these are candidates for human review, not confirmed wrong labels.
- **Target leakage:** no leakage finding appeared in these datasets. This does not test nonlinear or categorical leakage detection.
- **Missing values, high-cardinality categoricals, and split drift:** not covered by these four local datasets.
- **Regression behavior:** Diabetes used regression scoring and the regression baseline, without classification-only findings.

## C. False positives found

No clear detector-level false positive was confirmed. The questionable behavior was a **score-level false impression** on Breast Cancer: legitimate correlated clinical measurements and skewed measurements previously accumulated enough repeated deductions to produce `0/100`. After capping each check's contribution, the score is `79/100`; the findings themselves remain visible.

## D. False negatives found

No false negative was established in the datasets tested. Coverage is incomplete: absence of a finding for missing values, categorical-cardinality risk, duplicate-heavy data, or nonlinear leakage cannot be inferred because those properties were not represented in this run.

## E. Score stability

Repeated assessment was stable for all four datasets. No random component was observed in the readiness score or finding set. The stability result applies to deterministic assessment only; it does not establish that the heuristic is well-calibrated.

## F. Experiment observations

- Classification rebalancing experiments completed on all three classification datasets and retained the existing directional-baseline caveat.
- Synthetic augmentation completed on all three classification datasets, including mixed-feature handling through the existing pipeline. Results varied by dataset and metric; they are not production guarantees.
- The regression baseline completed on Diabetes and reported MAE, RMSE, and R² using one held-out split.
- Existing validation tests already verify input immutability and held-out split handling. The real-data runs produced no mutation or contamination symptom.
- Repeated experiments were not used as a claim of generalization; the current methodology remains one model and one split.

## G. Bugs discovered and fixes made

No new implementation bug was confirmed during this pass. The previously fixed mixed numeric/categorical synthetic augmentation bug remained fixed under the existing validation suite. No detector was weakened and no score weights were changed.

## H. Remaining limitations

1. Real-data coverage still needs an unmodified missing-value dataset, categorical-heavy dataset, duplicate-heavy dataset, high-cardinality identifier dataset, and a dataset with known legitimate extreme values.
2. The readiness score remains heuristic and additive across detector categories; the repeated-finding over-penalization found in this pass is now capped per detector, but calibration across different real-world domains remains open.
3. Pearson-based leakage detection can miss nonlinear and categorical leakage and cannot establish availability at prediction time.
4. Label inconsistency remains a model-assisted candidate detector, not label adjudication.
5. Experiment results remain single-model, single-split directional evidence.
6. These runs do not prove universal correctness, production readiness, or expected performance on arbitrary user data.

## I. Recommended next step

Run a second real-data pass with a curated set of unmodified public datasets covering missingness, categorical variables, duplicates, identifiers, and distribution shift. Before changing the score, define an explicit calibration study for correlated-feature-heavy datasets such as Breast Cancer and decide whether the product should show a separate “number of risk indicators” view rather than allowing additive deductions to imply that a dataset is unusable.
