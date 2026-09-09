"""
DataFit — core diagnostic engine.

Analyzes a pandas DataFrame for issues that commonly hurt ML model
performance: class imbalance, insufficient dataset size, missing
values, and distribution problems. Produces a structured report
with plain-language explanations, not just raw numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Issue:
    check: str
    severity: str  # "info" | "warning" | "critical"
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    n_rows: int
    n_cols: int
    target_column: Optional[str]
    issues: list[Issue] = field(default_factory=list)
    analyzable: bool = True
    analyzability_reason: str = ""

    def add(self, check: str, severity: str, message: str, **detail) -> None:
        self.issues.append(Issue(check, severity, message, detail))

    def critical_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "critical"]

    def summary(self) -> str:
        if not self.issues:
            return "No issues detected. Dataset looks healthy for modeling."
        lines = [f"Dataset: {self.n_rows} rows x {self.n_cols} columns"]
        for issue in sorted(self.issues, key=lambda i: -SEVERITY_ORDER[i.severity]):
            tag = issue.severity.upper()
            lines.append(f"[{tag}] {issue.check}: {issue.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "target_column": self.target_column,
            "issues": [
                {"check": i.check, "severity": i.severity, "message": i.message, "detail": i.detail}
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_dataset_size(df: pd.DataFrame, report: DiagnosticReport, min_rows_rule_of_thumb: int = 1000) -> None:
    """Flags datasets that are likely too small for stable model training.

    This is a heuristic, not a hard law — deep learning wants far more,
    simple linear models can work with less. We surface it as a
    warning with context, not a hard rule.
    """
    n = len(df)
    if n < 200:
        report.add(
            "dataset_size", "critical",
            f"Only {n} rows. Small sample sizes like this are a well-established risk factor for "
            f"unstable evaluation results and poor generalization — but whether a model trained on "
            f"this specific data actually generalizes can only be confirmed by training and "
            f"evaluating it, not from the row count alone.",
            n_rows=n,
        )
    elif n < min_rows_rule_of_thumb:
        report.add(
            "dataset_size", "warning",
            f"{n} rows is on the small side. Consider synthetic data augmentation or gathering more "
            f"real examples, especially if you have many features or classes.",
            n_rows=n,
        )
    else:
        report.add("dataset_size", "info", f"{n} rows — reasonable starting size.", n_rows=n)


def check_class_imbalance(df: pd.DataFrame, target_column: str, report: DiagnosticReport) -> None:
    """Flags class imbalance in a classification target column.

    Guards against being run on a continuous (regression-style) target:
    computing value_counts on a near-unique numeric column is both
    meaningless (every "class" has ~1 member) and wasteful for large
    datasets, so we detect that case and say so explicitly instead.
    """
    target = df[target_column].dropna()
    if pd.api.types.is_numeric_dtype(target) and len(target) > 0:
        uniqueness_ratio = target.nunique() / len(target)
        if target.nunique() > 20 and uniqueness_ratio > 0.2:
            report.add(
                "class_imbalance", "info",
                f"Target column '{target_column}' looks continuous ({target.nunique()} unique values, "
                f"{uniqueness_ratio*100:.0f}% of rows) rather than categorical — class-imbalance analysis "
                f"doesn't apply here. If this is a classification target, check it wasn't accidentally "
                f"loaded as a continuous value.",
                n_unique=int(target.nunique()), uniqueness_ratio=round(uniqueness_ratio, 3),
            )
            return

    counts = df[target_column].value_counts(dropna=True)
    if len(counts) < 2:
        report.add(
            "class_imbalance", "critical",
            f"Target column '{target_column}' has only one class present — a model cannot learn "
            f"to distinguish classes that don't exist in the data.",
            classes=counts.to_dict(),
        )
        return

    majority = counts.iloc[0]
    minority = counts.iloc[-1]
    minority_pct = minority / counts.sum() * 100
    ratio = majority / minority if minority > 0 else float("inf")

    detail = {
        "class_counts": counts.to_dict(),
        "minority_pct": round(minority_pct, 2),
        "imbalance_ratio": round(ratio, 1) if ratio != float("inf") else None,
    }

    if minority_pct < 2:
        report.add(
            "class_imbalance", "critical",
            f"Severe class imbalance — minority class is only {minority_pct:.1f}% of the data "
            f"(ratio {ratio:.0f}:1). This is a strong risk factor for a model learning to predict "
            f"only the majority class while still scoring high on accuracy — a known failure pattern "
            f"with this level of imbalance, though the actual outcome depends on the model and "
            f"training setup, and can only be confirmed by evaluating minority-class performance "
            f"directly (e.g. recall, F1) after training.",
            **detail,
        )
    elif minority_pct < 10:
        report.add(
            "class_imbalance", "warning",
            f"Class imbalance detected — minority class is {minority_pct:.1f}% of the data "
            f"(ratio {ratio:.0f}:1). This is a known risk factor for models under-performing on the "
            f"minority class specifically — worth checking minority-class recall after training, "
            f"rather than relying on overall accuracy alone.",
            **detail,
        )
    else:
        report.add(
            "class_imbalance", "info",
            f"Class balance looks reasonable (minority class {minority_pct:.1f}% of data).",
            **detail,
        )


def check_missing_values(df: pd.DataFrame, report: DiagnosticReport, warn_threshold: float = 0.05,
                          critical_threshold: float = 0.3) -> None:
    """Flags columns with concerning amounts of missing data."""
    missing_pct = df.isna().mean()
    for col, pct in missing_pct.items():
        if pct == 0:
            continue
        if pct >= critical_threshold:
            report.add(
                "missing_values", "critical",
                f"Column '{col}' is {pct*100:.1f}% missing. Imputation at this level is risky — "
                f"consider whether this column is usable at all.",
                column=col, missing_pct=round(pct * 100, 2),
            )
        elif pct >= warn_threshold:
            report.add(
                "missing_values", "warning",
                f"Column '{col}' has {pct*100:.1f}% missing values. Imputation is reasonable here, "
                f"but verify the missingness isn't systematic (e.g. correlated with the target).",
                column=col, missing_pct=round(pct * 100, 2),
            )


def check_distribution_outliers(df: pd.DataFrame, report: DiagnosticReport, z_threshold: float = 4.0,
                                 exclude_columns: tuple = ()) -> None:
    """Flags numeric columns with extreme skew or heavy outlier presence.

    Methodology note: z-score outlier detection assumes roughly-normal
    data — on a skewed distribution it systematically over- or
    under-flags points, since the mean/std it's built on are themselves
    distorted by the skew. So we check skew first and switch to a more
    robust IQR-based method for outlier counting on skewed columns,
    rather than reporting a z-score outlier count that our own skew
    check would call into question.
    """
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_columns]
    for col in numeric_cols:
        series = df[col].dropna()
        # Low-cardinality numeric fields are often binary/ordinal encodings
        # (for example, 0/1 flags). Treating their codes as continuous values
        # makes skew and IQR outlier language misleading.
        if len(series) < 20 or series.std() == 0 or series.nunique() <= 5:
            continue

        skew = stats.skew(series)
        is_skewed = abs(skew) > 1  # heuristic threshold: |skew| > 1 is commonly considered "substantial"

        if is_skewed:
            # IQR method: robust to skew, doesn't assume any particular distribution shape.
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_pct = ((series < lower) | (series > upper)).mean() * 100
            else:
                outlier_pct = 0.0
            method_note = "IQR method (1.5×IQR beyond Q1/Q3) — used here instead of z-score because this column is skewed enough that z-score's normality assumption would distort the count"
        else:
            z_scores = np.abs(stats.zscore(series))
            outlier_pct = (z_scores > z_threshold).mean() * 100
            method_note = f"z-score method (|z| > {z_threshold})"

        if outlier_pct > 1:
            report.add(
                "distribution", "warning",
                f"Column '{col}' has {outlier_pct:.1f}% extreme outliers, detected via {method_note}. "
                f"These can distort model training, especially for distance-based or linear models.",
                column=col, outlier_pct=round(outlier_pct, 2), method=method_note,
            )
        if is_skewed:
            report.add(
                "distribution", "info",
                f"Column '{col}' is heavily skewed (skew={skew:.2f}, heuristic threshold |skew|>1). "
                f"A log or power transform may help models that assume roughly normal inputs.",
                column=col, skew=round(float(skew), 2),
            )


def check_duplicate_rows(df: pd.DataFrame, report: DiagnosticReport) -> None:
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        pct = dup_count / len(df) * 100
        severity = "warning" if pct > 1 else "info"
        report.add(
            "duplicates", severity,
            f"{dup_count} duplicate rows found ({pct:.1f}% of data). Duplicates can inflate "
            f"apparent model performance during evaluation if not removed before splitting.",
            duplicate_count=int(dup_count), duplicate_pct=round(pct, 2),
        )


def check_constant_low_variance_features(df: pd.DataFrame, report: DiagnosticReport,
                                          exclude_columns: tuple = (), variance_threshold: float = 1e-8) -> None:
    """Flags columns that carry no (or almost no) information."""
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_columns]
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        n_unique = series.nunique()
        if n_unique == 1:
            report.add(
                "low_variance", "critical",
                f"Column '{col}' has only one unique value — it carries zero information and "
                f"should be dropped before training.",
                column=col, n_unique=1,
            )
        elif series.var() < variance_threshold:
            report.add(
                "low_variance", "warning",
                f"Column '{col}' has near-zero variance. It likely contributes little to the model "
                f"and may be worth dropping.",
                column=col, variance=float(series.var()),
            )

    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in exclude_columns]
    for col in cat_cols:
        series = df[col].dropna()
        if len(series) > 0 and series.nunique() == 1:
            report.add(
                "low_variance", "critical",
                f"Column '{col}' has only one unique category — it carries zero information.",
                column=col, n_unique=1,
            )


def check_high_cardinality_categoricals(df: pd.DataFrame, report: DiagnosticReport,
                                         exclude_columns: tuple = (), cardinality_ratio_threshold: float = 0.9) -> None:
    """Flags categorical columns that look like identifiers (near-unique per row)."""
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in exclude_columns]
    n = len(df)
    if n == 0:
        return
    for col in cat_cols:
        n_unique = df[col].nunique(dropna=True)
        ratio = n_unique / n
        if ratio >= cardinality_ratio_threshold and n_unique > 20:
            report.add(
                "high_cardinality", "warning",
                f"Column '{col}' has {n_unique} unique values across {n} rows ({ratio*100:.0f}% unique) — "
                f"this looks like an identifier rather than a real feature. Using it directly (e.g. "
                f"one-hot encoded) is a known overfitting risk, since a model could essentially "
                f"memorize individual rows rather than learn a generalizable pattern.",
                column=col, n_unique=int(n_unique), uniqueness_ratio=round(ratio, 3),
            )


def check_multicollinearity(df: pd.DataFrame, report: DiagnosticReport, exclude_columns: tuple = (),
                             corr_threshold: float = 0.95) -> None:
    """Flags pairs of numeric features that are highly correlated with each other.

    Methodology note: this checks pairwise correlation only. A feature
    can be a near-perfect linear combination of THREE OR MORE other
    features (true multicollinearity) without any single pair showing
    high correlation — that requires VIF (variance inflation factor)
    analysis, which this does not do. Treat this as "no obvious pairwise
    redundancy found," not a full multicollinearity clearance.
    """
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_columns]
    # Zero-variance columns produce divide-by-zero in correlation and are
    # already reported separately by check_constant_low_variance_features —
    # excluding them here avoids NaN/RuntimeWarning noise, not a real signal loss.
    numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    if len(numeric_cols) < 2:
        return
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = df[numeric_cols].corr().abs()
    seen = set()
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            if pd.isna(corr.loc[col_a, col_b]):
                continue
            if corr.loc[col_a, col_b] >= corr_threshold and (col_a, col_b) not in seen:
                seen.add((col_a, col_b))
                report.add(
                    "multicollinearity", "warning",
                    f"Columns '{col_a}' and '{col_b}' are highly correlated (r={corr.loc[col_a, col_b]:.2f}). "
                    f"They carry largely redundant information — consider dropping one.",
                    column_a=col_a, column_b=col_b, correlation=round(float(corr.loc[col_a, col_b]), 3),
                )


def check_target_leakage(df: pd.DataFrame, target_column: str, report: DiagnosticReport,
                          corr_threshold: float = 0.98) -> None:
    """Flags features suspiciously predictive of the target on their own —
    a common sign of data leakage.

    Methodology note: this uses Pearson correlation, which only catches
    LINEAR relationships. A feature that's a perfect non-linear function
    of the target (e.g. target squared, or a categorical recoding) can
    leak just as badly and this check will miss it. This is a real
    limitation, not a hypothetical one — treat a clean result here as
    "no linear leakage found," not "no leakage."
    """
    if target_column not in df.columns:
        return
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    target = df[target_column]
    if not pd.api.types.is_numeric_dtype(target):
        target = target.astype("category").cat.codes

    if target.nunique(dropna=True) <= 1:
        return  # zero-variance target: correlation is undefined, and class_imbalance already flags this case

    for col in numeric_cols:
        series = df[col]
        if series.nunique() <= 1 or series.isna().all():
            continue
        try:
            with np.errstate(invalid='ignore', divide='ignore'):
                corr = series.corr(target)
        except Exception:
            continue
        if pd.notna(corr) and abs(corr) >= corr_threshold:
            report.add(
                "target_leakage", "warning",
                f"Column '{col}' is a potential leakage indicator because it is highly correlated with "
                f"the target (r={corr:.3f}, Pearson correlation only). Correlation alone cannot prove "
                f"leakage: the relationship may be legitimately predictive, and nonlinear leakage may "
                f"be missed. Verify whether this feature would actually be available at prediction time.",
                column=col, correlation=round(float(corr), 3), method="pearson_linear_only",
            )


def check_label_noise(df: pd.DataFrame, target_column: str, report: DiagnosticReport,
                       contamination: float = 0.05) -> None:
    """Heuristic check for potentially mislabeled rows using cross-validated
    model disagreement. Doesn't prove mislabeling, but surfaces candidates
    worth a human look."""
    if target_column not in df.columns:
        return
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    if len(numeric_cols) < 1:
        return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import LabelEncoder

    X = df[numeric_cols].fillna(df[numeric_cols].median())
    y_raw = df[target_column].dropna()
    if y_raw.nunique() < 2 or len(y_raw) < 50:
        return

    X = X.loc[y_raw.index]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    try:
        cv = min(5, y_raw.value_counts().min())
        if cv < 2:
            return
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        predicted = proba.argmax(axis=1)
        confidence = proba.max(axis=1)

        disagree_mask = (predicted != y) & (confidence > 0.85)
        n_flagged = disagree_mask.sum()
        pct = n_flagged / len(y) * 100

        if pct > contamination * 100:
            report.add(
                "label_noise", "warning",
                f"{n_flagged} rows ({pct:.1f}%) show potential label inconsistency: a baseline model "
                f"confidently disagreed with their labels during cross-validation. These are candidates "
                f"for human review, not proof that any label is wrong.",
                n_flagged=int(n_flagged), flagged_pct=round(pct, 2),
            )
        elif n_flagged > 0:
            report.add(
                "label_noise", "info",
                f"{n_flagged} rows ({pct:.1f}%) show mild potential label inconsistency — within a "
                f"normal disagreement range; this does not establish that any label is wrong.",
                n_flagged=int(n_flagged), flagged_pct=round(pct, 2),
            )
    except Exception:
        return


def check_train_test_style_split_drift(df: pd.DataFrame, report: DiagnosticReport,
                                        exclude_columns: tuple = (), split_col: Optional[str] = None) -> None:
    """If the dataset has an explicit split marker column, checks whether
    feature distributions differ meaningfully between splits."""
    if split_col is None or split_col not in df.columns:
        return
    values = df[split_col].dropna().unique()
    if len(values) != 2:
        return

    a_mask = df[split_col] == values[0]
    b_mask = df[split_col] == values[1]
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_columns]

    drifted = []
    for col in numeric_cols:
        a = df.loc[a_mask, col].dropna()
        b = df.loc[b_mask, col].dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        stat, p_value = stats.ks_2samp(a, b)
        if p_value < 0.01:
            drifted.append(col)

    if drifted:
        report.add(
            "split_drift", "warning",
            f"{len(drifted)} feature(s) have significantly different distributions between "
            f"'{values[0]}' and '{values[1]}' splits: {', '.join(drifted[:5])}"
            f"{'...' if len(drifted) > 5 else ''}. This can make test performance misleading.",
            drifted_columns=drifted,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

MIN_ANALYZABLE_ROWS = 5  # Below this, statistics are not meaningful enough to score honestly


def diagnose(df: pd.DataFrame, target_column: Optional[str] = None, split_column: Optional[str] = None,
             run_label_noise_check: bool = True) -> DiagnosticReport:
    """Run the full diagnostic suite on a DataFrame.

    Args:
        df: the dataset to analyze.
        target_column: name of the classification target column, if any.
            Enables class imbalance, target leakage, and label noise checks.
        split_column: name of an explicit train/test split marker column, if any.
            Enables the split-drift check.
        run_label_noise_check: label noise detection trains a cross-validated
            model and is the slowest check — disable for quick runs on large data.

    Returns:
        A DiagnosticReport with all detected issues. If the input cannot be
        reliably analyzed (e.g. zero rows, too few rows for meaningful
        statistics), `report.analyzable` is False and `report.issues` will
        be empty — callers must check this rather than assume a score was
        computed. We do not fabricate a plausible-looking score for input
        we cannot honestly assess.
    """
    report = DiagnosticReport(n_rows=len(df), n_cols=len(df.columns), target_column=target_column)

    # --- Hard gates: cases where we genuinely cannot analyze, and must say so ---
    if len(df.columns) == 0:
        report.analyzable = False
        report.analyzability_reason = "Dataset has no columns — nothing to analyze."
        return report

    if len(df) == 0:
        report.analyzable = False
        report.analyzability_reason = "Dataset has zero rows — nothing to analyze."
        return report

    if len(df) < MIN_ANALYZABLE_ROWS:
        report.analyzable = False
        report.analyzability_reason = (
            f"Only {len(df)} row(s) provided. At least {MIN_ANALYZABLE_ROWS} rows are needed to compute "
            f"any statistic honestly — with fewer rows, a 'readiness score' would just be noise dressed "
            f"up as a number."
        )
        return report

    # --- Duplicate column names: fix defensively for analysis purposes, don't crash ---
    # This does not alter the user's data values — only how we internally
    # label columns during analysis, so we can inspect the data without
    # pandas' ambiguous-Series errors on duplicate labels.
    if df.columns.duplicated().any():
        dup_names = df.columns[df.columns.duplicated()].unique().tolist()
        new_cols = []
        seen = {}
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}__dup{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df = df.copy()
        df.columns = new_cols
        report.add(
            "duplicate_columns", "warning",
            f"Dataset had duplicate column name(s) ({', '.join(map(str, dup_names))}) — renamed "
            f"internally for analysis (e.g. 'col__dup1') since duplicate labels can't be reliably "
            f"distinguished. Consider renaming these columns in your source data.",
            duplicate_names=dup_names,
        )
        # target_column may have been a duplicated name itself; if so, we can't
        # know which copy was meant, so target-dependent checks are skipped below.
        if target_column in dup_names:
            report.add(
                "duplicate_columns", "critical",
                f"Target column '{target_column}' was duplicated — cannot reliably determine which "
                f"column was intended as the target. Please rename columns to be unique and retry.",
            )
            target_column = None

    exclude = tuple(c for c in (target_column, split_column) if c)

    # General checks — run regardless of target
    check_dataset_size(df, report)
    check_missing_values(df, report)
    check_distribution_outliers(df, report, exclude_columns=exclude)
    check_duplicate_rows(df, report)
    check_constant_low_variance_features(df, report, exclude_columns=exclude)
    check_high_cardinality_categoricals(df, report, exclude_columns=exclude)
    check_multicollinearity(df, report, exclude_columns=exclude)
    check_train_test_style_split_drift(df, report, exclude_columns=exclude, split_col=split_column)

    # Target-dependent checks
    if target_column is not None:
        if target_column not in df.columns:
            report.add("class_imbalance", "critical", f"Target column '{target_column}' not found in dataset.")
        else:
            check_class_imbalance(df, target_column, report)
            check_target_leakage(df, target_column, report)
            if run_label_noise_check:
                check_label_noise(df, target_column, report)

    return report
