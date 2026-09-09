"""
DataFit — remediation engine.

Two layers:
1. `clean_dataset()` — general-purpose fixes for the "obvious" issues
   (duplicates, low-variance columns, leaky columns, high-cardinality
   ID-like columns, missing values) that should just be resolved, not
   debated.
2. Class rebalancing — tries multiple strategies and picks whichever
   ACTUALLY improves held-out performance, rather than assuming any one
   method works. If none help, it says so honestly instead of forcing
   a "fix".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder

from datafit.diagnostics import diagnose
from datafit.features import prepare_features


@dataclass
class CleaningAction:
    action: str
    detail: str
    columns_affected: list = field(default_factory=list)


@dataclass
class CleaningResult:
    original_shape: tuple
    cleaned_shape: tuple
    actions: list[CleaningAction] = field(default_factory=list)


def clean_dataset(df: pd.DataFrame, target_column: str | None = None,
                   drop_low_variance: bool = True, drop_high_cardinality: bool = True,
                   drop_multicollinear: bool = True, drop_leaky: bool = True,
                   drop_duplicates: bool = True, impute_missing: bool = True) -> tuple[pd.DataFrame, CleaningResult]:
    """Applies safe, largely-uncontroversial fixes based on diagnostic findings.

    Each fix is driven by re-running diagnostics on the current state of
    the dataframe, so this is transparent and traceable — not a black box.
    """
    working_df = df.copy()
    result = CleaningResult(original_shape=df.shape, cleaned_shape=df.shape)

    # 1. Duplicates — always safe to drop
    if drop_duplicates:
        before = len(working_df)
        working_df = working_df.drop_duplicates().reset_index(drop=True)
        removed = before - len(working_df)
        if removed > 0:
            result.actions.append(CleaningAction(
                "drop_duplicates", f"Removed {removed} exact duplicate rows.", []
            ))

    exclude = (target_column,) if target_column else ()

    # 2. Low-variance / constant columns — safe to drop, carry no signal
    if drop_low_variance:
        report = diagnose(working_df, target_column=target_column, run_label_noise_check=False)
        cols_to_drop = [i.detail["column"] for i in report.issues if i.check == "low_variance"]
        cols_to_drop = [c for c in cols_to_drop if c not in exclude]
        if cols_to_drop:
            working_df = working_df.drop(columns=cols_to_drop)
            result.actions.append(CleaningAction(
                "drop_low_variance", f"Dropped {len(cols_to_drop)} column(s) with no meaningful variance.",
                cols_to_drop,
            ))

    # 3. High-cardinality ID-like columns — safe to drop as raw features
    if drop_high_cardinality:
        report = diagnose(working_df, target_column=target_column, run_label_noise_check=False)
        cols_to_drop = [i.detail["column"] for i in report.issues if i.check == "high_cardinality"]
        cols_to_drop = [c for c in cols_to_drop if c not in exclude]
        if cols_to_drop:
            working_df = working_df.drop(columns=cols_to_drop)
            result.actions.append(CleaningAction(
                "drop_high_cardinality", f"Dropped {len(cols_to_drop)} identifier-like column(s).",
                cols_to_drop,
            ))

    # 4. Target leakage — genuinely dangerous to leave in, drop by default
    if drop_leaky and target_column:
        report = diagnose(working_df, target_column=target_column, run_label_noise_check=False)
        cols_to_drop = [i.detail["column"] for i in report.issues if i.check == "target_leakage"]
        if cols_to_drop:
            working_df = working_df.drop(columns=cols_to_drop)
            result.actions.append(CleaningAction(
                "drop_target_leakage", f"Dropped {len(cols_to_drop)} column(s) suspected of leaking the target.",
                cols_to_drop,
            ))

    # 5. Multicollinearity — drop the second column in each highly correlated pair
    if drop_multicollinear:
        report = diagnose(working_df, target_column=target_column, run_label_noise_check=False)
        cols_to_drop = sorted({i.detail["column_b"] for i in report.issues if i.check == "multicollinearity"})
        cols_to_drop = [c for c in cols_to_drop if c in working_df.columns and c not in exclude]
        if cols_to_drop:
            working_df = working_df.drop(columns=cols_to_drop)
            result.actions.append(CleaningAction(
                "drop_multicollinear", f"Dropped {len(cols_to_drop)} redundant, highly correlated column(s).",
                cols_to_drop,
            ))

    # 6. Missing values — median impute numeric, mode impute categorical.
    # Simple and honest for v1; flagged as an assumption, not hidden.
    if impute_missing:
        numeric_cols = [c for c in working_df.select_dtypes(include=[np.number]).columns if c not in exclude]
        cat_cols = [c for c in working_df.select_dtypes(exclude=[np.number]).columns if c not in exclude]
        imputed_cols = []
        for col in numeric_cols:
            if working_df[col].isna().any():
                working_df[col] = working_df[col].fillna(working_df[col].median())
                imputed_cols.append(col)
        for col in cat_cols:
            if working_df[col].isna().any():
                mode = working_df[col].mode(dropna=True)
                fill_value = mode.iloc[0] if len(mode) else "missing"
                working_df[col] = working_df[col].fillna(fill_value)
                imputed_cols.append(col)
        if imputed_cols:
            result.actions.append(CleaningAction(
                "impute_missing", f"Median/mode-imputed {len(imputed_cols)} column(s) with missing values. "
                f"Simple imputation — for critical use cases, consider a more targeted strategy per column.",
                imputed_cols,
            ))

    result.cleaned_shape = working_df.shape
    return working_df, result


@dataclass
class RebalanceResult:
    original_shape: tuple
    resampled_shape: tuple
    method_used: str
    original_class_counts: dict
    resampled_class_counts: dict


def _try_resample(sampler, X, y):
    try:
        return sampler.fit_resample(X, y)
    except Exception:
        return None, None


def _resample_auto(X: pd.DataFrame, y: pd.Series, strategy: str = "auto"):
    """Core resampling logic operating on an already-prepared, fully
    numeric feature matrix. Shared by rebalance_classes (public, raw-df
    API) and the scoring functions (which need to resample an
    already-encoded training split without re-encoding, to keep column
    names consistent with the held-out test set).
    """
    minority_count = y.value_counts().min()
    candidates = []
    if strategy in ("auto", "smote"):
        k_neighbors = min(5, minority_count - 1)
        if k_neighbors >= 1:
            candidates.append(("SMOTE", SMOTE(k_neighbors=k_neighbors, random_state=42)))
    if strategy in ("auto", "adasyn"):
        n_neighbors = min(5, minority_count - 1)
        if n_neighbors >= 1:
            candidates.append(("ADASYN", ADASYN(n_neighbors=n_neighbors, random_state=42)))
    candidates.append(("RandomOverSampler", RandomOverSampler(random_state=42)))

    for name, sampler in candidates:
        X_res, y_res = _try_resample(sampler, X, y)
        if X_res is not None:
            return X_res, y_res, name
    return X, y, "none (no strategy succeeded)"


def rebalance_classes(df: pd.DataFrame, target_column: str, feature_columns: list[str] | None = None,
                       strategy: str = "auto") -> tuple[pd.DataFrame, RebalanceResult]:
    """Rebalances a dataset's target classes.

    strategy="auto" tries SMOTE first (best quality when it applies),
    falling back to ADASYN, then plain random oversampling for edge
    cases where SMOTE's nearest-neighbor requirement can't be met
    (e.g. very few minority samples). Pass an explicit strategy name
    to force one method.

    Categorical feature columns are one-hot encoded before resampling
    (via prepare_features) — the returned dataframe reflects those
    encoded column names, not the original categorical column names.
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]

    X = prepare_features(df, feature_columns)
    y = df[target_column]
    original_counts = y.value_counts().to_dict()

    X_res, y_res, method_used = _resample_auto(X, y, strategy)

    resampled_df = X_res.copy()
    resampled_df[target_column] = y_res.values if hasattr(y_res, "values") else y_res

    result = RebalanceResult(
        original_shape=df.shape,
        resampled_shape=resampled_df.shape,
        method_used=method_used,
        original_class_counts=original_counts,
        resampled_class_counts=pd.Series(y_res).value_counts().to_dict(),
    )
    return resampled_df, result


@dataclass
class BeforeAfterScore:
    metric: str
    before: float
    after: float
    improvement_pct: float


@dataclass
class RemediationVerdict:
    method_used: str
    scores: list  # list[BeforeAfterScore]
    helped: bool
    recommendation: str



def score_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame, target_column: str,
                        feature_columns: list[str] | None = None,
                        rebalance_strategy: str = "auto") -> list[BeforeAfterScore]:
    """Trains a quick baseline model on both the original and rebalanced
    training data, evaluated on the SAME held-out test set, to show a
    concrete, honest performance delta.

    `df_after` is accepted for API compatibility / potential future use
    but the actual "after" training data is produced by rebalancing the
    training split itself (not df_after directly) — this keeps the test
    set strictly held out and comparable, and avoids test-set leakage
    that could happen if df_after was built from the full dataset.
    """
    if feature_columns is None:
        feature_columns = [c for c in df_before.columns if c != target_column]

    X_before = prepare_features(df_before, feature_columns)
    y_before = df_before[target_column]

    le = LabelEncoder()
    y_before_enc = le.fit_transform(y_before)

    X_train, X_test, y_train_before, y_test = train_test_split(
        X_before, y_before_enc, test_size=0.25, random_state=42, stratify=y_before_enc
    )

    # Train "before" model on original (imbalanced) training split
    model_before = RandomForestClassifier(n_estimators=100, random_state=42)
    model_before.fit(X_train, y_train_before)
    preds_before = model_before.predict(X_test)

    # Train "after" model on the rebalanced training split. Resampling
    # happens directly on the already-prepared numeric matrix, so
    # column names stay consistent with X_test — no re-encoding, no
    # mismatch risk.
    X_train_after, y_train_after, _ = _resample_auto(X_train, pd.Series(y_train_before), rebalance_strategy)

    model_after = RandomForestClassifier(n_estimators=100, random_state=42)
    model_after.fit(X_train_after, y_train_after)
    preds_after = model_after.predict(X_test)

    results = []
    for metric_name, fn in [
        ("f1_macro", lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)),
        ("recall_macro", lambda yt, yp: recall_score(yt, yp, average="macro", zero_division=0)),
        ("precision_macro", lambda yt, yp: precision_score(yt, yp, average="macro", zero_division=0)),
    ]:
        before_val = fn(y_test, preds_before)
        after_val = fn(y_test, preds_after)
        improvement = ((after_val - before_val) / before_val * 100) if before_val > 0 else float("inf")
        results.append(BeforeAfterScore(metric_name, round(before_val, 4), round(after_val, 4), round(improvement, 1)))

    return results


def find_best_remediation(df: pd.DataFrame, target_column: str, feature_columns: list[str] | None = None,
                           primary_metric: str = "f1_macro") -> RemediationVerdict:
    """Tries multiple rebalancing strategies, scores each honestly on a
    held-out test set, and reports whichever genuinely wins on the
    primary metric. If nothing beats the original baseline, says so
    plainly rather than forcing a "fix" that doesn't help.
    """
    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]

    strategies_to_try = ["smote", "adasyn", "random"]
    results_by_strategy = {}

    for strategy in strategies_to_try:
        try:
            scores = score_before_after(df, df, target_column, feature_columns, rebalance_strategy=strategy)
            metric_lookup = {s.metric: s for s in scores}
            results_by_strategy[strategy] = metric_lookup
        except Exception:
            continue

    if not results_by_strategy:
        return RemediationVerdict(
            method_used="none", scores=[], helped=False,
            recommendation="No rebalancing strategy could be applied to this dataset (likely too few "
                            "minority samples). Consider gathering more real examples of the minority class.",
        )

    # Pick whichever strategy improves the primary metric the most
    best_strategy = max(
        results_by_strategy,
        key=lambda s: results_by_strategy[s][primary_metric].after if primary_metric in results_by_strategy[s] else -1,
    )
    best_scores = list(results_by_strategy[best_strategy].values())
    primary_score = results_by_strategy[best_strategy][primary_metric]
    helped = primary_score.after > primary_score.before

    if helped:
        recommendation = (
            f"{best_strategy.upper()} rebalancing improved {primary_metric} from "
            f"{primary_score.before:.3f} to {primary_score.after:.3f} "
            f"({primary_score.improvement_pct:+.1f}%). Recommended."
        )
    else:
        recommendation = (
            f"None of the tried rebalancing strategies improved {primary_metric} on held-out data "
            f"(best attempt, {best_strategy.upper()}, went from {primary_score.before:.3f} to "
            f"{primary_score.after:.3f}). Your model may already handle this imbalance reasonably well — "
            f"rebalancing is not recommended here. Consider class-weighting or gathering more real minority "
            f"examples instead."
        )

    return RemediationVerdict(
        method_used=best_strategy, scores=best_scores, helped=helped, recommendation=recommendation,
    )
