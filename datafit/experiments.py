"""
DataFit — impact experiments.

This is the "does it actually matter?" layer. Every function here is
explicitly opt-in: the user (or the calling UI) chooses to run an
experiment to see evidence of impact. Nothing here is called
automatically by assess_readiness(), and nothing here silently
replaces the user's original dataset — experiment results are returned
as a report; any modified data is clearly labeled "preview" and never
presented as a replacement for the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from datafit.remediation import find_best_remediation, clean_dataset
from datafit.synthetic import generate_synthetic_data, score_augmentation
from datafit.features import prepare_features


@dataclass
class ExperimentReport:
    experiment: str
    ran_successfully: bool
    evidence_summary: str
    metrics: list = field(default_factory=list)
    preview_data_available: bool = False
    caveat: str = ""


def run_regression_baseline_experiment(df: pd.DataFrame, target_column: str) -> ExperimentReport:
    """Run a minimal directional regression baseline on one held-out split.

    This is evidence about the disposable baseline and split only. It is not
    a prediction of the user's eventual model performance.
    """
    if target_column not in df.columns or not pd.api.types.is_numeric_dtype(df[target_column]):
        return ExperimentReport(
            experiment="regression_baseline",
            ran_successfully=False,
            evidence_summary="A numeric target column is required for the regression baseline.",
            caveat="No model was trained because the selected target is not numeric.",
        )

    usable = df.dropna(subset=[target_column]).copy()
    feature_columns = [c for c in usable.columns if c != target_column]
    X = prepare_features(usable, feature_columns)
    y = usable[target_column].astype(float)
    if len(usable) < 20 or X.shape[1] == 0 or not np.isfinite(y).all():
        return ExperimentReport(
            experiment="regression_baseline",
            ran_successfully=False,
            evidence_summary="Not enough usable numeric target and feature data for a baseline.",
            caveat="No model was trained because the data could not support this baseline safely.",
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = [
        {"metric": "mae", "value": round(float(mean_absolute_error(y_test, predictions)), 4)},
        {"metric": "rmse", "value": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4)},
        {"metric": "r2", "value": round(float(r2_score(y_test, predictions)), 4)},
    ]
    return ExperimentReport(
        experiment="regression_baseline",
        ran_successfully=True,
        evidence_summary="A disposable Random Forest regression baseline was evaluated on a held-out test split.",
        metrics=metrics,
        caveat="This is a directional baseline experiment using one model and one random split, not a guarantee of production performance.",
    )


def run_imbalance_experiment(df: pd.DataFrame, target_column: str) -> ExperimentReport:
    """Opt-in experiment: 'if I rebalanced this dataset's classes, would
    it actually help my model?' Tries multiple strategies honestly and
    reports whichever wins — or plainly reports that none helped.

    This does NOT alter the user's dataset. It trains disposable,
    temporary models purely to produce evidence.
    """
    verdict = find_best_remediation(df, target_column)

    return ExperimentReport(
        experiment="class_imbalance_rebalancing",
        ran_successfully=len(verdict.scores) > 0,
        evidence_summary=verdict.recommendation,
        metrics=[{"metric": s.metric, "before": s.before, "after": s.after, "change_pct": s.improvement_pct}
                  for s in verdict.scores],
        preview_data_available=verdict.helped,
        caveat="This experiment used a generic baseline model (Random Forest) to estimate impact. "
               "Your actual model choice may respond differently — treat this as directional evidence, "
               "not a guarantee.",
    )


def run_synthetic_data_experiment(df: pd.DataFrame, target_column: str,
                                   n_synthetic_rows: int | None = None) -> ExperimentReport:
    """Opt-in experiment: 'if I added synthetic data to this small
    dataset, would it actually help?' Generates synthetic rows via GMM,
    measures the honest before/after effect, and reports it — again,
    without touching the user's original dataset.
    """
    synthetic_df, gen_result = generate_synthetic_data(df, target_column=target_column,
                                                         n_synthetic_rows=n_synthetic_rows)

    if len(synthetic_df) == 0:
        return ExperimentReport(
            experiment="synthetic_data_augmentation",
            ran_successfully=False,
            evidence_summary="Could not generate synthetic data — likely too few real rows in one or "
                              "more classes to model a distribution from.",
            caveat="; ".join(gen_result.warnings) if gen_result.warnings else "",
        )

    scores = score_augmentation(df, synthetic_df, target_column)
    helped = any(s.after > s.before for s in scores)

    summary = (
        f"Generated {gen_result.synthetic_rows_added} synthetic rows using Gaussian Mixture Models "
        f"(one model fit per class, so class balance wasn't changed by this augmentation itself). "
    )
    if helped:
        best = max(scores, key=lambda s: s.improvement_pct)
        summary += f"Adding this synthetic data improved {best.metric} by {best.improvement_pct:+.1f}%."
    else:
        summary += "Adding this synthetic data did not measurably improve model performance on held-out data."

    return ExperimentReport(
        experiment="synthetic_data_augmentation",
        ran_successfully=True,
        evidence_summary=summary,
        metrics=[{"metric": s.metric, "before": s.before, "after": s.after, "change_pct": s.improvement_pct}
                  for s in scores],
        preview_data_available=True,
        caveat="Synthetic rows are statistically modeled approximations, not real observations. "
               "They should supplement, not replace, efforts to gather genuine additional data.",
    )


def preview_cleanup(df: pd.DataFrame, target_column: str | None = None) -> ExperimentReport:
    """Opt-in preview: 'what WOULD clean_dataset() do to this data?'
    Runs the cleanup logic to see what actions it would take, without
    this function itself being framed as the default path — the actual
    cleaned dataframe is available for the user to explicitly request,
    but is never applied silently as part of assessment.
    """
    _, cleaning_result = clean_dataset(df, target_column=target_column)

    if not cleaning_result.actions:
        summary = "No cleanup actions were identified — the dataset didn't trigger any of the automatic-fix rules."
    else:
        action_lines = [f"{a.action}: {a.detail}" for a in cleaning_result.actions]
        summary = f"If applied, {len(cleaning_result.actions)} action(s) would be taken:\n" + "\n".join(
            f"  - {line}" for line in action_lines
        )

    return ExperimentReport(
        experiment="cleanup_preview",
        ran_successfully=True,
        evidence_summary=summary,
        preview_data_available=len(cleaning_result.actions) > 0,
        caveat="This is a preview only. The original dataset is unchanged unless you explicitly "
               "request the cleaned version.",
    )
