"""
DataFit — ML Readiness scoring.

Converts raw diagnostic issues into a task-specific readiness score
(0-100) plus a structured, human-readable report. This is the
centerpiece of the product: not "here are some stats," but "here's
whether this dataset is fit for the ML task you're trying to do, and
why."

Core principle (deliberate, not accidental): this module NEVER modifies
the input data. It assesses and explains. Anything that could change
the dataset (rebalancing, imputation, synthetic data) lives in
remediation.py / synthetic.py and is only ever invoked explicitly, as
an opt-in "run this experiment" action — never automatically here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from datafit.diagnostics import diagnose, DiagnosticReport, Issue


class Task(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    GENERIC = "generic"


# How much each check matters for each task, and how severity translates
# to a point deduction. Deliberately simple and inspectable — a user (or
# you, debugging it) should be able to see exactly why a score came out
# the way it did, not treat it as a black box.
#
# Weight = max points this check can cost if triggered at "critical".
# "warning" costs half that, "info" costs a small flat amount (visibility
# without being punitive).
CHECK_WEIGHTS = {
    Task.CLASSIFICATION: {
        "dataset_size": 15,
        "class_imbalance": 25,
        "missing_values": 10,
        "distribution": 5,
        "duplicates": 8,
        "low_variance": 5,
        "high_cardinality": 8,
        "multicollinearity": 5,
        "target_leakage": 30,
        "label_noise": 12,
        "split_drift": 15,
    },
    Task.REGRESSION: {
        "dataset_size": 15,
        "class_imbalance": 0,  # not applicable to regression targets
        "missing_values": 12,
        "distribution": 12,  # skew/outliers matter more for regression targets
        "duplicates": 8,
        "low_variance": 5,
        "high_cardinality": 8,
        "multicollinearity": 10,  # hurts linear regression coefficients specifically
        "target_leakage": 30,
        "label_noise": 0,  # label noise check is classification-specific (predict_proba based)
        "split_drift": 15,
    },
    Task.GENERIC: {
        # Balanced defaults when the user hasn't specified a task
        "dataset_size": 15, "class_imbalance": 15, "missing_values": 12, "distribution": 8,
        "duplicates": 8, "low_variance": 6, "high_cardinality": 8, "multicollinearity": 8,
        "target_leakage": 25, "label_noise": 8, "split_drift": 12,
    },
}
CHECK_WEIGHTS[Task.CLUSTERING] = CHECK_WEIGHTS[Task.GENERIC]

SEVERITY_MULTIPLIER = {"critical": 1.0, "warning": 0.5, "info": 0.1}


@dataclass
class ReadinessFinding:
    check: str
    severity: str
    message: str
    points_deducted: float
    detail: dict = field(default_factory=dict)


@dataclass
class ReadinessReport:
    task: str
    score: Optional[int]
    n_rows: int
    n_cols: int
    findings: list[ReadinessFinding] = field(default_factory=list)
    recommended_next_steps: list[str] = field(default_factory=list)
    analyzable: bool = True
    analyzability_reason: str = ""
    scope_disclaimer: str = (
        "This task-specific heuristic readiness/risk score and these findings describe risk factors observed directly in your dataset. "
        "They are not a measurement of model bias, variance, overfitting, or accuracy — those "
        "depend on your model architecture, training procedure, and evaluation, none of which "
        "DataFit observes here. Confirming actual model impact requires training and evaluating "
        "a model; DataFit's opt-in experiments can help gather that evidence directly."
    )

    def status_label(self) -> str:
        if not self.analyzable:
            return "Cannot be assessed"
        if self.score >= 85:
            return "Ready for training"
        elif self.score >= 60:
            return "Usable, with caveats"
        else:
            return "Needs investigation before training"

    def to_markdown(self) -> str:
        if not self.analyzable:
            return (
                f"## ML READINESS: Cannot be assessed\n"
                f"**{self.analyzability_reason}**\n"
            )

        severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🟢"}
        lines = [
            f"## ML READINESS: {self.score}/100",
            f"**{self.status_label()}** — task: {self.task}",
            "",
        ]
        # group by check, show worst severity per check for a clean summary
        seen_checks = set()
        for f in sorted(self.findings, key=lambda x: -SEVERITY_MULTIPLIER[x.severity]):
            if f.check in seen_checks:
                continue
            seen_checks.add(f.check)
            icon = severity_icon.get(f.severity, "•")
            lines.append(f"{icon} **{f.check.replace('_', ' ').title()}**")
            lines.append(f"   {f.message}")
            lines.append("")

        if self.recommended_next_steps:
            lines.append("**Recommended next steps:**")
            for step in self.recommended_next_steps:
                lines.append(f"- {step}")

        lines.append("")
        lines.append(f"_{self.scope_disclaimer}_")

        return "\n".join(lines)


# Plain-language "why it matters" templates per check — this is the
# "data problem -> ML impact" translation that's the whole point of
# the product, not just restating the diagnostic message.
#
# Important: these describe KNOWN RISK FACTORS associated with each
# issue, based on established ML principles — not guaranteed outcomes.
# Whether a specific model actually ends up biased, overfit, or
# underperforming depends on the model architecture, training
# procedure, and evaluation methodology, none of which DataFit
# observes from the dataset alone. Confirming actual impact requires
# training and evaluating a model — see the opt-in experiments in
# experiments.py for a way to gather that evidence directly, rather
# than inferring it from the data alone.
IMPACT_EXPLANATIONS = {
    "class_imbalance": "A known risk factor for a model favoring the majority class, which can produce "
                        "misleadingly high accuracy while performing poorly on the minority class — but "
                        "the actual effect depends on the model and training setup, and is best confirmed "
                        "by checking minority-class recall after training, not assumed from this ratio alone.",
    "target_leakage": "If this feature is genuinely derived from the target, or unavailable at real "
                       "prediction time, it's a known cause of test performance that looks strong but "
                       "doesn't hold up in production. Worth verifying the feature's real availability "
                       "before training, not just its correlation here.",
    "dataset_size": "Small datasets are a known risk factor for unstable evaluation results — performance "
                     "estimates can vary significantly depending on how the data happens to be split. "
                     "Whether this specific dataset is 'enough' depends on the task complexity and model, "
                     "and is best assessed by checking variance across multiple train/test splits.",
    "missing_values": "Depending on how missingness is handled, this can bias what the model learns or "
                       "effectively shrink your usable training data — worth checking whether the "
                       "missingness is random or systematic (e.g. correlated with the target) before "
                       "choosing how to handle it.",
    "duplicates": "Duplicate rows that end up split across train and test can inflate apparent evaluation "
                  "performance in a way that doesn't hold up on genuinely new data — a known evaluation "
                  "pitfall, not a guaranteed one if duplicates are removed before splitting.",
    "low_variance": "Columns with no meaningful variance carry no signal for a model to learn from — this "
                     "is a direct observation about the data, not an inference about model behavior.",
    "high_cardinality": "Identifier-like columns encoded naively are a known overfitting risk, since a "
                         "model could effectively memorize individual rows rather than learn a "
                         "generalizable pattern — whether this actually happens depends on encoding "
                         "choices and model type.",
    "multicollinearity": "Redundant features are a known risk factor for destabilizing coefficient-based "
                          "models and confusing feature-importance interpretation in tree-based models — "
                          "the practical impact varies by model choice.",
    "label_noise": "Potential label inconsistency can limit achievable model performance, but this "
                    "heuristic only identifies rows whose labels disagree with high-confidence baseline "
                    "predictions. Review candidates manually; it does not confirm that any label is wrong.",
    "split_drift": "If train and test distributions genuinely differ, test performance is a known risk "
                    "for not reflecting how the model performs on real, new data — worth investigating "
                    "why the split differs before trusting the evaluation numbers.",
    "distribution": "Extreme outliers or skew are a known risk factor for dominating distance-based or "
                     "linear models unless explicitly handled (e.g. via transforms or robust scaling) — "
                     "tree-based models are typically far less sensitive to this.",
}


def assess_readiness(df: pd.DataFrame, task: str = "generic", target_column: str | None = None,
                      split_column: str | None = None) -> ReadinessReport:
    """The main entry point: assess a dataset's fitness for a specific ML
    task. Does NOT modify df in any way — pure read-only analysis.

    Args:
        df: dataset to assess (never modified).
        task: one of "classification", "regression", "clustering", "generic".
        target_column: name of the target/label column, if applicable.
        split_column: name of an explicit train/test split marker, if any.

    Returns:
        A ReadinessReport with a 0-100 score, findings, and next steps.
    """
    try:
        task_enum = Task(task.lower())
    except ValueError:
        task_enum = Task.GENERIC

    weights = CHECK_WEIGHTS[task_enum]

    diag_report = diagnose(
        df, target_column=target_column, split_column=split_column,
        run_label_noise_check=(task_enum == Task.CLASSIFICATION),
    )

    if not diag_report.analyzable:
        return ReadinessReport(
            task=task_enum.value, score=None, n_rows=diag_report.n_rows, n_cols=diag_report.n_cols,
            analyzable=False, analyzability_reason=diag_report.analyzability_reason,
        )

    readiness = ReadinessReport(task=task_enum.value, score=100, n_rows=len(df), n_cols=len(df.columns))

    # A check's configured weight is its maximum contribution. Checks that
    # emit one finding per column or pair remain visible without multiplying
    # the same check's full weight repeatedly.
    deductions_by_check: dict[str, float] = {}
    for issue in diag_report.issues:
        weight = weights.get(issue.check, 5)  # small default weight for unmapped checks
        if weight == 0:
            continue  # not applicable to this task (e.g. class_imbalance for regression)
        deduction = weight * SEVERITY_MULTIPLIER[issue.severity]
        deductions_by_check[issue.check] = min(
            weight,
            deductions_by_check.get(issue.check, 0.0) + deduction,
        )

        impact = IMPACT_EXPLANATIONS.get(issue.check, "")
        # Only attach the "why it matters" explanation to genuine problems —
        # attaching it to a positive "info" finding (e.g. "size looks fine")
        # reads as self-contradictory, since the explanation describes the
        # downside of the issue occurring, not confirming it's fine.
        show_impact = impact and issue.severity in ("critical", "warning")
        message = issue.message + (f" **Why it matters:** {impact}" if show_impact else "")

        readiness.findings.append(ReadinessFinding(
            check=issue.check, severity=issue.severity, message=message,
            points_deducted=round(deduction, 1), detail=issue.detail,
        ))

    readiness.score = max(0, round(100 - sum(deductions_by_check.values())))
    readiness.recommended_next_steps = _build_recommendations(readiness.findings, task_enum)

    return readiness


def _build_recommendations(findings: list[ReadinessFinding], task: Task) -> list[str]:
    """Translates findings into investigate-first recommendations —
    deliberately phrased as things to look into, not auto-fixes, per
    the core principle that DataFit doesn't decide what's wrong with
    the user's data on their behalf."""
    steps = []
    checks_present = {f.check for f in findings if f.severity in ("critical", "warning")}

    recommendation_templates = {
        "class_imbalance": "Investigate class representation — consider class weighting, resampling "
                            "experiments, or gathering more minority-class examples.",
        "target_leakage": "Review the flagged column(s) to confirm they would genuinely be available "
                           "at real prediction time, not just in this historical dataset.",
        "dataset_size": "Consider whether more real data can be gathered; if not, a synthetic data "
                         "experiment can show whether augmentation measurably helps for your model.",
        "missing_values": "Review whether missingness is random or systematic (e.g. correlated with "
                           "the target) before choosing an imputation strategy.",
        "duplicates": "Remove duplicate rows before splitting into train/test to avoid inflated "
                      "evaluation results.",
        "low_variance": "Consider dropping these columns — verify they aren't supposed to vary before removing.",
        "high_cardinality": "Avoid using these columns as raw categorical features; consider a different "
                             "encoding or excluding them.",
        "multicollinearity": "Consider dropping one column from each highly correlated pair, especially "
                              "if using a linear or coefficient-based model.",
        "label_noise": "Manually review the flagged high-confidence prediction disagreements for potential label inconsistency.",
        "split_drift": "Investigate why the splits differ — consider re-splitting the data, especially "
                        "if the split was based on time or another non-random factor.",
    }

    for check in checks_present:
        if check in recommendation_templates:
            steps.append(recommendation_templates[check])

    if not steps:
        steps.append("No significant issues detected — proceed with standard train/test validation practices.")

    return steps
