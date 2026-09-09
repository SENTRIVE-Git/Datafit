"""
DataFit — synthetic data generation.

For datasets that are too small to train on reliably (flagged by
`check_dataset_size`), generates additional synthetic rows that match
the real data's statistical properties per-class, using Gaussian
Mixture Models.

Why GMM instead of a deep generative model (GAN/VAE): it's fast, needs
no GPU, works well on small datasets (the exact case we're solving for
— deep generative models typically need MORE data than we have, which
would be backwards), and is easy to explain to a user: "we modeled the
distribution of each feature per class, then sampled new points from
it." Deferred to a later phase per the product plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder

from datafit.features import prepare_features


@dataclass
class SyntheticGenerationResult:
    original_rows: int
    synthetic_rows_added: int
    total_rows: int
    per_class_added: dict = field(default_factory=dict)
    n_components_used: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def generate_synthetic_data(df: pd.DataFrame, target_column: str | None = None,
                             feature_columns: list[str] | None = None,
                             n_synthetic_rows: int | None = None,
                             max_components: int = 5) -> tuple[pd.DataFrame, SyntheticGenerationResult]:
    """Generates synthetic rows matching the real data's statistical
    distribution, per class if a target column is given (so class
    balance isn't accidentally distorted by the augmentation itself).

    Args:
        df: source dataset (real data only).
        target_column: if given, fits a separate GMM per class so
            synthetic data respects per-class distributions.
        feature_columns: numeric columns to model. Defaults to all
            numeric columns except the target.
        n_synthetic_rows: total synthetic rows to generate. Defaults to
            doubling the dataset (a reasonable default for "too small").
        max_components: upper bound on GMM components — kept small since
            we're typically working with limited real data to fit from.

    Returns:
        (synthetic_df, result) — synthetic_df contains ONLY the new
        synthetic rows (concat with the original yourself if you want
        the combined dataset — keeping them separate makes it easy to
        audit what's real vs. synthetic).
    """
    if feature_columns is None:
        feature_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]

    if n_synthetic_rows is None:
        n_synthetic_rows = len(df)  # default: double the dataset

    result = SyntheticGenerationResult(original_rows=len(df), synthetic_rows_added=0, total_rows=len(df))
    categorical_columns = [
        c for c in df.columns
        if c != target_column and c not in feature_columns
        and not pd.api.types.is_numeric_dtype(df[c])
    ]
    rng = np.random.default_rng(42)

    synthetic_frames = []

    if target_column and target_column in df.columns:
        classes = df[target_column].unique()
        # split the requested synthetic rows proportionally to existing class sizes,
        # so we don't accidentally change the class balance ourselves
        class_counts = df[target_column].value_counts()
        class_proportions = class_counts / class_counts.sum()

        for cls in classes:
            cls_df = df[df[target_column] == cls]
            n_for_class = max(1, int(round(n_synthetic_rows * class_proportions[cls])))
            synth_cls, n_components = _fit_and_sample(cls_df[feature_columns], n_for_class, max_components, result)
            if synth_cls is not None:
                for col in categorical_columns:
                    values = cls_df[col].dropna().to_numpy()
                    synth_cls[col] = rng.choice(values, size=len(synth_cls)) if len(values) else None
                synth_cls[target_column] = cls
                synthetic_frames.append(synth_cls)
                result.per_class_added[str(cls)] = len(synth_cls)
                result.n_components_used[str(cls)] = n_components
    else:
        synth_all, n_components = _fit_and_sample(df[feature_columns], n_synthetic_rows, max_components, result)
        if synth_all is not None:
            for col in categorical_columns:
                values = df[col].dropna().to_numpy()
                synth_all[col] = rng.choice(values, size=len(synth_all)) if len(values) else None
            synthetic_frames.append(synth_all)
            result.n_components_used["all"] = n_components

    if synthetic_frames:
        synthetic_df = pd.concat(synthetic_frames, ignore_index=True)
    else:
        synthetic_df = pd.DataFrame(columns=list(feature_columns) + categorical_columns + ([target_column] if target_column else []))

    result.synthetic_rows_added = len(synthetic_df)
    result.total_rows = result.original_rows + result.synthetic_rows_added
    return synthetic_df, result


def _fit_and_sample(data: pd.DataFrame, n_samples: int, max_components: int,
                     result: SyntheticGenerationResult) -> tuple[pd.DataFrame | None, int]:
    """Fits a GMM to a numeric-only slice of data and samples n_samples
    new rows from it. Returns (None, 0) if there isn't enough data to
    fit anything meaningful."""
    clean_data = data.dropna()
    n_rows = len(clean_data)

    if n_rows < 5:
        result.warnings.append(
            f"Skipped a class/group with only {n_rows} real rows — too few to model a distribution from."
        )
        return None, 0

    # Don't fit more components than we have data to support (avoids
    # degenerate/overfit mixtures on small groups).
    n_components = max(1, min(max_components, n_rows // 5))

    gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type="full")
    gmm.fit(clean_data.values)

    samples, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(samples, columns=clean_data.columns)
    return synthetic_df, n_components


@dataclass
class AugmentationScore:
    metric: str
    before: float
    after: float
    improvement_pct: float


def score_augmentation(df_real: pd.DataFrame, df_augmented: pd.DataFrame, target_column: str,
                        feature_columns: list[str] | None = None) -> list[AugmentationScore]:
    """Same honesty principle as the rebalancing scorer: trains a model
    on the original data vs. the augmented (real + synthetic) data,
    evaluated on the SAME held-out real test set, so we measure the
    actual effect of adding synthetic data rather than assuming it helps.
    """
    if feature_columns is None:
        feature_columns = [c for c in df_real.columns if c != target_column]

    X_real = prepare_features(df_real, feature_columns)
    y_real = df_real[target_column]

    le = LabelEncoder()
    y_real_enc = le.fit_transform(y_real)

    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real_enc, test_size=0.25, random_state=42,
        stratify=y_real_enc if pd.Series(y_real_enc).value_counts().min() >= 2 else None,
    )

    model_before = RandomForestClassifier(n_estimators=100, random_state=42)
    model_before.fit(X_train, y_train)
    preds_before = model_before.predict(X_test)

    # Augmented training set = original train split + synthetic rows
    # (synthetic rows are never mixed into the held-out test set — that
    # would make the evaluation meaningless).
    train_df = X_train.copy()
    train_df[target_column] = le.inverse_transform(y_train)
    combined_train = pd.concat([train_df, df_augmented], ignore_index=True)

    X_train_aug = prepare_features(combined_train, feature_columns)
    y_train_aug = le.transform(combined_train[target_column])

    model_after = RandomForestClassifier(n_estimators=100, random_state=42)
    model_after.fit(X_train_aug, y_train_aug)
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
        results.append(AugmentationScore(metric_name, round(before_val, 4), round(after_val, 4), round(improvement, 1)))

    return results
