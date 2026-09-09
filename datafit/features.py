"""
DataFit — shared feature preparation.

Both remediation.py and synthetic.py need to turn a dataframe's columns
into a numeric matrix a model can train on. Previously this silently
dropped categorical columns, which is dishonest — a dataset with
meaningful categorical features would get a worse score than it should,
and the user would never know why. This module fixes that by properly
encoding categoricals instead of discarding them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_features(df: pd.DataFrame, feature_columns: list[str], max_onehot_cardinality: int = 20) -> pd.DataFrame:
    """Builds a fully numeric feature matrix from a mix of numeric and
    categorical columns.

    - Numeric columns: used as-is (missing values filled with 0 —
      callers doing careful missing-value handling should impute
      beforehand via clean_dataset()).
    - Low-cardinality categoricals (<= max_onehot_cardinality unique
      values): one-hot encoded.
    - High-cardinality categoricals: excluded, since these should have
      been caught and dropped upstream by clean_dataset(). We don't
      raise here to keep this a pure utility, but callers should run
      diagnostics/cleaning first.

    Returns a numeric DataFrame ready for model training.
    """
    if not feature_columns:
        return pd.DataFrame(index=df.index)

    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]

    numeric_part = df[numeric_cols].fillna(0) if numeric_cols else pd.DataFrame(index=df.index)

    onehot_parts = []
    for col in categorical_cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= max_onehot_cardinality:
            dummies = pd.get_dummies(df[col].astype("category"), prefix=col, dummy_na=True)
            onehot_parts.append(dummies)

    parts = [numeric_part] + onehot_parts
    non_empty = [p for p in parts if len(p.columns) > 0]
    result = pd.concat(non_empty, axis=1) if non_empty else numeric_part
    return result.astype(float)
