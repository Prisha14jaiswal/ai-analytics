"""
auto_trainer.py — Simple ML model training on user-selected target column.

Supports ONLY:
  - LogisticRegression
  - DecisionTreeClassifier
  - RandomForestClassifier

No deep learning. No neural nets. No AutoML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

# ── Model registry ───────────────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
}


# ── Data preparation ─────────────────────────────────────────────────────────


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """Prepare features and target for training.

    Steps:
    1. Drop rows where target is NaN.
    2. Select only numeric columns as features (simple, safe).
    3. Fill remaining NaNs in features with column median.
    4. If target is non-numeric, encode it with LabelEncoder.
    5. Split into train / test.

    Returns
    -------
    X_train, X_test, y_train, y_test, prep_info
    """
    work = df.copy()
    work = work.dropna(subset=[target_col])

    prep_info: Dict[str, Any] = {"target_col": target_col}

    # Encode target if non-numeric
    label_encoder = None
    if work[target_col].dtype == "object" or work[target_col].dtype.name == "category":
        label_encoder = LabelEncoder()
        work[target_col] = label_encoder.fit_transform(work[target_col])
        prep_info["label_classes"] = list(label_encoder.classes_)
    prep_info["label_encoder"] = label_encoder

    y = work[target_col]

    # Features = all numeric columns except target
    feature_cols = [
        c for c in work.select_dtypes(include="number").columns
        if c != target_col
    ]
    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found. "
            "The dataset needs at least one numeric column besides the target."
        )

    X = work[feature_cols].copy()
    X = X.fillna(X.median())

    prep_info["feature_cols"] = feature_cols
    prep_info["n_samples"] = len(X)
    prep_info["n_features"] = len(feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if y.nunique() <= 20 else None
    )

    prep_info["train_size"] = len(X_train)
    prep_info["test_size"] = len(X_test)
    prep_info["class_distribution"] = y.value_counts().to_dict()

    return X_train, X_test, y_train, y_test, prep_info


# ── Training ─────────────────────────────────────────────────────────────────


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    prep_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Train the chosen model and return results.

    Returns
    -------
    dict with: model, accuracy, classification_report, confusion_matrix,
    feature_importance (if available), prep_info.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(AVAILABLE_MODELS)}")

    ModelClass = AVAILABLE_MODELS[model_name]

    # Sensible defaults per model
    if model_name == "Logistic Regression":
        model = ModelClass(max_iter=1000, random_state=42)
    elif model_name == "Decision Tree":
        model = ModelClass(max_depth=5, random_state=42)
    else:  # Random Forest
        model = ModelClass(n_estimators=100, max_depth=5, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Feature importance
    feature_importance = {}
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        feature_importance = dict(zip(prep_info["feature_cols"], np.round(fi, 4).tolist()))
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
        feature_importance = dict(zip(prep_info["feature_cols"], np.round(coefs, 4).tolist()))

    # Sort by importance descending
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # Build target names for report
    target_names = None
    if "label_classes" in prep_info:
        target_names = [str(c) for c in prep_info["label_classes"]]

    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model_name": model_name,
        "model": model,
        "accuracy": round(acc, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance,
        "prep_info": prep_info,
    }


def results_to_text(results: Dict[str, Any]) -> str:
    """Convert training results to a readable text summary."""
    lines = []
    lines.append(f"Model: {results['model_name']}")
    lines.append(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    lines.append(f"Training samples: {results['prep_info']['train_size']:,}")
    lines.append(f"Test samples: {results['prep_info']['test_size']:,}")
    lines.append(f"Features used: {results['prep_info']['n_features']}")

    if results["feature_importance"]:
        lines.append("\nTop Feature Importances:")
        for feat, imp in list(results["feature_importance"].items())[:10]:
            bar = "█" * int(imp * 50)
            lines.append(f"  {feat:30s} {imp:.4f} {bar}")

    return "\n".join(lines)
