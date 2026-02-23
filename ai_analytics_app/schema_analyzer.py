"""
schema_analyzer.py — Automatic Dataset Schema Analysis.

Accepts any Pandas DataFrame and returns a structured summary
dictionary plus a human-readable text report (used downstream
by both the RAG system and the Streamlit UI).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def analyze_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze a DataFrame and return a structured summary.

    Returns
    -------
    dict with keys:
        n_rows, n_cols, columns, dtypes, missing_values,
        missing_pct, numeric_summary, categorical_summary,
        memory_usage_mb
    """
    n_rows, n_cols = df.shape

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)

    # Numeric summary
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        numeric_summary = desc.to_dict()

    # Categorical summary
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_summary = {}
    for col in cat_cols:
        top = df[col].value_counts().head(5)
        categorical_summary[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": top.to_dict(),
        }

    # Datetime columns
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": missing.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "datetime_cols": date_cols,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }


def summary_to_text(summary: Dict[str, Any]) -> str:
    """Convert a structured summary dict into a human-readable text report.

    This text is fed into the RAG pipeline as the source document.
    """
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("DATASET ANALYSIS REPORT")
    lines.append("=" * 60)

    # Overview
    lines.append(f"\nRows: {summary['n_rows']:,}")
    lines.append(f"Columns: {summary['n_cols']}")
    lines.append(f"Memory: {summary['memory_usage_mb']} MB")

    # Column types
    lines.append(f"\nNumeric columns ({len(summary['numeric_cols'])}): "
                 + ", ".join(summary["numeric_cols"]))
    lines.append(f"Categorical columns ({len(summary['categorical_cols'])}): "
                 + ", ".join(summary["categorical_cols"]))
    if summary["datetime_cols"]:
        lines.append(f"Datetime columns ({len(summary['datetime_cols'])}): "
                     + ", ".join(summary["datetime_cols"]))

    # Missing values
    lines.append("\n" + "=" * 60)
    lines.append("MISSING VALUES")
    lines.append("=" * 60)
    has_missing = False
    for col, count in summary["missing_values"].items():
        if count > 0:
            has_missing = True
            pct = summary["missing_pct"][col]
            lines.append(f"  {col}: {count:,} missing ({pct}%)")
    if not has_missing:
        lines.append("  No missing values found.")

    # Numeric summary
    if summary["numeric_summary"]:
        lines.append("\n" + "=" * 60)
        lines.append("NUMERIC COLUMN STATISTICS")
        lines.append("=" * 60)
        for col, stats in summary["numeric_summary"].items():
            lines.append(f"\n  {col}:")
            for stat, val in stats.items():
                lines.append(f"    {stat}: {val}")

    # Categorical summary
    if summary["categorical_summary"]:
        lines.append("\n" + "=" * 60)
        lines.append("CATEGORICAL COLUMN SUMMARY")
        lines.append("=" * 60)
        for col, info in summary["categorical_summary"].items():
            lines.append(f"\n  {col} ({info['unique_count']} unique values):")
            for val, count in info["top_values"].items():
                lines.append(f"    {val}: {count:,}")

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    return "\n".join(lines)
