"""
dataframe_agent.py — Sandboxed Pandas agent for live data queries.

Uses LangGraph's ReAct agent with a PythonAstREPLTool so the LLM
can write and execute Pandas code against the user's uploaded DataFrame.
"""

from __future__ import annotations

import os

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_experimental.tools import PythonAstREPLTool


# ── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are a senior data analyst. The user uploaded a CSV dataset.

DATASET SCHEMA
==============
{schema}

AVAILABLE COLUMNS
=================
{columns}

ANALYTICAL PROTOCOL
===================
1. Metric Computation: ALWAYS use `python_repl` to calculate actual values (mean, median, etc.) before answering.
2. Terminology: Define "High" as > 75th percentile and "Low" as < 25th percentile.
3. Evidence: No generic statements. Every claim must be backed by code output.

VISUALIZATION RULES
===================
1. Detection: Trigger on words like "show", "plot", "graph", "visualize", "bar", "line", "histogram", "scatter".
2. Constraints:
   - For "revenue" or "sales", use the first available column from: ['price', 'total_sales', 'amount'].
   - If y_column is numeric, you MUST provide an aggregation ('sum', 'mean', 'count').
3. OUTPUT FORMAT:
   - RETURN RAW JSON ONLY.
   - NO markdown backticks (e.g., NO ```json).
   - NO preamble (e.g., NO "Here is the chart:").
   - NO explanation text.
4. JSON SCHEMA:
{{
  "visualization": true,
  "chart_type": "bar | line | histogram | scatter",
  "x_column": "exact_column_name",
  "y_column": "exact_column_name_or_null",
  "aggregation": "sum | mean | count | null",
  "title": "Short descriptive title"
}}
"""


# ── LLM factory ──────────────────────────────────────────────────────────────


def _get_llm() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "groq")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


# ── Tool ─────────────────────────────────────────────────────────────────────


def _create_tool(df: pd.DataFrame) -> PythonAstREPLTool:
    """Sandboxed Python REPL with only df and pd in scope."""
    return PythonAstREPLTool(
        locals={"df": df, "pd": pd},
        name="python_repl",
        description=(
            "Run Python/Pandas code against the DataFrame `df`. "
            "Always print() your final answer. "
            "Available variables: `df`, `pd`. Do NOT import anything."
        ),
    )


# ── Schema helper ────────────────────────────────────────────────────────────


def _build_schema_text(df: pd.DataFrame) -> str:
    """Generate a concise schema string for the system prompt."""
    lines = [f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns", "", "Columns:"]
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        sample = str(df[col].dropna().iloc[0])[:50] if non_null > 0 else "N/A"
        lines.append(f"  - {col} ({dtype}) | non-null: {non_null:,} | e.g. {sample}")
    return "\n".join(lines)


# ── Agent builder ────────────────────────────────────────────────────────────


def build_dataframe_agent(df: pd.DataFrame):
    """Build and return a LangGraph ReAct agent for the given DataFrame.

    Returns a compiled graph that can be called with:
        result = agent.invoke({"messages": [("user", question)]})
        answer = result["messages"][-1].content
    """
    from langgraph.prebuilt import create_react_agent

    llm = _get_llm()
    tool = _create_tool(df)
    schema = _build_schema_text(df)
    columns = ", ".join(df.columns.tolist())
    system_prompt = _SYSTEM_TEMPLATE.format(schema=schema, columns=columns)

    agent = create_react_agent(
        model=llm,
        tools=[tool],
        prompt=system_prompt,
    )
    return agent


def ask_agent(agent, question: str) -> str:
    """Send a question to the agent and return the final answer string."""
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content
