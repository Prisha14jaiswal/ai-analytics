# Technical Walkthrough: Universal AI Analytics Assistant

This document provides a deep dive into the architecture, design decisions, and implementation details of the Universal AI Analytics Assistant. This is intended for interview preparation or technical documentation.

---

## 1. Project Objective
The goal was to build a "Swiss Army Knife" for data analysts—a tool that allows non-technical users to upload any CSV dataset and perform complex analysis, summary-based retrieval, and predictive modeling using simple natural language.

---

## 2. High-Level Architecture
The project follows a modular, capability-driven architecture. Each specialized task (RAG, Agentic Analysis, ML) is isolated into its own submodule.

### **Structure Overview**
- **`app.py`**: The orchestration layer. Manages the Streamlit lifecycle, session state, and UI rendering.
- **`schema_analyzer.py`**: The "brain" for initial data profiling. It generates both a JSON-like dict for the UI and a long-form text report for the RAG engine.
- **`rag/`**: Handles retrieval-based questions about the *metadata* and *summaries*.
- **`agent/`**: Handles live *computational* queries on the actual data entries using LLM-generated code.
- **`ml/`**: Provides automated, explainable classification models.

---

## 3. Core Capabilities & "How they work"

### **A. Automated Profiling (`schema_analyzer.py`)**
When a CSV is uploaded, the system immediately runs a comprehensive profile.
- **Dtype Detection**: Identifies numeric, categorical, and datetime columns.
- **Missing Value Analysis**: Calculates counts and percentages per column.
- **Statistics**: Generates `describe()` metrics for numeric data and value counts for categories.
- **Outcome**: This profile acts as the "context" for both the RAG system and the Data Agent's system prompt.

### **B. The Dual-Path Q&A System**
This is a critical architectural pattern. We separate questions into two types:

1. **RAG (Retrieval-Augmented Generation)**:
   - **Use Case**: "What columns have missing data?" or "Give me a summary of the prices."
   - **Process**: The text report from the analyzer is chunked using `RecursiveCharacterTextSplitter` and stored in an in-memory **FAISS** vector store. The LLM retrieves relevant chunks to answer.
   - **Why?**: It's faster and cheaper than scanning the whole 100k+ row dataset for simple metadata questions.

2. **The Data Agent (LangGraph ReAct)**:
   - **Use Case**: "What is the average shipping cost in São Paulo?"
   - **Process**: Uses a **LangGraph ReAct loop**. The LLM is given a `PythonAstREPLTool` with access to the DataFrame (`df`). It writes Pandas code, executes it in a sandbox, and interprets the results.
   - **Why?**: RAG cannot "calculate." An agent with a REPL is necessary for live data manipulation.

### **C. Safe Visualizations (JSON Instruction Pattern)**
Instead of allowing the LLM to generate raw `matplotlib` code (which is brittle and dangerous), we implemented a **Structured Output Pattern**:
1. The Agent is told in its system prompt: *"If asked for a chart, return JSON only."*
2. It returns a schema: `{"chart_type": "bar", "x_col": "state", "agg": "mean"}`.
3. The Streamlit app parses this, validates columns against the DataFrame, and executes a pre-written, safe plotting function.
4. **Benefit**: 100% safe, consistent styling, and no "code hallucination" errors for the user to see.

### **D. Auto ML Trainer (`ml/auto_trainer.py`)**
We focused on **Explainability over Complexity**.
- **Models**: Limited to Logistic Regression, Decision Trees, and Random Forests.
- **Preprocessing**: Automatic handling of missing values (median imputation) and categorical targets (Label Encoding).
- **Diagnostics**: The UI reveals the **Classification Report** and **Feature Importance** charts so the user understands *why* the model predicts a certain way.

---

## 4. Key Design Decisions (The "Why")

| Choice | Rationale |
| :--- | :--- |
| **Streamlit** | Chosen for its "Data-First" primitives (st.dataframe, st.pyplot) and rapid prototyping speed. |
| **FAISS (In-Memory)** | Since datasets are uploaded per session, we don't need a persistent vector DB like Pinecone. FAISS is extremely fast for local, session-based retrieval. |
| **PythonAstREPLTool** | Safer than `exec()` because it uses Abstract Syntax Tree parsing to reject potentially malicious or invalid syntax. |
| **Groq (Llama 3.3)** | Used for its incredible inference speed (sub-second responses), which makes the Agentic loop feel snappy rather than sluggish. |

---

## 5. Potential Interview Questions

**Q: How do you handle large datasets in the Data Agent?**
*A: We don't feed the whole dataset to the LLM. We only feed the **Schema** (column names, types, and sample values). The LLM writes code that runs locally on the data. Only the **results** (e.g., a single numeric value or a small grouped table) are sent back to the LLM for explanation.*

**Q: Why didn't you use LangChain's built-in `create_pandas_dataframe_agent`?**
*A: Customization and stability. By building our own with **LangGraph**, we have finer control over the system prompt, tool access, and the loop logic. It also allowed us to implement the "JSON-for-Vis" pattern easily.*

**Q: How do you handle security for code execution?**
*A: We use several layers: (1) The `PythonAstREPLTool` which lacks access to dangerous modules like `os` or `subprocess`. (2) A strict system prompt. (3) The code runs within the local Python environment of the app, never reaching the LLM provider's internal infrastructure.*
