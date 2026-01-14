
# Multi-Agent Financial Research (LangGraph)

A **LangGraph-based deterministic multi-agent research pipeline** for financial, macroeconomic, and alternative data analysis.
This project demonstrates how **explicit state machines and graph-based orchestration** can be used to build **auditable, reproducible, and production-ready financial research systems**.

Designed for **quantitative research, regulated environments, and ML feature pipelines**.

---

## One-Sentence Audience Framing

**For quantitative researchers, ML engineers, and AI architects who need deterministic, inspectable, and production-grade multi-agent workflows for financial research.**

---

## Business Impact

This system enables:

* Fully reproducible research pipelines
* Auditable data lineage and feature generation
* Leakage-safe target construction
* Deterministic execution suitable for regulated domains
* Seamless transition from research to production ML

It is especially valuable for **finance, commodities, macro strategy, and enterprise analytics** where correctness and traceability matter.

---

## Architecture Overview (LangGraph)

In this implementation:

* Agents are implemented as **deterministic graph nodes**
* Execution follows an **explicit DAG**
* State is **typed, versioned, and inspectable**
* Failures are **localized and debuggable**

**Pipeline Flow**

```
START
  → Yahoo OHLCV Extraction
  → FRED + World Bank Macro/FX
  → Open-Meteo Multi-Region Weather
  → Feature Engineering
  → Validation
END
```

Each node:

* Accepts a typed state object
* Produces explicit state updates
* Records metadata, warnings, and errors

---

## Key Features

* LangGraph-based orchestration
* Explicit state transitions
* Deterministic execution
* Typed error handling
* Multi-region weather aggregation
* Macro + FX enrichment
* Leakage-safe multi-horizon targets
* ML-ready feature datasets
* Built-in validation checks

---

## Technologies Used

* **LangGraph**
* **LangChain**
* **Python 3.11+**
* **uv** (dependency & environment management)
* **Yahoo Finance**
* **FRED API**
* **World Bank API**
* **Open-Meteo API**
* **Pandas / NumPy**

---

## Quick Start (Using `uv` – Recommended)

This project **does not use pip**.
All dependencies are managed via **`uv`**.

---

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell, then verify:

```bash
uv --version
```

---

### 2. Clone the Repository

```bash
git clone <YOUR_LANGGRAPH_REPO_URL>
cd multi-agent-research-langgraph
```

---

### 3. Create and Sync the Virtual Environment

```bash
uv venv
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
FRED_API_KEY=your_fred_api_key
OPENAI_API_KEY=your_openai_api_key
```

Optional:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

### 5. Run the Pipeline

If exposed as a CLI:

```bash
uv run commodity-pipeline \
  --topic coffee \
  --ticker "KC=F" \
  --start 2015-01-01 \
  --end 2025-01-01 \
  --enable-exogenous \
  --weather-6-regions
```

Or directly:

```bash
uv run python src/langgraphagenticai/main.py
```

---

## Output Artifacts

Generated outputs include:

```
data/raw/
data/processed/features_daily.csv
data/processed/features_daily_data_dictionary.md
```

These datasets are:

* Deterministically generated
* Fully auditable
* Leakage-safe
* Ready for linear or nonlinear models
* Suitable for long-horizon research

---

## How This Differs from the CrewAI Version

| Dimension         | LangGraph                 | CrewAI         |
| ----------------- | ------------------------- | -------------- |
| Control Flow      | Explicit DAG              | Task sequence  |
| State             | Typed & explicit          | Implicit       |
| Determinism       | Strong                    | Moderate       |
| Debuggability     | Excellent                 | Moderate       |
| Failure Isolation | High                      | Medium         |
| Best For          | Production & regulated ML | Rapid research |

This repo exists to provide a **direct, apples-to-apples comparison** with the CrewAI implementation.

---

## Use Cases

* Commodity price forecasting
* Macro-driven return modeling
* Weather-impact analysis
* Feature engineering pipelines
* Regulated financial research
* ML dataset generation
* Research reproducibility

---

## Why LangGraph

LangGraph is ideal when you need:

* Explicit execution graphs
* Strong guarantees about ordering
* Deterministic behavior
* Clear state evolution
* Debuggable, inspectable pipelines

It bridges the gap between **LLM orchestration** and **data engineering pipelines**.

---

## Why `uv`

This project uses **`uv`** instead of `pip` because:

* Faster installs
* Deterministic dependency resolution
* Built-in virtual environments
* Cleaner CI/CD
* Modern Python best practice

---

## Repository Structure

```
.
├── src/langgraphagenticai/
│   ├── main.py
│   ├── graph/
│   │   └── pipeline.py
│   ├── nodes/
│   │   ├── yahoo_data_extraction_node.py
│   │   ├── fred_data_extraction_node.py
│   │   ├── open_meteo_weather_node.py
│   │   ├── build_features_node.py
│   │   └── validate_data_node.py
│   └── state.py
├── data/
│   ├── raw/
│   └── processed/
├── pyproject.toml
└── README.md
```

---

## Related Projects

* **CrewAI version:** `<CREWAI_REPO_URL>`
* **Blog post:** `<BLOG_URL>`
* **Architecture diagrams:** `<DIAGRAMS_URL>`

---

## License

MIT License

