# 📊 Financial Research Assistant

A local-first AI system for querying financial documents in natural language — with a built-in Trust Layer that scores, validates, and routes every answer before surfacing it to the user.

Financial professionals, students, and analysts spend hours reading through hundreds of pages of annual reports just to find one revenue figure or compare two companies' margins. This project was built to change that — to make financial research faster, more reliable, and accessible without a Bloomberg terminal or expensive data subscription. What makes it different from a generic "chat with PDFs" tool is accountability: every answer carries a confidence score, a source citation, and a human review flag when evidence is thin. That matters in finance, where acting on a wrong number has real consequences. The target users are equity analysts, finance students, small advisory firms, and independent investors who need document-grounded answers they can actually trust.

---

## What It Does

| Feature | Example query |
|---|---|
| Chat Q&A with Trust Scoring | `What was Cognizant revenue in FY2024?` |
| Multi-company Comparison | `Compare TCS and Infosys for FY2025` |
| Knowledge Graph | `Show knowledge graph for Cognizant` |
| Human-in-the-Loop Review | Low-confidence answers queued for human validation |
| Analytics Dashboard | `Show dashboard for Infosys` |
| Analyst Report Generation | `Generate report for Cognizant` |

---

## File Reference

| File | Responsibility |
|---|---|
| `document_processor.py` | Loads PDFs/TXTs, resolves company and fiscal year from folder structure and document content |
| `vector_store.py` | Chunks text, embeds, stores in ChromaDB; handles filtered similarity search |
| `rag_pipeline.py` | All RAG logic: Q&A, insights, ratios, comparisons, reports, knowledge graph |
| `trust_engine.py` | Five-factor confidence scoring, citation validation, contradiction detection, Approve/Warn/Review decision |
| `langgraph_flow.py` | 4-node LangGraph workflow (classify → route → trust → response); plain-Python fallback if LangGraph unavailable |
| `dashboard.py` | Streamlit single-window UI — Chat, Report, Knowledge Graph, Dashboard, HITL Queue tabs |
| `hitl_queue.py` | JSON-backed persistent queue for human review |

---

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) for local LLM

### Install

```bash
pip install -r requirements.txt
ollama pull llama3
```

### Ingest documents

Place documents in this structure:

```
documents/
├── Annual_reports/
│   ├── Cognizant/
│   ├── Infosys/
│   └── TCS/
├── Earning_calls/
└── Financial_News/
```

Company name comes from the immediate parent folder. Source type comes from the grandparent folder. Filenames can be anything.

```bash
rm -rf chroma_db/
python rag_pipeline.py --ingest --docs_folder ./documents
```

### Run

```bash
ollama serve        # separate terminal
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Using OpenAI Instead of Ollama

Set your API key, then toggle **Use OpenAI** in the sidebar:

```bash
export OPENAI_API_KEY=your_key_here
streamlit run dashboard.py
```

---

## Trust Scoring

Every answer is scored out of 100 across five factors:

| Factor | Max | What it measures |
|---|---|---|
| Source Quality | 25 | Proportion of chunks from primary sources (annual reports, earnings calls) |
| Retrieval Relevance | 21 | Mean cosine similarity between query and retrieved chunks |
| Citation Completeness | 20 | Every retrieved chunk has a matching citation entry |
| Cross-Source Agreement | 15 | All chunks agree on company name and fiscal year |
| Calculation Validation | 15 | Numeric figures in the answer appear in the source chunks |

| Band | Score | Decision |
|---|---|---|
| High | ≥ 85 | Approve |
| Medium | 70–84 | Warn |
| Low | < 70 | Review |

---

## Tech Stack

Streamlit · ChromaDB · SentenceTransformers · Ollama · LangGraph · PyMuPDF · Plotly · NetworkX · PyVis · OpenAI (optional)

---

## Known Limitations

- Company alias matching is exact — `TCS` and `Tata Consultancy Services` are treated as different companies
- Trust score weights are heuristic, not derived from a labeled evaluation dataset
- INR→USD conversion uses a fixed ~83 rate for chart display only
