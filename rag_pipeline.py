"""
rag_pipeline.py
───────────────
Part 1 — Steps 6 & 7: Context Assembly + LLM Answer Generation

This is the main entry point.  Run it from the terminal:

  # INGEST: process documents and build the ChromaDB collection
  python rag_pipeline.py --ingest --docs_folder ./documents

  # QUERY: ask a question against the built collection
  python rag_pipeline.py --query "What was Infosys revenue in FY2024?"

  # QUERY with OpenAI LLM + OpenAI embeddings
  python rag_pipeline.py --query "..." --use_openai

  # QUERY with local LLM (Ollama) + local embeddings (default)
  python rag_pipeline.py --query "..."

  # QUERY filtered to one company only
  python rag_pipeline.py --query "..." --company Infosys

Environment variables:
  OPENAI_API_KEY  — required only when --use_openai is passed
  OLLAMA_HOST     — optional, defaults to http://localhost:11434


Install:
  pip install chromadb sentence-transformers openai requests pymupdf numpy
"""

import os
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── Part 1 modules ────────────────────────────────────────────────────────────
from document_processor import load_documents_from_folder
from vector_store import (
    chunk_pages,
    embed_texts,
    build_chroma_collection,
    load_chroma_collection,
    retrieve_chunks,
    CHROMA_DIR,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
PIPELINE_VERSION   = "1.0.0"
OUTPUT_FOLDER      = "answer_packages"   # where answer_package JSONs are saved
DEFAULT_TOP_K      = 5
LOCAL_LLM_MODEL    = "llama3"            # Ollama model — change as needed
OPENAI_LLM_MODEL   = "gpt-4o"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL  = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# STEP 6: CONTEXT ASSEMBLY
# ─────────────────────────────────────────────
def assemble_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Stitch retrieved chunk texts into one context string for the LLM prompt.
    Each chunk is labelled with its source so the LLM can reference it clearly.

    Developer 2's Trust Layer will later filter/rerank chunks by confidence;
    we simply concatenate here in similarity order.
    """
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        source_label = (
            f"[Source {i}: {chunk['file_name']}"
            + (f", page {chunk['page_number']}" if chunk["page_number"] else "")
            + f" | {chunk['company_name']} {chunk['fiscal_year']}]"
        )
        context_parts.append(f"{source_label}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


# ─────────────────────────────────────────────
# STEP 7a: LLM — OpenAI backend
# ─────────────────────────────────────────────
def _call_openai(system_prompt: str, user_prompt: str,
                 model: str = OPENAI_LLM_MODEL) -> str:
    """
    Call the OpenAI Chat Completions API.
    Requires OPENAI_API_KEY in environment.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client   = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# STEP 7b: LLM — Local Ollama backend
# ─────────────────────────────────────────────
def _call_ollama(system_prompt: str, user_prompt: str,
                 model: str = LOCAL_LLM_MODEL) -> str:
    """
    Call a locally running Ollama server.
    Install: https://ollama.com  then run: ollama pull llama3 && ollama serve
    Override server address with OLLAMA_HOST env var.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    host    = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    payload = {
        "model":   model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream":  False,
        "options": {"temperature": 0.2},
    }

    try:
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to Ollama at {host}. "
            "Start it with: ollama serve"
        )


# ─────────────────────────────────────────────
# STEP 7: ANSWER GENERATION
# ─────────────────────────────────────────────
def generate_answer(question:        str,
                    retrieved_chunks: List[Dict[str, Any]],
                    use_openai:       bool = False) -> str:
    """
    Assemble context from retrieved chunks and call the LLM to generate an answer.
    """
    context = assemble_context(retrieved_chunks)

    system_prompt = (
        "You are a financial research assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "Be precise and cite figures and fiscal years where relevant. "
        "If the context does not contain enough information, say so clearly — "
        "do NOT fabricate financial data. "
        "Keep your answer concise (3–6 sentences unless more detail is needed)."
    )

    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    print(f"[LLM] Generating answer with "
          f"{'OpenAI ' + OPENAI_LLM_MODEL if use_openai else 'Ollama ' + LOCAL_LLM_MODEL} …")

    if use_openai:
        return _call_openai(system_prompt, user_prompt, model=OPENAI_LLM_MODEL)
    else:
        return _call_ollama(system_prompt, user_prompt, model=LOCAL_LLM_MODEL)


# ─────────────────────────────────────────────
# ASSEMBLE answer_package  (frozen contract output)
# ─────────────────────────────────────────────
def build_answer_package(question:        str,
                         answer:          str,
                         retrieved_chunks: List[Dict[str, Any]],
                         total_in_store:   int,
                         use_openai:       bool = False) -> Dict[str, Any]:
    """
    Build the frozen answer_package dict as defined in project_contract.json.
    Field names MUST NOT change — Developer 2 depends on them.
    """
    citations = [
        {
            "chunk_id":     c["chunk_id"],
            "file_name":    c["file_name"],
            "page_number":  c["page_number"],
            "company_name": c["company_name"],
            "fiscal_year":  c["fiscal_year"],
        }
        for c in retrieved_chunks
    ]

    metadata = {
        "model_used":            OPENAI_LLM_MODEL if use_openai else LOCAL_LLM_MODEL,
        "retrieval_top_k":       len(retrieved_chunks),
        "pipeline_version":      PIPELINE_VERSION,
        "timestamp":             datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_chunks_in_store": total_in_store,
        "embedding_model":       OPENAI_EMBED_MODEL if use_openai else LOCAL_EMBED_MODEL,
    }

    return {
        "question":         question,
        "answer":           answer,
        "retrieved_chunks": retrieved_chunks,
        "citations":        citations,
        "metadata":         metadata,
    }


# ─────────────────────────────────────────────
# SAVE answer_package to disk
# ─────────────────────────────────────────────
def save_answer_package(answer_package: Dict[str, Any]) -> str:
    """
    Write the answer_package to a timestamped JSON file.
    Developer 2 reads from this file or imports run_query() directly.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    ts        = answer_package["metadata"]["timestamp"].replace(":", "-")
    file_path = os.path.join(OUTPUT_FOLDER, f"answer_package_{ts}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(answer_package, f, indent=2, ensure_ascii=False)

    print(f"[Pipeline] answer_package saved → {file_path}")
    return file_path


# ─────────────────────────────────────────────
# PUBLIC API — for Developer 2 or tests
# ─────────────────────────────────────────────
def run_query(question:          str,
              top_k:             int  = DEFAULT_TOP_K,
              use_openai:        bool = False,
              save:              bool = True,
              filter_company:    str  = None,
              filter_fiscal_year: str = None) -> Dict[str, Any]:
    """
    End-to-end RAG query.

    Parameters
    ----------
    question           : user's natural-language question
    top_k              : number of chunks to retrieve (default 5)
    use_openai         : True → OpenAI embeddings + GPT-4o
                         False → local SentenceTransformer + Ollama (default)
    save               : save answer_package JSON to ./answer_packages/
    filter_company     : optional — limit retrieval to one company
                         e.g. "Infosys", "TCS", "Cognizant"
    filter_fiscal_year : optional — limit retrieval to one fiscal year
                         e.g. "FY2024". When both are provided, ChromaDB
                         applies an $and filter so only chunks matching
                         BOTH company AND fiscal year are considered —
                         this prevents wrong-year boilerplate chunks from
                         crowding out the actually relevant pages.

    Returns
    -------
    dict — answer_package conforming to project_contract.json schema
    """
    collection = load_chroma_collection()

    retrieved = retrieve_chunks(
        query=question,
        collection=collection,
        top_k=top_k,
        use_openai=use_openai,
        openai_model=OPENAI_EMBED_MODEL,
        local_model=LOCAL_EMBED_MODEL,
        filter_company=filter_company,
        filter_fiscal_year=filter_fiscal_year,
    )

    # Generate answer
    answer = generate_answer(question, retrieved, use_openai=use_openai)

    # Build the frozen contract package
    package = build_answer_package(
        question=question,
        answer=answer,
        retrieved_chunks=retrieved,
        total_in_store=collection.count(),
        use_openai=use_openai,
    )

    if save:
        save_answer_package(package)

    return package


def run_ingest(docs_folder: str,
               use_openai:  bool = False) -> None:
    """
    Full ingest pipeline:
      load documents → chunk → embed → build ChromaDB collection → persist to disk.

    Run once per document set (or whenever documents change).
    ChromaDB automatically saves to ./chroma_db/ — no manual step needed.
    """
    # Step 1: Load all PDF + TXT documents from the nested folder structure
    pages = load_documents_from_folder(docs_folder)
    if not pages:
        print("[Pipeline] No documents found in folder. Exiting.")
        return

    # Step 2: Chunk pages into overlapping text windows
    chunks = chunk_pages(pages)

    # Steps 3 + 4: Embed and store in ChromaDB (combined in one function)
    build_chroma_collection(
        chunks=chunks,
        use_openai=use_openai,
        openai_model=OPENAI_EMBED_MODEL,
        local_model=LOCAL_EMBED_MODEL,
    )

    print(f"\n[Pipeline] Ingestion complete. "
          f"ChromaDB saved to ./{CHROMA_DIR}/\n"
          f"           Run with --query to start asking questions.")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Financial Research Pipeline — Part 1 (ChromaDB)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ingest", action="store_true",
                      help="Load docs, embed, build ChromaDB collection")
    mode.add_argument("--query",  type=str, metavar="QUESTION",
                      help="Ask a question against the built collection")

    parser.add_argument("--docs_folder", type=str, default="./data",
                        help="Root folder containing company subfolders (used with --ingest)")
    parser.add_argument("--top_k",       type=int, default=DEFAULT_TOP_K,
                        help=f"Chunks to retrieve (default {DEFAULT_TOP_K})")
    parser.add_argument("--use_openai",  action="store_true",
                        help="Use OpenAI API for embeddings & LLM (default: local)")
    parser.add_argument("--company",     type=str, default=None,
                        help="Filter retrieval to one company e.g. --company Infosys")
    parser.add_argument("--no_save",     action="store_true",
                        help="Don't save answer_package JSON to disk")

    args = parser.parse_args()

    if args.ingest:
        run_ingest(docs_folder=args.docs_folder, use_openai=args.use_openai)

    elif args.query:
        package = run_query(
            question=args.query,
            top_k=args.top_k,
            use_openai=args.use_openai,
            save=not args.no_save,
            filter_company=args.company,
        )
        print("\n" + "═" * 60)
        print("ANSWER:")
        print(package["answer"])
        print("\nANSWER PACKAGE (for Developer 2):")
        print(json.dumps(package, indent=2))


# ═════════════════════════════════════════════════════════════════════════════
# EXTENDED FEATURE FUNCTIONS
# Added to support: insight extraction, ratio analysis, company comparison,
# report generation, and query classification.
# All functions reuse run_query() and the existing LLM backends.
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# QUERY CLASSIFIER
# ─────────────────────────────────────────────

# Maps user intent keywords → route label used by LangGraph
_ROUTE_PATTERNS = [
    ("comparison",   ["compare", "vs", "versus", "difference between", "better than"]),
    ("ratio",        ["ratio", "roe", "roa", "debt to equity", "current ratio",
                      "profit margin", "return on", "calculate"]),
    ("report",       ["generate report", "write report", "analyst report",
                      "research report", "report for"]),
    ("graph",        ["knowledge graph", "show graph", "graph for", "entity graph"]),
    ("dashboard",    ["show dashboard", "analytics dashboard", "trends for",
                      "dashboard for", "charts for"]),
    ("analysis",     ["analyze", "analyse", "insights", "overview", "tell me about",
                      "summarize", "summary of", "deep dive"]),
    ("qa",           []),   # default fallback
]


def classify_query(question: str) -> str:
    """
    Classify a user question into one of these route labels:
      qa | analysis | ratio | comparison | report | graph | dashboard

    Uses simple keyword matching — no LLM call needed.
    Returns a route string consumed by LangGraph and the dashboard.

    Examples
    --------
    "Compare TCS and Infosys"          → "comparison"
    "Show knowledge graph for Infosys" → "graph"
    "What was Infosys revenue?"        → "qa"
    "Analyze Infosys FY2024"           → "analysis"
    """
    q_lower = question.lower()
    for route, keywords in _ROUTE_PATTERNS:
        if any(kw in q_lower for kw in keywords):
            return route
    return "qa"


# ─────────────────────────────────────────────
# FEATURE 1: FINANCIAL INSIGHT EXTRACTION
# ─────────────────────────────────────────────

_INSIGHT_SECTIONS = {
    "Company Overview": (
        "Give a brief overview of {company}'s business model, core services, "
        "and key market segments as described in {fy} documents."
    ),
    "Financial Highlights": (
        "What were the key financial results for {company} in {fy}? "
        "Include revenue, net income, operating margin, and growth rates."
    ),
    "Key Strengths": (
        "What are the key competitive strengths and advantages of {company} "
        "mentioned in {fy} annual reports or earnings calls?"
    ),
    "Challenges & Risks": (
        "What challenges, risks, and headwinds did {company} face or mention in {fy}?"
    ),
    "Management Commentary": (
        "What did {company} management say about business performance, "
        "strategy, and priorities during {fy}?"
    ),
    "Outlook": (
        "What guidance or forward-looking commentary did {company} provide "
        "after {fy}? Include any targets or expectations mentioned."
    ),
}


def extract_insights(company: str,
                     fiscal_year: str = "FY2024",
                     use_openai: bool = False) -> dict:
    """
    Generate structured analyst-style insights for a company.

    Fires one RAG question per insight section.
    Each section gets its own trust card for granular confidence tracking.

    Returns
    -------
    {
      "company":     str,
      "fiscal_year": str,
      "sections": {
          "Company Overview":      {"content": str, "confidence": int, "band": str},
          "Financial Highlights":  { ... },
          ...
      },
      "overall_confidence": int,
    }
    """
    sections = {}

    for section_name, question_template in _INSIGHT_SECTIONS.items():
        question = question_template.format(company=company, fy=fiscal_year)
        try:
            from trust_engine import process_answer_package
            # FIX: filter_fiscal_year was missing -- insight sections could
            # retrieve chunks from any fiscal year, mixing data across years
            # in a single "FY2025 insights" response.
            ap = run_query(question, top_k=8, use_openai=use_openai,
                           save=False, filter_company=company,
                           filter_fiscal_year=fiscal_year)
            tc = process_answer_package(ap)
            sections[section_name] = {
                "content":    ap["answer"],
                "confidence": tc["confidence_score"],
                "band":       tc["confidence_band"],
                "decision":   tc["decision"]["action"],
                "trust_card": tc,
            }
        except Exception as e:
            sections[section_name] = {
                "content":    f"Could not retrieve information: {e}",
                "confidence": 0,
                "band":       "Low",
                "decision":   "Review",
                "trust_card": {},
            }

    scores  = [s["confidence"] for s in sections.values() if s["confidence"] > 0]
    overall = int(sum(scores) / len(scores)) if scores else 0

    return {
        "company":            company,
        "fiscal_year":        fiscal_year,
        "sections":           sections,
        "overall_confidence": overall,
    }


# ─────────────────────────────────────────────
# FEATURE 2: FINANCIAL RATIO ANALYSIS
# ─────────────────────────────────────────────

import re as _re

_RATIO_QUERIES = {
    "Revenue":          "What was the total revenue or net sales of {company} in {fy}?",
    "Net Income":       "What was the net income or net profit of {company} in {fy}?",
    "Total Assets":     "What were the total assets of {company} in {fy}?",
    "Total Equity":     "What was the total shareholders equity of {company} in {fy}?",
    "Total Debt":       "What was the total debt or borrowings of {company} in {fy}?",
    "Current Assets":   "What were the current assets of {company} in {fy}?",
    "Current Liab":     "What were the current liabilities of {company} in {fy}?",
    "Operating Income": "What was the operating income or EBIT of {company} in {fy}?",
}


def _is_no_data_answer(text: str) -> bool:
    """
    Detect when the LLM is saying "I don't have this information" rather
    than giving an actual figure. Checking this FIRST prevents the regex
    parser from accidentally matching a comma or stray digit inside an
    apology sentence (e.g. "Cognizant's" matching as a number).

    Also catches raw error/exception text (e.g. "Error: Could not connect
    to Ollama...") as a defense-in-depth measure -- this is the second
    line of defense after _ask()'s connection_failed tracking in
    build_knowledge_graph_data(). Without this check, error message
    fragments like "Error", "Could", "Ollama", "Start" were being picked
    up by the Title-Case fallback in _extract_items() and turned into
    fake graph nodes.
    """
    if not text or not text.strip():
        return True
    no_data_phrases = [
        "does not provide", "do not provide", "context does not",
        "i apologize", "i don't have", "i do not have",
        "cannot find", "could not find", "no information",
        "not available in", "not provided in", "insufficient information",
        "context provided does not", "unable to find",
        # Error/exception text patterns -- never treat these as real content
        "error:", "could not connect", "connection error",
        "traceback", "exception:", "failed to", "timed out",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in no_data_phrases)


def _parse_number(text: str) -> Optional[float]:
    """
    Extract the first financial figure from a text string.
    Returns None if no number found (avoids hallucination).

    CRITICAL: checks _is_no_data_answer() first. Without this guard,
    apology sentences like "I apologize, but Cognizant's financial data..."
    would have the comma in "apologize," and "Cognizant's" falsely matched
    by the regex (a lone comma satisfies [\\d,]+), producing a crash on
    float(',') or, worse, silently returning a garbage non-None value.
    """
    if _is_no_data_answer(text):
        return None

    # Match patterns like $383.3 billion, 21.5%, 85,200
    # NOTE: require at least one DIGIT (not just commas) to avoid the
    # "apologize," false-match bug — \d (not [\d,]) anchors each pattern.
    patterns = [
        r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(billion|million|trillion|bn|mn|b|m|t)\b",
        r"(\d[\d,]*(?:\.\d+)?)\s*(billion|million|trillion|bn|mn|b|m|t)\b",
        r"(\d[\d,]*(?:\.\d+)?)\s*%",
        r"(\d[\d,]*(?:\.\d+)?)",
    ]
    multipliers = {
        "billion": 1e9, "bn": 1e9, "b": 1e9,
        "million": 1e6, "mn": 1e6, "m": 1e6,
        "trillion": 1e12, "t": 1e12,
    }
    for pattern in patterns:
        m = _re.search(pattern, text, _re.IGNORECASE)
        if m:
            digits_only = m.group(1).replace(",", "")
            if not digits_only:   # safety net — should never trigger now
                continue
            try:
                raw = float(digits_only)
            except ValueError:
                continue
            unit = m.group(2).lower() if len(m.groups()) > 1 and m.group(2) else ""
            return raw * multipliers.get(unit, 1)
    return None


def calculate_ratios(company: str,
                     fiscal_year: str = "FY2024",
                     use_openai: bool = False) -> dict:
    """
    Retrieve key financial figures via RAG and compute standard ratios.
    Does NOT hallucinate — if a figure cannot be found, the ratio is marked N/A.

    Computed ratios (when data is available):
      - Profit Margin (%)    = Net Income / Revenue × 100
      - ROA (%)              = Net Income / Total Assets × 100
      - ROE (%)              = Net Income / Total Equity × 100
      - Debt-to-Equity       = Total Debt / Total Equity
      - Current Ratio        = Current Assets / Current Liabilities
      - Operating Margin (%) = Operating Income / Revenue × 100

    Returns
    -------
    {
      "company":     str,
      "fiscal_year": str,
      "raw_figures": { "Revenue": float|None, ... },
      "ratios": {
          "Profit Margin (%)": float|None,
          ...
      },
      "answers":     { "Revenue": str, ... },   ← raw LLM answers for transparency
    }
    """
    raw_figures = {}
    answers     = {}

    for label, question_template in _RATIO_QUERIES.items():
        question = question_template.format(company=company, fy=fiscal_year)
        try:
            # FIX: filter_fiscal_year was missing -- ratio figures (Revenue,
            # Net Income, Total Assets, etc.) could be pulled from different
            # fiscal years for the same "ratio calculation," producing
            # mathematically meaningless ratios (e.g. FY2025 revenue divided
            # by FY2023 total assets).
            ap  = run_query(question, top_k=5, use_openai=use_openai,
                            save=False, filter_company=company,
                            filter_fiscal_year=fiscal_year)
            ans = ap["answer"]
            val = _parse_number(ans)
            raw_figures[label] = val
            answers[label]     = ans
        except Exception as e:
            raw_figures[label] = None
            answers[label]     = f"Error: {e}"

    # Compute ratios — only when both numerator and denominator are available
    def _ratio(num_key, den_key, pct=False):
        n = raw_figures.get(num_key)
        d = raw_figures.get(den_key)
        if n is not None and d and d != 0:
            result = n / d
            return round(result * 100, 2) if pct else round(result, 4)
        return None

    ratios = {
        "Profit Margin (%)":    _ratio("Net Income",       "Revenue",       pct=True),
        "ROA (%)":              _ratio("Net Income",       "Total Assets",  pct=True),
        "ROE (%)":              _ratio("Net Income",       "Total Equity",  pct=True),
        "Debt-to-Equity":       _ratio("Total Debt",       "Total Equity",  pct=False),
        "Current Ratio":        _ratio("Current Assets",   "Current Liab",  pct=False),
        "Operating Margin (%)": _ratio("Operating Income", "Revenue",       pct=True),
    }

    return {
        "company":     company,
        "fiscal_year": fiscal_year,
        "raw_figures": raw_figures,
        "ratios":      ratios,
        "answers":     answers,
    }


# ─────────────────────────────────────────────
# FEATURE 3: COMPANY COMPARISON
# ─────────────────────────────────────────────

_COMPARISON_DIMENSIONS = [
    ("Financial Performance",
     "What were the key financial results (revenue, profit, margins) of {company} in {fy}?"),
    ("Revenue Growth",
     "What was the revenue growth rate of {company} in {fy} compared to prior year?"),
    ("Profitability",
     "How profitable was {company} in {fy}? Include operating margin and net margin."),
    ("Management Commentary",
     "What did {company} management say about strategy and outlook in {fy}?"),
    ("Key Risks",
     "What were the main risks and challenges {company} mentioned in {fy}?"),
    ("Sentiment",
     "What was the overall business sentiment for {company} in {fy} "
     "based on earnings calls and annual reports?"),
]


def compare_companies(companies: List[str],
                      fiscal_year: str = "FY2024",
                      use_openai: bool = False) -> dict:
    """
    Analyst-style comparison of two or more companies across key dimensions.

    For each dimension, fires company-filtered RAG queries so chunks from
    Company A never contaminate Company B's answer.

    Finally asks the LLM to write an overall assessment comparing all companies.

    Returns
    -------
    {
      "companies":   [str, ...],
      "fiscal_year": str,
      "dimensions": {
          "Financial Performance": {
              "Infosys": {"answer": str, "confidence": int},
              "TCS":     { ... },
          },
          ...
      },
      "overall_assessment": str,
      "overall_confidence": int,
    }
    """
    from trust_engine import process_answer_package

    dimensions: dict = {dim: {} for dim, _ in _COMPARISON_DIMENSIONS}

    for company in companies:
        for dim_name, question_template in _COMPARISON_DIMENSIONS:
            question = question_template.format(company=company, fy=fiscal_year)
            try:
                # FIX: filter_fiscal_year was missing here -- comparisons
                # were retrieving chunks from ANY fiscal year for each
                # company, not just the requested one, contaminating the
                # comparison with mixed-year data (e.g. comparing TCS
                # FY2025 revenue against an Infosys chunk that happened
                # to score well but was actually from FY2023).
                ap = run_query(question, top_k=6, use_openai=use_openai,
                               save=False, filter_company=company,
                               filter_fiscal_year=fiscal_year)
                tc = process_answer_package(ap)
                dimensions[dim_name][company] = {
                    "answer":     ap["answer"],
                    "confidence": tc["confidence_score"],
                    "band":       tc["confidence_band"],
                }
            except Exception as e:
                dimensions[dim_name][company] = {
                    "answer": f"Could not retrieve: {e}",
                    "confidence": 0, "band": "Low",
                }

    # Overall assessment — unfiltered by COMPANY so the LLM sees all
    # compared companies together, but DOES filter by fiscal year so
    # the comparison doesn't mix in chunks from unrelated years.
    companies_str = " and ".join(companies)
    summary_q = (
        f"Based on the available financial documents, compare {companies_str} "
        f"for {fiscal_year}. Which company performed better overall in terms of "
        f"revenue, profitability, growth, and business outlook? "
        f"Provide a balanced analyst-style assessment."
    )
    try:
        ap_sum = run_query(summary_q, top_k=10, use_openai=use_openai,
                           save=False, filter_fiscal_year=fiscal_year)
        overall_assessment = ap_sum["answer"]
        from trust_engine import process_answer_package
        tc_sum = process_answer_package(ap_sum)
        overall_conf = tc_sum["confidence_score"]
    except Exception as e:
        overall_assessment = f"Could not generate overall assessment: {e}"
        overall_conf = 0

    return {
        "companies":          companies,
        "fiscal_year":        fiscal_year,
        "dimensions":         dimensions,
        "overall_assessment": overall_assessment,
        "overall_confidence": overall_conf,
    }


# ─────────────────────────────────────────────
# FEATURE 4: REPORT GENERATION
# ─────────────────────────────────────────────

_REPORT_SECTIONS = [
    ("Executive Summary",
     "Write a 3-sentence executive summary of {company}'s financial performance in {fy}."),
    ("Business Overview",
     "Describe {company}'s core business, segments, and key services as of {fy}."),
    ("Financial Performance",
     "Summarise {company}'s revenue, profit, margins, and growth rates in {fy}."),
    ("Operational Highlights",
     "What were the key operational achievements and deal wins for {company} in {fy}?"),
    ("Risks & Challenges",
     "What are the key risks and challenges {company} faces as mentioned in {fy} documents?"),
    ("Analyst Verdict",
     "Based on {fy} performance, provide a short analyst verdict on {company}: "
     "strengths, concerns, and overall outlook. Be balanced and factual."),
]


def generate_report(company: str,
                    fiscal_year: str = "FY2024",
                    use_openai: bool = False) -> dict:
    """
    Generate a short (~1 page) analyst-style research report.

    Each section is a separate RAG call for precision.
    Returns both a structured dict and a ready-to-display markdown string.

    Returns
    -------
    {
      "company":         str,
      "fiscal_year":     str,
      "sections":        { "Executive Summary": {"content": str, "confidence": int}, ... },
      "report_markdown": str,
      "overall_confidence": int,
    }
    """
    from trust_engine import process_answer_package

    sections = {}
    for section_name, question_template in _REPORT_SECTIONS:
        question = question_template.format(company=company, fy=fiscal_year)
        try:
            # FIX: filter_fiscal_year was missing -- report sections could
            # retrieve chunks from any fiscal year, producing a "FY2025
            # report" that actually describes FY2023 risks or FY2024
            # financial performance mixed together.
            ap = run_query(question, top_k=7, use_openai=use_openai,
                           save=False, filter_company=company,
                           filter_fiscal_year=fiscal_year)
            tc = process_answer_package(ap)
            sections[section_name] = {
                "content":    ap["answer"],
                "confidence": tc["confidence_score"],
                "band":       tc["confidence_band"],
                "decision":   tc["decision"]["action"],
            }
        except Exception as e:
            sections[section_name] = {
                "content":    f"Section unavailable: {e}",
                "confidence": 0, "band": "Low", "decision": "Review",
            }

    # Build markdown
    import datetime as _dt
    lines = [
        f"# {company} — Analyst Research Report",
        f"**Fiscal Year:** {fiscal_year}  |  "
        f"**Generated:** {_dt.datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
    ]
    for name, sec in sections.items():
        conf_tag = f"*[Confidence: {sec['confidence']}/100 · {sec['band']}]*"
        lines += [f"## {name}", conf_tag, "", sec["content"], ""]

    scores  = [s["confidence"] for s in sections.values() if s["confidence"] > 0]
    overall = int(sum(scores) / len(scores)) if scores else 0

    return {
        "company":            company,
        "fiscal_year":        fiscal_year,
        "sections":           sections,
        "report_markdown":    "\n".join(lines),
        "overall_confidence": overall,
    }


# ─────────────────────────────────────────────
# FEATURE 5: KNOWLEDGE GRAPH DATA BUILDER
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# FEATURE 5: KNOWLEDGE GRAPH DATA BUILDER
# ─────────────────────────────────────────────

def _extract_items(text: str, max_items: int = 5) -> List[str]:
    """
    Extract short 1-4 word labels from an LLM answer for use as graph nodes.

    The LLM is prompted to return bullet-point lists like:
      - Financial Services
      - Healthcare
      - AI and Cloud

    This function pulls those items and cleans them to short labels.
    If the LLM returns prose instead of bullets, it falls back to
    extracting proper-noun-like phrases (Title Case words).

    Returns empty list if LLM said it has no data.
    """
    if _is_no_data_answer(text):
        return []

    # ── Try bullet points first (ideal case — this is what we prompt for) ──
    bullets = _re.findall(r"[-•*]\s*(.+)", text)
    if bullets:
        clean = []
        for b in bullets[:max_items]:
            # Take only the first part before any colon/dash explanation
            # e.g. "Financial Services: includes banking..." → "Financial Services"
            label = _re.split(r"[:\-–(]", b)[0].strip()
            # Strip trailing punctuation — cap at 50 chars (up from 30)
            # so full names like "Communications Media Technology" aren't cut off
            label = label.rstrip(".,;").strip()[:50]
            if label and len(label) > 2:
                clean.append(label)
        if clean:
            return clean

    # ── Try numbered lists ──────────────────────────────────────────────────
    numbered = _re.findall(r"(?:^|\n)\s*\d+[.)]\s*(.+)", text)
    if numbered:
        clean = []
        for n in numbered[:max_items]:
            label = _re.split(r"[:\-–(]", n)[0].strip().rstrip(".,;")[:50]
            if label and len(label) > 2:
                clean.append(label)
        if clean:
            return clean

    # ── Fallback: extract Title Case phrases from prose ────────────────────
    # Matches 1-4 consecutive Title-Cased or ALL-CAPS words
    # e.g. "Financial Services", "AI", "Cloud Computing"
    title_phrases = _re.findall(
        r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,3})\b", text
    )
    # Filter out noise words that are Title Case but not useful labels
    noise = {"The", "This", "According", "Based", "However", "While",
             "Although", "Since", "During", "For", "With", "From",
             "Their", "These", "Those", "Such", "Any", "All", "Some"}
    clean = []
    seen = set()
    for phrase in title_phrases:
        phrase = phrase.strip()
        if phrase not in noise and phrase not in seen and len(phrase) > 2:
            clean.append(phrase)
            seen.add(phrase)
        if len(clean) >= max_items:
            break

    return clean


def build_knowledge_graph_data(company: str,
                               fiscal_year: str = "FY2024",
                               use_openai: bool = False) -> dict:
    """
    Gather the raw data needed to build a NetworkX knowledge graph.
    Fires 6 targeted RAG questions about the company, each asking for
    short bullet-point labels specifically so node names are clean.

    Returns
    -------
    {
      "company":           str,
      "fiscal_year":       str,
      "segments":          [str, ...],   e.g. ["Financial Services", "Healthcare"]
      "products":          [str, ...],   e.g. ["Cognizant Neuro AI", "TriZetto"]
      "risks":             [str, ...],   e.g. ["Macro uncertainty", "AI disruption"]
      "opportunities":     [str, ...],   e.g. ["Generative AI", "Cloud migration"]
      "management_themes": [str, ...],   e.g. ["NextGen", "Cost discipline"]
      "revenue_label":     str,          e.g. "$19.7 billion"
    }
    """
    connection_failed = {"value": False}   # mutable flag closures can write to

    def _ask(question: str, filter_fy: str = None) -> str:
        try:
            ap = run_query(question, top_k=6, use_openai=use_openai,
                           save=False, filter_company=company,
                           filter_fiscal_year=filter_fy)
            return ap["answer"]
        except ConnectionError as e:
            # Ollama (or whichever local LLM) is unreachable. Mark this so
            # we can raise a clear, single error after all calls instead of
            # silently feeding "Error: Could not connect to Ollama..." text
            # into _extract_items(), which previously turned fragments like
            # "Error", "Could", "Ollama", "Start" into fake graph nodes.
            connection_failed["value"] = True
            return ""   # empty string -> _is_no_data_answer() treats as no data
        except Exception:
            # Any other failure (retrieval error, etc.) -> treat as no data
            # rather than injecting the exception text as graph content.
            return ""

    # Each prompt explicitly asks for a SHORT bullet-point list of labels.
    # "2-4 words each" is the key instruction that prevents prose responses.
    seg_answer = _ask(
        f"List the main business segments or industry verticals of {company} in {fiscal_year}. "
        f"Return ONLY a bullet-point list. Each item should be 2-4 words. No explanations.",
        filter_fy=fiscal_year,
    )
    prod_answer = _ask(
        f"List the key products, platforms, or branded service offerings of {company} in {fiscal_year}. "
        f"Return ONLY a bullet-point list. Each item should be 2-4 words. No explanations.",
        filter_fy=fiscal_year,
    )
    risk_answer = _ask(
        f"List the main risks and challenges facing {company} in {fiscal_year}. "
        f"Return ONLY a bullet-point list. Each risk in 2-5 words. No explanations.",
        filter_fy=fiscal_year,
    )
    opp_answer = _ask(
        f"List the key growth opportunities and strategic priorities for {company} in {fiscal_year}. "
        f"Return ONLY a bullet-point list. Each item in 2-4 words. No explanations.",
        filter_fy=fiscal_year,
    )
    mgmt_answer = _ask(
        f"List the top strategic themes or management priorities for {company} in {fiscal_year}. "
        f"Return ONLY a bullet-point list. Each theme in 2-4 words. No explanations.",
        filter_fy=fiscal_year,
    )
    rev_answer = _ask(
        f"What was the total revenue of {company} in {fiscal_year}? Give just the number.",
        filter_fy=fiscal_year,
    )

    # If EVERY call failed due to a connection error, fail loudly with one
    # clear message instead of silently returning an empty/garbage graph.
    # The dashboard's try/except around build_knowledge_graph_data() will
    # catch this and show st.error() with the real reason.
    if connection_failed["value"]:
        raise ConnectionError(
            "Could not connect to the local LLM (Ollama) while building the "
            "knowledge graph. Start it with: ollama serve — or enable "
            "'Use OpenAI' in the sidebar."
        )

    # Extract the revenue figure for the central metric node label
    # Try INR crore format first (Infosys), then USD millions/billions
    rev_match = _re.search(
        r"₹\s*[\d,]+(?:\.\d+)?\s*(?:crore|billion|million)?|"
        r"\$\s*[\d,.]+\s*(?:billion|million|B|M)?",
        rev_answer, _re.IGNORECASE
    )
    if rev_match:
        rev_label = rev_match.group(0).strip()
    else:
        # Fallback: grab first number-like token
        num_match = _re.search(r"[\d,]+(?:\.\d+)?\s*(?:crore|billion|million)?", rev_answer)
        rev_label = num_match.group(0).strip() if num_match else "Revenue (N/A)"

    return {
        "company":           company,
        "fiscal_year":       fiscal_year,
        "segments":          _extract_items(seg_answer, 5),
        "products":          _extract_items(prod_answer, 4),
        "risks":             _extract_items(risk_answer, 4),
        "opportunities":     _extract_items(opp_answer, 4),
        "management_themes": _extract_items(mgmt_answer, 3),
        "revenue_label":     rev_label,
    }
