"""
langgraph_flow.py
─────────────────
Simple LangGraph workflow for the Financial Research Assistant.

Flow:
  START → classify_node → route_node → trust_node → response_node → END

Each node is a plain Python function that receives and returns a state dict.
LangGraph wires them together — that's all it does here.

Why LangGraph?
  - Teaches the concept of stateful, node-based AI workflows
  - Each step is testable in isolation
  - Easy to extend with new routes later
  - Portfolio-worthy: shows awareness of agentic frameworks

Install:
  pip install langgraph

Supported routes:
  qa          → standard RAG question answering
  analysis    → extract_insights() — company deep dive
  ratio       → calculate_ratios() — financial ratio computation
  comparison  → compare_companies() — analyst comparison
  report      → generate_report() — 1-page research report
  dashboard   → signals dashboard to open analytics page
  graph       → signals dashboard to open knowledge graph page
"""

from __future__ import annotations
from typing import TypedDict, Any

# ── LangGraph imports ─────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("[WARNING] langgraph not installed. Run: pip install langgraph")

# ── Pipeline imports ──────────────────────────────────────────────────────────
from rag_pipeline import (
    run_query,
    classify_query,
    extract_insights,
    calculate_ratios,
    compare_companies,
    generate_report,
)
from trust_engine import process_answer_package


# ─────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────
class ResearchState(TypedDict):
    """
    The state dict that flows through every node.
    Each node reads from it and writes back to it.
    """
    question:     str           # original user question
    route:        str           # classified route: qa | analysis | ratio | ...
    company:      str           # extracted company name (may be empty)
    companies:    list[str]     # list of companies (for comparison)
    fiscal_year:  str           # e.g. "FY2024"
    use_openai:   bool          # LLM backend choice
    raw_result:   Any           # output from the feature function
    trust_card:   dict          # trust evaluation result
    final_answer: str           # the text shown to the user in chat
    needs_page:   str           # "graph" | "report" | "dashboard" | ""


# ─────────────────────────────────────────────
# HELPER: extract company name from question
# ─────────────────────────────────────────────
import re

# Known companies in the dataset — extend this list as needed
_KNOWN_COMPANIES = ["infosys", "tcs", "cognizant", "wipro", "hcl", "accenture"]

def _extract_company(question: str) -> str:
    """Pull the first known company name out of the question."""
    q = question.lower()
    for company in _KNOWN_COMPANIES:
        if company in q:
            return company.title()
    return ""

def _extract_companies(question: str) -> list[str]:
    """Pull all company names from a comparison question."""
    q = question.lower()
    found = [c.title() for c in _KNOWN_COMPANIES if c in q]
    return found if found else []

def _extract_fiscal_year(question: str) -> str:
    """Extract a fiscal year like FY2024 or 2023 from the question."""
    m = re.search(r"(FY\s?\d{4}|\d{4})", question, re.IGNORECASE)
    if m:
        fy = m.group(1).replace(" ", "").upper()
        return fy if fy.startswith("FY") else f"FY{fy}"
    return "FY2024"   # sensible default


# ─────────────────────────────────────────────
# NODE 1: CLASSIFY
# ─────────────────────────────────────────────
def classify_node(state: ResearchState) -> ResearchState:
    """
    Determine what kind of request this is.
    Sets state["route"] to one of: qa | analysis | ratio | comparison | report | graph | dashboard
    Also extracts company name(s) and fiscal year from the question.
    """
    question = state["question"]
    route    = classify_query(question)

    state["route"]       = route
    state["company"]     = _extract_company(question)
    state["companies"]   = _extract_companies(question)
    state["fiscal_year"] = _extract_fiscal_year(question)

    print(f"[LangGraph] classify_node → route={route}, "
          f"company={state['company']}, fy={state['fiscal_year']}")
    return state


# ─────────────────────────────────────────────
# NODE 2: ROUTE (feature execution)
# ─────────────────────────────────────────────
def route_node(state: ResearchState) -> ResearchState:
    """
    Execute the correct feature function based on the classified route.
    Stores the raw result in state["raw_result"].

    For graph/dashboard requests, sets state["needs_page"] so the dashboard
    knows to open the appropriate Streamlit page automatically.
    """
    route      = state["route"]
    company    = state["company"]
    companies  = state["companies"]
    fy         = state["fiscal_year"]
    use_openai = state.get("use_openai", False)

    state["needs_page"] = ""   # default: no special page needed

    if route == "analysis" and company:
        state["raw_result"] = extract_insights(company, fy, use_openai)
        # Build a text summary for the chat bubble
        sections = state["raw_result"]["sections"]
        summary  = sections.get("Financial Highlights", {}).get("content", "")
        overview = sections.get("Company Overview", {}).get("content", "")
        state["final_answer"] = (
            f"**{company} — {fy} Insights**\n\n"
            f"**Overview:** {overview}\n\n"
            f"**Financial Highlights:** {summary}\n\n"
            f"*Full breakdown available in the Insights section below.*"
        )

    elif route == "ratio" and company:
        state["raw_result"] = calculate_ratios(company, fy, use_openai)
        ratios = state["raw_result"]["ratios"]
        ratio_lines = []
        for name, val in ratios.items():
            display = f"{val:.2f}" if val is not None else "N/A (insufficient data)"
            ratio_lines.append(f"- **{name}:** {display}")
        state["final_answer"] = (
            f"**{company} — Financial Ratios ({fy})**\n\n"
            + "\n".join(ratio_lines)
            + "\n\n*Ratios computed from retrieved financial documents only. "
            "N/A means the figure could not be found in the documents.*"
        )

    elif route == "comparison" and len(companies) >= 2:
        state["raw_result"] = compare_companies(companies, fy, use_openai)
        state["final_answer"] = (
            f"**Comparison: {' vs '.join(companies)} — {fy}**\n\n"
            + state["raw_result"]["overall_assessment"]
            + "\n\n*Detailed dimension-by-dimension breakdown available below.*"
        )

    elif route == "report" and company:
        state["raw_result"]   = generate_report(company, fy, use_openai)
        state["final_answer"] = (
            f"Research report generated for **{company} ({fy})**. "
            f"Opening report page…"
        )
        state["needs_page"] = "report"

    elif route == "graph" and company:
        state["raw_result"]   = {"company": company, "fiscal_year": fy}
        state["final_answer"] = (
            f"Building knowledge graph for **{company} ({fy})**. "
            f"Opening graph page…"
        )
        state["needs_page"] = "graph"

    elif route == "dashboard" and company:
        state["raw_result"]   = {"company": company, "fiscal_year": fy}
        state["final_answer"] = (
            f"Generating analytics dashboard for **{company} ({fy})**. "
            f"Opening dashboard page…"
        )
        state["needs_page"] = "dashboard"

    else:
        # Default: standard RAG question answering
        try:
            fc = company if company else None

            # Apply fiscal year filter only when a year was explicitly
            # mentioned in the user's question. We check by seeing if the
            # extracted fy string (e.g. "FY2024") or its plain year part
            # ("2024") appears in the original question text.
            fy_in_question = (
                fy and (
                    fy in state["question"] or
                    fy.replace("FY", "") in state["question"]
                )
            )
            fy_filter = fy if fy_in_question else None

            ap = run_query(
                state["question"],
                top_k=8,
                use_openai=use_openai,
                save=False,
                filter_company=fc,
                filter_fiscal_year=fy_filter,
            )
            state["raw_result"]   = ap
            state["final_answer"] = ap["answer"]
        except Exception as e:
            state["raw_result"]   = {}
            state["final_answer"] = f"Error retrieving answer: {e}"

    print(f"[LangGraph] route_node → needs_page={state['needs_page']}")
    return state


# ─────────────────────────────────────────────
# NODE 3: TRUST EVALUATION
# ─────────────────────────────────────────────
def trust_node(state: ResearchState) -> ResearchState:
    """
    Run the trust engine on the raw result.
    For qa route: runs directly on the answer_package.
    For other routes: wraps the final_answer in a minimal package for scoring.
    """
    route      = state["route"]
    raw_result = state.get("raw_result", {})

    try:
        if route == "qa" and isinstance(raw_result, dict) and "answer" in raw_result:
            # Full trust evaluation on the answer_package
            state["trust_card"] = process_answer_package(raw_result)

        elif route in ("analysis", "ratio", "comparison", "report"):
            # For multi-call features, use overall_confidence from the result
            # and build a lightweight trust card for display
            overall_conf = raw_result.get("overall_confidence", 50) if isinstance(raw_result, dict) else 50
            band = "High" if overall_conf >= 85 else "Medium" if overall_conf >= 70 else "Low"
            action = "Approve" if overall_conf >= 85 else "Warn" if overall_conf >= 70 else "Review"
            state["trust_card"] = {
                "confidence_score": overall_conf,
                "confidence_band":  band,
                "decision":         {"action": action, "reason": f"Overall confidence: {overall_conf}/100"},
                "issues":           [],
                "rag_output":       {"question": state["question"],
                                     "answer":   state["final_answer"],
                                     "retrieved_chunks": [], "citations": [],
                                     "metadata": {}},
            }
        else:
            # graph / dashboard routes — no trust evaluation needed
            state["trust_card"] = {
                "confidence_score": 100,
                "confidence_band":  "High",
                "decision":         {"action": "Approve", "reason": "Navigation request."},
                "issues":           [],
                "rag_output":       {"question": state["question"], "answer": state["final_answer"],
                                     "retrieved_chunks": [], "citations": [], "metadata": {}},
            }

    except Exception as e:
        state["trust_card"] = {
            "confidence_score": 0,
            "confidence_band":  "Low",
            "decision":         {"action": "Review", "reason": f"Trust evaluation error: {e}"},
            "issues":           [str(e)],
            "rag_output":       {},
        }

    print(f"[LangGraph] trust_node → confidence={state['trust_card']['confidence_score']}")
    return state


# ─────────────────────────────────────────────
# NODE 4: RESPONSE ASSEMBLY
# ─────────────────────────────────────────────
def response_node(state: ResearchState) -> ResearchState:
    """
    Final node — packages everything for the dashboard to render.
    The dashboard reads state["final_answer"] and state["trust_card"].
    Nothing changes here — just a clean termination point.
    """
    print(f"[LangGraph] response_node → done. "
          f"answer length={len(state.get('final_answer',''))}")
    return state


# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────
def build_graph():
    """
    Assemble the 4-node LangGraph workflow.
    Returns a compiled graph ready to invoke.

    Graph structure:
      START → classify_node → route_node → trust_node → response_node → END
    """
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph not installed. Run: pip install langgraph")

    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("classify",  classify_node)
    graph.add_node("route",     route_node)
    graph.add_node("trust",     trust_node)
    graph.add_node("response",  response_node)

    # Wire edges — linear flow
    graph.set_entry_point("classify")
    graph.add_edge("classify", "route")
    graph.add_edge("route",    "trust")
    graph.add_edge("trust",    "response")
    graph.add_edge("response", END)

    return graph.compile()


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def run_research_flow(question: str, use_openai: bool = False) -> ResearchState:
    """
    Main entry point called by dashboard.py.

    Parameters
    ----------
    question   : the user's natural-language question
    use_openai : True → OpenAI backend, False → Ollama + local embeddings

    Returns
    -------
    ResearchState dict with:
      - final_answer : str  — text to show in chat bubble
      - trust_card   : dict — confidence + validation results
      - needs_page   : str  — "graph" | "report" | "dashboard" | ""
      - raw_result   : Any  — structured data for the relevant Streamlit page
      - route        : str  — what type of query this was
    """
    graph = build_graph()

    initial_state: ResearchState = {
        "question":     question,
        "route":        "",
        "company":      "",
        "companies":    [],
        "fiscal_year":  "FY2024",
        "use_openai":   use_openai,
        "raw_result":   {},
        "trust_card":   {},
        "final_answer": "",
        "needs_page":   "",
    }

    result = graph.invoke(initial_state)
    return result


# ─────────────────────────────────────────────
# FALLBACK: when LangGraph is not installed
# ─────────────────────────────────────────────
def run_research_flow_simple(question: str, use_openai: bool = False) -> dict:
    """
    Fallback that runs the same logic without LangGraph.
    Called automatically by dashboard.py if langgraph is not installed.
    Produces an identical output dict to run_research_flow().
    """
    state: dict = {
        "question":     question,
        "route":        "",
        "company":      "",
        "companies":    [],
        "fiscal_year":  "FY2024",
        "use_openai":   use_openai,
        "raw_result":   {},
        "trust_card":   {},
        "final_answer": "",
        "needs_page":   "",
    }
    state = classify_node(state)
    state = route_node(state)
    state = trust_node(state)
    state = response_node(state)
    return state


def run_flow(question: str, use_openai: bool = False) -> dict:
    """
    Smart dispatcher: uses LangGraph if available, fallback otherwise.
    This is what dashboard.py calls.
    """
    if LANGGRAPH_AVAILABLE:
        return run_research_flow(question, use_openai)
    else:
        return run_research_flow_simple(question, use_openai)
