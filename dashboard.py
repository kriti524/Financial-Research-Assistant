"""
dashboard.py  —  Financial Research Assistant (Chat-First, Single Window)
===========================================================================
The chatbot is the PRIMARY interface, always visible in the first tab.
Report, Knowledge Graph, and Analytics Dashboard render in OTHER TABS
of the SAME window — no sidebar pages, no separate browser pages.

When the user asks for a report/graph/dashboard in chat, the matching
tab is automatically activated so the result appears immediately.
"""

import re
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Core imports ──────────────────────────────────────────────────────────────
from langgraph_flow import run_flow, LANGGRAPH_AVAILABLE
from rag_pipeline    import (
    run_query, calculate_ratios, generate_report, build_knowledge_graph_data,
)
from hitl_queue      import (
    add_to_queue, get_queue, get_pending_items, submit_review, VALID_ACTIONS,
)
from vector_store    import load_chroma_collection, get_available_fiscal_years

# ── Optional libraries with graceful fallback ─────────────────────────────────
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

try:
    from pyvis.network import Network as PyvisNetwork
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
code, pre { font-family: 'IBM Plex Mono', monospace; }

.bubble-user {
  background:#1864ab; color:#fff; padding:10px 16px;
  border-radius:18px 18px 4px 18px; margin:6px 0 6px 20%;
  font-size:0.92rem; line-height:1.5;
}
.bubble-assistant {
  background:#f1f3f5; color:#212529; padding:12px 16px;
  border-radius:18px 18px 18px 4px; margin:6px 20% 6px 0;
  font-size:0.92rem; line-height:1.7;
}
.trust-bar { display:flex; align-items:center; gap:10px; padding:7px 14px;
  border-radius:8px; margin:6px 0; font-size:0.82rem; font-weight:600; }
.trust-high   { background:#d3f9d8; color:#0a6640; border:1px solid #b2f2bb; }
.trust-medium { background:#fff3bf; color:#7c5c0a; border:1px solid #ffe066; }
.trust-low    { background:#ffe3e3; color:#7c1616; border:1px solid #ffc9c9; }
.pill { padding:2px 10px; border-radius:20px; font-weight:700; font-size:0.75rem; }
.pill-approve { background:#1864ab; color:#fff; }
.pill-warn    { background:#e67700; color:#fff; }
.pill-review  { background:#862e2e; color:#fff; }
.badge-high   { background:#0a6640; color:#d3f9d8; padding:2px 9px; border-radius:20px; font-weight:700; font-size:0.75rem; }
.badge-medium { background:#7c5c0a; color:#fff3bf; padding:2px 9px; border-radius:20px; font-weight:700; font-size:0.75rem; }
.badge-low    { background:#7c1616; color:#ffe3e3; padding:2px 9px; border-radius:20px; font-weight:700; font-size:0.75rem; }
.score-bar-wrap { background:#e9ecef; border-radius:6px; height:12px; overflow:hidden; margin-top:3px; }
.score-bar-fill { height:12px; border-radius:6px; }
.chunk-card { border-left:4px solid #339af0; background:#f8f9fa; padding:9px 13px;
  border-radius:0 8px 8px 0; margin-bottom:8px; font-size:0.83rem; line-height:1.55; }
.insight-card { border-left:4px solid #339af0; background:#f8f9fa;
  padding:12px 16px; border-radius:0 8px 8px 0; margin-bottom:10px; }
.insight-high   { border-left-color:#2f9e44; }
.insight-medium { border-left-color:#f08c00; }
.insight-low    { border-left-color:#c92a2a; }
.route-tag { display:inline-block; font-size:0.68rem; font-weight:700;
  padding:1px 8px; border-radius:10px; margin-right:6px;
  background:#e9ecef; color:#495057; text-transform:uppercase; }
.graph-node-tag { display:inline-block; padding:3px 11px; border-radius:14px;
  font-size:0.78rem; font-weight:600; color:white; margin:3px 4px 3px 0; }
hr.light { border:none; border-top:1px solid #dee2e6; margin:0.7rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "chat_history":  [],
    "use_openai":    False,
    "active_tab":    "💬 Chat",   # which tab should be active right now
    # cached results for the other tabs — populated by chat triggers
    "report_data":   None,
    "graph_data":    None,
    "dashboard_data": None,
    "dashboard_company": "",
    "dashboard_fys":     "FY2022, FY2023, FY2024",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────

def _trust_bar_html(score: int, band: str, decision: str) -> str:
    cls_map  = {"High":"trust-high","Medium":"trust-medium","Low":"trust-low"}
    pill_map = {"Approve":"pill-approve","Warn":"pill-warn","Review":"pill-review"}
    icon_map = {"High":"✅","Medium":"⚠️","Low":"🔴"}
    cls  = cls_map.get(band, "trust-low")
    pill = pill_map.get(decision, "pill-review")
    icon = icon_map.get(band, "🔴")
    return (
        f"<div class='trust-bar {cls}'>{icon} "
        f"Trust: <strong>{score}/100</strong> &nbsp;·&nbsp; "
        f"<strong>{band}</strong> &nbsp;·&nbsp; "
        f"<span class='pill {pill}'>{decision}</span></div>"
    )


def _score_bar(label: str, score: int, max_score: int, reason: str = "") -> None:
    pct   = int(score / max_score * 100) if max_score else 0
    color = "#2f9e44" if pct >= 70 else "#f08c00" if pct >= 45 else "#c92a2a"
    st.markdown(f"""
        <div style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;
                      font-size:0.78rem;font-weight:600;color:#495057;">
            <span>{label}</span><span>{score}/{max_score}</span>
          </div>
          <div class="score-bar-wrap">
            <div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>
          </div>
          <div style="font-size:0.72rem;color:#868e96;margin-top:1px;">{reason}</div>
        </div>""", unsafe_allow_html=True)


def _render_trust_details(tc: dict, idx: str) -> None:
    ap = tc.get("rag_output", {})

    with st.expander("📎 Citations & Evidence", expanded=False):
        cit = tc.get("validation_results", {}).get("citations", {})
        for r in cit.get("results", []):
            icon = "✅" if r["valid"] else "❌"
            st.markdown(f"{icon} `{r['chunk_id']}`")
            for issue in r.get("issues", []):
                st.caption(f"  ⚠ {issue}")
        if not cit.get("results"):
            st.caption("No citations to validate.")

    with st.expander("📊 Confidence Breakdown", expanded=False):
        breakdown = tc.get("confidence_breakdown", {})
        label_map = {
            "source_quality":"Source Quality",
            "retrieval_relevance":"Retrieval Relevance",
            "citation_completeness":"Citation Completeness",
            "cross_source_agreement":"Cross-Source Agreement",
            "calculation_validation":"Calculation Validation",
        }
        for key_name, display in label_map.items():
            entry = breakdown.get(key_name, {})
            _score_bar(display, entry.get("score",0),
                       entry.get("max",0), entry.get("reason",""))

    with st.expander("📄 Retrieved Chunks", expanded=False):
        for chunk in ap.get("retrieved_chunks", []):
            sim = chunk.get("similarity_score", 0)
            sc  = "#2f9e44" if sim >= 0.8 else "#f08c00" if sim >= 0.6 else "#c92a2a"
            st.markdown(f"""
                <div class="chunk-card">
                  <strong>{chunk.get('chunk_id','?')}</strong>
                  &nbsp;|&nbsp;{chunk.get('company_name','?')}
                  &nbsp;|&nbsp;{chunk.get('fiscal_year','?')}
                  &nbsp;|&nbsp;{chunk.get('source_type','?').upper()}
                  &nbsp;|&nbsp;Page:{chunk.get('page_number','null')}
                  &nbsp;|&nbsp;<span style="color:{sc};font-weight:700;">
                    Sim:{sim:.2f}</span><br>
                  <span style="color:#495057;">{chunk.get('text','')[:280]}…</span>
                </div>""", unsafe_allow_html=True)

    if tc.get("validation_results"):
        with st.expander("🔍 Validation Details", expanded=False):
            fin = tc["validation_results"].get("financials", {})
            if fin.get("warnings"):
                for w in fin["warnings"]: st.warning(w)
            else:
                st.success("No financial figure warnings.")

            contra = tc["validation_results"].get("contradictions", {})
            if contra.get("has_contradictions"):
                for flag in contra.get("flags", []):
                    st.error(f"**{flag['type']}** — {flag['detail']}")
            else:
                st.success("No contradictions detected.")

            fwd = tc["validation_results"].get("forward_looking", {})
            if fwd.get("detected"):
                st.warning(f"⚠ Forward-looking language: {', '.join(fwd.get('matches',[]))}")

    with st.expander("👤 HITL Actions", expanded=False):
        decision = tc.get("decision", {})
        action   = decision.get("action", "?")
        pill_cls = {"Approve":"pill-approve","Warn":"pill-warn",
                    "Review":"pill-review"}.get(action,"pill-review")
        st.markdown(
            f"<span class='pill {pill_cls}'>{action}</span> &nbsp; {decision.get('reason','')}",
            unsafe_allow_html=True
        )
        st.markdown("")
        if st.button("➕ Add to HITL Queue", key=f"hitl_{idx}"):
            rid = add_to_queue(tc)
            st.success(f"Added. Review ID: `{rid}`")

    with st.expander("⚙ Metadata", expanded=False):
        meta = ap.get("metadata", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Model",     meta.get("model_used","—"))
        c2.metric("Embedding", meta.get("embedding_model","—"))
        c3.metric("Top-K",     meta.get("retrieval_top_k","—"))
        with st.expander("Raw JSON", expanded=False):
            st.json(tc)


def _render_structured_result(entry: dict, idx: str) -> None:
    route      = entry.get("route", "qa")
    raw_result = entry.get("raw_result", {})
    if not isinstance(raw_result, dict):
        return

    if route == "analysis":
        with st.expander("📊 Full Insight Breakdown", expanded=False):
            sections = raw_result.get("sections", {})
            for name, sec in sections.items():
                band  = sec.get("band","Low")
                conf  = sec.get("confidence",0)
                color = "#2f9e44" if band=="High" else "#f08c00" if band=="Medium" else "#c92a2a"
                cls   = f"insight-card insight-{band.lower()}"
                st.markdown(
                    f"<div class='{cls}'>"
                    f"<div style='font-weight:700;font-size:0.86rem;color:#212529;'>{name} "
                    f"<span style='font-size:0.70rem;color:{color};'>({conf}/100)</span></div>"
                    f"<div style='font-size:0.82rem;color:#495057;margin-top:4px;line-height:1.6;'>"
                    f"{sec.get('content','')[:500]}{'…' if len(sec.get('content',''))>500 else ''}"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

    elif route == "ratio":
        with st.expander("🔢 Ratio Details", expanded=False):
            ratios = raw_result.get("ratios", {})
            valid  = {k: v for k, v in ratios.items() if v is not None}
            na     = {k: v for k, v in ratios.items() if v is None}
            if valid:
                cols = st.columns(min(len(valid), 3))
                for col, (name, val) in zip(cols * 10, valid.items()):
                    suffix = "%" if "%" in name else ("x" if "Ratio" in name else "")
                    col.metric(name, f"{val:.2f}{suffix}")
            if na:
                st.caption("N/A (data not found): " + ", ".join(na.keys()))
            with st.expander("Raw LLM answers per figure", expanded=False):
                for label, ans in raw_result.get("answers", {}).items():
                    st.markdown(f"**{label}:** {ans[:200]}")

    elif route == "comparison":
        with st.expander("📊 Dimension-by-Dimension Comparison", expanded=False):
            dims = raw_result.get("dimensions", {})
            for dim_name, company_data in dims.items():
                st.markdown(f"**{dim_name}**")
                for company, data in company_data.items():
                    conf   = data.get("confidence", 0)
                    color  = "#2f9e44" if conf>=70 else "#f08c00" if conf>=50 else "#c92a2a"
                    answer = data.get("answer","")[:300]
                    st.markdown(
                        f"<div style='margin-left:12px;margin-bottom:4px;'>"
                        f"<span style='font-weight:700;color:#1864ab;'>{company}</span> "
                        f"<span style='font-size:0.72rem;color:{color};'>({conf}/100)</span><br>"
                        f"<span style='font-size:0.83rem;color:#495057;'>{answer}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                st.markdown("<hr class='light'>", unsafe_allow_html=True)


def _extract_number(text: str):
    """
    Extract the first financial figure from text, returning a value
    ALWAYS normalized to plain USD-equivalent units (so downstream code
    can safely divide by 1e9 for "billions" display).

    FIX (rupee/crore bug): Infosys reports revenue in Indian Rupees using
    "crore" (1 crore = 10,000,000) as the unit, e.g. "Revenue from
    operations: ₹153,670 crore". The old version of this function had NO
    rupee/crore handling at all — it didn't recognize ₹ as a currency
    symbol or "crore"/"lakh" as unit words, so it fell through to the
    bare-number fallback and returned the raw figure "153670" with NO
    multiplier applied. Dashboard code then divided that by 1e9 expecting
    a raw-dollar value, producing 153670 / 1e9 ≈ 0.00015 → displayed as
    "$0.0B" even though the real value was largely correct, just off by
    a massive unit-conversion factor.

    NOTE: applies a rough fixed INR→USD rate (~83) for chart proportionality
    only — not a live exchange rate. The purpose is purely to make TREND
    CHARTS show proportionally correct shapes across years and across
    companies reporting in different currencies, instead of one currency's
    figures vanishing to zero or ballooning to an absurd magnitude. For
    exact reported figures in their original currency, the Raw Data tab
    still shows the original LLM answer text unmodified.
    """
    # ── Indian Rupee with crore/lakh unit (check first — most specific) ──────
    # 1 crore = 1,00,00,000 = 10 million ; 1 lakh = 100,000
    #
    # SECOND FIX: my first version of this fix multiplied by the crore/lakh
    # factor but treated 1 INR as numerically equal to 1 USD, producing
    # nonsense like "$1536.7B" for Infosys (whose real revenue is ~$18-19B).
    # An approximate INR→USD rate must be applied too, otherwise the chart's
    # /1e9 "billions" divisor (calibrated for raw USD) produces a magnitude
    # error of roughly 83x (the typical USD/INR exchange rate). This is a
    # ROUGH approximation for chart proportionality only — not a live FX
    # rate — but it keeps trend lines in a sane, comparable range instead of
    # showing either $0.0B (old bug) or $1500B (first fix attempt).
    INR_TO_USD = 1 / 83.0   # approximate, fixed rate — good enough for chart shape
    m = re.search(
        r"₹\s*([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|lac)\b",
        text, re.IGNORECASE
    )
    if not m:
        # Same units sometimes appear without the ₹ symbol in LLM answers
        m = re.search(
            r"([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|lac)\b",
            text, re.IGNORECASE
        )
    if m:
        val  = float(m.group(1).replace(",", ""))
        unit = m.group(2).lower()
        mult = {"crore": 1e7, "cr": 1e7, "lakh": 1e5, "lac": 1e5}.get(unit, 1)
        return val * mult * INR_TO_USD

    # ── USD with $ sign or billion/million word (existing logic) ─────────────
    m = re.search(
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|bn|mn|b|m)?",
        text, re.IGNORECASE
    )
    if not m:
        m = re.search(
            r"([\d,]+(?:\.\d+)?)\s*(billion|million|bn|mn|b|m)\b",
            text, re.IGNORECASE
        )
    if m:
        val  = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        mult = {"billion":1e9,"bn":1e9,"b":1e9,
                "million":1e6,"mn":1e6,"m":1e6}.get(unit, 1)
        return val * mult

    # No $ or unit found — fall back to scanning all numbers and picking
    # the LARGEST plausible one, skipping bare years.
    #
    # FIX: returning the FIRST number found was unreliable — a date like
    # "December 31, 2024" has "31" appear before the real metric value
    # "336,800", so the old first-match strategy grabbed the day-of-month
    # instead of the headcount figure. Financial metrics we ask about
    # (revenue, headcount, net income) are almost always the LARGEST
    # number in a sentence; incidental numbers (dates, page refs, small
    # percentages already handled by _extract_pct) are smaller. Bare years
    # (1900-2099, exactly 4 digits, no thousands grouping) are still
    # always excluded regardless of size.
    candidates = []
    for raw_match in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        candidate   = raw_match.group(0).replace(",", "")
        digit_count = len(re.sub(r"\D", "", candidate))
        try:
            val = float(candidate)
        except ValueError:
            continue
        is_bare_year = (1900 <= val <= 2099 and digit_count == 4)
        if is_bare_year:
            continue
        candidates.append(val)

    return max(candidates) if candidates else None


def _extract_pct(text: str):
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*%", text)
    return float(m.group(1).replace(",","")) if m else None


def _gather_trend(company: str, years: list, metric: str,
                  use_openai: bool, extract_fn=_extract_number):
    """
    Gather a metric across multiple fiscal years for a trend chart.

    FIX: previously only filter_company was passed to run_query(), with
    no fiscal year filter. Revenue and Headcount happened to work because
    those terms appear in distinctive, easily-matched chunks even without
    a year filter, but "net income" and "operating margin" appear in
    generic financial-statement boilerplate across MANY years, so without
    a year filter the retrieved chunks were a mix of years and the LLM
    correctly said "data not found" rather than guess.
    """
    x_vals, y_vals = [], []
    for fy in years:
        try:
            ap = run_query(f"What was the {metric} of {company} in {fy}?",
                           top_k=5, use_openai=use_openai, save=False,
                           filter_company=company,
                           filter_fiscal_year=fy)
            val = extract_fn(ap["answer"])
        except Exception:
            val = None
        if val is not None:
            x_vals.append(fy)
            y_vals.append(val)
    return x_vals, y_vals


# ─────────────────────────────────────────────
# SIDEBAR  (settings only — no page navigation here)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Financial Research")
    st.markdown("*RAG · Trust Layer · LangGraph*")
    st.markdown("---")
    st.markdown("**Settings**")

    st.session_state.use_openai = st.toggle(
        "Use OpenAI", value=st.session_state.use_openai,
        help="ON = GPT-4o + OpenAI embeddings. OFF = Ollama + local (CPU)."
    )

    if LANGGRAPH_AVAILABLE:
        st.success("✅ LangGraph active")
    else:
        st.warning("⚠ LangGraph not installed\n`pip install langgraph`\n(fallback active)")

    st.markdown("---")
    pending = len(get_pending_items())
    if pending:
        st.warning(f"⏳ {pending} pending review")
    else:
        st.success("✅ Queue clear")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Try these queries:**")
    st.caption(
        "💬 What was Infosys revenue FY2024?\n"
        "📊 Analyze Infosys FY2024\n"
        "🔢 Calculate ratios for TCS FY2023\n"
        "⚖ Compare TCS and Infosys\n"
        "📝 Generate report for Cognizant\n"
        "🕸 Show knowledge graph for Infosys\n"
        "📈 Show dashboard for TCS"
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW — ALL FEATURES AS TABS IN ONE VIEW (Option 1: same window)
# ═════════════════════════════════════════════════════════════════════════════
st.title("📊 Financial Research Assistant")

tab_chat, tab_report, tab_graph, tab_dashboard, tab_hitl = st.tabs(
    ["💬 Chat", "📝 Report", "🕸 Knowledge Graph", "📈 Dashboard", "🗂 HITL Queue"]
)


# ─────────────────────────────────────────────
# TAB 1: CHAT  (primary interface)
# ─────────────────────────────────────────────
with tab_chat:
    st.caption(
        "Ask anything. The assistant automatically routes your question. "
        + ("*(LangGraph active)*" if LANGGRAPH_AVAILABLE else "*(simple fallback active)*")
    )

    for idx, entry in enumerate(st.session_state.chat_history):
        tc       = entry.get("trust_card", {})
        score    = tc.get("confidence_score", 0)
        band     = tc.get("confidence_band", "Low")
        decision = tc.get("decision", {}).get("action", "Review")
        route    = entry.get("route", "qa")

        st.markdown(f"<div class='bubble-user'>🧑 {entry['question']}</div>",
                    unsafe_allow_html=True)

        route_colors = {
            "qa":"#495057","analysis":"#1864ab","ratio":"#2f9e44",
            "comparison":"#862e2e","report":"#5f3dc4",
            "graph":"#0c8599","dashboard":"#e67700",
        }
        route_color = route_colors.get(route, "#495057")
        st.markdown(
            f"<span class='route-tag' style='background:{route_color};color:white;'>{route}</span>",
            unsafe_allow_html=True
        )
        st.markdown(f"<div class='bubble-assistant'>🤖 {entry['answer']}</div>",
                    unsafe_allow_html=True)
        st.markdown(_trust_bar_html(score, band, decision), unsafe_allow_html=True)

        _render_structured_result(entry, str(idx))

        if "validation_results" in tc or "confidence_breakdown" in tc:
            _render_trust_details(tc, str(idx))

        needs_page = entry.get("needs_page", "")
        if needs_page == "report":
            st.success("📝 Report ready — see the **Report** tab above.")
        elif needs_page == "graph":
            st.success("🕸 Graph ready — see the **Knowledge Graph** tab above.")
        elif needs_page == "dashboard":
            st.success("📈 Dashboard ready — see the **Dashboard** tab above.")

        st.markdown("<hr class='light'>", unsafe_allow_html=True)

    user_input = st.chat_input(
        "Ask a financial question, request an analysis, comparison, report, or graph…"
    )

    if user_input:
        st.markdown(f"<div class='bubble-user'>🧑 {user_input}</div>", unsafe_allow_html=True)

        with st.spinner("🔍 Processing…"):
            try:
                state = run_flow(user_input, use_openai=st.session_state.use_openai)
                needs = state.get("needs_page", "")

                # Pre-populate the OTHER tabs' data — no navigation needed,
                # the data is just sitting there ready when the user clicks the tab.
                if needs == "graph":
                    company = state.get("company", "")
                    fy      = state.get("fiscal_year", "FY2024")
                    st.session_state["graph_data"] = build_knowledge_graph_data(
                        company, fy, st.session_state.use_openai
                    )
                elif needs == "report":
                    st.session_state["report_data"] = state.get("raw_result")
                elif needs == "dashboard":
                    st.session_state["dashboard_company"] = state.get("company", "")
                    st.session_state["dashboard_data"] = None  # force tab to recompute

                st.session_state.chat_history.append({
                    "question":   user_input,
                    "answer":     state.get("final_answer", ""),
                    "trust_card": state.get("trust_card", {}),
                    "route":      state.get("route", "qa"),
                    "raw_result": state.get("raw_result", {}),
                    "needs_page": needs,
                })
                st.rerun()

            except FileNotFoundError:
                st.error(
                    "⚠ ChromaDB collection not found.\n\n"
                    "Run: `python rag_pipeline.py --ingest --docs_folder ./data`"
                )
            except ConnectionError as e:
                st.error(f"⚠ Cannot reach Ollama: {e}\n\nRun: `ollama serve`  or enable OpenAI in sidebar.")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    if not st.session_state.chat_history and not user_input:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#868e96;">
          <div style="font-size:3rem;">📊</div>
          <div style="font-size:1.1rem;font-weight:600;color:#495057;margin:12px 0 8px;">
            Ask a financial research question</div>
          <div style="font-size:0.85rem;line-height:2;">
            💬 <em>What was Infosys revenue in FY2024?</em><br>
            📊 <em>Analyze Infosys FY2024</em><br>
            ⚖ <em>Compare TCS and Infosys</em><br>
            🔢 <em>Calculate ratios for Cognizant FY2023</em><br>
            📝 <em>Generate report for Infosys</em><br>
            🕸 <em>Show knowledge graph for TCS</em><br>
            📈 <em>Show dashboard for Infosys</em>
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TAB 2: REPORT
# ─────────────────────────────────────────────
with tab_report:
    st.caption("Generated automatically when you ask for a report in chat. You can also generate one directly here.")

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        rpt_company = st.text_input("Company", value="Infosys", key="rpt_company_input")
    with col2:
        rpt_fy = st.text_input("Fiscal Year", value="FY2024", key="rpt_fy_input")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📝 Generate", key="rpt_gen_btn", use_container_width=True):
            with st.spinner(f"Generating report for {rpt_company} {rpt_fy}…"):
                try:
                    st.session_state["report_data"] = generate_report(
                        rpt_company, rpt_fy, st.session_state.use_openai
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

    result = st.session_state.get("report_data")
    if result:
        overall = result["overall_confidence"]
        band    = "High" if overall >= 85 else "Medium" if overall >= 70 else "Low"
        color   = "#2f9e44" if band=="High" else "#f08c00" if band=="Medium" else "#c92a2a"

        st.markdown(f"### {result['company']} — {result['fiscal_year']} Research Report")
        st.markdown(
            f"<div style='background:#f8f9fa;padding:10px 16px;border-radius:8px;"
            f"border-left:4px solid {color};margin-bottom:16px;'>"
            f"Overall Confidence: <strong style='color:{color}'>{overall}/100 · {band}</strong>"
            f"</div>", unsafe_allow_html=True
        )

        for section_name, sec in result["sections"].items():
            conf, sband = sec["confidence"], sec["band"]
            scolor = "#2f9e44" if sband=="High" else "#f08c00" if sband=="Medium" else "#c92a2a"
            st.markdown(
                f"#### {section_name} "
                f"<span style='font-size:0.75rem;background:{scolor};color:white;"
                f"padding:2px 9px;border-radius:12px;font-weight:700;'>{conf}/100</span>",
                unsafe_allow_html=True
            )
            st.markdown(sec["content"])
            if sec["decision"] == "Review":
                st.warning("⚠ Low confidence — human review recommended for this section.")
            st.markdown("<hr class='light'>", unsafe_allow_html=True)

        st.download_button(
            "⬇ Download as Markdown",
            data=result["report_markdown"],
            file_name=f"{result['company']}_{result['fiscal_year']}_report.md",
            mime="text/markdown",
        )

        if st.button("➕ Add to HITL Review Queue", key="rpt_hitl_btn"):
            dummy_tc = {
                "confidence_score": overall, "confidence_band": band,
                "decision": {"action": "Warn" if overall < 85 else "Approve",
                             "reason": f"Report overall confidence: {overall}/100"},
                "issues": [],
                "rag_output": {"question": f"Report: {result['company']} {result['fiscal_year']}",
                               "answer": result["report_markdown"][:500],
                               "retrieved_chunks": [], "citations": [], "metadata": {}},
            }
            rid = add_to_queue(dummy_tc)
            st.success(f"Added to HITL queue. ID: `{rid}`")
    else:
        st.info("No report generated yet. Ask in chat (\"Generate report for Infosys\") or use the form above.")


# ─────────────────────────────────────────────
# TAB 3: KNOWLEDGE GRAPH
# ─────────────────────────────────────────────
with tab_graph:
    st.caption("Entity relationships extracted from financial documents via RAG.")

    if not NX_AVAILABLE:
        st.error("NetworkX not installed. Run: `pip install networkx`")
    else:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            kg_company = st.text_input("Company", value="Infosys", key="kg_company_input")
        with col2:
            kg_fy = st.text_input("Fiscal Year", value="FY2024", key="kg_fy_input")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🕸 Build Graph", key="kg_build_btn", use_container_width=True):
                with st.spinner(f"Building knowledge graph for {kg_company} {kg_fy}…"):
                    try:
                        st.session_state["graph_data"] = build_knowledge_graph_data(
                            kg_company, kg_fy, st.session_state.use_openai
                        )
                    except Exception as e:
                        st.error(f"Error: {e}")

        gdata = st.session_state.get("graph_data")
        if gdata:
            # Build NetworkX graph from the plain-dict data
            G = nx.Graph()
            company = gdata["company"]
            G.add_node(company, type="company", color="#1864ab", size=40)

            node_groups = [
                ("segments",          "seg_",  "#862e2e", 20, "segment"),
                ("products",          "prod_", "#5f3dc4", 18, "product"),
                ("risks",             "risk_", "#c92a2a", 16, "risk"),
                ("opportunities",     "opp_",  "#e67700", 16, "opportunity"),
                ("management_themes", "mgmt_", "#0c8599", 14, "management"),
            ]
            for key, prefix, color, size, ntype in node_groups:
                for item in gdata.get(key, []):
                    # Build a clean slug for the node ID (letters+digits only,
                    # no spaces or special chars) so PyVis doesn't corrupt it.
                    # The full label is stored separately for display.
                    slug    = re.sub(r"[^a-z0-9]", "_", item.lower())[:25]
                    node_id = f"{prefix}{slug}"
                    G.add_node(node_id, label=item, type=ntype, color=color, size=size)
                    G.add_edge(company, node_id, label=ntype)

            G.add_node("revenue", label=f"Revenue\n{gdata.get('revenue_label','N/A')}",
                       type="revenue", color="#2f9e44", size=22)
            G.add_edge(company, "revenue", label="reported")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Nodes", G.number_of_nodes())
            col_b.metric("Edges", G.number_of_edges())
            col_c.metric("Company", company)
            st.markdown("---")

            if PYVIS_AVAILABLE:
                net = PyvisNetwork(height="560px", width="100%", bgcolor="#1a1a2e",
                                   font_color="white", notebook=False)
                net.barnes_hut(gravity=-8000, central_gravity=0.3,
                               spring_length=200, spring_strength=0.05)
                for node_id, attrs in G.nodes(data=True):
                    net.add_node(str(node_id),
                                label=attrs.get("label", str(node_id)),
                                color=attrs.get("color", "#495057"),
                                size=attrs.get("size", 15),
                                title=f"Type: {attrs.get('type','?')}")
                for src, tgt, attrs in G.edges(data=True):
                    net.add_edge(str(src), str(tgt),
                                label=attrs.get("label",""), color="#4dabf7", width=1.5)

                import tempfile, os as _os
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                                 mode="w", encoding="utf-8") as f:
                    net.save_graph(f.name)
                    tmp_path = f.name
                html = open(tmp_path, encoding="utf-8").read()
                _os.unlink(tmp_path)
                st.components.v1.html(html, height=580, scrolling=False)
            else:
                st.warning("PyVis not installed — showing text list. Run: `pip install pyvis`")
                for node_id, attrs in G.nodes(data=True):
                    icon = {"company":"🔵","segment":"🟤","product":"🟣",
                            "risk":"🔴","opportunity":"🟠","management":"🟢",
                            "revenue":"🟢"}.get(attrs.get("type"), "⚪")
                    st.markdown(f"{icon} **{attrs.get('label', node_id)}** `({attrs.get('type','?')})`")

            st.markdown("---")
            legend = [("company","#1864ab"),("segment","#862e2e"),("product","#5f3dc4"),
                     ("risk","#c92a2a"),("opportunity","#e67700"),
                     ("management","#0c8599"),("revenue","#2f9e44")]
            legend_cols = st.columns(len(legend))
            for col, (ntype, color) in zip(legend_cols, legend):
                col.markdown(
                    f"<span class='graph-node-tag' style='background:{color};'>{ntype}</span>",
                    unsafe_allow_html=True
                )

            with st.expander("📋 Edge List", expanded=False):
                # Look up each node's display LABEL (full text, e.g.
                # "Banking and Financial Services") instead of printing
                # the raw node ID (a truncated/slugified internal key like
                # "seg_banking_and_financial_ser"). The graph visualization
                # already does this correctly via attrs.get("label", ...)
                # on the NODE — this edge list was printing src/tgt (the
                # IDs) directly instead of looking up their labels.
                for src, tgt, attrs in G.edges(data=True):
                    src_label = G.nodes[src].get("label", src)
                    tgt_label = G.nodes[tgt].get("label", tgt)
                    st.markdown(f"- **{src_label}** → **{tgt_label}** *({attrs.get('label','')})*")
        else:
            st.info("No graph generated yet. Ask in chat (\"Show knowledge graph for Infosys\") or use the form above.")


# ─────────────────────────────────────────────
# TAB 4: ANALYTICS DASHBOARD
# ─────────────────────────────────────────────
with tab_dashboard:
    st.caption("Financial trends and ratio analysis from retrieved documents.")

    if not PLOTLY_AVAILABLE:
        st.error("Plotly not installed. Run: `pip install plotly`")
    else:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            dash_company = st.text_input(
                "Company",
                value=st.session_state.get("dashboard_company") or "Infosys",
                key="dash_company_input"
            )
        with col2:
            # ── Auto-discover real fiscal years for this company ───────────
            # FIX: previously defaulted to a HARDCODED "FY2022, FY2023, FY2024"
            # string, which silently excluded any years actually present in
            # the ingested documents (e.g. FY2025, FY2026) unless the user
            # manually retyped the field. Now we query ChromaDB directly for
            # every distinct fiscal_year value stored for this company and
            # use that as the default -- so trend charts cover the FULL
            # historical range present in the data, not a guessed window.
            discovered_key = f"discovered_years_{dash_company.lower()}"
            if discovered_key not in st.session_state:
                try:
                    _collection = load_chroma_collection()
                    _years = get_available_fiscal_years(_collection, company=dash_company)
                    st.session_state[discovered_key] = _years
                except Exception:
                    st.session_state[discovered_key] = []

            discovered_years = st.session_state[discovered_key]
            default_years_str = (
                ", ".join(discovered_years) if discovered_years
                else "FY2023, FY2024, FY2025"   # last-resort fallback only if discovery fails
            )

            dash_years_input = st.text_input(
                "Fiscal Years (comma-separated)",
                value=st.session_state.get("dashboard_fys") or default_years_str,
                key="dash_years_input",
                help=(
                    f"Auto-detected {len(discovered_years)} year(s) in your documents "
                    f"for {dash_company}. Edit to narrow the range if needed."
                    if discovered_years else
                    "Could not auto-detect years -- enter manually or check ingestion."
                ),
            )
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            load_clicked = st.button("📈 Load", key="dash_load_btn", use_container_width=True)

        if discovered_years:
            st.caption(f"✅ Found {len(discovered_years)} fiscal year(s) in documents: "
                      f"{', '.join(discovered_years)}")
        else:
            st.caption("⚠️ No fiscal years auto-detected for this company — "
                      "check spelling or re-ingest documents.")

        # Auto-load if chat triggered this tab and no data cached yet
        should_load = load_clicked or (
            st.session_state.get("dashboard_company") and
            st.session_state.get("dashboard_data") is None
        )

        if should_load and dash_company:
            fiscal_years = [y.strip() for y in dash_years_input.split(",") if y.strip()]
            with st.spinner(f"Loading analytics for {dash_company} across "
                           f"{len(fiscal_years)} fiscal year(s)…"):
                try:
                    ratio_result = calculate_ratios(
                        dash_company, fiscal_years[-1], st.session_state.use_openai
                    )
                    st.session_state["dashboard_data"] = {
                        "company": dash_company,
                        "fiscal_years": fiscal_years,
                        "ratio_result": ratio_result,
                    }
                except Exception as e:
                    st.error(f"Error: {e}")

        ddata = st.session_state.get("dashboard_data")
        if ddata:
            company      = ddata["company"]
            fiscal_years = ddata["fiscal_years"]
            ratio_result = ddata["ratio_result"]

            tab_trends, tab_ratios, tab_raw = st.tabs(["📈 Trends", "🔢 Ratios", "📄 Raw Data"])

            with tab_trends:
                st.markdown(f"### {company} — Financial Trends ({', '.join(fiscal_years)})")
                col_r, col_m = st.columns(2)

                with col_r:
                    rx, ry = _gather_trend(company, fiscal_years, "total revenue",
                                          st.session_state.use_openai, _extract_number)
                    if rx:
                        ry_b = [v/1e9 for v in ry]
                        fig = go.Figure(go.Scatter(
                            x=rx, y=ry_b, mode="lines+markers+text",
                            text=[f"${v:.1f}B" for v in ry_b], textposition="top center",
                            line=dict(color="#1864ab", width=3), marker=dict(size=10),
                        ))
                        fig.update_layout(
                            title=dict(
                                text="Revenue (USD Billions)",
                                font=dict(color="#212529", size=16),
                            ),
                            plot_bgcolor="white", paper_bgcolor="white",
                            height=300, margin=dict(t=40,b=20,l=40,r=10),
                            font=dict(color="#495057"),
                            xaxis=dict(tickfont=dict(color="#495057")),
                            yaxis=dict(tickfont=dict(color="#495057")),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        _missing = [y for y in fiscal_years if y not in rx]
                        if _missing:
                            st.caption(f"⚠️ No revenue data found for: {', '.join(_missing)}")
                    else:
                        st.info("Revenue data not found in documents.")

                with col_m:
                    mx, my = _gather_trend(company, fiscal_years, "operating margin",
                                          st.session_state.use_openai, _extract_pct)
                    if mx:
                        fig = go.Figure(go.Scatter(
                            x=mx, y=my, mode="lines+markers+text",
                            text=[f"{v:.1f}%" for v in my], textposition="top center",
                            line=dict(color="#2f9e44", width=3), marker=dict(size=10),
                        ))
                        fig.update_layout(
                            title=dict(
                                text="Operating Margin (%)",
                                font=dict(color="#212529", size=16),
                            ),
                            plot_bgcolor="white", paper_bgcolor="white",
                            height=300, margin=dict(t=40,b=20,l=40,r=10),
                            font=dict(color="#495057"),
                            xaxis=dict(tickfont=dict(color="#495057")),
                            yaxis=dict(ticksuffix="%", tickfont=dict(color="#495057")),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        _missing = [y for y in fiscal_years if y not in mx]
                        if _missing:
                            st.caption(f"⚠️ No operating margin data found for: {', '.join(_missing)}")
                    else:
                        st.info("Operating margin data not found.")

                col_n, col_h = st.columns(2)
                with col_n:
                    nxv, ny = _gather_trend(company, fiscal_years, "net income",
                                           st.session_state.use_openai, _extract_number)
                    if nxv:
                        ny_b = [v/1e9 for v in ny]
                        fig = go.Figure(go.Bar(
                            x=nxv, y=ny_b, marker_color="#339af0",
                            text=[f"${v:.1f}B" for v in ny_b], textposition="outside",
                        ))
                        fig.update_layout(
                            title=dict(
                                text="Net Income (USD Billions)",
                                font=dict(color="#212529", size=16),
                            ),
                            plot_bgcolor="white", paper_bgcolor="white",
                            height=300, margin=dict(t=40,b=20,l=40,r=10),
                            font=dict(color="#495057"),
                            xaxis=dict(tickfont=dict(color="#495057")),
                            yaxis=dict(tickfont=dict(color="#495057")),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        _missing = [y for y in fiscal_years if y not in nxv]
                        if _missing:
                            st.caption(f"⚠️ No net income data found for: {', '.join(_missing)}")
                    else:
                        st.info("Net income data not found.")

                with col_h:
                    hx, hy = _gather_trend(company, fiscal_years, "total employee headcount",
                                          st.session_state.use_openai, _extract_number)
                    if hx:
                        fig = go.Figure(go.Scatter(
                            x=hx, y=hy, mode="lines+markers+text",
                            text=[f"{int(v):,}" for v in hy], textposition="top center",
                            line=dict(color="#e67700", width=3), marker=dict(size=10),
                        ))
                        fig.update_layout(
                            title=dict(
                                text="Total Headcount",
                                font=dict(color="#212529", size=16),
                            ),
                            plot_bgcolor="white", paper_bgcolor="white",
                            height=300, margin=dict(t=40,b=20,l=40,r=10),
                            font=dict(color="#495057"),
                            xaxis=dict(tickfont=dict(color="#495057")),
                            yaxis=dict(tickfont=dict(color="#495057")),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        _missing = [y for y in fiscal_years if y not in hx]
                        if _missing:
                            st.caption(f"⚠️ No headcount data found for: {', '.join(_missing)}")
                    else:
                        st.info("Headcount data not found.")

            with tab_ratios:
                st.markdown(f"### {company} — Financial Ratios ({fiscal_years[-1]})")
                ratios = ratio_result["ratios"]
                valid  = {k: v for k, v in ratios.items() if v is not None}
                na     = {k: v for k, v in ratios.items() if v is None}

                if valid:
                    cols = st.columns(min(len(valid), 3))
                    for col, (name, val) in zip(cols * 10, valid.items()):
                        suffix = "%" if "%" in name else ("x" if "Ratio" in name else "")
                        col.metric(name, f"{val:.2f}{suffix}")
                if na:
                    st.caption("N/A (insufficient data): " + ", ".join(na.keys()))

                pct_ratios = {k: v for k, v in valid.items() if "%" in k}
                if pct_ratios:
                    fig = go.Figure(go.Bar(
                        x=list(pct_ratios.keys()), y=list(pct_ratios.values()),
                        marker_color=["#1864ab","#2f9e44","#e67700"][:len(pct_ratios)],
                        text=[f"{v:.1f}%" for v in pct_ratios.values()], textposition="outside",
                    ))
                    fig.update_layout(
                        title=dict(
                            text=f"{company} — Margin Ratios ({fiscal_years[-1]})",
                            font=dict(color="#212529", size=16),
                        ),
                        yaxis_title="%",
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=320, margin=dict(t=40,b=30,l=40,r=10),
                        font=dict(color="#495057"),
                        xaxis=dict(tickfont=dict(color="#495057")),
                        yaxis=dict(tickfont=dict(color="#495057"),
                                  title_font=dict(color="#495057")),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab_raw:
                st.markdown("### Raw LLM Answers")
                for label, answer in ratio_result.get("answers", {}).items():
                    st.markdown(f"**{label}**")
                    st.caption(answer)
                    st.markdown("<hr class='light'>", unsafe_allow_html=True)
        else:
            st.info("No dashboard data yet. Ask in chat (\"Show dashboard for Infosys\") or click Load above.")


# ─────────────────────────────────────────────
# TAB 5: HITL QUEUE
# ─────────────────────────────────────────────
with tab_hitl:
    st.caption("Human-in-the-Loop review. Action pending trust cards.")

    tab_pending, tab_reviewed, tab_all = st.tabs(["⏳ Pending", "✅ Reviewed", "📋 All"])

    def _render_queue_items(items: list, allow_review: bool = False) -> None:
        if not items:
            st.info("No items found.")
            return
        for item in items:
            tc       = item.get("trust_card", {})
            ap       = tc.get("rag_output", {})
            band     = tc.get("confidence_band", "?")
            score    = tc.get("confidence_score", "?")
            rid      = item["review_id"]
            decision = tc.get("decision", {}).get("action", "?")
            badge_cls = {"High":"badge-high","Medium":"badge-medium",
                        "Low":"badge-low"}.get(band,"badge-low")
            pill_cls  = {"Approve":"pill pill-approve","Warn":"pill pill-warn",
                        "Review":"pill pill-review"}.get(decision,"pill pill-review")

            with st.container():
                ca, cb, cc = st.columns([4,1,1])
                with ca:
                    q = ap.get("question", "—")
                    st.markdown(f"**Q:** {q[:120]}{'…' if len(q)>120 else ''}")
                    st.caption(f"ID: `{rid}` | {item.get('queued_at','?')}")
                with cb:
                    st.markdown(f"<span class='{badge_cls}'>{band} ({score})</span>", unsafe_allow_html=True)
                with cc:
                    st.markdown(f"<span class='{pill_cls}'>{decision}</span>", unsafe_allow_html=True)

                if allow_review and item["status"] == "Pending":
                    with st.expander(f"Review `{rid[:8]}…`"):
                        st.markdown(f"**Answer:** {ap.get('answer','—')[:400]}")
                        sel = st.selectbox("Action", list(VALID_ACTIONS), key=f"a_{rid}")
                        note = st.text_area("Note", key=f"n_{rid}", height=70)
                        edited = ""
                        if sel == "Edit":
                            edited = st.text_area("Edited Answer", value=ap.get("answer",""),
                                                  key=f"e_{rid}", height=100)
                        if st.button("Submit", key=f"s_{rid}"):
                            try:
                                submit_review(rid, sel, reviewer_note=note,
                                              edited_answer=edited if sel=="Edit" else None)
                                st.success(f"✅ {sel}")
                                st.rerun()
                            except ValueError as e:
                                st.error(str(e))
                elif not allow_review and item.get("action"):
                    st.caption(f"Action: **{item['action']}** | {item.get('reviewed_at','?')} | "
                              f"Note: {item.get('reviewer_note') or '—'}")
                st.markdown("---")

    with tab_pending:
        pending_items = get_pending_items()
        c1, c2 = st.columns([3,1])
        c1.markdown(f"**{len(pending_items)} pending**")
        if c2.button("🔄 Refresh", key="refresh_pending"): st.rerun()
        _render_queue_items(pending_items, allow_review=True)

    with tab_reviewed:
        reviewed = [i for i in get_queue() if i["status"] != "Pending"]
        st.markdown(f"**{len(reviewed)} reviewed**")
        _render_queue_items(reviewed)

    with tab_all:
        st.json(get_queue())
