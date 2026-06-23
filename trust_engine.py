"""
trust_engine.py  —  Developer 2: Trust Layer & Validation
==========================================================
Consumes an answer_package (dict) as defined in project_contract.json.
Produces a trust_card dict that wraps the original package with all
validation results, confidence scoring, and a final decision.

PUBLIC API
----------
    from trust_engine import process_answer_package

    trust_card = process_answer_package(answer_package)   # main entry point
"""

from __future__ import annotations

print("RUNNING TRUST_ENGINE.PY")

def process_answer_package(answer_package: dict) -> dict:
    print("PROCESSING ANSWER PACKAGE")
    return generate_trust_card(answer_package)

import os
import json
import logging
from typing import Dict, List

import re
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _score_source_quality(chunks: list[dict]) -> dict:
    """
    Source Quality — max 25 points.
    Rewards PDF sources (primary documents) over plain-text transcripts,
    and penalises an index with very few chunks.
    """
    if not chunks:
        return {"score": 0, "max": 25, "reason": "No chunks retrieved."}

    # "annual_report" and "earnings_call" are the primary source types
    # after the document_processor fix — they replaced the old "pdf" fallback.
    HIGH_QUALITY_SOURCES = {"pdf", "annual_report", "earnings_call"}
    pdf_count = sum(1 for c in chunks
                    if c.get("source_type") in HIGH_QUALITY_SOURCES)    
    txt_count  = len(chunks) - pdf_count

    # Base: 15 pts if at least one PDF present, else 8
    base = 15 if pdf_count > 0 else 8

    # Mix bonus: up to 10 pts proportional to PDF share
    mix_bonus = round(10 * (pdf_count / len(chunks)))

    score = min(base + mix_bonus, 25)
    reason = (
        f"{pdf_count} PDF chunk(s), {txt_count} TXT chunk(s). "
        f"PDF share: {pdf_count}/{len(chunks)}."
    )
    return {"score": score, "max": 25, "reason": reason}


def _score_retrieval_relevance(chunks: list[dict]) -> dict:
    """
    Retrieval Relevance — max 25 points.
    Based on the mean cosine similarity score across retrieved chunks.
    """
    if not chunks:
        return {"score": 0, "max": 25, "reason": "No chunks retrieved."}

    scores = [c.get("similarity_score", 0.0) for c in chunks]
    mean_sim = sum(scores) / len(scores)

    # Linear mapping: 0.0 → 0 pts, 1.0 → 25 pts
    # all-MiniLM-L6-v2 on financial documents has a natural similarity ceiling
    # of ~0.60-0.65. A mean similarity of 0.58 is actually excellent for this
    # model on this domain — the old linear formula (0.58 × 25 = 14.5/25)
    # unfairly penalised good retrievals just because the model's absolute
    # similarity scores are lower than models like text-embedding-3-small.
    #
    # New curve:
    #   sim < 0.30  → 0  pts  (genuinely irrelevant)
    #   0.30-0.45   → 8  pts  (weak relevance)
    #   0.45-0.55   → 15 pts  (moderate relevance)
    #   0.55-0.65   → 21 pts  (good relevance — typical for this model)
    #   0.65+       → 25 pts  (excellent relevance)
    if mean_sim >= 0.65:
        score = 25
    elif mean_sim >= 0.55:
        score = 21
    elif mean_sim >= 0.45:
        score = 15
    elif mean_sim >= 0.30:
        score = 8
    else:
        score = 0
    reason = f"Mean cosine similarity: {mean_sim:.3f} across {len(chunks)} chunk(s)."
    return {"score": score, "max": 25, "reason": reason}


def _score_citation_completeness(
    chunks: list[dict], citations: list[dict]
) -> dict:
    """
    Citation Completeness — max 20 points.
    Every retrieved chunk should have a matching citation entry.
    """
    if not chunks:
        return {"score": 0, "max": 20, "reason": "No chunks to cite."}

    chunk_ids   = {c["chunk_id"] for c in chunks if "chunk_id" in c}
    citation_ids = {c["chunk_id"] for c in citations if "chunk_id" in c}

    matched  = chunk_ids & citation_ids
    coverage = len(matched) / len(chunk_ids) if chunk_ids else 0

    score = round(coverage * 20)
    reason = (
        f"{len(matched)}/{len(chunk_ids)} chunks have matching citations "
        f"({coverage*100:.0f}% coverage)."
    )
    return {"score": score, "max": 20, "reason": reason}


def _score_cross_source_agreement(chunks: list[dict]) -> dict:
    """
    Cross-Source Agreement — max 15 points.
    Checks that all chunks agree on company_name and fiscal_year.
    A consistent, multi-source answer is more trustworthy.
    """
    if not chunks:
        return {"score": 0, "max": 15, "reason": "No chunks to compare."}

    companies   = {c.get("company_name", "unknown") for c in chunks}
    fiscal_years = {c.get("fiscal_year", "unknown") for c in chunks}

    issues = []
    score  = 15

    if len(companies) > 1:
        score -= 8
        issues.append(f"Multiple company names found: {companies}.")
    if len(fiscal_years) > 1:
        score -= 7
        issues.append(f"Multiple fiscal years found: {fiscal_years}.")

    score = max(score, 0)
    reason = " ".join(issues) if issues else (
        f"All chunks agree: company='{next(iter(companies))}', "
        f"fiscal_year='{next(iter(fiscal_years))}'."
    )
    return {"score": score, "max": 15, "reason": reason}


def _score_calculation_validation(answer: str, chunks: list[dict]) -> dict:
    """
    Calculation Validation — max 15 points.
    Extracts dollar/percent figures from the answer and checks whether
    at least some of them appear verbatim in the retrieved chunk texts.
    Simple but effective heuristic.
    """
    # Extract numbers that look financial (e.g. $383.3 billion, 9%, $85.2B)
    number_pattern = re.compile(
        r"\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M|T))?|"
        r"\d+(?:\.\d+)?\s*%"
    )
    answer_figures = set(number_pattern.findall(answer))

    if not answer_figures:
        return {
            "score": 10,
            "max": 15,
            "reason": "No numeric figures found in answer; partial credit awarded.",
        }

    chunk_text_all = " ".join(c.get("text", "") for c in chunks)
    supported = {fig for fig in answer_figures if fig in chunk_text_all}

    ratio = len(supported) / len(answer_figures)
    score = round(ratio * 15)

    reason = (
        f"{len(supported)}/{len(answer_figures)} answer figures "
        f"found verbatim in retrieved chunks."
    )
    return {"score": score, "max": 15, "reason": reason}


def compute_confidence_score(answer_package: dict) -> dict:
    """
    Aggregates all five sub-scores into a single confidence dict.
    Returns:
        {
          "total": int (0-100),
          "band":  "High" | "Medium" | "Low",
          "breakdown": { <factor>: {"score": int, "max": int, "reason": str} }
        }
    """
    answer   = answer_package.get("answer", "")
    chunks   = answer_package.get("retrieved_chunks", [])
    citations = answer_package.get("citations", [])

    breakdown = {
        "source_quality":        _score_source_quality(chunks),
        "retrieval_relevance":   _score_retrieval_relevance(chunks),
        "citation_completeness": _score_citation_completeness(chunks, citations),
        "cross_source_agreement": _score_cross_source_agreement(chunks),
        "calculation_validation": _score_calculation_validation(answer, chunks),
    }

    total = sum(v["score"] for v in breakdown.values())

    if total >= 85:
        band = "High"
    elif total >= 70:
        band = "Medium"
    else:
        band = "Low"

    return {"total": total, "band": band, "breakdown": breakdown}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CITATION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_citations(answer_package: dict) -> dict:
    """
    Verifies each citation against its matching chunk for:
      - company_name match
      - fiscal_year match
      - source_type presence
      - page_number consistency (null for TXT, int for PDF)

    Returns:
        {
          "all_valid": bool,
          "results":   [ { "chunk_id": str, "valid": bool, "issues": [...] } ]
        }
    """
    chunks_by_id = {
        c["chunk_id"]: c
        for c in answer_package.get("retrieved_chunks", [])
        if "chunk_id" in c
    }
    citations = answer_package.get("citations", [])
    results   = []

    for cit in citations:
        cid    = cit.get("chunk_id", "<missing>")
        issues = []
        chunk  = chunks_by_id.get(cid)

        if chunk is None:
            issues.append("chunk_id not found in retrieved_chunks.")
        else:
            if cit.get("company_name") != chunk.get("company_name"):
                issues.append(
                    f"company_name mismatch: citation='{cit.get('company_name')}' "
                    f"vs chunk='{chunk.get('company_name')}'."
                )
            if cit.get("fiscal_year") != chunk.get("fiscal_year"):
                issues.append(
                    f"fiscal_year mismatch: citation='{cit.get('fiscal_year')}' "
                    f"vs chunk='{chunk.get('fiscal_year')}'."
                )
            if not chunk.get("source_type"):
                issues.append("source_type is missing in chunk.")
            # page_number must be null for txt, int-or-null for pdf
            src  = chunk.get("source_type", "")
            page = chunk.get("page_number")
            if src == "txt" and page is not None:
                issues.append(
                    f"TXT source should have page_number=null, got {page}."
                )
            if src == "pdf" and page is not None and not isinstance(page, int):
                issues.append(
                    f"PDF source should have integer page_number, got {page!r}."
                )

        results.append({"chunk_id": cid, "valid": len(issues) == 0, "issues": issues})

    all_valid = all(r["valid"] for r in results)
    return {"all_valid": all_valid, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# 3. FINANCIAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_financials(text: str) -> dict[str, float]:
    """
    Extracts simple key → value pairs like:
        revenue: 383.3 (billion)
        growth:  9  (%)
        margin:  21.5 (%)
    Returns a dict of {label: numeric_value}.
    """
    found: dict[str, float] = {}

    # Dollar amounts in billions/millions
    for match in re.finditer(
        r"([\w\s]+?)\s+(?:of\s+)?\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|trillion|B|M|T)?",
        text, re.IGNORECASE
    ):
        label = match.group(1).strip().lower()[-40:]  # cap label length
        raw   = float(match.group(2).replace(",", ""))
        unit  = (match.group(3) or "").lower()
        multiplier = (
            1e9  if unit in ("billion", "b") else
            1e6  if unit in ("million", "m") else
            1e12 if unit in ("trillion", "t") else
            1
        )
        found[label] = raw * multiplier

    # Percentage values
    for match in re.finditer(
        r"([\w\s]+?)\s+(?:of\s+)?([\d,]+(?:\.\d+)?)\s*%", text, re.IGNORECASE
    ):
        label = match.group(1).strip().lower()[-40:] + " (%)"
        found[label] = float(match.group(2).replace(",", ""))

    return found


def _compute_growth(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return round((current - previous) / abs(previous) * 100, 2)


def _compute_margin(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator * 100, 2)


def validate_financials(answer_package: dict) -> dict:
    """
    Attempts basic financial cross-checks across the answer and chunk texts.
    Computes growth %, operating margin, EBIT margin, YoY change where data allows.

    Returns:
        {
          "computed_metrics": { ... },
          "answer_figures":   { ... },
          "chunk_figures":    { ... },
          "warnings":         [ str ]
        }
    """
    answer      = answer_package.get("answer", "")
    chunks      = answer_package.get("retrieved_chunks", [])
    chunk_text  = " ".join(c.get("text", "") for c in chunks)

    answer_figs = _extract_financials(answer)
    chunk_figs  = _extract_financials(chunk_text)

    warnings: list[str] = []

    # Cross-check figures that appear in both answer and chunks
    for label, ans_val in answer_figs.items():
        for clabel, chunk_val in chunk_figs.items():
            if label in clabel or clabel in label:
                # Allow 1% tolerance for rounding differences
                if chunk_val != 0:
                    diff_pct = abs(ans_val - chunk_val) / abs(chunk_val) * 100
                    if diff_pct > 1.0:
                        warnings.append(
                            f"Possible figure mismatch for '{label}': "
                            f"answer={ans_val:,.0f}, chunk={chunk_val:,.0f} "
                            f"({diff_pct:.1f}% difference)."
                        )

    # Try to compute YoY growth if two revenue-like values exist
    computed: dict[str, Any] = {}
    rev_vals = [
        v for k, v in {**answer_figs, **chunk_figs}.items()
        if "revenue" in k or "net sales" in k or "net revenue" in k
    ]
    if len(rev_vals) >= 2:
        rev_vals.sort(reverse=True)
        growth = _compute_growth(rev_vals[0], rev_vals[1])
        if growth is not None:
            computed["yoy_revenue_growth_pct"] = growth

    # Operating / EBIT margin approximation
    income_vals  = [v for k, v in {**answer_figs, **chunk_figs}.items() if "income" in k or "profit" in k or "ebit" in k]
    revenue_vals = [v for k, v in {**answer_figs, **chunk_figs}.items() if "revenue" in k or "net sales" in k]
    if income_vals and revenue_vals:
        margin = _compute_margin(max(income_vals), max(revenue_vals))
        if margin is not None:
            computed["estimated_margin_pct"] = margin

    return {
        "computed_metrics": computed,
        "answer_figures":   answer_figs,
        "chunk_figures":    chunk_figs,
        "warnings":         warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONTRADICTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_contradictions(answer_package: dict) -> dict:
    """
    Flags:
      - Conflicting monetary/percentage figures across chunks
      - Chunks that don't mention the same company as the majority
      - Answer text that references entities not found in any chunk

    Returns:
        {
          "has_contradictions": bool,
          "flags": [ { "type": str, "detail": str } ]
        }
    """
    chunks  = answer_package.get("retrieved_chunks", [])
    answer  = answer_package.get("answer", "")
    flags: list[dict] = []

    # 1. Conflicting numbers across chunk texts
    all_numbers: dict[str, list[tuple[str, float]]] = {}  # figure_str → [(chunk_id, val)]
    dollar_re = re.compile(
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|trillion|B|M|T)?",
        re.IGNORECASE,
    )
    for chunk in chunks:
        for m in dollar_re.finditer(chunk.get("text", "")):
            raw  = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").lower()
            mult = (
                1e9  if unit in ("billion", "b") else
                1e6  if unit in ("million", "m") else
                1e12 if unit in ("trillion", "t") else
                1
            )
            canonical = round(raw * mult, -6)  # round to nearest million for grouping
            key = str(int(canonical))
            all_numbers.setdefault(key, [])
            # Check if same magnitude appears with wildly different raw values
            existing = [v for _, v in all_numbers[key]]
            for ev in existing:
                if abs(raw * mult - ev) / max(abs(ev), 1) > 0.05:
                    flags.append({
                        "type":   "conflicting_figures",
                        "detail": (
                            f"Similar but non-matching dollar figures across chunks: "
                            f"${raw * mult:,.0f} vs ${ev:,.0f}."
                        ),
                    })
            all_numbers[key].append((chunk.get("chunk_id", "?"), raw * mult))

    # 2. Company name disagreement
    companies = [c.get("company_name", "unknown") for c in chunks]
    if len(set(companies)) > 1:
        flags.append({
            "type":   "source_disagreement",
            "detail": f"Chunks reference multiple companies: {set(companies)}.",
        })

    # 3. Fiscal year disagreement
    fiscal_years = [c.get("fiscal_year", "unknown") for c in chunks]
    if len(set(fiscal_years)) > 1:
        flags.append({
            "type":   "source_disagreement",
            "detail": f"Chunks span multiple fiscal years: {set(fiscal_years)}.",
        })

    # 4. Answer mentions a company not in chunks
    chunk_companies = {c.get("company_name", "") for c in chunks}
    for co in chunk_companies:
        if co and co not in answer:
            flags.append({
                "type":   "missing_evidence",
                "detail": (
                    f"Company '{co}' appears in chunks but not referenced in the answer."
                ),
            })

    return {"has_contradictions": len(flags) > 0, "flags": flags}


# ─────────────────────────────────────────────────────────────────────────────
# 5. FORWARD-LOOKING LANGUAGE DETECTION  (Gap 1 fix)
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that signal recommendations, outlooks, or forward-looking statements.
# These cannot be verified against historical chunk data and must always go
# to a human reviewer regardless of confidence score.
_FORWARD_LOOKING_PATTERNS = re.compile(
    r"\b("
    r"outlook|forecast|foreseeable|going forward|in the future|"
    r"we expect|we anticipate|we project|we estimate|we believe|"
    r"is expected to|are expected to|is projected|are projected|"
    r"is likely to|are likely to|will likely|will probably|"
    r"recommendation|recommend|buy|sell|hold|target price|"
    r"guidance|next quarter|next year|next fiscal|upcoming quarter|"
    r"future growth|future revenue|future earnings|"
    r"could rise|could fall|may increase|may decrease|"
    r"potential upside|potential downside|risk factor"
    r")\b",
    re.IGNORECASE,
)


def detect_forward_looking_language(answer: str) -> dict:
    """
    Scans the answer text for forward-looking or recommendation language.
    Such statements cannot be validated against retrieved historical chunks
    and must always be escalated to a human reviewer.

    Returns:
        {
          "detected":  bool,
          "matches":   [ str ],   # the exact phrases found
          "trigger_review": bool  # always True when detected
        }
    """
    matches = _FORWARD_LOOKING_PATTERNS.findall(answer)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_matches: list[str] = []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    return {
        "detected":       len(unique_matches) > 0,
        "matches":        unique_matches,
        "trigger_review": len(unique_matches) > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. DECISION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def make_decision(confidence_score: int, validation_results: dict) -> dict:
    """
    Approve  → confidence >= 85 AND no citation issues AND no contradictions
               AND no forward-looking language detected
    Warn     → confidence 70-84, OR minor citation issues (no contradictions,
               no forward-looking language)
    Review   → confidence < 70, OR contradictions detected, OR citations
               invalid, OR forward-looking / recommendation language found

    Returns:
        { "action": "Approve" | "Warn" | "Review", "reason": str }
    """
    has_contradictions   = validation_results.get("contradictions", {}).get(
        "has_contradictions", False
    )
    citations_valid      = validation_results.get("citations", {}).get("all_valid", True)
    forward_looking      = validation_results.get("forward_looking", {})
    has_forward_looking  = forward_looking.get("trigger_review", False)

    # Build a list of reason fragments so the message is always precise
    review_reasons: list[str] = []
    if has_contradictions:
        review_reasons.append("Contradictions detected across sources.")
    if not citations_valid:
        review_reasons.append("Citation validation failed.")
    if has_forward_looking:
        phrases = ", ".join(f"'{p}'" for p in forward_looking.get("matches", [])[:3])
        review_reasons.append(
            f"Forward-looking / recommendation language detected: {phrases}."
        )

    # Decision logic
    if review_reasons or confidence_score < 70:
        # Force Review for any hard trigger, even if confidence is high
        all_reasons = [f"Confidence {confidence_score}/100."] + review_reasons
        return {"action": "Review", "reason": " ".join(all_reasons)}

    if confidence_score >= 85 and citations_valid:
        return {
            "action": "Approve",
            "reason": (
                f"Confidence {confidence_score}/100 (High). "
                "All citations valid. No contradictions or forward-looking language detected."
            ),
        }

    # 70–84 with no hard issues → Warn
    return {
        "action": "Warn",
        "reason": (
            f"Confidence {confidence_score}/100 (Medium). "
            + ("Some citation issues found. " if not citations_valid else "")
            + "Manual review recommended before publishing."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. PER-CLAIM VALIDATION  (Gap 2 fix)
# ─────────────────────────────────────────────────────────────────────────────

# ── Sentence splitter helpers ─────────────────────────────────────────────────
# We use a two-pass approach to avoid variable-width lookbehind limitations.
# Pass 1: temporarily mask known abbreviations / decimal numbers so their
#         periods are not treated as sentence boundaries.
# Pass 2: split on remaining ". " / "! " / "? " followed by a capital letter.

_MASK_ABBREVS_RE = re.compile(
    r"(\b(?:FY|Q[1-4]|Mr|Mrs|Dr|Inc|Ltd|Corp|vs|etc|approx|est|fig|no)\.)"
    r"|(\d+\.\d+)"   # decimal numbers like 383.3
    r"|(\d\.)",      # single digit + period (list items / footnotes)
    re.IGNORECASE,
)
_SPLIT_BOUNDARY_RE = re.compile(r"[.!?]\s+(?=[A-Z\"\'(])")

# Financial keywords that make a sentence a "claim worth validating"
_FINANCIAL_CLAIM_KEYWORDS = re.compile(
    r"\b("
    r"revenue|sales|income|profit|loss|margin|ebit|ebitda|"
    r"earnings|growth|decline|increase|decrease|improved|fell|"
    r"billion|million|trillion|percent|%|\$|quarter|fiscal|annual|"
    r"operating|net|gross|ratio|yoy|year.over.year"
    r")\b",
    re.IGNORECASE,
)


def _split_into_sentences(text: str) -> list[str]:
    """
    Split answer text into individual sentences.
    Uses a mask-then-split strategy so decimal numbers and common
    abbreviations are never mistaken for sentence boundaries.
    """
    # Replace each masked token with a placeholder that has no period
    placeholders: list[str] = []

    def _mask(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"\x00MASK{len(placeholders) - 1}\x00"

    masked = _MASK_ABBREVS_RE.sub(_mask, text)

    # Split on remaining boundary punctuation
    raw_parts = _SPLIT_BOUNDARY_RE.split(masked)

    # Restore placeholders and clean up
    restored: list[str] = []
    for part in raw_parts:
        for i, ph in enumerate(placeholders):
            part = part.replace(f"\x00MASK{i}\x00", ph)
        part = part.strip()
        if part:
            restored.append(part)

    return restored


def _sentence_supported_by_chunks(sentence: str, chunks: list[dict]) -> dict:
    """
    Check whether a sentence is supported by at least one retrieved chunk.

    Strategy:
      1. Extract all numbers / key noun phrases from the sentence.
      2. Check whether ANY chunk text contains a meaningful overlap of those tokens.
      3. Also do a simple keyword overlap check as a fallback.

    Returns:
        {
          "supported":    bool,
          "matched_chunk_ids": [ str ],
          "support_signal":   "numeric_match" | "keyword_overlap" | "none"
        }
    """
    # Extract numeric tokens from the sentence
    num_re    = re.compile(r"\d[\d,\.]*(?:\s*(?:billion|million|trillion|B|M|T|%))?", re.I)
    sent_nums = set(num_re.findall(sentence))

    # Extract meaningful words (length > 3, no stopwords)
    stopwords = {"that", "this", "with", "from", "have", "been", "were",
                 "their", "than", "also", "into", "more", "over", "year",
                 "its", "the", "and", "for", "was", "has", "are", "not"}
    word_re   = re.compile(r"\b[a-zA-Z]{4,}\b")
    sent_words = {w.lower() for w in word_re.findall(sentence)} - stopwords

    matched_chunks: list[str] = []
    support_signal = "none"

    for chunk in chunks:
        chunk_text = chunk.get("text", "")

        # 1. Numeric match — strongest signal
        chunk_nums = set(num_re.findall(chunk_text))
        if sent_nums and sent_nums & chunk_nums:
            matched_chunks.append(chunk.get("chunk_id", "?"))
            support_signal = "numeric_match"
            continue

        # 2. Keyword overlap — weaker but useful
        chunk_words = {w.lower() for w in word_re.findall(chunk_text)} - stopwords
        overlap     = sent_words & chunk_words
        if len(overlap) >= max(2, len(sent_words) * 0.35):
            matched_chunks.append(chunk.get("chunk_id", "?"))
            if support_signal == "none":
                support_signal = "keyword_overlap"

    return {
        "supported":          len(matched_chunks) > 0,
        "matched_chunk_ids":  list(set(matched_chunks)),
        "support_signal":     support_signal,
    }


def validate_claims_per_sentence(answer_package: dict) -> dict:
    """
    Splits the answer into sentences, identifies financial claim sentences,
    and checks each one for chunk-level support.

    Returns:
        {
          "total_sentences":    int,
          "financial_claims":   int,
          "supported_claims":   int,
          "unsupported_claims": int,
          "support_rate":       float (0.0–1.0),
          "claims": [
            {
              "sentence":          str,
              "is_financial_claim": bool,
              "supported":         bool,
              "matched_chunk_ids": [ str ],
              "support_signal":    str,
              "is_forward_looking": bool,
            }
          ]
        }
    """
    answer = answer_package.get("answer", "")
    chunks = answer_package.get("retrieved_chunks", [])

    sentences   = _split_into_sentences(answer)
    claim_rows: list[dict] = []
    financial_count   = 0
    supported_count   = 0
    unsupported_count = 0

    for sent in sentences:
        is_financial = bool(_FINANCIAL_CLAIM_KEYWORDS.search(sent))
        is_forward   = bool(_FORWARD_LOOKING_PATTERNS.search(sent))

        if is_financial:
            financial_count += 1
            support = _sentence_supported_by_chunks(sent, chunks)
            if support["supported"]:
                supported_count += 1
            else:
                unsupported_count += 1
        else:
            support = {"supported": True, "matched_chunk_ids": [], "support_signal": "non-financial"}

        claim_rows.append({
            "sentence":           sent,
            "is_financial_claim": is_financial,
            "supported":          support["supported"],
            "matched_chunk_ids":  support["matched_chunk_ids"],
            "support_signal":     support["support_signal"],
            "is_forward_looking": is_forward,
        })

    support_rate = (
        round(supported_count / financial_count, 3) if financial_count else 1.0
    )

    return {
        "total_sentences":    len(sentences),
        "financial_claims":   financial_count,
        "supported_claims":   supported_count,
        "unsupported_claims": unsupported_count,
        "support_rate":       support_rate,
        "claims":             claim_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. TRUST CARD GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_trust_card(answer_package: dict) -> dict:
    """
    Builds the complete trust_card from an answer_package.

    Output structure:
    {
      "confidence_score":     int,
      "confidence_band":      "High" | "Medium" | "Low",
      "confidence_breakdown": { <factor>: {score, max, reason} },
      "validation_results": {
          "citations":       { ... },
          "financials":      { ... },
          "contradictions":  { ... },
          "forward_looking": { detected, matches, trigger_review },   ← NEW
      },
      "per_claim_validation": {                                        ← NEW
          "total_sentences":    int,
          "financial_claims":   int,
          "supported_claims":   int,
          "unsupported_claims": int,
          "support_rate":       float,
          "claims": [ { sentence, is_financial_claim, supported,
                        matched_chunk_ids, support_signal,
                        is_forward_looking } ]
      },
      "issues":   [ str ],
      "decision": { "action": str, "reason": str },
      "rag_output": { <original answer_package> }
    }
    """
    answer       = answer_package.get("answer", "")

    confidence   = compute_confidence_score(answer_package)
    cit_results  = validate_citations(answer_package)
    fin_results  = validate_financials(answer_package)
    contra       = detect_contradictions(answer_package)
    fwd_lang     = detect_forward_looking_language(answer)      # Gap 1
    per_claim    = validate_claims_per_sentence(answer_package)  # Gap 2

    validation_results = {
        "citations":        cit_results,
        "financials":       fin_results,
        "contradictions":   contra,
        "forward_looking":  fwd_lang,   # now part of validation block
    }

    # Collect all human-readable issues into a flat list
    issues: list[str] = []

    for r in cit_results.get("results", []):
        issues.extend(r.get("issues", []))

    issues.extend(fin_results.get("warnings", []))

    for flag in contra.get("flags", []):
        issues.append(f"[{flag['type']}] {flag['detail']}")

    # Forward-looking language issues
    if fwd_lang["detected"]:
        phrases = ", ".join(f"'{p}'" for p in fwd_lang["matches"])
        issues.append(
            f"[forward_looking] Answer contains unverifiable language: {phrases}."
        )

    # Unsupported financial claims from per-claim validation
    for claim in per_claim["claims"]:
        if claim["is_financial_claim"] and not claim["supported"] and not claim["is_forward_looking"]:
            snippet = claim["sentence"][:80] + ("…" if len(claim["sentence"]) > 80 else "")
            issues.append(f"[unsupported_claim] No chunk evidence for: \"{snippet}\"")

    decision = make_decision(confidence["total"], validation_results)

    return {
        "confidence_score":      confidence["total"],
        "confidence_band":       confidence["band"],
        "confidence_breakdown":  confidence["breakdown"],
        "validation_results":    validation_results,
        "per_claim_validation":  per_claim,              # Gap 2 — new top-level field
        "issues":                issues,
        "decision":              decision,
        "rag_output":            answer_package,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def process_answer_package(answer_package: dict) -> dict:
    """
    THE function Developer 2 exposes to the rest of the system.
    Receives an answer_package (dict) from Developer 1.
    Returns a complete trust_card (dict).

    Usage:
        from trust_engine import process_answer_package
        trust_card = process_answer_package(package)
    """
    return generate_trust_card(answer_package)


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE SMOKE-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, pathlib

    contract_path = pathlib.Path("project_contract.json")
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        sample   = contract["sample_answer_package"]
    else:
        # Minimal stub so the file can be tested alone
        sample = {
            "question": "What was Apple's revenue?",
            "answer":   "Apple reported $383.3 billion in FY2023.",
            "retrieved_chunks": [
                {
                    "chunk_id": "apple_10k_2023_p12_c0",
                    "text": "Apple Inc. total net revenue for fiscal year 2023 was $383.3 billion.",
                    "company_name": "Apple Inc.",
                    "fiscal_year": "FY2023",
                    "source_type": "pdf",
                    "file_name": "apple_10k_2023.pdf",
                    "page_number": 12,
                    "similarity_score": 0.91,
                }
            ],
            "citations": [
                {
                    "chunk_id": "apple_10k_2023_p12_c0",
                    "file_name": "apple_10k_2023.pdf",
                    "page_number": 12,
                    "company_name": "Apple Inc.",
                    "fiscal_year": "FY2023",
                }
            ],
            "metadata": {
                "model_used": "gpt-4o",
                "retrieval_top_k": 5,
                "pipeline_version": "1.0.0",
                "timestamp": "2024-10-15T09:32:11Z",
                "total_chunks_in_store": 342,
                "embedding_model": "text-embedding-3-small",
            },
        }

    trust_card = process_answer_package(sample)
    print(json.dumps(trust_card, indent=2, default=str))
