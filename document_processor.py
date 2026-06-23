"""
document_processor.py
─────────────────────
Part 1 — Step 1: Document Loading & Metadata Extraction

Responsibilities:
  - Load PDF files and extract text page-by-page
  - Load TXT transcript files
  - Generate metadata for each document (company_name, fiscal_year, source_type, file_name)

"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# We use PyMuPDF (fitz) for PDF reading -- it's fast and beginner-friendly
# Install with: pip install pymupdf
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("[WARNING] PyMuPDF not installed. PDF support disabled.")
    print("          Run: pip install pymupdf")


# ─────────────────────────────────────────────
# FISCAL YEAR EXTRACTION
# ─────────────────────────────────────────────

def _normalize_year_token(raw: str) -> str:
    """
    Convert any recognized year token into the canonical "FY<4-digit>" form.
    This is what makes '2025' and 'FY2025' the SAME stored value -- fixing
    the fragmentation seen in earlier ingests (8 different fiscal_year
    strings for what was really 4-5 real years).

    Examples:
      "2025"   -> "FY2025"
      "FY25"   -> "FY2025"
      "25"     -> "FY2025"   (2-digit, assumed 20xx)
      "FY2025" -> "FY2025"
    """
    raw = raw.strip().upper().replace("FY", "").replace(" ", "")
    if len(raw) == 4 and raw.isdigit():
        return f"FY{raw}"
    if len(raw) == 2 and raw.isdigit():
        yy = int(raw)
        century = 2000 if yy <= 50 else 1900
        return f"FY{century + yy}"
    return f"FY{raw}" if raw.isdigit() else "unknown"


def _extract_year_from_filename(file_name: str) -> Optional[str]:
    """
    Try several patterns against the filename ONLY. Returns None if no
    year-like token is found -- caller then falls back to document text.

    Patterns tried, in order:
      FY2025, FY25                  -> explicit fiscal year markers
      2025-2026, 2025_2026          -> year ranges (takes the FIRST year)
      2025                          -> plain 4-digit year
      -25, _25, -26-                -> 2-digit year suffixes near separators
    """
    m = re.search(r"FY[\s_-]?(\d{2,4})", file_name, re.IGNORECASE)
    if m:
        return _normalize_year_token(m.group(1))

    m = re.search(r"(20\d{2})[\s_-](?:20\d{2}|\d{2})", file_name)
    if m:
        return _normalize_year_token(m.group(1))

    m = re.search(r"(20\d{2})", file_name)
    if m:
        return _normalize_year_token(m.group(1))

    m = re.search(r"[-_](\d{2})(?:[-_.]|$)", file_name)
    if m:
        return _normalize_year_token(m.group(1))

    return None


def _extract_year_from_text(text: str) -> Optional[str]:
    """
    Scan the first 300-500 LINES of document text (not a character count)
    for a fiscal year or reporting-period statement. This is the PRIMARY,
    authoritative extraction method -- filename is only the fallback when
    this finds nothing (see _extract_fiscal_year).

    Using a LINE-based window (not a character count) matters because PDF
    text extraction produces many short lines (headers, table cells, single
    words per line from multi-column layouts) -- 500 lines of real PDF
    content is typically only ~1-3 pages, which comfortably covers the
    cover page and immediate following content where the reporting period
    is stated, without pulling in unrelated later sections.

    Patterns tried, in order (most specific first):
      "year ended March 31, 2025"     -> standard 10-K / annual report
                                          consolidated-statements header
      "fiscal year 2025" / "FY 2025"  -> explicit fiscal year label
      "Annual Report 2025"            -> title-page style
      A plain 4-digit year            -> last resort
    """
    lines        = text.splitlines()
    window_lines = lines[:500]
    snippet      = "\n".join(window_lines)

    # Most authoritative: standard reporting-period statement
    m = re.search(
        r"year ended[^,]{0,40},?\s*(20\d{2})", snippet, re.IGNORECASE
    )
    if m:
        return _normalize_year_token(m.group(1))

    m = re.search(r"fiscal(?:\s+year)?\s*[:\-]?\s*(20\d{2})", snippet, re.IGNORECASE)
    if m:
        return _normalize_year_token(m.group(1))

    m = re.search(r"FY\s?(20\d{2}|\d{2})\b", snippet, re.IGNORECASE)
    if m:
        return _normalize_year_token(m.group(1))

    m = re.search(r"annual report\s+(20\d{2})", snippet, re.IGNORECASE)
    if m:
        return _normalize_year_token(m.group(1))

    m = re.search(r"\b(20\d{2})\b", snippet)
    if m:
        return _normalize_year_token(m.group(1))

    return None


def _extract_publication_date(text: str) -> Optional[str]:
    """
    Extract a publication/dateline date from news article text.
    Used ONLY for financial_news source_type, where fiscal_year is
    intentionally unreliable (see _extract_fiscal_year docstring).

    Reuters-style articles almost always open with a dateline like:
      "BENGALURU, April 17 (Reuters) -"
    or contain an explicit date near the top of the article.

    Returns a string like "2026-04-17" if found, else None.
    This is informational metadata, NOT used for retrieval filtering --
    it tells a human or the LLM when the article was published, separate
    from any fiscal year it discusses or forecasts.
    """
    snippet = text[:1000]

    months = (
        "january|february|march|april|may|june|july|"
        "august|september|october|november|december"
    )

    # Dateline style: "BENGALURU, April 17 (Reuters)" -- no year given,
    # Reuters omits the year in datelines on the publication day itself.
    # We can't recover the year from the dateline alone, so we only use
    # this pattern combined with a nearby 4-digit year elsewhere in the
    # snippet as supporting evidence.
    m = re.search(
        rf"\b({months})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b",
        snippet, re.IGNORECASE,
    )
    if m:
        month_name, day, year = m.group(1), m.group(2), m.group(3)
        month_num = _MONTH_NAME_TO_NUM.get(month_name.lower())
        if month_num:
            return f"{year}-{month_num}-{int(day):02d}"

    # Numeric date style: "17/04/2026" or "2026-04-17"
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", snippet)
    if m:
        return m.group(0)

    return None


_MONTH_NAME_TO_NUM = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}


def _extract_fiscal_year(file_name: str,
                          sample_text: str = "",
                          source_type: str = "") -> str:
    """
    Public entry point used by load_pdf() and load_txt().

    Order of operations (content is authoritative):
      1. Scan the document's own CONTENT first -- the first 300-500 lines
         of extracted text. This is the AUTHORITATIVE source: the cover
         page or opening remarks of an annual report / earnings call
         transcript reliably state the exact reporting period, which is
         more trustworthy than any filename convention (filenames vary
         wildly across companies in this dataset and don't reliably
         indicate which single fiscal year a document covers).
      2. If the content scan finds nothing (e.g. a scanned/image-only PDF
         with no extractable text), fall back to the FILENAME as a
         secondary signal.
      3. If neither source has a year, return "unknown" explicitly --
         never silently default to a guessed or current year.

    WHY financial_news SKIPS THE CONTENT-SCANNING:
      News articles routinely mention 3-4 different years in one piece --
      prior year actuals, the current year, forward guidance for next year,
      analyst estimates for the year after. A regex scan has no way to tell
      "the year this article is about" from "a year mentioned in passing."
      For news, we only check the filename (rarely has a year anyway) and
      otherwise mark fiscal_year="unknown" -- publication_date (extracted
      separately) is the reliable temporal signal for news instead.
    """
    if source_type == "financial_news":
        # News content is unreliable for fiscal year (see docstring above).
        # Filename is the only signal we trust here; otherwise unknown.
        year_from_filename = _extract_year_from_filename(file_name)
        return year_from_filename if year_from_filename else "unknown"

    # ── Content-first extraction (authoritative) ────────────────────────────
    year_from_text = _extract_year_from_text(sample_text)
    if year_from_text:
        return year_from_text

    # ── Filename fallback (only when content scan found nothing) ────────────
    year_from_filename = _extract_year_from_filename(file_name)
    if year_from_filename:
        return year_from_filename

    return "unknown"


# ─────────────────────────────────────────────
# COMPANY NAME -- FOLDER-ONLY, NO FILENAME FALLBACK
# ─────────────────────────────────────────────

_SOURCE_TYPE_FOLDER_NAMES = {
    "annual_report", "annual_reports",
    "earnings_call", "earnings_calls",
    "earning_call", "earning_calls",
    "financial_news", "news",
}

_ROOT_FOLDER_NAMES = {"data", "documents", "docs", ".", ""}


def _resolve_company_name(file_path: Path) -> str:
    """
    Extract company name STRICTLY from folder structure. Never falls back
    to guessing from the filename -- filenames in real datasets are far too
    inconsistent (tickers like "CTSH", news headlines, internal codes like
    "PRO013686_8_...", leading/trailing whitespace) to parse reliably.

    REAL DATASET STRUCTURE (confirmed):
        documents/<source_type>/<company>/<file>
    Example: documents/Annual_reports/Infosys/infosys-ar-25.pdf

    So the COMPANY folder is the file's IMMEDIATE PARENT directory
    (NOT the grandparent -- the source-type folder sits ABOVE the
    company folder in this dataset, the reverse of a company-first layout).

    If the immediate parent folder looks like a source-type or root folder
    name (meaning the dataset isn't structured as expected, e.g. a file
    sitting directly under documents/ with no company subfolder), returns
    "unknown" rather than guessing -- this makes structural problems visible
    in the ingest log instead of silently producing garbage company names.
    """
    parent = file_path.parent.name.strip()
    parent_lower = parent.lower().replace(" ", "_")

    if parent_lower in _ROOT_FOLDER_NAMES:
        return "unknown"
    if parent_lower in _SOURCE_TYPE_FOLDER_NAMES:
        # File sits directly under a source-type folder with no company
        # folder beneath it -- structure mismatch, not a real company name.
        return "unknown"

    return parent.title()


# ─────────────────────────────────────────────
# SOURCE TYPE -- FROM GRANDPARENT FOLDER
# ─────────────────────────────────────────────

_FOLDER_TO_SOURCE_TYPE = {
    "annual_report":  "annual_report",
    "annual_reports": "annual_report",
    "earnings_call":  "earnings_call",
    "earnings_calls": "earnings_call",
    "earning_call":   "earnings_call",
    "earning_calls":  "earnings_call",
    "financial_news": "financial_news",
    "news":           "financial_news",
}


def _resolve_source_type(file_path: Path, file_ext: str) -> str:
    """
    Determine source_type from the file's GRANDPARENT folder name.

    REAL DATASET STRUCTURE (confirmed):
        documents/<source_type>/<company>/<file>
    Example: documents/Annual_reports/Infosys/infosys-ar-25.pdf

    So the SOURCE TYPE folder is the file's GRANDPARENT directory
    (two levels up -- the company folder sits BETWEEN the source-type
    folder and the file itself).

    Falls back to the raw file extension only if the grandparent folder
    name isn't recognized -- this fallback is logged so silent mis-tagging
    (e.g. earnings calls ending up labeled 'pdf') is visible in the
    ingest output rather than hidden.
    """
    grandparent_name = file_path.parent.parent.name.lower().replace(" ", "_")
    resolved = _FOLDER_TO_SOURCE_TYPE.get(grandparent_name)

    if resolved is None:
        print(f"[Loader] WARNING: grandparent folder '{file_path.parent.parent.name}' "
              f"not recognized as a source type for '{file_path.name}'. "
              f"Falling back to file extension '{file_ext.lstrip('.')}'. "
              f"Expected folder names: {sorted(_FOLDER_TO_SOURCE_TYPE.keys())}")
        return file_ext.lstrip(".")

    return resolved


# ─────────────────────────────────────────────
# CORE: Load a single PDF file
# ─────────────────────────────────────────────
def load_pdf(file_path: str, source_type: str = "") -> List[Dict[str, Any]]:
    """
    Extract text from every page of a PDF file.
    company_name is filled in by load_documents_from_folder() based on
    folder position -- this function handles text + fiscal_year (and
    publication_date for news articles).

    Parameters
    ----------
    file_path   : path to the PDF file
    source_type : 'annual_report' | 'earnings_call' | 'financial_news' | ''
                  Passed in by load_documents_from_folder() (already
                  resolved from folder structure) so fiscal_year extraction
                  can apply different rules for news vs. official reports.
                  Pass '' (default) only when calling this function
                  standalone outside the folder-scan pipeline.
    """
    if not PDF_SUPPORT:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    file_name = Path(file_path).name
    pages     = []
    all_text  = ""

    doc = fitz.open(file_path)

    for page_index in range(len(doc)):
        page     = doc[page_index]
        raw_text = page.get_text("text").strip()

        if not raw_text:
            continue

        if page_index == 0:
            all_text = raw_text
        elif page_index < 4:
            # Capture first 4 pages so _extract_year_from_text() reliably
            # has 300-500 lines available to scan -- this is now the
            # PRIMARY extraction method (not just a fallback), so we want
            # enough content even when page 1 is a low-text cover image.
            all_text += "\n" + raw_text

        pages.append({
            "text":             raw_text,
            "page_number":      page_index + 1,
            "file_name":        file_name,
            "source_type":      source_type or "pdf",
            "company_name":     None,
            "fiscal_year":      None,
            "publication_date": None,
        })

    doc.close()

    fiscal_year = _extract_fiscal_year(file_name, all_text, source_type=source_type)

    # publication_date is only meaningful (and only attempted) for news --
    # annual reports/earnings calls already have a reliable fiscal_year,
    # so we don't bother extracting a date for those source types.
    pub_date = _extract_publication_date(all_text) if source_type == "financial_news" else None

    for page in pages:
        page["fiscal_year"]      = fiscal_year
        page["publication_date"] = pub_date

    log_suffix = f" | published={pub_date}" if pub_date else ""
    print(f"[PDF Loader] '{file_name}' -> {len(pages)} pages | year={fiscal_year}{log_suffix}")
    return pages


# ─────────────────────────────────────────────
# CORE: Load a single TXT file
# ─────────────────────────────────────────────
def load_txt(file_path: str, source_type: str = "") -> List[Dict[str, Any]]:
    """
    Load a plain-text transcript file as a single 'page'.
    company_name is filled in by load_documents_from_folder().

    Parameters
    ----------
    file_path   : path to the TXT file
    source_type : 'annual_report' | 'earnings_call' | 'financial_news' | ''
                  See load_pdf() docstring for why this matters.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    file_name = Path(file_path).name

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read().strip()

    if not raw_text:
        print(f"[TXT Loader] WARNING: '{file_name}' is empty. Skipping.")
        return []

    fiscal_year = _extract_fiscal_year(file_name, raw_text, source_type=source_type)
    pub_date    = _extract_publication_date(raw_text) if source_type == "financial_news" else None

    log_suffix = f" | published={pub_date}" if pub_date else ""
    print(f"[TXT Loader] '{file_name}' -> 1 document | year={fiscal_year}{log_suffix}")

    return [{
        "text":             raw_text,
        "page_number":      None,
        "file_name":        file_name,
        "source_type":      source_type or "txt",
        "company_name":     None,
        "fiscal_year":      fiscal_year,
        "publication_date": pub_date,
    }]


# ─────────────────────────────────────────────
# CORE: Load an entire folder tree of documents
# ─────────────────────────────────────────────
def load_documents_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Recursively scan a folder and load all .pdf and .txt files.

    REAL DATASET STRUCTURE (confirmed):
      documents/
      |-- Annual_reports/
      |   |-- Cognizant/         <- company folder, source_type='annual_report'
      |   |-- Infosys/
      |   `-- TCS/
      |-- Earning_calls/
      |   |-- Cognizant/         <- company folder, source_type='earnings_call'
      |   |-- Infosys/
      |   `-- TCS/
      `-- Financial_News/
          |-- Cognizant/         <- company folder, source_type='financial_news'
          |-- Infosys/
          `-- TCS/

    source_type comes from the GRANDPARENT folder (e.g. 'Annual_reports').
    company_name comes from the IMMEDIATE PARENT folder (e.g. 'Infosys').
    Filenames are NEVER used to guess company name -- only used for fiscal
    year extraction as a first attempt before falling back to document text.

    Returns a flat list of page-dicts for all documents found.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_pages = []
    skipped_unknown_company = []

    files = sorted(folder.rglob("*"))

    for file in files:
        if not file.is_file():
            continue

        suffix = file.suffix.lower()
        if suffix not in (".pdf", ".txt"):
            continue

        company_name = _resolve_company_name(file)
        source_type  = _resolve_source_type(file, suffix)

        if company_name == "unknown":
            skipped_unknown_company.append(str(file))

        try:
            # source_type is now passed IN so fiscal_year extraction can
            # apply different rules for news vs. official reports (see
            # _extract_fiscal_year docstring for why this matters).
            if suffix == ".pdf":
                pages = load_pdf(str(file), source_type=source_type)
            else:
                pages = load_txt(str(file), source_type=source_type)

            for page in pages:
                page["company_name"] = company_name
                # source_type is already set correctly by load_pdf/load_txt
                # using the value we passed in -- no need to overwrite here.

            all_pages.extend(pages)

        except Exception as e:
            print(f"[Loader] ERROR loading '{file.name}': {e}")

    print(f"\n[Loader] Total pages/documents loaded: {len(all_pages)}")

    if skipped_unknown_company:
        print(f"\n[Loader] WARNING: {len(skipped_unknown_company)} file(s) had "
              f"an unresolved company name (folder structure mismatch). "
              f"These were still loaded with company_name='unknown' -- check "
              f"their folder placement:")
        for path in skipped_unknown_company[:10]:
            print(f"    - {path}")
        if len(skipped_unknown_company) > 10:
            print(f"    ... and {len(skipped_unknown_company) - 10} more")

    return all_pages
