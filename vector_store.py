"""
vector_store.py
───────────────
Part 1 — Steps 2, 3, 4, 5: Chunking → Embeddings → ChromaDB → Retrieval

Responsibilities:
  - Split page text into overlapping chunks
  - Attach frozen metadata fields to every chunk
  - Embed chunks with OpenAI or a local SentenceTransformer model
  - Build and persist a ChromaDB vector store (replaces FAISS)
  - Retrieve the top-k most similar chunks for a given query

Why ChromaDB instead of FAISS?
  - ChromaDB stores vectors AND metadata AND text together in one place
  - No need for a separate chunk_metadata.json file
  - Persistent by default — survives restarts automatically
  - Beginner-friendly: no manual numpy normalization needed
  - Supports metadata filtering (e.g. query only Infosys documents)

Install:
  pip install chromadb sentence-transformers openai numpy
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# ChromaDB — vector store with built-in persistence
# Install: pip install chromadb
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[WARNING] chromadb not installed. Run: pip install chromadb")

import numpy as np


# ─────────────────────────────────────────────
# CONFIGURATION  (edit these as needed)
# ─────────────────────────────────────────────
CHUNK_SIZE        = 500      # max characters per chunk
CHUNK_OVERLAP     = 100      # characters of overlap between consecutive chunks
DEFAULT_TOP_K     = 5        # how many chunks to retrieve per query

# ChromaDB persistence folder and collection name
CHROMA_DIR        = "chroma_db"          # folder where ChromaDB saves its files
COLLECTION_NAME   = "financial_docs"     # name of the collection inside ChromaDB


# ─────────────────────────────────────────────
# STEP 2: CHUNKING
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# COMPANY NAME NORMALIZATION
# ─────────────────────────────────────────────
def _normalize_company(name: str) -> str:
    """
    Normalize a company name for reliable exact-match filtering in ChromaDB.

    WHY THIS EXISTS: ChromaDB's where-filter only supports exact string
    match ($eq) — no case-insensitivity, no partial match. If documents
    were ingested with company_name="Cognizant" (from a folder named
    "cognizant") but a query asks for filter_company="cognizant Ltd" or
    "COGNIZANT", the $eq filter silently returns ZERO chunks — no error,
    just an empty result that makes the LLM say "I don't have this data."

    Fix: lowercase + strip both the stored metadata value AND the query
    filter value before comparing, so casing/whitespace differences can
    never cause a silent retrieval failure.
    """
    if not name:
        return ""
    return name.strip().lower()


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split each page's text into overlapping fixed-size character chunks.

    Each chunk gets the frozen metadata fields required by project_contract.json:
      chunk_id, text, company_name, fiscal_year, source_type, file_name, page_number

    Input:  list of page-dicts from document_processor.py
    Output: flat list of chunk-dicts
    """
    all_chunks = []

    for page in pages:
        text       = page["text"]
        file_stem  = Path(page["file_name"]).stem   # e.g. 'infosys_ar_2024'
        page_label = page["page_number"] if page["page_number"] else 0

        # Slide a window over the text to produce overlapping chunks
        start       = 0
        chunk_index = 0

        while start < len(text):
            end        = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:   # skip whitespace-only windows
                # chunk_id format is frozen — defined in project_contract.json
                chunk_id = f"{file_stem}_p{page_label}_c{chunk_index}"

                all_chunks.append({
                    # ── frozen fields (contract schema) ──────────────────────
                    "chunk_id":         chunk_id,
                    "text":             chunk_text,
                    "company_name":     page["company_name"],
                    "fiscal_year":      page["fiscal_year"],
                    "source_type":      page["source_type"],
                    "file_name":        page["file_name"],
                    "page_number":      page["page_number"],
                    # ── internal field — filled at retrieval time ─────────────
                    "similarity_score": None,
                })

            start       += CHUNK_SIZE - CHUNK_OVERLAP
            chunk_index += 1

    print(f"[Chunker] Created {len(all_chunks)} chunks from {len(pages)} pages/docs")
    return all_chunks


# ─────────────────────────────────────────────
# STEP 3: EMBEDDINGS
# Two backends: OpenAI (cloud) or SentenceTransformers (local/free)
# ─────────────────────────────────────────────

# ── Module-level model cache ──────────────────────────────────────────────────
# CRITICAL FIX: SentenceTransformer was being instantiated fresh on every single
# call to _embed_local() — once per chat message, once per ratio question, once
# per report section. Each load pushes the model onto the GPU (MPS on Mac,
# CUDA on Windows/Linux) and never releases the old one, causing memory to pile
# up until the GPU runs out and crashes with kIOGPUCommandBufferCallbackErrorOutOfMemory.
#
# Fix: load the model ONCE per process and reuse the same instance forever.
# Also force CPU explicitly — this embedding model is tiny (~80MB) and CPU
# inference is fast enough that GPU offload isn't worth the instability.
_LOCAL_MODEL_CACHE: dict = {}   # {model_name: SentenceTransformer instance}

# Force CPU for local embeddings — prevents MPS/CUDA out-of-memory crashes.
# This model is small enough that CPU inference is plenty fast for RAG retrieval.
EMBEDDING_DEVICE = "cpu"


def _get_local_model(model_name: str = "all-MiniLM-L6-v2") -> "SentenceTransformer":
    """
    Return a cached SentenceTransformer instance, loading it only once.

    This is the fix for the GPU memory leak: previously every call to
    _embed_local() created a brand new model instance and loaded it onto
    the GPU (MPS/CUDA), never releasing the previous one. Now the model is
    loaded once per model_name and reused for the lifetime of the process.
    """
    if model_name not in _LOCAL_MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        print(f"[Embedder] Loading local model '{model_name}' on "
              f"device='{EMBEDDING_DEVICE}' (one-time load, cached for reuse) …")
        _LOCAL_MODEL_CACHE[model_name] = SentenceTransformer(
            model_name, device=EMBEDDING_DEVICE
        )
    return _LOCAL_MODEL_CACHE[model_name]


def _embed_openai(texts: List[str],
                  model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Embed a list of strings using the OpenAI Embeddings API.
    Requires OPENAI_API_KEY environment variable.

    Returns a plain Python list of float lists (ChromaDB-friendly format).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client   = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def _embed_local(texts: List[str],
                 model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Embed a list of strings using a local SentenceTransformer model.
    Runs entirely offline — no API key needed. Runs on CPU (forced) to
    avoid GPU/MPS out-of-memory crashes on repeated calls.

    The model is loaded ONCE via _get_local_model() and reused for every
    subsequent call — this is what prevents the memory leak.

    Default model 'all-MiniLM-L6-v2' is ~80MB and very fast even on CPU.
    For better quality try 'BAAI/bge-small-en-v1.5'.

    Returns a plain Python list of float lists (ChromaDB-friendly format).
    """
    model   = _get_local_model(model_name)   # cached — no reload, no GPU push
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    # ChromaDB accepts plain Python lists — convert from numpy
    return vectors.tolist()


def embed_texts(texts: List[str],
                use_openai:   bool = False,
                openai_model: str  = "text-embedding-3-small",
                local_model:  str  = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Embed a list of raw text strings.

    Returns a list of float lists ready for ChromaDB's .add() method.
    """
    print(f"[Embedder] Embedding {len(texts)} texts "
          f"({'OpenAI: ' + openai_model if use_openai else 'Local: ' + local_model}) …")

    if use_openai:
        vectors = _embed_openai(texts, model=openai_model)
    else:
        vectors = _embed_local(texts, model_name=local_model)

    print(f"[Embedder] Done. {len(vectors)} vectors produced.")
    return vectors


# ─────────────────────────────────────────────
# STEP 4: BUILD & PERSIST ChromaDB COLLECTION
# ─────────────────────────────────────────────

def get_chroma_client() -> "chromadb.PersistentClient":
    """
    Create (or reconnect to) a persistent ChromaDB client.

    ChromaDB automatically saves everything to CHROMA_DIR on disk.
    No manual save() call needed — it persists after every .add().
    """
    if not CHROMA_AVAILABLE:
        raise RuntimeError("chromadb not installed. Run: pip install chromadb")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client


def build_chroma_collection(chunks:      List[Dict[str, Any]],
                             use_openai:  bool = False,
                             openai_model:str  = "text-embedding-3-small",
                             local_model: str  = "all-MiniLM-L6-v2"):
    """
    Embed all chunks and store them in a ChromaDB persistent collection.

    ChromaDB stores text + embeddings + metadata all in one place.
    No separate JSON metadata file needed (unlike FAISS).

    Parameters
    ----------
    chunks       : list of chunk-dicts from chunk_pages()
    use_openai   : True → OpenAI embeddings, False → local SentenceTransformer
    openai_model : OpenAI model name (used only if use_openai=True)
    local_model  : local model name (used only if use_openai=False)

    Returns
    -------
    chromadb.Collection — the populated collection, ready to query
    """
    client = get_chroma_client()

    # Delete existing collection if it exists so we start fresh on re-ingest
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"[ChromaDB] Deleted existing collection '{COLLECTION_NAME}' for fresh ingest.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        # cosine distance = 1 - cosine_similarity; lower distance = more similar
        metadata={"hnsw:space": "cosine"}
    )

    # ── Prepare data for ChromaDB .add() ────────────────────────────────────
    # ChromaDB .add() takes parallel lists: ids, documents, embeddings, metadatas

    ids        = [c["chunk_id"] for c in chunks]
    documents  = [c["text"]     for c in chunks]

    # ChromaDB metadata values must be str, int, float, or bool — not None
    # We convert page_number=None → -1 (sentinel) so ChromaDB accepts it
    #
    # company_name_lower is a NEW field added alongside company_name.
    # company_name keeps the original display casing (e.g. "Cognizant")
    # for citations/UI. company_name_lower is used for the actual $eq
    # filter at query time, so casing differences in the user's question
    # can never cause a silent zero-result retrieval failure.
    metadatas  = [
        {
            "company_name":       c["company_name"] or "unknown",
            "company_name_lower": _normalize_company(c["company_name"]) or "unknown",
            "fiscal_year":        c["fiscal_year"]  or "unknown",
            "source_type":        c["source_type"]  or "unknown",
            "file_name":          c["file_name"]    or "unknown",
            "page_number":        c["page_number"]  if c["page_number"] is not None else -1,
        }
        for c in chunks
    ]

    # Embed all chunk texts
    embeddings = embed_texts(
        documents,
        use_openai=use_openai,
        openai_model=openai_model,
        local_model=local_model,
    )

    # ── Add to ChromaDB in batches of 500 (safe for large document sets) ────
    BATCH_SIZE = 500
    total      = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            ids        = ids[start:end],
            documents  = documents[start:end],
            embeddings = embeddings[start:end],
            metadatas  = metadatas[start:end],
        )
        print(f"[ChromaDB] Added chunks {start + 1}–{end} of {total}")

    print(f"\n[ChromaDB] Collection '{COLLECTION_NAME}' built. "
          f"Total vectors: {collection.count()} | Saved to ./{CHROMA_DIR}/")
    return collection


def load_chroma_collection():
    """
    Load the existing ChromaDB collection from disk.

    Returns
    -------
    chromadb.Collection — ready for querying
    """
    client = get_chroma_client()

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing:
        raise FileNotFoundError(
            f"ChromaDB collection '{COLLECTION_NAME}' not found. "
            "Run the pipeline with --ingest first."
        )

    collection = client.get_collection(COLLECTION_NAME)
    print(f"[ChromaDB] Loaded collection '{COLLECTION_NAME}' "
          f"with {collection.count()} vectors from ./{CHROMA_DIR}/")
    return collection


def get_available_fiscal_years(collection: "chromadb.Collection",
                                company: str = None) -> List[str]:
    """
    Discover every distinct fiscal_year value actually present in the
    vector store, optionally scoped to one company.

    WHY THIS EXISTS: trend charts, comparisons, and reports previously
    used a HARDCODED default year list (e.g. "FY2022, FY2023, FY2024")
    in the dashboard UI, which silently EXCLUDED any years actually
    present in the ingested documents (e.g. FY2025, FY2026) unless the
    user manually retyped the field. This function queries the real
    data so callers can build charts that cover every year that's
    actually there -- completeness over guessing.

    Parameters
    ----------
    collection : ChromaDB collection (from load_chroma_collection())
    company    : optional -- restrict to one company's fiscal years.
                 When provided, uses the normalized company_name_lower
                 field so casing differences don't cause empty results.

    Returns
    -------
    Sorted list of fiscal year strings, e.g. ["FY2023", "FY2024", "FY2025"].
    "unknown" is excluded from the returned list (it's not a real,
    chartable year). Sorted chronologically (oldest first) so charts
    render left-to-right in the correct order.
    """
    get_kwargs = {"include": ["metadatas"], "limit": 100000}
    if company:
        get_kwargs["where"] = {
            "company_name_lower": {"$eq": _normalize_company(company)}
        }

    result = collection.get(**get_kwargs)
    metadatas = result.get("metadatas", [])

    years = {
        m.get("fiscal_year", "unknown")
        for m in metadatas
        if m.get("fiscal_year", "unknown") != "unknown"
    }

    # Sort chronologically: "FY2023" -> 2023 for sort key, "FY2024" -> 2024, etc.
    def _year_sort_key(fy: str) -> int:
        digits = "".join(ch for ch in fy if ch.isdigit())
        return int(digits) if digits else 0

    return sorted(years, key=_year_sort_key)


# ─────────────────────────────────────────────
# STEP 5: RETRIEVER
# ─────────────────────────────────────────────

def retrieve_chunks(query:        str,
                    collection:      "chromadb.Collection",
                    top_k:           int  = DEFAULT_TOP_K,
                    use_openai:      bool = False,
                    openai_model:    str  = "text-embedding-3-small",
                    local_model:     str  = "all-MiniLM-L6-v2",
                    filter_company:  str  = None,
                    filter_fiscal_year: str = None) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve the top-k most similar chunks from ChromaDB.

    Parameters
    ----------
    query              : the user's natural-language question
    collection         : ChromaDB collection (from load_chroma_collection)
    top_k              : number of chunks to return
    use_openai         : which embedding backend to use (must match ingest)
    filter_company     : optional — restrict results to one company
    filter_fiscal_year : optional — restrict results to one fiscal year
                         e.g. "FY2024". When both company and fiscal year
                         are provided, ChromaDB applies BOTH filters with $and,
                         which is what makes "Cognizant revenue in FY2024"
                         return FY2024 chunks instead of boilerplate pages
                         from FY2023/FY2025 that score similarly on embedding.

    Fallback strategy  : if company+year filter returns zero chunks, retry
                         with company filter only; if still zero, retry
                         with no filter at all and log a warning.

    Returns
    -------
    List of chunk-dicts with 'similarity_score' filled in.
    """
    # Embed the query using the same backend as ingest
    query_vector = embed_texts(
        [query],
        use_openai=use_openai,
        openai_model=openai_model,
        local_model=local_model,
    )

    # ── Build metadata filter ────────────────────────────────────────────────
    # Use ChromaDB's $and operator when both company and fiscal year are known.
    # This prevents the "wrong year" problem: without a year filter, semantically
    # similar boilerplate pages from FY2023/FY2025 crowd out the FY2024-specific
    # revenue chunk — even when the relevant chunk clearly exists in the index.
    def _build_filter(company: str = None, fiscal_year: str = None):
        conditions = []
        if company:
            conditions.append(
                {"company_name_lower": {"$eq": _normalize_company(company)}}
            )
        if fiscal_year and fiscal_year.upper() != "UNKNOWN":
            conditions.append(
                {"fiscal_year": {"$eq": fiscal_year.upper()}}
            )
        if len(conditions) == 0:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    where_filter = _build_filter(filter_company, filter_fiscal_year)

    # ── Query ChromaDB ───────────────────────────────────────────────────────
    actual_k = min(top_k, collection.count())

    query_kwargs = {
        "query_embeddings": query_vector,
        "n_results":        actual_k,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    # ── Fallback 1: company+year filter returned zero → try company only ─────
    if (where_filter and filter_fiscal_year
            and len(results.get("ids", [[]])[0]) == 0):
        print(
            f"[Retriever] WARNING: company='{filter_company}' + "
            f"fiscal_year='{filter_fiscal_year}' filter matched ZERO chunks. "
            f"Retrying with company filter only."
        )
        fallback_filter = _build_filter(filter_company, None)
        query_kwargs["where"] = fallback_filter
        results = collection.query(**query_kwargs)

    # ── Fallback 2: company filter returned zero → try unfiltered ────────────
    if where_filter and len(results.get("ids", [[]])[0]) == 0:
        print(
            f"[Retriever] WARNING: company filter '{filter_company}' matched "
            f"ZERO chunks. This usually means the company was never ingested. "
            f"Falling back to unfiltered search."
        )
        query_kwargs.pop("where", None)
        results = collection.query(**query_kwargs)

    # ── Parse ChromaDB results into contract-format chunk-dicts ─────────────
    # ChromaDB returns parallel lists inside results["ids"][0], etc.
    # distances are cosine distances (0 = identical, 2 = opposite)
    # We convert to similarity_score = 1 - distance  (higher = more similar)

    retrieved = []
    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        similarity_score = round(1 - distance, 4)   # cosine similarity

        # Restore page_number: -1 sentinel → None (as defined in contract)
        page_number = meta.get("page_number", -1)
        if page_number == -1:
            page_number = None

        retrieved.append({
            # ── frozen contract fields ────────────────────────────────────
            "chunk_id":         chunk_id,
            "text":             text,
            "company_name":     meta.get("company_name", "unknown"),
            "fiscal_year":      meta.get("fiscal_year",  "unknown"),
            "source_type":      meta.get("source_type",  "unknown"),
            "file_name":        meta.get("file_name",    "unknown"),
            "page_number":      page_number,
            "similarity_score": similarity_score,
        })

    if retrieved:
        print(f"[Retriever] Query: '{query[:60]}…'\n"
              f"            Retrieved {len(retrieved)} chunks "
              f"(top score: {retrieved[0]['similarity_score']:.4f})")
    else:
        print(f"[Retriever] No results found for query: '{query[:60]}'")

    return retrieved
