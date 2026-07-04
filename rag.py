"""Scoped Spanish RAG fallback for the investment-analysis agent.

This module adds a **retrieval-augmented** capability that the LangGraph agent
uses as a *fallback* — when the live financial MCP tools can't answer a request,
Claude can consult a local corpus of Spanish Argentine legal documents in
``rag_data/`` (noisy-OCR notarial deeds: mostly *sociedad anónima* constitution
documents and statutes under Ley 19.550).

The capability is exposed as a single LangChain tool via :func:`build_rag_tool`,
which the agent binds alongside the MCP tools. Claude selects it by its
description only when the finance tools don't apply.

Two design choices are baked in (per project decision):

- **Local embeddings.** Retrieval uses ``sentence-transformers`` running on CPU,
  so there is no embeddings API key and no per-call cost. The FAISS index is
  built once and persisted to disk, then reloaded on subsequent runs.
- **Two-layer out-of-scope guardrail**, enforced *inside* the tool so the
  "only answer commercial-societies questions" rule lives at the retrieval
  boundary:

  1. a cheap Claude **scope classifier** (Haiku) that rejects any query not
     about Argentine commercial societies *before* retrieval runs, and
  2. a **relevance floor** — if the best-matching chunk is below a cosine
     similarity threshold, we return "no relevant records" instead of feeding
     weak matches to the model (guards against hallucination on in-domain but
     absent subjects).

Configuration (all optional, via environment):

- ``RAG_EMBED_MODEL``          sentence-transformers model name.
- ``RAG_GUARD_MODEL``          Claude model for the scope classifier.
- ``RAG_RELEVANCE_THRESHOLD``  cosine floor in [0, 1] (default 0.30).
- ``RAG_TOP_K``                number of chunks to return (default 4).
- ``RAG_INDEX_DIR``            where the FAISS index is persisted.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# The default embedding models are PUBLIC Hugging Face repos. If the machine has
# a stale/expired HF token cached (from a prior `huggingface-cli login` or an
# HF_TOKEN env var), the hub sends it implicitly and public downloads fail with a
# spurious 401. Disable implicit-token sending so anonymous access is used. A
# user who needs a *private* RAG_EMBED_MODEL can override this to "0" and provide
# a valid token.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).parent
CORPUS_DIR = _HERE / "rag_data"
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", _HERE / ".rag_index"))

# Lightweight multilingual model — good on Spanish, no prompt-prefix needs, and
# small enough to run on CPU. Swap for higher quality via RAG_EMBED_MODEL, e.g.
# "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" or
# "intfloat/multilingual-e5-base" (the latter expects "query:"/"passage:" prefixes).
EMBED_MODEL = os.environ.get(
    "RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Haiku is plenty for a binary in/out-of-scope decision, and keeps the guardrail cheap.
GUARD_MODEL = os.environ.get("RAG_GUARD_MODEL", "claude-haiku-4-5")

RELEVANCE_THRESHOLD = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.30"))
TOP_K = int(os.environ.get("RAG_TOP_K", "4"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TOOL_NAME = "consultar_registro_sociedades"
TOOL_DESCRIPTION = (
    "Fallback knowledge base of Argentine COMMERCIAL SOCIETIES records "
    "(registro de sociedades comerciales). Use this ONLY when the live financial "
    "tools cannot answer, and only for questions about Argentine commercial "
    "companies / sociedades comerciales — e.g. constitution deeds (constitución), "
    "bylaws/statutes (estatutos), corporate purpose (objeto social), directors "
    "(directorio), share capital (capital social), partners/shareholders "
    "(socios/accionistas), or Ley 19.550. The source documents are in Spanish, so "
    "the retrieved excerpts are in Spanish. Input: the user's question (Spanish or "
    "English). The tool enforces its own scope: unrelated questions are rejected."
)

# Guardrail responses (returned to the model as the tool result).
_OUT_OF_SCOPE_MSG = (
    "[FUERA DE ALCANCE] Esta herramienta solo responde consultas sobre sociedades "
    "comerciales argentinas (constitución, estatutos, objeto social, directorio, "
    "capital social, Ley 19.550). La consulta no corresponde a ese ámbito; no se "
    "realizó ninguna búsqueda."
)
_NO_RESULTS_MSG = (
    "[SIN RESULTADOS] No se encontraron registros relevantes de sociedades "
    "comerciales para esta consulta en el corpus disponible."
)

_SCOPE_SYSTEM_PROMPT = (
    "You are a strict scope classifier for a document retrieval tool. The tool "
    "only serves questions about ARGENTINE COMMERCIAL SOCIETIES / sociedades "
    "comerciales: company constitution deeds, bylaws/statutes, corporate purpose "
    "(objeto social), directors (directorio), share capital, partners or "
    "shareholders, and Ley 19.550 (sociedades comerciales). "
    "Reply with exactly one word: YES if the user's query is within that scope, "
    "or NO otherwise. Investment/markets/finance questions, general knowledge, "
    "weather, coding, etc. are all NO."
)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #

def _iter_corpus_files() -> list[Path]:
    """Return the sorted list of ``.txt`` files under the corpus directory."""
    return sorted(CORPUS_DIR.rglob("*.txt"))


def _corpus_signature(files: list[Path]) -> str:
    """Fingerprint the corpus (paths + sizes + mtimes) to detect changes.

    Lets us skip re-embedding when nothing has changed since the index was built.
    """
    h = hashlib.sha256()
    h.update(EMBED_MODEL.encode("utf-8"))
    for path in files:
        stat = path.stat()
        rel = path.relative_to(CORPUS_DIR).as_posix()
        h.update(f"{rel}|{stat.st_size}|{int(stat.st_mtime)}\n".encode("utf-8"))
    return h.hexdigest()


def _load_documents(files: list[Path]):
    """Read each corpus file into a chunked list of LangChain ``Document`` objects.

    The OCR text contains odd bytes, so decoding uses ``errors="replace"``. Each
    chunk keeps a ``source`` metadata field (folder/file) for citation.
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    docs: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        source = path.relative_to(CORPUS_DIR).as_posix()
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": source}))
    return docs


# --------------------------------------------------------------------------- #
# Embeddings + FAISS index (built once, persisted)
# --------------------------------------------------------------------------- #

def _build_embeddings():
    """Instantiate the local sentence-transformers embeddings.

    ``normalize_embeddings=True`` makes the FAISS L2 distance a direct function
    of cosine similarity (see :func:`_l2_to_cosine`).
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    # ``token=False`` forces anonymous Hugging Face access, which propagates
    # through every hub lookup (including transformers' PEFT-adapter probe that
    # ignores HF_HUB_DISABLE_IMPLICIT_TOKEN). This prevents a stale/expired
    # cached token from turning public-model downloads into spurious 401s. Set
    # RAG_HF_TOKEN to authenticate when using a private RAG_EMBED_MODEL.
    hf_token = os.environ.get("RAG_HF_TOKEN") or False
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu", "token": hf_token},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_or_build_index(embeddings):
    """Load the persisted FAISS index, rebuilding it if the corpus changed."""
    from langchain_community.vectorstores import FAISS

    files = _iter_corpus_files()
    if not files:
        raise RuntimeError(
            f"No .txt documents found under {CORPUS_DIR}. "
            "The RAG corpus is required to build the retrieval index."
        )

    signature = _corpus_signature(files)
    sig_path = INDEX_DIR / "corpus.sig"
    faiss_path = INDEX_DIR / "index.faiss"

    if faiss_path.exists() and sig_path.exists():
        try:
            if sig_path.read_text(encoding="utf-8").strip() == signature:
                return FAISS.load_local(
                    str(INDEX_DIR),
                    embeddings,
                    allow_dangerous_deserialization=True,  # our own local index
                )
        except Exception:
            pass  # fall through and rebuild on any load/read problem

    docs = _load_documents(files)
    if not docs:
        raise RuntimeError(f"Corpus under {CORPUS_DIR} produced no usable text.")

    store = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    sig_path.write_text(signature, encoding="utf-8")
    return store


def _l2_to_cosine(distance: float) -> float:
    """Convert FAISS squared-L2 distance to cosine similarity for unit vectors.

    For normalized vectors, ``||a-b||^2 = 2 - 2*cos``, so ``cos = 1 - d/2``.
    """
    return 1.0 - distance / 2.0


# --------------------------------------------------------------------------- #
# Guardrail: scope classifier
# --------------------------------------------------------------------------- #

def _in_scope(query: str) -> bool:
    """Return True if the query is about Argentine commercial societies.

    Uses a cheap Claude classification call. On any failure we *fail open* to
    retrieval — the relevance floor still guards against irrelevant answers.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        model = ChatAnthropic(model=GUARD_MODEL, max_tokens=5)
        resp = model.invoke(
            [SystemMessage(content=_SCOPE_SYSTEM_PROMPT), HumanMessage(content=query)]
        )
        verdict = str(resp.content).strip().upper()
        return verdict.startswith("YES") or verdict.startswith("SI")
    except Exception:
        return True  # fail open; relevance floor is the second line of defense


def _format_hits(hits) -> str:
    """Render retrieved (Document, cosine) hits into a cited, model-friendly block."""
    parts = [
        "Fragmentos recuperados del corpus de sociedades comerciales "
        "(en español, OCR; cita las fuentes en tu respuesta):",
        "",
    ]
    for doc, score in hits:
        source = doc.metadata.get("source", "desconocido")
        parts.append(f"[Documento: {source} · similitud {score:.2f}]")
        parts.append(doc.page_content.strip())
        parts.append("---")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Tool factory
# --------------------------------------------------------------------------- #

def build_rag_tool():
    """Build the scoped RAG fallback tool (index is built/loaded once here).

    Returns a LangChain ``StructuredTool``. The tool body is synchronous
    (sentence-transformers/FAISS are CPU-bound); the agent's ``tools`` node calls
    it via ``ainvoke``, which runs sync tools in a threadpool, so it never blocks
    the event loop.
    """
    from langchain_core.tools import StructuredTool

    embeddings = _build_embeddings()
    store = _load_or_build_index(embeddings)

    def _consultar(query: str) -> str:
        """Consult the commercial-societies corpus, enforcing scope + relevance."""
        query = (query or "").strip()
        if not query:
            return _NO_RESULTS_MSG

        # Layer 1 — scope guardrail: reject off-topic queries before retrieval.
        if not _in_scope(query):
            return _OUT_OF_SCOPE_MSG

        # Retrieve, then Layer 2 — relevance floor on the best cosine similarity.
        results = store.similarity_search_with_score(query, k=TOP_K)
        hits = [(doc, _l2_to_cosine(dist)) for doc, dist in results]
        hits = [(doc, cos) for doc, cos in hits if cos >= RELEVANCE_THRESHOLD]
        if not hits:
            return _NO_RESULTS_MSG

        return _format_hits(hits)

    return StructuredTool.from_function(
        func=_consultar,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
    )
