"""Scoped Spanish RAG fallback for the investment-analysis agent.

This module adds a **retrieval-augmented** capability that the LangGraph agent
uses as a *fallback* — when the live financial MCP tools can't answer a request,
Claude can consult a local corpus of Spanish Argentine legal documents in
``rag_data/`` (noisy-OCR notarial deeds: mostly *sociedad anónima* constitution
documents and statutes under Ley 19.550).

The capability is exposed as a single LangChain tool via :func:`build_rag_tool`,
which the agent binds alongside the MCP tools. Claude selects it by its
description only when the finance tools don't apply.

Retrieval is **hybrid + reranked**, which suits noisy OCR where exact tokens
(company names, CUIT/DNI numbers, article references) matter as much as meaning:

1. **Dense** — semantic search over local ``sentence-transformers`` embeddings in
   a persisted FAISS index (cosine similarity). No embeddings API key, no
   per-call cost.
2. **Sparse** — BM25 lexical search over the same chunks (``rank_bm25``), which
   catches exact-string matches dense vectors miss.
3. **Fusion** — the two rankings are merged with **Reciprocal Rank Fusion (RRF)**,
   robust to the different score scales of cosine vs BM25.
4. **Rerank** — a multilingual **cross-encoder** re-scores the fused candidates
   against the query and picks the final top-k.

A **two-layer out-of-scope guardrail** is enforced *inside* the tool so the
"only answer commercial-societies questions" rule lives at the retrieval
boundary:

- a cheap Claude **scope classifier** (Haiku) rejects any query not about
  Argentine commercial societies *before* retrieval runs, and
- a **relevance floor** — a candidate is kept only if its reranker probability or
  its dense cosine clears a threshold; if none clear it, the tool returns "no
  relevant records" instead of feeding weak matches to the model (guards against
  hallucination on in-domain but absent subjects).

If ``rank_bm25`` or the reranker model is unavailable, retrieval degrades
gracefully (dense-only, and/or fusion order without reranking).

Configuration (all optional, via environment):

- ``RAG_EMBED_MODEL``          sentence-transformers embedding model.
- ``RAG_RERANK_MODEL``         cross-encoder reranker (``""``/``none`` disables it).
- ``RAG_GUARD_MODEL``          Claude model for the scope classifier.
- ``RAG_RELEVANCE_THRESHOLD``  dense cosine floor in [0, 1] (default 0.30).
- ``RAG_RERANK_THRESHOLD``     reranker probability floor in [0, 1] (default 0.30).
- ``RAG_DENSE_K`` / ``RAG_SPARSE_K``  first-stage candidates per retriever (20 each).
- ``RAG_FUSE_K``               candidates kept after fusion, fed to the reranker (12).
- ``RAG_TOP_K``                final chunks returned (default 4).
- ``RAG_INDEX_DIR``            where the FAISS index + chunk cache are persisted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

# The default models are PUBLIC Hugging Face repos. If the machine has a
# stale/expired HF token cached (from a prior `huggingface-cli login` or an
# HF_TOKEN env var), the hub sends it implicitly and public downloads fail with a
# spurious 401. Disable implicit-token sending so anonymous access is used. A
# user who needs a *private* model can set RAG_HF_TOKEN (see _hf_token()).
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).parent
CORPUS_DIR = _HERE / "rag_data"
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", _HERE / ".rag_index"))

# Lightweight multilingual embedder — good on Spanish, no prompt-prefix needs,
# small enough for CPU. Swap via RAG_EMBED_MODEL, e.g.
# "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" for higher quality.
EMBED_MODEL = os.environ.get(
    "RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Multilingual cross-encoder reranker (trained on mMARCO, incl. Spanish). Set
# RAG_RERANK_MODEL to "" / "none" to disable reranking (fusion order is kept).
RERANK_MODEL = os.environ.get(
    "RAG_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

# Haiku is plenty for a binary in/out-of-scope decision, and keeps the guardrail cheap.
GUARD_MODEL = os.environ.get("RAG_GUARD_MODEL", "claude-haiku-4-5")

# Relevance floor (guardrail). A candidate is kept if EITHER signal clears its
# threshold — the reranker catches semantic/lexical relevance, cosine backs it up.
RELEVANCE_THRESHOLD = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.30"))
RERANK_THRESHOLD = float(os.environ.get("RAG_RERANK_THRESHOLD", "0.30"))

DENSE_K = int(os.environ.get("RAG_DENSE_K", "20"))   # dense first-stage candidates
SPARSE_K = int(os.environ.get("RAG_SPARSE_K", "20"))  # BM25 first-stage candidates
FUSE_K = int(os.environ.get("RAG_FUSE_K", "12"))      # kept after fusion → reranker
TOP_K = int(os.environ.get("RAG_TOP_K", "4"))         # final chunks returned
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))        # RRF damping constant

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


def _hf_token():
    """Token for Hugging Face downloads: a value from RAG_HF_TOKEN, else ``False``.

    ``False`` forces anonymous access, which propagates through every hub lookup
    (including transformers' config/tokenizer/PEFT probes that ignore
    ``HF_HUB_DISABLE_IMPLICIT_TOKEN``) and prevents a stale/expired cached token
    from turning public-model downloads into spurious 401s.
    """
    return os.environ.get("RAG_HF_TOKEN") or False


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
# Embeddings + FAISS index + chunk cache (built once, persisted)
# --------------------------------------------------------------------------- #

def _build_embeddings():
    """Instantiate the local sentence-transformers embeddings.

    ``normalize_embeddings=True`` makes the FAISS L2 distance a direct function
    of cosine similarity (see :func:`_l2_to_cosine`).
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu", "token": _hf_token()},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_or_build_index(embeddings):
    """Load the persisted FAISS index + chunk cache, rebuilding if the corpus changed.

    Returns ``(store, chunks)`` where ``chunks`` is a list of
    ``{"content": str, "source": str}`` — the same text the FAISS index holds,
    cached so BM25 and the reranker can be rebuilt in-process without re-reading
    the corpus.
    """
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
    chunks_path = INDEX_DIR / "chunks.json"

    if faiss_path.exists() and sig_path.exists() and chunks_path.exists():
        try:
            if sig_path.read_text(encoding="utf-8").strip() == signature:
                store = FAISS.load_local(
                    str(INDEX_DIR),
                    embeddings,
                    allow_dangerous_deserialization=True,  # our own local index
                )
                chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
                return store, chunks
        except Exception:
            pass  # fall through and rebuild on any load/read problem

    docs = _load_documents(files)
    if not docs:
        raise RuntimeError(f"Corpus under {CORPUS_DIR} produced no usable text.")

    store = FAISS.from_documents(docs, embeddings)
    chunks = [
        {"content": d.page_content, "source": d.metadata.get("source", "desconocido")}
        for d in docs
    ]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    sig_path.write_text(signature, encoding="utf-8")
    return store, chunks


def _l2_to_cosine(distance: float) -> float:
    """Convert FAISS squared-L2 distance to cosine similarity for unit vectors.

    For normalized vectors, ``||a-b||^2 = 2 - 2*cos``, so ``cos = 1 - d/2``.
    """
    return 1.0 - distance / 2.0


# --------------------------------------------------------------------------- #
# Sparse retrieval (BM25) + reranker
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer (Unicode-aware, keeps Spanish accents)."""
    return _TOKEN_RE.findall(text.lower())


def _build_bm25(chunks):
    """Build a BM25 index over the chunk texts, or ``None`` if ``rank_bm25`` is absent."""
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return None
    return BM25Okapi([_tokenize(c["content"]) for c in chunks])


def _load_cross_encoder():
    """Load the multilingual cross-encoder reranker, or ``None`` if disabled/unavailable."""
    if RERANK_MODEL.strip().lower() in ("", "none", "off", "0", "false"):
        return None
    try:
        import inspect

        from sentence_transformers import CrossEncoder

        token = _hf_token()
        # Pass token=False on every from_pretrained path (model/config/tokenizer)
        # so a stale cached HF token can't 401 the public reranker download. The
        # tokenizer kwarg was renamed tokenizer_kwargs → processor_kwargs across
        # sentence-transformers versions, so pass whichever the installed one
        # accepts (keeps this working — and quiet — across versions).
        params = inspect.signature(CrossEncoder.__init__).parameters
        kwargs = {
            name: {"token": token}
            for name in ("model_kwargs", "config_kwargs")
            if name in params
        }
        if "processor_kwargs" in params:
            kwargs["processor_kwargs"] = {"token": token}
        elif "tokenizer_kwargs" in params:
            kwargs["tokenizer_kwargs"] = {"token": token}
        return CrossEncoder(RERANK_MODEL, **kwargs)
    except Exception:
        return None  # degrade to fusion order + cosine floor


def _reciprocal_rank_fusion(ranked_lists) -> dict:
    """Fuse several ranked lists of keys into ``{key: fused_score}`` via RRF.

    RRF adds ``1 / (RRF_K + rank)`` per list, so it combines rankings without
    needing the underlying scores to share a scale (cosine vs BM25).
    """
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    return scores


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
    """Render retrieved hit dicts into a cited, model-friendly block."""
    parts = [
        "Fragmentos recuperados del corpus de sociedades comerciales "
        "(en español, OCR; cita las fuentes en tu respuesta):",
        "",
    ]
    for hit in hits:
        bits = []
        if hit.get("rerank") is not None:
            bits.append(f"rerank {hit['rerank']:.2f}")
        if hit.get("cosine") is not None:
            bits.append(f"coseno {hit['cosine']:.2f}")
        score_str = (" · " + ", ".join(bits)) if bits else ""
        parts.append(f"[Documento: {hit['source']}{score_str}]")
        parts.append(hit["content"].strip())
        parts.append("---")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Tool factory
# --------------------------------------------------------------------------- #

def build_rag_tool():
    """Build the scoped hybrid-RAG fallback tool (indexes are built/loaded once here).

    Returns a LangChain ``StructuredTool``. The tool body is synchronous
    (sentence-transformers/FAISS/BM25 are CPU-bound); the agent's ``rag`` node
    calls it via ``ainvoke``, which runs sync tools in a threadpool, so it never
    blocks the event loop.
    """
    import numpy as np
    from langchain_core.tools import StructuredTool

    embeddings = _build_embeddings()
    store, chunks = _load_or_build_index(embeddings)
    bm25 = _build_bm25(chunks)
    cross_encoder = _load_cross_encoder()

    def _search(query: str):
        """Hybrid retrieve → fuse → rerank → relevance floor. Returns hit dicts or None."""
        text_by_key: dict = {}
        source_by_key: dict = {}
        cosine_by_key: dict = {}

        # 1. Dense (semantic) candidates.
        dense_rank = []
        for doc, dist in store.similarity_search_with_score(query, k=DENSE_K):
            source = doc.metadata.get("source", "desconocido")
            key = f"{source}\x00{doc.page_content}"
            text_by_key.setdefault(key, doc.page_content)
            source_by_key.setdefault(key, source)
            cosine_by_key[key] = _l2_to_cosine(dist)
            dense_rank.append(key)

        # 2. Sparse (BM25 lexical) candidates.
        sparse_rank = []
        if bm25 is not None:
            scores = bm25.get_scores(_tokenize(query))
            for i in np.argsort(scores)[::-1][:SPARSE_K]:
                if scores[i] <= 0:
                    continue
                chunk = chunks[i]
                key = f"{chunk['source']}\x00{chunk['content']}"
                text_by_key.setdefault(key, chunk["content"])
                source_by_key.setdefault(key, chunk["source"])
                sparse_rank.append(key)

        # 3. Reciprocal Rank Fusion of the two rankings.
        fused = _reciprocal_rank_fusion([dense_rank, sparse_rank])
        candidates = sorted(fused, key=fused.get, reverse=True)[:FUSE_K]
        if not candidates:
            return None

        # 4. Cross-encoder rerank (probability via sigmoid), else keep fusion order.
        rerank_by_key: dict = {}
        if cross_encoder is not None:
            logits = cross_encoder.predict([(query, text_by_key[k]) for k in candidates])
            probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))
            rerank_by_key = {k: float(p) for k, p in zip(candidates, probs)}
            ordered = sorted(candidates, key=lambda k: rerank_by_key[k], reverse=True)
        else:
            ordered = candidates

        # 5. Relevance floor (guardrail): keep a candidate if either signal clears
        #    its threshold; if none do, the subject is treated as absent.
        def passes(key: str) -> bool:
            if key in rerank_by_key and rerank_by_key[key] >= RERANK_THRESHOLD:
                return True
            return cosine_by_key.get(key, 0.0) >= RELEVANCE_THRESHOLD

        kept = [k for k in ordered if passes(k)][:TOP_K]
        if not kept:
            return None

        return [
            {
                "source": source_by_key[k],
                "content": text_by_key[k],
                "cosine": cosine_by_key.get(k),
                "rerank": rerank_by_key.get(k),
            }
            for k in kept
        ]

    def _consultar(query: str) -> str:
        """Consult the commercial-societies corpus, enforcing scope + relevance."""
        query = (query or "").strip()
        if not query:
            return _NO_RESULTS_MSG

        # Layer 1 — scope guardrail: reject off-topic queries before retrieval.
        if not _in_scope(query):
            return _OUT_OF_SCOPE_MSG

        # Layers 2+ — hybrid retrieval, rerank, and the relevance floor.
        hits = _search(query)
        if not hits:
            return _NO_RESULTS_MSG
        return _format_hits(hits)

    return StructuredTool.from_function(
        func=_consultar,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
    )
