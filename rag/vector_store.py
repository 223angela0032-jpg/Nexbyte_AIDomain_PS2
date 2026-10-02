# rag/vector_store.py
"""
FAISS vector store for official scholarship documents.

Files (inside vector_db/):
    scholarship.index   - FAISS index of all chunk embeddings
    metadata.json       - [{"text": ..., "metadata": {...}}, ...] in the SAME order as the index

Each chunk's metadata carries 'opportunity_id', 'source' (file name) and 'page', so search
can be limited to one scholarship's own documents.
"""
import json
from pathlib import Path

import numpy as np

from config import RAG_TOP_K, VECTOR_DB_DIR
from rag.embeddings import embed_text, embed_texts

_STORE_DIR = Path(VECTOR_DB_DIR)
INDEX_FILE = _STORE_DIR / "scholarship.index"
METADATA_FILE = _STORE_DIR / "metadata.json"


def _faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required for RAG. Run: pip install faiss-cpu") from exc
    return faiss


# ---------------------------------------------------------------- build / persist
def create_index(embeddings):
    """Inner-product index; with normalized vectors this equals cosine similarity."""
    faiss = _faiss()
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def _write(index, items):
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    _faiss().write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(items, file, ensure_ascii=False, indent=2)


def _split_chunks(chunks):
    """Accept strings or dicts; return (texts, metadata) with empty chunks dropped."""
    texts, metadata = [], []
    for chunk in chunks:
        if isinstance(chunk, str):
            text, meta = chunk, {}
        elif isinstance(chunk, dict):
            text = chunk.get("text", "")
            meta = {k: v for k, v in chunk.items() if k != "text"}
        else:
            raise TypeError("Each chunk must be a string or dictionary.")
        text = (text or "").strip()
        if text:
            texts.append(text)
            metadata.append(meta)
    return texts, metadata


def save_vector_store(texts, metadata=None):
    """Replace the whole store with these texts. Returns the number stored."""
    if not texts:
        raise ValueError("No texts provided.")
    metadata = metadata if metadata is not None else [{} for _ in texts]
    if len(texts) != len(metadata):
        raise ValueError("texts and metadata must have the same length.")

    index = create_index(embed_texts(texts))
    _write(index, [{"text": t, "metadata": m} for t, m in zip(texts, metadata)])
    return len(texts)


def rebuild_vector_store(chunks):
    """Replace the whole store with these chunks (strings or dicts). Returns the count."""
    if not chunks:
        raise ValueError("No chunks provided.")
    texts, metadata = _split_chunks(chunks)
    if not texts:
        raise ValueError("No valid text chunks found.")
    return save_vector_store(texts, metadata)


# ---------------------------------------------------------------- load / check
def vector_store_exists() -> bool:
    return INDEX_FILE.exists() and METADATA_FILE.exists()


def load_vector_store():
    """(index, metadata) or (None, None) if the store does not exist / is inconsistent."""
    if not vector_store_exists():
        return None, None
    index = _faiss().read_index(str(INDEX_FILE))
    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if index.ntotal != len(metadata):
        return None, None
    return index, metadata


def delete_vector_store():
    for path in (INDEX_FILE, METADATA_FILE):
        if path.exists():
            path.unlink()


# ---------------------------------------------------------------- incremental updates
def add_chunks(chunks) -> int:
    """
    Append chunks to the existing store (creating it if needed).
    This is how an admin-uploaded PDF becomes searchable without
    wiping other scholarships' documents. Returns the number added.
    """
    texts, metadata = _split_chunks(chunks or [])
    if not texts:
        return 0

    new_vectors = np.ascontiguousarray(embed_texts(texts), dtype=np.float32)
    _faiss().normalize_L2(new_vectors)

    index, existing = load_vector_store()
    if index is None:
        index, existing = create_index(new_vectors), []
    else:
        index.add(new_vectors)

    existing = list(existing) + [{"text": t, "metadata": m} for t, m in zip(texts, metadata)]
    _write(index, existing)
    return len(texts)


def remove_chunks(opportunity_id=None, source=None) -> int:
    """
    Remove chunks matching opportunity_id and/or source file name.
    Returns the number removed. Vectors are reused (not re-embedded).
    """
    index, metadata = load_vector_store()
    if index is None or (opportunity_id is None and source is None):
        return 0

    keep = []
    for position, item in enumerate(metadata):
        meta = item.get("metadata", {})
        matches = (opportunity_id is None or meta.get("opportunity_id") == opportunity_id) and (
            source is None or meta.get("source") == source
        )
        if not matches:
            keep.append(position)

    removed = len(metadata) - len(keep)
    if removed == 0:
        return 0
    if not keep:
        delete_vector_store()
        return removed

    vectors = index.reconstruct_n(0, index.ntotal)
    new_index = create_index(vectors[keep])
    _write(new_index, [metadata[i] for i in keep])
    return removed


# ---------------------------------------------------------------- search
def search_vector_store(query, top_k=None, opportunity_id=None):
    """
    Most similar chunks to the query.

    opportunity_id: if given, only chunks uploaded for that scholarship are returned.
    Result: [{"text", "similarity", "metadata"}, ...] best first.
    """
    if not query or not query.strip():
        return []
    top_k = RAG_TOP_K if top_k is None else top_k

    index, metadata = load_vector_store()
    if index is None or index.ntotal == 0:
        return []

    query_vector = np.ascontiguousarray(embed_text(query), dtype=np.float32).reshape(1, -1)
    _faiss().normalize_L2(query_vector)

    # When filtering we must look at every chunk, then filter and trim.
    search_k = index.ntotal if opportunity_id is not None else min(top_k, index.ntotal)
    similarities, positions = index.search(query_vector, search_k)

    results = []
    for similarity, position in zip(similarities[0], positions[0]):
        if position < 0 or position >= len(metadata):
            continue
        item = metadata[position]
        meta = item.get("metadata", {})
        if opportunity_id is not None and meta.get("opportunity_id") != opportunity_id:
            continue
        results.append(
            {"text": item.get("text", ""), "similarity": float(similarity), "metadata": meta}
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    demo = [
        {"text": "The applicant must be a resident of Maharashtra.", "source": "demo.pdf", "page": 1},
        {"text": "The annual family income should not exceed Rs. 2,50,000.", "source": "demo.pdf", "page": 2},
        {"text": "Applicants must be studying in an undergraduate course.", "source": "demo.pdf", "page": 3},
    ]
    print("Stored:", rebuild_vector_store(demo))
    for hit in search_vector_store("What is the family income limit?", top_k=2):
        print(round(hit["similarity"], 3), hit["metadata"], hit["text"])
