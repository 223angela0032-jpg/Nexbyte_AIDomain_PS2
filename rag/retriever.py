# rag/retriever.py
"""
Retriever for the scholarship RAG pipeline.

    query -> vector_store (FAISS) -> relevant official clauses -> formatted for the LLM
"""
from config import RAG_MIN_SIMILARITY
from rag.vector_store import search_vector_store, vector_store_exists


def retrieve(query, top_k=None, opportunity_id=None):
    """Most relevant chunks for a query ([] if no vector store exists yet)."""
    if not query or not query.strip():
        return []
    if not vector_store_exists():
        return []
    return search_vector_store(query=query, top_k=top_k, opportunity_id=opportunity_id)


def retrieve_relevant(query, top_k=None, min_similarity=RAG_MIN_SIMILARITY, opportunity_id=None):
    """Like retrieve(), but drops weak matches."""
    results = retrieve(query=query, top_k=top_k, opportunity_id=opportunity_id)
    return [r for r in results if r.get("similarity", 0) >= min_similarity]


def retrieve_opportunity_rules(opportunity_name, top_k=None, opportunity_id=None):
    """Official eligibility clauses for one scholarship."""
    query = (
        f"Scholarship or scheme eligibility rules for: {opportunity_name}. "
        "Find official information about eligibility, income criteria, category, gender, "
        "state, course, academic year, required documents and application conditions."
    )
    return retrieve(query=query, top_k=top_k, opportunity_id=opportunity_id)


def retrieve_eligibility_evidence(
    profile,
    opportunity_name,
    eligibility_status=None,
    top_k=None,
    opportunity_id=None,
):
    """
    Evidence for explaining WHY a student is eligible / not eligible / needs information.
    Pass opportunity_id so only that scholarship's own documents are searched.
    """
    profile = profile or {}
    query = (
        f"Official eligibility rules for the scholarship: {opportunity_name}\n"
        "Applicant information:\n"
        f"Gender: {profile.get('gender', '')}\n"
        f"Family income: {profile.get('family_income', '')}\n"
        f"State: {profile.get('state', '')}\n"
        f"Category: {profile.get('category', '')}\n"
        f"Course: {profile.get('course', '')}\n"
        f"Year: {profile.get('year', '')}\n"
        f"Eligibility evaluation: {eligibility_status or 'Not specified'}\n"
        "Find the official clauses about who can apply, income limits, category, gender, "
        "state, course, academic year, required documents and reasons an applicant may not qualify."
    )
    return retrieve(query=query, top_k=top_k, opportunity_id=opportunity_id)


def format_retrieved_context(results):
    """Retrieved chunks -> one text block for the LLM prompt."""
    if not results:
        return "No relevant official document clauses were found."

    parts = []
    for number, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        header = (
            f"[Source: {metadata.get('source', 'Unknown source')} | "
            f"Page: {metadata.get('page', 'Unknown page')}"
        )
        if metadata.get("section"):
            header += f" | Section: {metadata['section']}"
        header += "]"
        parts.append(
            f"CLAUSE {number}\n{header}\n"
            f"Similarity: {result.get('similarity', 0):.3f}\n"
            f"{result.get('text', '').strip()}"
        )
    return "\n\n".join(parts)


def get_rag_context(query, top_k=None, min_similarity=RAG_MIN_SIMILARITY, opportunity_id=None):
    """Query -> retrieve -> filter -> formatted context, in one call."""
    return format_retrieved_context(
        retrieve_relevant(query, top_k, min_similarity, opportunity_id)
    )


def get_citations(results):
    """[{'source', 'page', 'section', 'text'}, ...] for display."""
    return [
        {
            "source": r.get("metadata", {}).get("source", "Unknown source"),
            "page": r.get("metadata", {}).get("page", "Unknown page"),
            "section": r.get("metadata", {}).get("section", ""),
            "text": r.get("text", ""),
        }
        for r in results
    ]


# ---------------------------------------------------------------- indexing an uploaded PDF
def index_source_pdf(pdf_path, opportunity_id) -> int:
    """
    Make an admin-uploaded official PDF searchable.
    Re-uploading the same file name for the same scholarship replaces its old chunks.
    Returns the number of chunks indexed.
    """
    from pathlib import Path

    from rag.vector_store import add_chunks, remove_chunks
    from utils.pdf_parser import parse_pdf_to_chunks

    chunks = parse_pdf_to_chunks(pdf_path, opportunity_id=opportunity_id)
    if not chunks:
        return 0
    remove_chunks(opportunity_id=opportunity_id, source=Path(pdf_path).name)
    return add_chunks(chunks)


if __name__ == "__main__":
    if not vector_store_exists():
        print("No vector store found. Upload an official PDF in the admin dashboard first.")
    else:
        hits = retrieve("Can a female SC student with income below Rs. 2,50,000 apply?", top_k=3)
        print(format_retrieved_context(hits))
