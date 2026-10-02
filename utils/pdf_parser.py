# utils/pdf_parser.py
"""Reads official PDF documents and splits them into RAG-ready chunks."""
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE


def clean_text(text) -> str:
    """Remove blank lines and surrounding whitespace, keep the content."""
    if not text:
        return ""
    lines = [line.strip() for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line)


def _open_pdf(pdf_path):
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required to read PDFs. Run: pip install pymupdf") from exc
    return fitz.open(str(pdf_path))


def extract_text_from_pdf(pdf_path) -> list:
    """[{'text': ..., 'page': 1}, ...] for every page that has text."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The provided file is not a PDF.")

    pages = []
    document = _open_pdf(pdf_path)
    try:
        for page_number, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append({"text": text, "page": page_number})
    finally:
        document.close()
    return pages


def chunk_text(text, chunk_size=None, chunk_overlap=None) -> list:
    """Split text into overlapping chunks so sentences aren't cut off between chunks."""
    if not text or not text.strip():
        return []

    chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
    chunk_overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text = clean_text(text)
    chunks, start, length = [], 0, len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = end - chunk_overlap

    return chunks


def parse_pdf_to_chunks(pdf_path, chunk_size=None, chunk_overlap=None, opportunity_id=None) -> list:
    """PDF -> [{'text', 'source', 'page', 'opportunity_id'}], ready for the vector store."""
    pdf_path = Path(pdf_path)
    all_chunks = []
    for page_data in extract_text_from_pdf(pdf_path):
        for chunk in chunk_text(page_data["text"], chunk_size, chunk_overlap):
            item = {"text": chunk, "source": pdf_path.name, "page": page_data["page"]}
            if opportunity_id is not None:
                item["opportunity_id"] = opportunity_id
            all_chunks.append(item)
    return all_chunks


def extract_full_text(pdf_path) -> str:
    return "\n\n".join(page["text"] for page in extract_text_from_pdf(pdf_path))


def get_pdf_info(pdf_path) -> dict:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    document = _open_pdf(pdf_path)
    try:
        return {
            "file_name": pdf_path.name,
            "page_count": len(document),
            "file_size_bytes": pdf_path.stat().st_size,
        }
    finally:
        document.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.pdf_parser <path_to_pdf>")
        sys.exit(0)

    for key, value in get_pdf_info(sys.argv[1]).items():
        print(f"{key}: {value}")
    chunks = parse_pdf_to_chunks(sys.argv[1], opportunity_id=1)
    print(f"\nCreated {len(chunks)} chunks.")
    for number, chunk in enumerate(chunks[:3], start=1):
        print(f"\n--- CHUNK {number} (page {chunk['page']}) ---\n{chunk['text'][:500]}")
