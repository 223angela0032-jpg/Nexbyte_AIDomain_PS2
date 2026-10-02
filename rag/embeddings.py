# rag/embeddings.py
"""Text -> embedding vectors (small CPU-friendly sentence-transformer model)."""
from typing import List, Union

import numpy as np

from config import EMBEDDING_MODEL_NAME

MODEL_NAME = EMBEDDING_MODEL_NAME  # all-MiniLM-L6-v2 -> 384 dimensions
EMBEDDING_DIMENSION = 384

_model = None


def get_embedding_model():
    """Load the model once (downloaded on first use, then cached by Hugging Face)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for RAG. Run: pip install sentence-transformers"
            ) from exc
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed one text. Returns a normalized float32 vector."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = text.strip()
    if not text:
        raise ValueError("Cannot create an embedding for empty text.")

    embedding = get_embedding_model().encode(
        text, convert_to_numpy=True, normalize_embeddings=True
    )
    return embedding.astype(np.float32)


def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """
    Embed many texts -> array of shape (len(texts), dim).

    Row i always belongs to texts[i]; every text must be a non-empty string
    (callers filter empty chunks first so rows never shift).
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    cleaned = []
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("Every item in texts must be a string.")
        text = text.strip()
        if not text:
            raise ValueError("Cannot embed an empty text chunk.")
        cleaned.append(text)

    embeddings = get_embedding_model().encode(
        cleaned,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def get_embedding_dimension() -> int:
    return int(get_embedding_model().get_sentence_embedding_dimension())


def cosine_similarity(
    embedding_a: Union[List[float], np.ndarray],
    embedding_b: Union[List[float], np.ndarray],
) -> float:
    """Cosine similarity of two embeddings."""
    a = np.asarray(embedding_a, dtype=np.float32)
    b = np.asarray(embedding_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("Embedding dimensions do not match.")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


if __name__ == "__main__":
    vector = embed_text(
        "Students from Maharashtra with family income below three lakh rupees are eligible."
    )
    print("Embedding shape:", vector.shape)
