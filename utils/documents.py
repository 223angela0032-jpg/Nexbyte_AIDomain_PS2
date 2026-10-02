# utils/documents.py
"""Student document handling: validation, storage on disk, and metadata in the database."""
import re
from pathlib import Path

from config import UPLOAD_DIR
from database import get_document_record, get_user_documents, save_document_metadata
from utils.helpers import is_allowed_document


def _safe_filename(filename: str) -> str:
    """Strip any path and unsafe characters from an uploaded filename."""
    filename = Path(filename or "").name
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return filename or "document"


def _get_user_upload_dir(user_id: int) -> Path:
    user_dir = Path(UPLOAD_DIR) / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def save_document(user_id: int, document_type: str, uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to disk and record it in the database.
    Re-uploading the same document type replaces the old file.
    Returns the saved file path.
    """
    if not user_id:
        raise ValueError("User ID is required.")
    if not document_type:
        raise ValueError("Document type is required.")
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    original_name = getattr(uploaded_file, "name", "document")
    if not is_allowed_document(original_name):
        raise ValueError("Unsupported file type. Please upload a PDF, PNG, JPG or WEBP file.")

    user_dir = _get_user_upload_dir(user_id)
    safe_type = re.sub(r"[^A-Za-z0-9_-]", "_", document_type.strip())
    file_path = user_dir / f"{safe_type}_{_safe_filename(original_name)}"

    previous = get_document_record(user_id, document_type)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    save_document_metadata(user_id=user_id, document_type=document_type, file_path=str(file_path))

    # Remove the replaced file (if it had a different name) so orphans don't pile up.
    if previous and previous.get("file_path") and Path(previous["file_path"]) != file_path:
        delete_document(previous["file_path"])

    return str(file_path)


def get_user_document_list(user_id: int):
    return get_user_documents(user_id)


def get_document_path(user_id: int, document_type: str):
    """Path of the uploaded file for a document type, or None."""
    for document in get_user_documents(user_id):
        if document.get("document_type") == document_type:
            file_path = document.get("file_path")
            if file_path and Path(file_path).exists():
                return file_path
    return None


def check_required_documents(user_id: int, required_documents) -> list:
    """Required document types the user has not uploaded."""
    from ai.evaluator import check_missing_documents

    return check_missing_documents(get_user_documents(user_id), required_documents)


def document_exists(file_path: str) -> bool:
    return bool(file_path) and Path(file_path).is_file()


def delete_document(file_path: str) -> bool:
    if not file_path:
        return False
    path = Path(file_path)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False
