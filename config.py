# config.py
"""Central configuration. Every other module imports its settings from here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "scholarship.db"
UPLOAD_DIR = BASE_DIR / "uploads"
SOURCE_DOCUMENTS_DIR = BASE_DIR / "source_documents"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
DATA_DIR = BASE_DIR / "data"
SAMPLE_OPPORTUNITIES_FILE = DATA_DIR / "sample_opportunities.json"

for _directory in (DATABASE_DIR, UPLOAD_DIR, SOURCE_DOCUMENTS_DIR, VECTOR_DB_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- app info
APP_NAME = "Scholarship & Scheme Eligibility Assistant"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------- Gemini LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------------- RAG
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RAG_TOP_K = 5
RAG_MIN_SIMILARITY = 0.25
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------- application defaults
DEFAULT_APPLICATION_LIMIT = 100
DEFAULT_APPLICATIONS_ENABLED = True

# ---------------------------------------------------------------- allowed values
ROLES = ["ADMIN", "USER"]
GENDERS = ["Male", "Female", "Other", "Prefer not to say"]
CATEGORIES = ["General", "SC", "ST", "OBC", "EWS", "Other"]
YEARS = [1, 2, 3, 4, 5]

DOCUMENT_TYPES = [
    "Aadhaar Card",
    "PAN Card",
    "Income Certificate",
    "Caste Certificate",
    "Domicile Certificate",
    "Birth Certificate",
    "College ID",
    "Marksheet",
]

# ---------------------------------------------------------------- eligibility statuses
ELIGIBLE = "ELIGIBLE"
NOT_ELIGIBLE = "NOT_ELIGIBLE"
NEEDS_MORE_INFORMATION = "NEEDS_MORE_INFORMATION"
ELIGIBILITY_STATUSES = [ELIGIBLE, NOT_ELIGIBLE, NEEDS_MORE_INFORMATION]

# ---------------------------------------------------------------- application statuses
APPLICATION_PENDING = "PENDING"
APPLICATION_APPROVED = "APPROVED"
APPLICATION_REJECTED = "REJECTED"
APPLICATION_STATUSES = [APPLICATION_PENDING, APPLICATION_APPROVED, APPLICATION_REJECTED]
