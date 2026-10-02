# Nexbyte – Scholarship & Scheme Eligibility Assistant

Streamlit app where **admins** publish scholarships (rules written in plain English, interpreted by
Gemini, then verified by the admin) and **students** see – deterministically – whether they are
eligible, why, which documents are missing, and an AI explanation backed by the official PDF clause.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then put your GEMINI_API_KEY in .env
streamlit run app.py
```

The first start creates `database/scholarship.db`, 8 demo accounts and 8 sample scholarships
(`data/sample_opportunities.json`). Without a Gemini key everything except the AI rule
interpretation / AI explanation works (the admin can type rules manually).

| Role    | Username                | Password     |
|---------|-------------------------|--------------|
| Admin   | `admin`                 | `admin123`   |
| Student | `student01`–`student07` | `student123` |

The first PDF upload / explanation downloads the `all-MiniLM-L6-v2` embedding model (needs internet once).

## How the files connect

```
app.py ── login ──► views/login.py ──► database.authenticate_user
   ├── ADMIN ► views/admin.py
   │            ├─ ai/rule_parser.py ─► ai/llm.py ─► Gemini      (plain English → structured rules)
   │            ├─ database.py                                   (opportunities, rules, documents, applications)
   │            └─ rag/retriever.index_source_pdf ─► utils/pdf_parser.py ─► rag/embeddings.py ─► rag/vector_store.py (FAISS)
   └── USER  ► views/user.py
                ├─ ai/evaluator.py        deterministic ELIGIBLE / NOT_ELIGIBLE / NEEDS_MORE_INFORMATION
                ├─ utils/documents.py     student uploads
                └─ rag/retriever.py ─► ai/llm.py                  (official clause + explanation)
config.py: settings used everywhere · utils/helpers.py: formatting & date helpers · ai/prompts.py: all prompts
```

## Tests

```bash
python tests/check_imports.py      # every cross-file import resolves, no unused imports
python tests/test_project.py       # 31 end-to-end checks (DB, evaluator, rule parser, RAG, all pages)
```
`tests/fakes.py` stands in for Streamlit/FAISS/sentence-transformers/PyMuPDF so the tests run
offline; they exercise this project's own code paths.

> The folder is called `views/` (not `pages/`) on purpose: Streamlit treats a `pages/` folder as
> automatic extra pages, which would expose the admin screen without login.
