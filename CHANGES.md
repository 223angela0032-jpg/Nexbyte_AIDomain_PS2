# What was fixed (compared with the code in Sure.pdf)

## Things that would crash or silently break
1. `pages/` is a magic Streamlit folder → admin/user screens were reachable as standalone pages without login. Renamed to `views/`.
2. `config.py` lacked `APPLICATION_STATUSES` and `ELIGIBILITY_STATUSES` (imported by admin/user) → ImportError, hidden by `app.py` catching `ImportError` and showing "being developed". Added; removed the masking `try/except`.
3. `authenticate_user` returned `user_id`, every page reads `user["id"]` → KeyError after login. Returns both.
4. `admin.py` imported `parse_rules`, `rule_parser.py` only defined `generate_rules`. Added `parse_rules`.
5. `user.py` imported `generate_eligibility_explanation` from `ai.llm`; it did not exist. Written.
6. `RULE_GENERATION_PROMPT.format(...)` raised `KeyError` because the prompt contains literal JSON braces. Added `render_prompt()`.
7. `create_opportunity(**kwargs)` / `save_user_profile(**kwargs)` were called with keywords but defined to take a dict. Both styles accepted.
8. In the PDF the SQL section headers (e.g. a bare `USERS` line) appear without `--`; if that is how your file really looks, `init_db` fails with a syntax error. The schema is rewritten with proper comments.
9. `utils/documents.get_document_path` could raise `UnboundLocalError`.
10. `sample_opportunities.json` was invalid JSON (line breaks inside strings) and was never loaded. Regenerated and loaded by `seed_sample_opportunities()`.

## Logic bugs
11. Three different operator vocabularies (`less_equal`, `contains`, `less_than_or_equal`, …) – the shared evaluator rejected the sample data's operators. One evaluator (`ai/evaluator.py`) now serves the UI, with aliases; the duplicate evaluator in `user.py` is gone.
12. `course contains "STEM" / "undergraduate"` never matched a real course like "Engineering". Added course groups.
13. Admin rule editor stored every value as text (`"300000"`, `"[2, 3]"`) – now parsed back to numbers/lists; numeric/text comparisons are tolerant (`3 == "3"`).
14. Source PDFs were saved but never indexed → RAG could never return anything. Upload now indexes them; the vector store supports add/remove (the old one could only wipe everything) and search is filtered per scholarship so one scheme's clauses never explain another.
15. Application limit was displayed but never enforced; duplicate/limit checks are now atomic in `create_application`.
16. If AI rule parsing failed the admin had no way to add rules afterwards. Added manual rule editor + "Interpret with AI" on existing opportunities.
17. Messages shown before `st.rerun()` vanished; added `flash()`.
18. SQLite connections were never closed; `get_connection()` now commits/rolls back/closes.
19. Embeddings helper dropped empty chunks silently, shifting vectors against metadata. Empty chunks are filtered before embedding.
20. Missing `requirements.txt` entries (PyMuPDF, FAISS, sentence-transformers, numpy); `use_container_width` (deprecated) removed.
