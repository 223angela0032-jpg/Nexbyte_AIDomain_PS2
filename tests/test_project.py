"""
End-to-end checks of the project's own code.
Run:  python tests/test_project.py
(Third-party packages are replaced by tests/fakes.py; see that file.)
"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fakes  # noqa: E402

st = fakes.install_all()

# ---- redirect every writable path into a temp dir BEFORE using the project
TMP = Path(tempfile.mkdtemp(prefix="scholarship_test_"))
import database  # noqa: E402
import utils.documents as documents_mod  # noqa: E402
import rag.vector_store as vector_store  # noqa: E402
import views.admin as admin_view  # noqa: E402

database.DATABASE_PATH = TMP / "test.db"
documents_mod.UPLOAD_DIR = TMP / "uploads"
admin_view.SOURCE_DOCUMENTS_DIR = TMP / "source_documents"
vector_store._STORE_DIR = TMP / "vector_db"
vector_store.INDEX_FILE = vector_store._STORE_DIR / "scholarship.index"
vector_store.METADATA_FILE = vector_store._STORE_DIR / "metadata.json"

from ai import evaluator, llm, rule_parser  # noqa: E402
from rag import retriever  # noqa: E402
from utils import helpers, pdf_parser  # noqa: E402
import views.login as login_view  # noqa: E402
import views.user as user_view  # noqa: E402

PASSED, FAILED = [], []


def check(name):
    def wrap(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except Exception:
            FAILED.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc()
        return fn
    return wrap


class FakeUpload:
    def __init__(self, name, data=b"data"):
        self.name, self._data = name, data
    def getbuffer(self): return self._data


def run(fn, *a, **k):
    """Run a page function; return True if it ended in st.rerun()."""
    try:
        fn(*a, **k)
        return False
    except fakes.Rerun:
        return True


def real_errors():
    """st.error messages that are not just the red 'NOT ELIGIBLE' verdict badge."""
    return [t for t in st.texts("error") if "NOT ELIGIBLE" not in t]


def login_as(username, password):
    user = database.authenticate_user(username, password)
    st.session_state["user"] = user
    return user


# =====================================================================
print("\nDatabase")
database.init_db()
database.seed_demo_users()
SEEDED = database.seed_sample_opportunities()


@check("tables created, demo users + 8 sample opportunities seeded")
def _():
    assert SEEDED == 8
    assert database.seed_sample_opportunities() == 0          # idempotent
    assert len(database.get_all_opportunities()) == 8
    for o in database.get_all_opportunities():
        assert database.get_rules(o["id"]), o["name"]
        assert database.get_required_documents(o["id"]), o["name"]
        assert o["rules_verified"] == 1


@check("authenticate_user returns 'id' (needed by pages) and rejects bad passwords")
def _():
    user = database.authenticate_user("student01", "student123")
    assert user["id"] == user["user_id"] and user["role"] == "USER"
    assert database.authenticate_user("student01", "wrong") is None
    assert database.authenticate_user("nobody", "x") is None
    assert database.authenticate_user("admin", "admin123")["role"] == "ADMIN"


@check("create_opportunity / save_user_profile accept dict AND keyword arguments")
def _():
    a = database.create_opportunity({"name": "Dict Style"})
    b = database.create_opportunity(name="Kwargs Style", application_limit=5)
    assert database.get_opportunity(a)["name"] == "Dict Style"
    assert database.get_opportunity(b)["application_limit"] == 5
    uid = database.authenticate_user("student01", "student123")["id"]
    database.save_user_profile(uid, {"gender": "Male", "family_income": 1, "state": "X",
                                     "category": "SC", "course": "B.Tech", "year": 2})
    database.save_user_profile(uid, gender="Male", family_income=250000, state="Maharashtra",
                               district="Mumbai", category="SC", course="Engineering", year=3)
    assert database.get_user_profile(uid)["family_income"] == 250000


@check("rules round-trip with JSON values (numbers, lists, strings)")
def _():
    oid = database.create_opportunity(name="Rules RT")
    database.save_rules(oid, [
        {"field": "year", "operator": "in", "value": [2, 3, 4], "human_readable": "Year 2-4"},
        {"field": "family_income", "operator": "less_than", "value": 300000},
    ])
    rules = database.get_rules(oid)
    assert rules[0]["value"] == [2, 3, 4] and rules[1]["value"] == 300000
    assert rules[1]["human_readable"]  # auto-filled


# =====================================================================
print("\nEligibility evaluator")
OPP = {o["name"]: o["id"] for o in database.get_all_opportunities()}


def verdict(username, opp_name):
    uid = database.authenticate_user(username, "student123")["id"]
    profile = database.get_user_profile(uid)
    oid = OPP[opp_name]
    return evaluator.evaluate_opportunity(
        profile, database.get_rules(oid), database.get_user_documents(uid),
        database.get_required_documents(oid))["status"]


@check("demo students get the expected verdicts")
def _():
    E, N = "ELIGIBLE", "NOT_ELIGIBLE"
    expected = {
        ("student06", "State Merit Scholarship"): E,                  # income exactly 3,00,000
        ("student06", "Women in STEM Scholarship"): E,                # 'STEM' matches Engineering
        ("student06", "First Generation College Student Grant"): E,
        ("student06", "Engineering Excellence Scholarship"): N,      # year 1 < 2
        ("student06", "SC Higher Education Support Scheme"): N,      # category General
        ("student01", "SC Higher Education Support Scheme"): E,
        ("student04", "SC Higher Education Support Scheme"): N,      # Gujarat
        ("student04", "Women in STEM Scholarship"): E,
        ("student03", "State Merit Scholarship"): N,                 # income 5,00,000
        ("student05", "Women in STEM Scholarship"): N,               # male
        ("student05", "State Merit Scholarship"): E,                 # Arts counts as undergraduate
        ("student02", "OBC Student Support Scholarship"): E,
        ("student07", "Engineering Excellence Scholarship"): E,
    }
    for (user, opp), want in expected.items():
        got = verdict(user, opp)
        assert got == want, f"{user} / {opp}: expected {want}, got {got}"


@check("operator aliases, numeric text, list text and case-insensitivity")
def _():
    ev = evaluator.evaluate_rule
    assert ev(300000, "less_equal", "300000") and not ev(300001, "less_equal", 300000)
    assert ev(3, "greater_equal", 2) and ev("Female", "equals", "female")
    assert ev(3, "in", "[2, 3, 4]") and ev("SC", "in", "SC, ST") and not ev("OBC", "in", "SC, ST")
    assert ev("OBC", "not_in", ["SC", "ST"]) and ev(1, "equals", "1")
    assert not ev("abc", "less_than", 5) and not ev(5, "no_such_operator", 5)


@check("missing profile data -> NEEDS_MORE_INFORMATION; a failed rule still wins")
def _():
    rules = [{"field": "course", "operator": "contains", "value": "STEM", "human_readable": "c"},
             {"field": "family_income", "operator": "less_than", "value": 100, "human_readable": "i"}]
    res = evaluator.evaluate_opportunity({"course": None, "family_income": 50}, rules)
    assert res["status"] == "NEEDS_MORE_INFORMATION" and res["missing_information"][0]["field"] == "course"
    res = evaluator.evaluate_opportunity({"course": None, "family_income": 500}, rules)
    assert res["status"] == "NOT_ELIGIBLE"
    assert evaluator.evaluate_opportunity({}, [])["status"] == "NEEDS_MORE_INFORMATION"


@check("missing documents are detected case-insensitively")
def _():
    missing = evaluator.check_missing_documents(
        [{"document_type": "aadhaar card"}], ["Aadhaar Card", "Marksheet"])
    assert missing == ["Marksheet"]


# =====================================================================
print("\nRule parser (LLM mocked)")


@check("messy LLM output is validated and normalized")
def _():
    rule_parser.ask_llm_json = lambda prompt: {"rules": [
        {"field": "gender", "operator": "==", "value": "Women"},
        {"field": "family_income", "operator": "less_equal", "value": "₹5,00,000"},
        {"field": "year", "operator": "in", "value": ["2", 3, 4]},
        {"field": "hair_colour", "operator": "equals", "value": "red"},   # unknown field -> dropped
        {"field": "state", "operator": "bogus", "value": "X"},            # unknown operator -> dropped
        {"field": "course", "operator": "equals", "value": None},         # no value -> dropped
    ]}
    rules = rule_parser.parse_rules("anything")
    assert [r["field"] for r in rules] == ["gender", "family_income", "year"]
    assert rules[0]["value"] == "female" and rules[0]["operator"] == "equals"
    assert rules[1]["value"] == 500000 and rules[1]["operator"] == "less_than_or_equal"
    assert rules[2]["value"] == [2, 3, 4]
    assert "₹5,00,000" in rules[1]["human_readable"]


@check("prompt filling survives the JSON braces in the template")
def _():
    from ai.prompts import RULE_GENERATION_PROMPT, render_prompt
    try:
        RULE_GENERATION_PROMPT.format(admin_text="x")
        raise AssertionError("str.format should have failed (this is the original bug)")
    except KeyError:
        pass
    assert "Female students" in render_prompt(RULE_GENERATION_PROMPT, admin_text="Female students")


# =====================================================================
print("\nRAG pipeline (FAISS / embeddings / PyMuPDF faked)")
PDF1 = TMP / "merit.pdf"
PDF2 = TMP / "stem.pdf"
fakes.FAKE_PDF_PAGES[str(PDF1)] = ["Applicants must be residents of Maharashtra.\nFamily income must not exceed Rs 3,00,000.",
                                   "Required documents: income certificate and domicile certificate."]
fakes.FAKE_PDF_PAGES[str(PDF2)] = ["Only women students pursuing STEM courses may apply for this scholarship."]
for p in (PDF1, PDF2):
    p.write_bytes(b"%PDF fake")


@check("PDF -> chunks -> index -> search is scoped to one scholarship")
def _():
    o1, o2 = OPP["State Merit Scholarship"], OPP["Women in STEM Scholarship"]
    assert retriever.retrieve("income", opportunity_id=o1) == []          # nothing indexed yet
    assert retriever.index_source_pdf(PDF1, o1) == 2
    assert retriever.index_source_pdf(PDF2, o2) == 1
    hits = retriever.retrieve("family income limit", top_k=5, opportunity_id=o1)
    assert hits and all(h["metadata"]["opportunity_id"] == o1 for h in hits)
    assert hits[0]["metadata"]["source"] == "merit.pdf" and "page" in hits[0]["metadata"]
    other = retriever.retrieve("women STEM", top_k=5, opportunity_id=o2)
    assert len(other) == 1 and other[0]["metadata"]["source"] == "stem.pdf"
    assert "Source: merit.pdf" in retriever.format_retrieved_context(hits)


@check("re-uploading the same file replaces its chunks instead of duplicating")
def _():
    o1 = OPP["State Merit Scholarship"]
    retriever.index_source_pdf(PDF1, o1)
    _, meta = vector_store.load_vector_store()
    assert sum(1 for m in meta if m["metadata"]["source"] == "merit.pdf") == 2
    assert len(meta) == 3                                                  # other scholarship untouched


@check("chunking: overlap, bounds and validation")
def _():
    chunks = pdf_parser.chunk_text("word " * 1000, chunk_size=200, chunk_overlap=50)
    assert all(len(c) <= 200 for c in chunks) and len(chunks) > 20
    try:
        pdf_parser.chunk_text("x", chunk_size=10, chunk_overlap=10)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# =====================================================================
print("\nPages (Streamlit faked)")


@check("login: wrong password shows an error, right password logs in")
def _():
    st.reset()
    st.session_state["user"] = None
    st.inputs.update({"Username": "student01", "Password": "nope"})
    st.clicked.add("🔐 Login")
    assert not run(login_view.render_login)
    assert any("Incorrect" in t for t in st.texts("error")) and st.session_state["user"] is None
    st.inputs["Password"] = "student123"
    assert run(login_view.render_login)                                    # rerun after success
    assert st.session_state["user"]["username"] == "student01"


@check("student dashboard renders every tab for all 7 demo students")
def _():
    for n in range(1, 8):
        st.reset()
        login_as(f"student0{n}", "student123")
        run(user_view.render_user)
        assert not real_errors(), (n, real_errors())
        assert any("ELIGIBLE" in t for t in st.texts("success") + st.texts("error") + st.texts("warning"))


@check("profile form saves and the new values are used for eligibility")
def _():
    st.reset()
    user = login_as("student06", "student123")
    st.inputs.update({"Annual family income (₹)": 450000, "Course": "B.Tech", "State": "Maharashtra"})
    st.clicked.add("💾 Save Profile")
    assert run(user_view.show_profile)
    profile = database.get_user_profile(user["id"])
    assert profile["family_income"] == 450000 and profile["course"] == "B.Tech" and profile["year"] == 1
    database.save_user_profile(user["id"], gender="Female", family_income=300000, state="Maharashtra",
                               district="Mumbai", category="General", course="Engineering", year=1)


@check("profile form rejects an empty state")
def _():
    st.reset()
    login_as("student06", "student123")
    st.inputs["State"] = "  "
    st.clicked.add("💾 Save Profile")
    assert not run(user_view.show_profile)
    assert any("state" in t.lower() for t in st.texts("error"))


@check("document upload stores file + metadata and replaces on re-upload")
def _():
    st.reset()
    user = login_as("student06", "student123")
    st.uploads["user_doc_0"] = FakeUpload("my income cert.pdf", b"A")
    st.inputs["Document type"] = "Income Certificate"
    st.clicked.add("📥 Upload Document")
    assert run(user_view.show_document_upload)
    docs = database.get_user_documents(user["id"])
    assert [d["document_type"] for d in docs] == ["Income Certificate"]
    first = Path(docs[0]["file_path"])
    assert first.exists() and " " not in first.name
    # bad extension is refused
    st.reset(); login_as("student06", "student123")
    st.uploads["user_doc_0"] = FakeUpload("virus.exe")
    st.inputs["Document type"] = "PAN Card"
    st.clicked.add("📥 Upload Document")
    # (the real uploader filters by type; the server-side check is the second line of defence)
    st.session_state["upload_nonce"] = 0
    assert not run(user_view.show_document_upload)
    assert any("Unsupported" in t for t in st.texts("error"))
    # replace
    st.reset(); login_as("student06", "student123")
    st.uploads["user_doc_0"] = FakeUpload("other.pdf", b"B")
    st.inputs["Document type"] = "Income Certificate"
    st.clicked.add("📥 Upload Document")
    run(user_view.show_document_upload)
    docs = database.get_user_documents(user["id"])
    assert len(docs) == 1 and not first.exists() and Path(docs[0]["file_path"]).exists()


@check("Apply button creates one application; second render says already applied")
def _():
    st.reset()
    user = login_as("student06", "student123")
    oid = OPP["State Merit Scholarship"]
    st.clicked.add(f"apply_{oid}")
    assert run(user_view.render_user)
    apps = database.get_user_applications(user["id"])
    assert len(apps) == 1 and apps[0]["status"] == "PENDING"
    st.reset(); login_as("student06", "student123")
    run(user_view.render_user)
    assert any("already applied" in t for t in st.texts("success"))


@check("application limit is enforced in the database")
def _():
    oid = database.create_opportunity(name="Tiny", application_limit=1)
    database.save_rules(oid, [{"field": "state", "operator": "equals", "value": "Maharashtra"}])
    u1 = database.authenticate_user("student01", "student123")["id"]
    u2 = database.authenticate_user("student02", "student123")["id"]
    database.create_application(u1, oid)
    try:
        database.create_application(u2, oid)
        raise AssertionError("limit not enforced")
    except ValueError as e:
        assert "limit" in str(e)
    try:
        database.create_application(u1, oid)
        raise AssertionError("duplicate not blocked")
    except ValueError:
        pass


@check("AI explanation: RAG evidence + LLM text are shown; LLM failure degrades gracefully")
def _():
    oid = OPP["State Merit Scholarship"]
    llm.ask_llm = lambda prompt: "You qualify. See the official income clause."
    st.reset(); login_as("student06", "student123")
    st.clicked.add(f"explain_{oid}")
    run(user_view.render_user)
    exp = st.session_state[f"explanation_{oid}"]
    assert exp["text"].startswith("You qualify") and exp["sources"] and not exp["error"]
    assert exp["sources"][0]["metadata"]["opportunity_id"] == oid

    def boom(prompt): raise RuntimeError("GEMINI_API_KEY is not configured.")
    llm.ask_llm = boom
    st.reset(); login_as("student06", "student123")
    st.clicked.add(f"explain_{oid}")
    run(user_view.render_user)
    exp = st.session_state[f"explanation_{oid}"]
    assert exp["text"] == "" and "GEMINI_API_KEY" in exp["error"] and exp["sources"]
    assert not real_errors()                                               # page itself did not break


@check("document Q&A uses only the scholarship's own documents")
def _():
    oid = OPP["State Merit Scholarship"]
    seen = {}
    def fake_llm(prompt): seen["prompt"] = prompt; return "Rs 3,00,000 per the official PDF."
    llm.ask_llm = fake_llm
    st.reset(); login_as("student06", "student123")
    st.inputs[f"question_{oid}"] = "family income limit"
    st.clicked.add(f"ask_{oid}")
    run(user_view.render_user)
    assert st.session_state[f"answer_{oid}"]["text"].startswith("Rs 3,00,000")
    assert "merit.pdf" in seen["prompt"] and "stem.pdf" not in seen["prompt"]
    assert "{question}" not in seen["prompt"] and "{evidence}" not in seen["prompt"]


@check("admin page is refused for students")
def _():
    st.reset(); login_as("student01", "student123")
    run(admin_view.render_admin)
    assert any("Admin access" in t for t in st.texts("error"))


@check("admin: create opportunity -> AI interprets -> verify -> rules active")
def _():
    rule_parser.ask_llm_json = lambda prompt: {"rules": [
        {"field": "gender", "operator": "equals", "value": "female", "human_readable": "Gender must be Female"},
        {"field": "family_income", "operator": "less_than", "value": 500000},
    ]}
    st.reset(); login_as("admin", "admin123")
    st.inputs.update({"Opportunity name *": "Test Women Grant",
                      "Describe eligibility rules in normal language *": "Female, income below 5 lakh",
                      "Required documents": ["Aadhaar Card", "Income Certificate"]})
    st.clicked.add("Create Opportunity")
    assert run(admin_view.render_admin)                                    # rerun after creation
    oid = next(o["id"] for o in database.get_all_opportunities() if o["name"] == "Test Women Grant")
    assert database.get_required_documents(oid) == ["Aadhaar Card", "Income Certificate"]
    assert database.get_rules(oid) == []                                   # NOT active before verification
    assert database.get_opportunity(oid)["rules_verified"] == 0
    assert len(st.session_state["create_pending_rules"]) == 2

    # next run (session state kept): editor appears; the admin verifies
    saved = dict(st.session_state)
    st.reset(); st.session_state.update(saved)
    st.clicked.add("create_rules_v1_verify")
    assert run(admin_view.render_admin)
    rules = database.get_rules(oid)
    assert [r["field"] for r in rules] == ["gender", "family_income"]
    assert database.get_opportunity(oid)["rules_verified"] == 1
    assert "create_pending_rules" not in st.session_state


@check("admin: if the AI fails, a blank rule editor is offered (no dead end)")
def _():
    def fail(prompt): raise RuntimeError("GEMINI_API_KEY is not configured.")
    rule_parser.ask_llm_json = fail
    st.reset(); login_as("admin", "admin123")
    st.inputs.update({"Opportunity name *": "Manual Rules Grant",
                      "Describe eligibility rules in normal language *": "Must be from Goa"})
    st.clicked.add("Create Opportunity")
    assert run(admin_view.render_admin)
    assert any("AI rule parsing failed" in t for t in st.texts("error"))
    saved = dict(st.session_state)
    st.reset(); st.session_state.update(saved)
    st.inputs.update({"create_rules_v1_field_0": "state", "create_rules_v1_value_0": "Goa"})
    st.clicked.add("create_rules_v1_verify")
    assert run(admin_view.render_admin)
    oid = next(o["id"] for o in database.get_all_opportunities() if o["name"] == "Manual Rules Grant")
    rule = database.get_rules(oid)[0]
    assert (rule["field"], rule["value"]) == ("state", "Goa") and rule["human_readable"]


@check("admin: editing rule values keeps numbers/lists as real JSON values")
def _():
    assert admin_view.parse_rule_value("300000") == 300000
    assert admin_view.parse_rule_value("[2, 3, 4]") == [2, 3, 4]
    assert admin_view.parse_rule_value("Maharashtra") == "Maharashtra"
    assert admin_view.format_rule_value([2, 3]) == "[2, 3]" and admin_view.format_rule_value("x") == "x"


@check("admin: manage tab renders every section for every opportunity")
def _():
    for o in database.get_all_opportunities():
        st.reset(); login_as("admin", "admin123")
        st.inputs["Select an opportunity"] = o["id"]
        run(admin_view.render_admin)
        assert not real_errors(), (o["name"], real_errors())


@check("admin: uploading a source PDF saves it AND indexes it for RAG")
def _():
    oid = OPP["Rural Student Assistance Scheme"]
    target = TMP / "source_documents" / str(oid) / "rural_rules.pdf"
    fakes.FAKE_PDF_PAGES[str(target)] = ["Rural students from Maharashtra qualify if income is below Rs 4,00,000."]
    st.reset(); login_as("admin", "admin123")
    st.inputs["Select an opportunity"] = oid
    st.uploads[f"source_pdf_{oid}"] = FakeUpload("rural rules.pdf", b"%PDF")
    st.clicked.add(f"save_source_{oid}")
    assert run(admin_view.render_admin)
    assert database.get_opportunity_sources(oid)[0]["file_name"] == "rural_rules.pdf"
    hits = retriever.retrieve("income", opportunity_id=oid)
    assert hits and hits[0]["metadata"]["source"] == "rural_rules.pdf"


@check("admin: settings change closes applications; student then sees it closed")
def _():
    oid = OPP["OBC Student Support Scholarship"]
    st.reset(); login_as("admin", "admin123")
    st.inputs.update({"Select an opportunity": oid, f"enabled_{oid}": False, f"limit_{oid}": 7})
    st.clicked.add(f"save_settings_{oid}")
    assert run(admin_view.render_admin)
    o = database.get_opportunity(oid)
    assert o["applications_enabled"] == 0 and o["application_limit"] == 7
    st.reset(); login_as("student02", "student123")
    run(user_view.render_user)
    assert any("closed by the admin" in t for t in st.texts("warning"))
    st.reset(); login_as("student02", "student123")
    st.clicked.add(f"apply_{oid}")
    run(user_view.render_user)
    assert not database.has_applied(database.authenticate_user("student02", "student123")["id"], oid)


@check("admin: approve an application; student sees the new status")
def _():
    oid = OPP["State Merit Scholarship"]
    app = database.get_opportunity_applications(oid)[0]
    st.reset(); login_as("admin", "admin123")
    st.inputs.update({"Select an opportunity": oid, f"status_{app['id']}": "APPROVED",
                      f"reason_{app['id']}": "Documents verified"})
    st.clicked.add(f"update_app_{app['id']}")
    assert run(admin_view.render_admin)
    st.reset(); login_as("student06", "student123")
    run(user_view.show_application_history)
    assert any("Approved" in t for t in st.texts("write"))
    assert any("Documents verified" in t for t in st.texts("caption"))


@check("app.py imports and routes: logged-out -> login, student -> user page, admin -> admin page")
def _():
    import importlib
    for who, expect in ((None, "Login"), ("student01", "Welcome"), ("admin", "Admin Dashboard")):
        st.reset()
        st.session_state["user"] = None
        if who:
            login_as(who, "admin123" if who == "admin" else "student123")
        sys.modules.pop("app", None)
        importlib.import_module("app")
        titles = " ".join(st.texts("title") + st.texts("subheader"))
        assert expect in titles, (who, titles)


# =====================================================================
print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
sys.exit(1 if FAILED else 0)
