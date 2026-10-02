# views/user.py
"""
Student dashboard.

    Profile -> Documents -> Opportunities
    deterministic eligibility check (ai.evaluator)
    -> RAG retrieves official clauses -> LLM explains the result
    -> Apply
"""
import os
import streamlit as st

from ai.evaluator import evaluate_opportunity
from config import (
    CATEGORIES,
    DOCUMENT_TYPES,
    ELIGIBLE,
    GENDERS,
    NOT_ELIGIBLE,
    YEARS,
)
from database import (
    create_application,
    get_all_opportunities,
    get_application_count,
    get_required_documents,
    get_rules,
    get_user_applications,
    get_user_documents,
    get_user_profile,
    has_applied,
    save_user_profile,
)
from utils.documents import save_document
from utils.helpers import (
    format_date,
    get_missing_profile_fields,
    get_status_emoji,
    get_status_label,
    parse_date,
    registration_status,
)
from utils.ui import flash, show_flash


# =========================================================
# Helpers
# =========================================================
def get_current_user():
    return st.session_state.get("user")


def _index_of(options, value, default=0):
    """Position of value in options (compared as text), else default."""
    for position, option in enumerate(options):
        if str(option) == str(value):
            return position
    return default


# =========================================================
# PROFILE
# =========================================================
def show_profile():
    user = get_current_user()
    user_id = user["id"]
    profile = get_user_profile(user_id) or {}

    st.header("👤 My Profile")
    st.write("Enter your information so the system can check scholarship and scheme eligibility.")

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox(
                "Gender", GENDERS, index=_index_of(GENDERS, profile.get("gender"))
            )
            family_income = st.number_input(
                "Annual family income (₹)",
                min_value=0,
                value=int(profile.get("family_income") or 0),
                step=1000,
            )
            state = st.text_input("State", value=str(profile.get("state") or ""))
            district = st.text_input("District", value=str(profile.get("district") or ""))
        with col2:
            category = st.selectbox(
                "Category", CATEGORIES, index=_index_of(CATEGORIES, profile.get("category"))
            )
            course = st.text_input(
                "Course", value=str(profile.get("course") or ""), placeholder="Example: B.Tech"
            )
            year = st.selectbox(
                "Academic year", YEARS, index=_index_of(YEARS, profile.get("year"))
            )
        submitted = st.form_submit_button("💾 Save Profile", type="primary")

    if not submitted:
        return

    if not state.strip():
        st.error("Please enter your state.")
        return
    if not course.strip():
        st.error("Please enter your course.")
        return

    try:
        save_user_profile(
            user_id,
            gender=gender,
            family_income=family_income,
            state=state.strip(),
            district=district.strip(),
            category=category,
            course=course.strip(),
            year=int(year),
        )
    except Exception as error:
        st.error(f"Could not save profile: {error}")
        return

    flash("success", "Profile saved successfully.")
    st.rerun()


# =========================================================
# DOCUMENT UPLOAD
# =========================================================
def show_document_upload():
    user = get_current_user()
    user_id = user["id"]

    st.header("📄 My Documents")
    st.write("Upload documents that may be required by different scholarships and schemes.")

    nonce = st.session_state.setdefault("upload_nonce", 0)  # changing the key clears the uploader
    col1, col2 = st.columns(2)
    with col1:
        document_type = st.selectbox("Document type", DOCUMENT_TYPES)
    with col2:
        uploaded_file = st.file_uploader(
            "Choose file", type=["pdf", "png", "jpg", "jpeg", "webp"], key=f"user_doc_{nonce}"
        )

    if uploaded_file is not None:
        st.write(f"Selected file: **{uploaded_file.name}**")
        if st.button("📥 Upload Document"):
            try:
                save_document(user_id, document_type, uploaded_file)
            except Exception as error:
                st.error(f"Could not upload document: {error}")
            else:
                st.session_state["upload_nonce"] = nonce + 1
                flash("success", f"{document_type} uploaded successfully.")
                st.rerun()

    st.subheader("Uploaded Documents")
    documents = get_user_documents(user_id)
    if not documents:
        st.info("You have not uploaded any documents yet.")
        return
    for document in documents:
        st.write(f"✅ **{document['document_type']}**")
        if document.get("file_path"):
            st.caption(f"File: {os.path.basename(document['file_path'])}")


# =========================================================
# RAG + LLM EXPLANATION
# =========================================================
def generate_ai_explanation(profile, opportunity, evaluation):
    """
    Retrieve official clauses (RAG) and ask the LLM to explain the deterministic result.
    Never raises: problems are returned in 'error' so the page keeps working.
    """
    sources, notes = [], []

    try:
        from rag.retriever import retrieve_eligibility_evidence

        sources = retrieve_eligibility_evidence(
            profile=profile,
            opportunity_name=opportunity.get("name", ""),
            eligibility_status=evaluation["status"],
            opportunity_id=opportunity["id"],
        )
    except Exception as error:
        notes.append(f"Official-document search is unavailable: {error}")

    text = ""
    try:
        from ai.llm import generate_eligibility_explanation
        from rag.retriever import format_retrieved_context

        text = generate_eligibility_explanation(
            profile=profile,
            opportunity=opportunity,
            evaluation=evaluation,
            rag_context=format_retrieved_context(sources) if sources else "",
        )
    except Exception as error:
        notes.append(f"AI explanation is unavailable: {error}")

    return {"text": text, "sources": sources, "error": " ".join(notes)}


def answer_document_question(opportunity_id, question):
    """Free-text question answered ONLY from the scholarship's official documents."""
    try:
        from ai.llm import answer_question_with_rag
        from rag.retriever import get_rag_context

        context = get_rag_context(question, opportunity_id=opportunity_id)
        return {"text": answer_question_with_rag(question, context), "error": ""}
    except Exception as error:
        return {"text": "", "error": str(error)}


def _show_sources(sources):
    st.markdown("### 📚 Official supporting clauses")
    for number, source in enumerate(sources, start=1):
        metadata = source.get("metadata", {})
        st.markdown(
            f"**Clause {number} — {metadata.get('source', 'Unknown source')}, "
            f"Page {metadata.get('page', '?')}**"
        )
        st.write(source.get("text", ""))
        st.divider()


def _show_ai_section(opportunity, profile, evaluation):
    opportunity_id = opportunity["id"]
    explanation_key = f"explanation_{opportunity_id}"
    answer_key = f"answer_{opportunity_id}"

    with st.expander("🤖 AI explanation & official evidence"):
        if st.button("Generate Explanation", key=f"explain_{opportunity_id}"):
            with st.spinner("Retrieving official rules and generating explanation..."):
                st.session_state[explanation_key] = generate_ai_explanation(
                    profile, opportunity, evaluation
                )

        explanation = st.session_state.get(explanation_key)
        if explanation:
            if explanation["text"]:
                st.markdown(explanation["text"])
            if explanation["error"]:
                st.info(explanation["error"] + " The rule-based result shown above is still correct.")
            if explanation["sources"]:
                _show_sources(explanation["sources"])
            elif not explanation["error"]:
                st.info("No official documents have been uploaded for this scholarship yet.")

        st.markdown("**Ask about the official documents**")
        question = st.text_input("Your question", key=f"question_{opportunity_id}")
        if st.button("Ask", key=f"ask_{opportunity_id}") and question.strip():
            with st.spinner("Searching official documents..."):
                st.session_state[answer_key] = answer_document_question(
                    opportunity_id, question.strip()
                )
        answer = st.session_state.get(answer_key)
        if answer:
            if answer["text"]:
                st.markdown(answer["text"])
            if answer["error"]:
                st.info(f"Could not answer: {answer['error']}")


# =========================================================
# OPPORTUNITY CARD
# =========================================================
_WINDOW_MESSAGES = {
    "NOT_STARTED": ("info", "Applications have not opened yet."),
    "ENDED": ("warning", "The application period has ended."),
    "CLOSED": ("warning", "Applications are currently closed by the admin."),
}


def show_opportunity(opportunity, profile, user_id):
    opportunity_id = opportunity["id"]

    rules = get_rules(opportunity_id)
    required_documents = get_required_documents(opportunity_id)
    user_documents = get_user_documents(user_id)
    evaluation = evaluate_opportunity(profile, rules, user_documents, required_documents)
    status = evaluation["status"]
    missing_documents = evaluation["missing_documents"]

    with st.container():
        st.subheader(f"🎓 {opportunity.get('name', 'Scholarship')}")
        if opportunity.get("organization"):
            st.caption(opportunity["organization"])
        if opportunity.get("description"):
            st.write(opportunity["description"])

        # ---- application window and limit
        window = registration_status(
            opportunity.get("registration_start"),
            opportunity.get("registration_end"),
            bool(opportunity.get("applications_enabled", 1)),
        )
        limit = opportunity.get("application_limit") or 0
        count = get_application_count(opportunity_id)
        limit_reached = bool(limit) and count >= limit

        if window == "OPEN" and limit_reached:
            st.warning("The application limit has been reached.")
        elif window == "OPEN":
            end_date = parse_date(opportunity.get("registration_end"))
            st.success(
                f"Applications open until {format_date(end_date)}."
                if end_date
                else "Applications are open."
            )
        elif window == "NOT_STARTED":
            st.info(f"Applications open from {format_date(opportunity.get('registration_start'))}.")
        elif window == "ENDED":
            st.warning(f"Applications closed on {format_date(opportunity.get('registration_end'))}.")
        else:
            st.warning(_WINDOW_MESSAGES["CLOSED"][1])
        can_apply_now = window == "OPEN" and not limit_reached

        # ---- eligibility verdict
        label = f"{get_status_emoji(status)} {get_status_label(status).upper()}"
        if status == ELIGIBLE:
            st.success(label)
        elif status == NOT_ELIGIBLE:
            st.error(label)
        else:
            st.warning(label)

        if not evaluation["has_rules"]:
            st.info("The administrator has not published verified eligibility rules yet.")

        if evaluation["passed_rules"]:
            st.markdown("**Requirements you meet:**")
            for rule in evaluation["passed_rules"]:
                st.write(f"✅ {rule['human_readable']}")
        if evaluation["failed_rules"]:
            st.markdown("**Requirements you do not meet:**")
            for rule in evaluation["failed_rules"]:
                st.write(f"❌ {rule['human_readable']}")
        if evaluation["missing_information"]:
            st.markdown("**Missing information:**")
            for item in evaluation["missing_information"]:
                st.write(f"⚠️ {item['message']}")

        # ---- documents
        if required_documents:
            st.markdown("**Required documents:**")
            for document in required_documents:
                if document in missing_documents:
                    st.write(f"❌ {document} — missing")
                else:
                    st.write(f"✅ {document}")

        _show_ai_section(opportunity, profile, evaluation)

        # ---- apply
        if has_applied(user_id, opportunity_id):
            st.success("✅ You have already applied.")
        elif status == ELIGIBLE:
            if missing_documents:
                st.warning(
                    "You appear eligible, but some required documents are missing: "
                    + ", ".join(missing_documents)
                )
            if can_apply_now:
                if st.button(
                    "📝 Apply Now", key=f"apply_{opportunity_id}", type="primary"
                ):
                    try:
                        create_application(
                            user_id=user_id,
                            opportunity_id=opportunity_id,
                            reason="Application submitted through eligibility assistant.",
                        )
                    except Exception as error:
                        st.error(f"Could not submit application: {error}")
                    else:
                        flash("success", f"Application submitted for {opportunity['name']}.")
                        st.rerun()
        st.divider()


# =========================================================
# OPPORTUNITIES TAB
# =========================================================
def show_opportunities():
    user = get_current_user()
    user_id = user["id"]
    profile = get_user_profile(user_id) or {}

    st.header("🎓 Scholarships & Schemes")

    missing = get_missing_profile_fields(profile, required_only=True)
    if missing:
        st.warning("Please complete your profile before checking eligibility.")
        st.write("Missing:")
        for label in missing:
            st.write(f"• {label}")
        return

    opportunities = get_all_opportunities()
    if not opportunities:
        st.info("No scholarships or schemes have been added yet.")
        return

    st.write(f"Found **{len(opportunities)}** opportunities.")
    for opportunity in opportunities:
        show_opportunity(opportunity, profile, user_id)


# =========================================================
# APPLICATION HISTORY
# =========================================================
def show_application_history():
    user = get_current_user()
    applications = get_user_applications(user["id"])

    st.header("📝 My Applications")
    if not applications:
        st.info("You have not submitted any applications yet.")
        return

    for application in applications:
        with st.container():
            st.subheader(f"🎓 {application['opportunity_name']}")
            st.write(
                f"**Status:** {get_status_emoji(application['status'])} "
                f"{get_status_label(application['status'])}"
            )
            st.write(f"**Applied:** {format_date(application.get('applied_at'))}")
            if application.get("reason"):
                st.caption(application["reason"])
            st.divider()


# =========================================================
# MAIN
# =========================================================
def render_user():
    """Called by app.py for logged-in students."""
    user = get_current_user()
    if not user:
        st.error("Please login first.")
        return

    st.title(f"🎓 Welcome, {user.get('name') or user.get('username', 'Student')}")
    st.caption("Find scholarships and welfare schemes you may be eligible for.")
    show_flash()

    tab_profile, tab_documents, tab_opportunities, tab_applications = st.tabs(
        ["👤 Profile", "📄 Documents", "🎓 Opportunities", "📝 My Applications"]
    )
    with tab_profile:
        show_profile()
    with tab_documents:
        show_document_upload()
    with tab_opportunities:
        show_opportunities()
    with tab_applications:
        show_application_history()
