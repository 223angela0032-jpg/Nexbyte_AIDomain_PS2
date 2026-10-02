# views/admin.py
"""
Admin dashboard.

Admin can:
  1. Create scholarship/scheme opportunities
  2. Enter eligibility rules in plain language
  3. Let the AI interpret the rules
  4. Review / edit the interpreted rules and VERIFY them (rules are active only after this)
  5. Choose required documents
  6. Upload official source PDFs (indexed for RAG)
  7. Set application limits and dates, open/close applications
  8. View applications and approve / reject them
"""
import json
import re
from datetime import date
from pathlib import Path

import streamlit as st

from ai.evaluator import OPERATOR_ALIASES, normalize_operator
from ai.rule_parser import ALLOWED_FIELDS, build_human_readable
from config import (
    APPLICATION_STATUSES,
    DEFAULT_APPLICATION_LIMIT,
    DOCUMENT_TYPES,
    SOURCE_DOCUMENTS_DIR,
)
from database import (
    create_opportunity,
    get_all_opportunities,
    get_opportunity,
    get_opportunity_applications,
    get_opportunity_sources,
    get_required_documents,
    get_rules,
    save_opportunity_source,
    save_required_documents,
    save_rules,
    update_application_status,
    update_opportunity,
)
from utils.helpers import (
    format_currency,
    format_date,
    get_status_emoji,
    get_status_label,
    parse_date,
    safe_int,
)
from utils.ui import flash, show_flash

FIELD_OPTIONS = sorted(ALLOWED_FIELDS)
OPERATOR_OPTIONS = sorted(set(OPERATOR_ALIASES.values()))


# =========================================================
# Helpers
# =========================================================
def parse_rule_value(text):
    """
    Text typed by the admin -> Python value.
    '300000' -> 300000, '[2, 3, 4]' -> [2, 3, 4], 'Maharashtra' -> 'Maharashtra'
    """
    text = (text or "").strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def format_rule_value(value):
    """Python value -> text shown in the editor (inverse of parse_rule_value)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def parse_rules_with_ai(rule_text):
    """Send plain-language rules to the LLM-based parser. Returns [] (and shows why) on failure."""
    if not rule_text or not rule_text.strip():
        return []
    try:
        from ai.rule_parser import parse_rules

        parsed = parse_rules(rule_text)
        if not parsed:
            st.warning("The AI did not find any usable rules in that text.")
        return parsed
    except Exception as error:
        st.error(f"AI rule parsing failed: {error}")
        return []


def _safe_filename(name):
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "").name)
    return name or "document.pdf"


def _describe_rule(rule):
    readable = rule.get("human_readable")
    if readable:
        return readable
    return f"{rule.get('field')} {rule.get('operator')} {format_rule_value(rule.get('value'))}"


# =========================================================
# Rule verification editor
# =========================================================
def set_pending(state_key, rules):
    """Store a working copy of rules and bump the version so editor widgets start fresh."""
    st.session_state[state_key] = rules
    st.session_state[state_key + "_v"] = st.session_state.get(state_key + "_v", 0) + 1


def clear_pending(state_key):
    st.session_state.pop(state_key, None)
    st.session_state[state_key + "_v"] = st.session_state.get(state_key + "_v", 0) + 1


def show_rule_editor(rules, state_key, opportunity_id, key_prefix):
    """
    Editable list of rules. Rules become active only when the admin clicks 'Verify'.
    `rules` is the working copy stored in st.session_state[state_key].
    """
    st.warning("Review every rule carefully. Rules become active only after verification.")

    # Widget keys include a version so that adding/removing rules never reuses stale values.
    key_prefix = f"{key_prefix}_v{st.session_state.get(state_key + '_v', 0)}"

    edited = []
    for index, rule in enumerate(rules):
        st.markdown(f"### Rule {index + 1}")
        col1, col2 = st.columns(2)
        with col1:
            field_default = rule.get("field", "")
            field_options = FIELD_OPTIONS if field_default in FIELD_OPTIONS else [field_default] + FIELD_OPTIONS
            field = st.selectbox(
                "Field",
                field_options,
                index=field_options.index(field_default) if field_default in field_options else 0,
                key=f"{key_prefix}_field_{index}",
            )
            operator_default = normalize_operator(rule.get("operator")) or "equals"
            operator = st.selectbox(
                "Operator",
                OPERATOR_OPTIONS,
                index=OPERATOR_OPTIONS.index(operator_default),
                key=f"{key_prefix}_operator_{index}",
            )
        with col2:
            value_text = st.text_input(
                "Value (numbers as digits, lists like [2, 3, 4])",
                value=format_rule_value(rule.get("value", "")),
                key=f"{key_prefix}_value_{index}",
            )
            readable = st.text_input(
                "Human-readable rule (leave empty to auto-generate)",
                value=str(rule.get("human_readable", "")),
                key=f"{key_prefix}_readable_{index}",
            )
        remove = st.checkbox("Remove this rule", key=f"{key_prefix}_remove_{index}")
        if not remove:
            edited.append(
                {
                    "field": field,
                    "operator": operator,
                    "value": parse_rule_value(value_text),
                    "human_readable": readable.strip(),
                }
            )
        st.divider()

    col_add, col_verify, col_discard = st.columns(3)
    with col_add:
        if st.button("➕ Add another rule", key=f"{key_prefix}_add"):
            set_pending(
                state_key,
                edited + [{"field": "gender", "operator": "equals", "value": "", "human_readable": ""}],
            )
            st.rerun()
    with col_discard:
        if st.button("🗑️ Discard", key=f"{key_prefix}_discard"):
            clear_pending(state_key)
            st.rerun()
    with col_verify:
        verify = st.button("✅ Verify and Activate Rules", type="primary", key=f"{key_prefix}_verify")

    if not verify:
        return

    valid = []
    for rule in edited:
        if rule["value"] in ("", None, []):
            st.error(f"Rule on '{rule['field']}' has no value. Fill it in or remove the rule.")
            return
        if not rule["human_readable"]:
            rule["human_readable"] = build_human_readable(
                rule["field"], rule["operator"], rule["value"]
            )
        valid.append(rule)

    if not valid:
        st.error("At least one rule is required.")
        return

    try:
        save_rules(opportunity_id, valid)
    except Exception as error:
        st.error(f"Could not save rules: {error}")
        return

    clear_pending(state_key)
    flash("success", "Rules verified and activated successfully.")
    st.rerun()


# =========================================================
# Tab 1: create opportunity
# =========================================================
_CREATE_ID_KEY = "create_pending_opportunity_id"
_CREATE_RULES_KEY = "create_pending_rules"


def show_create_opportunity():
    st.header("➕ Add New Scholarship / Scheme")

    with st.form("create_opportunity_form"):
        name = st.text_input("Opportunity name *", placeholder="Example: State Merit Scholarship")
        organization = st.text_input("Organization", placeholder="Example: Government of Maharashtra")
        description = st.text_area("Description", placeholder="Briefly describe the scholarship or scheme.")
        official_url = st.text_input("Official website URL", placeholder="https://...")

        st.markdown("### Application settings")
        col1, col2 = st.columns(2)
        with col1:
            application_limit = st.number_input(
                "Maximum applications", min_value=1, value=DEFAULT_APPLICATION_LIMIT, step=1
            )
        with col2:
            applications_enabled = st.checkbox("Accept applications", value=True)

        col1, col2 = st.columns(2)
        with col1:
            registration_start = st.date_input("Registration start", value=date.today())
        with col2:
            registration_end = st.date_input("Registration end", value=date.today())

        required_documents = st.multiselect("Required documents", options=DOCUMENT_TYPES)

        st.markdown("### Eligibility rules")
        rule_text = st.text_area(
            "Describe eligibility rules in normal language *",
            height=180,
            placeholder=(
                "Example:\nApplicant must be female.\nApplicant must belong to SC category.\n"
                "Family income must be below Rs. 2,50,000.\nApplicant must be studying B.Tech.\n"
                "Applicant must be from Maharashtra."
            ),
        )
        submitted = st.form_submit_button("Create Opportunity", type="primary")

    if submitted:
        if not name.strip():
            st.error("Opportunity name is required.")
        elif registration_end < registration_start:
            st.error("Registration end date cannot be before start date.")
        elif not rule_text.strip():
            st.error("Please enter eligibility rules.")
        else:
            _create_and_interpret(
                name=name.strip(),
                organization=organization.strip(),
                description=description.strip(),
                official_url=official_url.strip(),
                application_limit=int(application_limit),
                applications_enabled=applications_enabled,
                registration_start=registration_start.isoformat(),
                registration_end=registration_end.isoformat(),
                natural_language_rules=rule_text.strip(),
                required_documents=required_documents,
            )

    # AI-interpreted rules waiting for the admin's verification
    pending_id = st.session_state.get(_CREATE_ID_KEY)
    pending_rules = st.session_state.get(_CREATE_RULES_KEY)
    if pending_id and pending_rules:
        st.divider()
        opportunity = get_opportunity(pending_id)
        st.subheader(f"3️⃣ Verify AI-interpreted rules for “{opportunity['name'] if opportunity else pending_id}”")
        show_rule_editor(pending_rules, _CREATE_RULES_KEY, pending_id, key_prefix="create_rules")


def _create_and_interpret(required_documents, **data):
    try:
        opportunity_id = create_opportunity(**data)
        save_required_documents(opportunity_id, required_documents)
    except Exception as error:
        st.error(f"Could not create opportunity: {error}")
        return

    parsed = parse_rules_with_ai(data["natural_language_rules"])
    if parsed:
        st.session_state[_CREATE_ID_KEY] = opportunity_id
        set_pending(_CREATE_RULES_KEY, parsed)
        flash("success", "Opportunity created. The AI interpreted the rules — please review and verify them below.")
    else:
        # Don't leave the admin stuck: open a blank editor so rules can be typed manually.
        st.session_state[_CREATE_ID_KEY] = opportunity_id
        set_pending(
            _CREATE_RULES_KEY,
            [{"field": "gender", "operator": "equals", "value": "", "human_readable": ""}],
        )
        flash(
            "warning",
            "Opportunity created, but the AI could not interpret the rules. "
            "Enter them manually below (or use 'Manage Opportunities' later).",
        )
    st.rerun()


# =========================================================
# Tab 2: manage — rules section
# =========================================================
def show_rules_section(opportunity):
    opportunity_id = opportunity["id"]
    state_key = f"manage_pending_rules_{opportunity_id}"

    st.subheader("📋 Eligibility Rules")

    rules = get_rules(opportunity_id)
    if rules:
        for number, rule in enumerate(rules, start=1):
            st.write(f"**{number}.** {_describe_rule(rule)}")
        st.success("Rules are verified and active.")
    else:
        st.warning("No verified eligibility rules yet. Students cannot be checked until rules are verified.")

    # pending editor (AI output or manual edit) — shown instead of the action buttons
    pending = st.session_state.get(state_key)
    if pending is not None:
        st.divider()
        show_rule_editor(pending, state_key, opportunity_id, key_prefix=f"manage_rules_{opportunity_id}")
        return

    rule_text = st.text_area(
        "Plain-language rules",
        value=opportunity.get("natural_language_rules") or "",
        height=140,
        key=f"nl_rules_{opportunity_id}",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🤖 Interpret with AI", key=f"interpret_{opportunity_id}"):
            if not rule_text.strip():
                st.error("Please enter the rules first.")
            else:
                update_opportunity(opportunity_id, natural_language_rules=rule_text.strip())
                parsed = parse_rules_with_ai(rule_text)
                if parsed:
                    set_pending(state_key, parsed)
                    st.rerun()
    with col2:
        if st.button("✏️ Edit / add rules manually", key=f"edit_rules_{opportunity_id}"):
            set_pending(
                state_key,
                [{k: r[k] for k in ("field", "operator", "value", "human_readable")} for r in rules]
                or [{"field": "gender", "operator": "equals", "value": "", "human_readable": ""}],
            )
            st.rerun()


# =========================================================
# Tab 2: manage — documents, sources, settings, applications
# =========================================================
def show_required_documents(opportunity_id):
    st.subheader("📄 Required Documents")
    existing = get_required_documents(opportunity_id)
    known = [d for d in existing if d in DOCUMENT_TYPES]
    extra = [d for d in existing if d not in DOCUMENT_TYPES]  # keep custom types selectable

    selected = st.multiselect(
        "Select required documents",
        options=DOCUMENT_TYPES + extra,
        default=known + extra,
        key=f"required_docs_{opportunity_id}",
    )
    if st.button("💾 Save Required Documents", key=f"save_docs_{opportunity_id}"):
        try:
            save_required_documents(opportunity_id, selected)
        except Exception as error:
            st.error(f"Could not save documents: {error}")
        else:
            flash("success", "Required documents saved.")
            st.rerun()


def show_source_documents(opportunity_id):
    st.subheader("📚 Official Source Documents for RAG")
    st.caption(
        "Upload official PDFs, notifications, guidelines or scheme documents. "
        "They are indexed so the AI can quote the official clause that explains each decision."
    )

    uploaded = st.file_uploader("Upload official PDF", type=["pdf"], key=f"source_pdf_{opportunity_id}")
    if uploaded is not None:
        st.write(f"Selected: **{uploaded.name}**")
        if st.button("📥 Save & Index Official Document", key=f"save_source_{opportunity_id}"):
            try:
                folder = Path(SOURCE_DOCUMENTS_DIR) / str(opportunity_id)
                folder.mkdir(parents=True, exist_ok=True)
                file_path = folder / _safe_filename(uploaded.name)
                file_path.write_bytes(uploaded.getbuffer())
                save_opportunity_source(opportunity_id, file_path.name, str(file_path))
            except Exception as error:
                st.error(f"Could not save source document: {error}")
            else:
                with st.spinner("Indexing document for search..."):
                    try:
                        from rag.retriever import index_source_pdf

                        count = index_source_pdf(file_path, opportunity_id)
                    except Exception as error:
                        flash(
                            "warning",
                            f"Document saved, but it could not be indexed for search: {error}",
                        )
                    else:
                        if count:
                            flash("success", f"Official document saved and indexed ({count} text chunks).")
                        else:
                            flash(
                                "warning",
                                "Document saved, but no text could be read from it "
                                "(is it a scanned image PDF?).",
                            )
                st.rerun()

    sources = get_opportunity_sources(opportunity_id)
    if sources:
        st.markdown("**Uploaded source documents:**")
        for source in sources:
            st.write(f"📄 {source.get('file_name', 'Unknown')}")


def show_opportunity_settings(opportunity):
    opportunity_id = opportunity["id"]
    st.subheader("⚙️ Application Settings")

    col1, col2 = st.columns(2)
    with col1:
        new_limit = st.number_input(
            "Maximum applications",
            min_value=1,
            value=max(1, safe_int(opportunity.get("application_limit"), DEFAULT_APPLICATION_LIMIT)),
            step=1,
            key=f"limit_{opportunity_id}",
        )
    with col2:
        new_enabled = st.checkbox(
            "Applications currently open",
            value=bool(opportunity.get("applications_enabled", 1)),
            key=f"enabled_{opportunity_id}",
        )

    col1, col2 = st.columns(2)
    with col1:
        new_start = st.date_input(
            "Registration starts",
            value=parse_date(opportunity.get("registration_start")) or date.today(),
            key=f"start_{opportunity_id}",
        )
    with col2:
        new_end = st.date_input(
            "Registration ends",
            value=parse_date(opportunity.get("registration_end")) or date.today(),
            key=f"end_{opportunity_id}",
        )

    if st.button("💾 Save Application Settings", key=f"save_settings_{opportunity_id}"):
        if new_end < new_start:
            st.error("End date cannot be before start date.")
            return
        try:
            update_opportunity(
                opportunity_id,
                application_limit=int(new_limit),
                registration_start=new_start.isoformat(),
                registration_end=new_end.isoformat(),
                applications_enabled=new_enabled,
            )
        except Exception as error:
            st.error(f"Could not update settings: {error}")
        else:
            flash("success", "Application settings updated.")
            st.rerun()


def show_applications(opportunity_id):
    st.subheader("👥 Applications")
    try:
        applications = get_opportunity_applications(opportunity_id)
    except Exception as error:
        st.error(f"Could not load applications: {error}")
        return

    if not applications:
        st.info("No applications received yet.")
        return

    st.write(f"Total applications: **{len(applications)}**")
    for application in applications:
        status = application.get("status") or APPLICATION_STATUSES[0]
        title = (
            f"👤 {application.get('name') or application.get('username')} — "
            f"{get_status_emoji(status)} {get_status_label(status)}"
        )
        with st.expander(title):
            st.write(f"**Application ID:** {application['id']}")
            st.write(f"**Username:** {application.get('username', '')}")
            st.write(f"**Applied:** {format_date(application.get('applied_at'))}")
            st.write(
                f"**Profile:** {application.get('gender') or '—'}, "
                f"{application.get('category') or '—'}, "
                f"{application.get('course') or '—'} (year {application.get('year') or '—'}), "
                f"{application.get('state') or '—'}, "
                f"income {format_currency(application.get('family_income'))}"
            )

            new_status = st.selectbox(
                "Application status",
                APPLICATION_STATUSES,
                index=APPLICATION_STATUSES.index(status) if status in APPLICATION_STATUSES else 0,
                key=f"status_{application['id']}",
            )
            reason = st.text_area(
                "Admin note / reason",
                value=application.get("reason") or "",
                key=f"reason_{application['id']}",
            )
            if st.button("Update Application", key=f"update_app_{application['id']}"):
                try:
                    update_application_status(application["id"], new_status, reason)
                except Exception as error:
                    st.error(f"Could not update application: {error}")
                else:
                    flash("success", "Application updated.")
                    st.rerun()


def show_opportunity_detail(opportunity_id):
    opportunity = get_opportunity(opportunity_id)
    if not opportunity:
        st.error("Opportunity not found.")
        return

    st.header(f"🎓 {opportunity.get('name', 'Opportunity')}")
    if opportunity.get("organization"):
        st.caption(f"Organization: {opportunity['organization']}")
    if opportunity.get("description"):
        st.write(opportunity["description"])
    if opportunity.get("official_url"):
        st.caption(opportunity["official_url"])

    applications = get_opportunity_applications(opportunity_id)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Application Limit", opportunity.get("application_limit") or "—")
    with col2:
        st.metric("Applications Received", len(applications))
    with col3:
        st.metric("Status", "OPEN" if opportunity.get("applications_enabled") else "CLOSED")

    st.divider()
    show_rules_section(opportunity)
    st.divider()
    show_required_documents(opportunity_id)
    st.divider()
    show_source_documents(opportunity_id)
    st.divider()
    show_opportunity_settings(opportunity)
    st.divider()
    show_applications(opportunity_id)


# =========================================================
# MAIN
# =========================================================
def render_admin():
    """Called by app.py for logged-in admins."""
    user = st.session_state.get("user")
    if not user or user.get("role") != "ADMIN":
        st.error("Admin access required.")
        return

    st.title("🛠️ Admin Dashboard")
    st.caption("Manage scholarships, eligibility rules, documents, applications and RAG sources.")
    show_flash()

    tab_create, tab_manage = st.tabs(["➕ Add Opportunity", "📚 Manage Opportunities"])

    with tab_create:
        show_create_opportunity()

    with tab_manage:
        st.header("Existing Opportunities")
        opportunities = get_all_opportunities()
        if not opportunities:
            st.info("No opportunities have been created yet.")
            return

        # Select by id (names are not unique).
        labels = {o["id"]: f"{o['name']}  (#{o['id']})" for o in opportunities}
        selected_id = st.selectbox(
            "Select an opportunity",
            options=list(labels),
            format_func=lambda opportunity_id: labels[opportunity_id],
        )
        show_opportunity_detail(selected_id)
