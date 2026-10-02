# app.py
"""Entry point:  streamlit run app.py"""
import streamlit as st

from database import init_db, seed_demo_users, seed_sample_opportunities

# Must be the first Streamlit command.
st.set_page_config(
    page_title="Scholarship AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def bootstrap():
    """Create tables and demo data once per server process."""
    init_db()
    seed_demo_users()
    seed_sample_opportunities()
    return True


bootstrap()

if "user" not in st.session_state:
    st.session_state["user"] = None


def logout():
    st.session_state["user"] = None
    st.rerun()


def show_sidebar(user):
    with st.sidebar:
        st.title("🎓 Scholarship AI")
        st.divider()
        st.write(f"**Name:** {user['name']}")
        st.write(f"**Username:** {user['username']}")
        st.write(f"**Role:** {user['role']}")
        st.divider()
        if st.button("🚪 Logout"):
            logout()


def main():
    user = st.session_state["user"]

    if user is None:
        from views.login import render_login

        render_login()
        return

    show_sidebar(user)

    if user["role"] == "ADMIN":
        from views.admin import render_admin

        render_admin()
    elif user["role"] == "USER":
        from views.user import render_user

        render_user()
    else:
        st.error("Invalid account role.")
        logout()


main()
