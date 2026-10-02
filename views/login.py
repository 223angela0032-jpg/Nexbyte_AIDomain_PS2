# views/login.py
"""Login page: checks credentials via database.py and stores the user in session state."""
import streamlit as st

from database import authenticate_user
from utils.ui import show_flash


def show_demo_credentials():
    """Demo logins for development. Remove before production."""
    with st.expander("🧪 Demo login credentials"):
        st.write("**Admin**")
        st.code("Username: admin\nPassword: admin123")
        st.write("**Student**")
        st.code("Username: student01\nPassword: student123")
        st.caption("student01 … student07 all use the password student123.")


def render_login():
    """Show the login form. On success the user is stored in st.session_state['user']."""
    st.title("🎓 Scholarship & Scheme Assistant")
    st.subheader("Login")
    st.write("Login to check scholarship and welfare-scheme eligibility.")
    show_flash()
    st.divider()

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("🔐 Login", type="primary")

    if submitted:
        username = username.strip()
        if not username:
            st.error("Please enter your username.")
        elif not password:
            st.error("Please enter your password.")
        else:
            try:
                user = authenticate_user(username, password)
            except Exception as error:
                st.error(f"Login failed: {error}")
                user = None
            else:
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                st.error("Incorrect username or password.")

    show_demo_credentials()
    return False
