import os
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from streamlit_js_eval import streamlit_js_eval
from utils import init_db, authenticate_user, register_user

# Initialize the database connection
conn = init_db()

# Initialize cookies manager
cookies = EncryptedCookieManager(
    prefix="myprefix", 
    password=os.environ.get("COOKIES_PASSWORD", "mypassword"),
)

# Wait for the component to load and send us current cookies
if not cookies.ready():
    st.stop()
st.session_state.cookies = cookies

# Define pages based on user role
def check_role(cookies_role):
    if cookies_role == "user":
        return {
            f"WELCOME , {authenticated_user}": [
                st.Page("page/prediksi.py", title="Prediksi"),
                st.Page("page/report.py", title="Analisis Sentiment"),
                st.Page("page/analisis.py", title="Sentiment Analisis"),
                st.Page("page/profile.py", title="Account")
            ]
        }
    elif cookies_role == "admin":
        return {
            f"WELCOME , {authenticated_user}": [
                st.Page("page/prediksi.py", title="Prediksi"),
                st.Page("page/report.py", title="Analisis Sentiment"),
                st.Page("page/analisis.py", title="Sentiment Analisis"),
                st.Page("page/profile.py", title="Account"),
                st.Page("page/user_management.py", title="User Management")
            ]
        }

# Main UI
authenticated_user = cookies.get('authenticated')
authenticated_role = cookies.get('role')

if authenticated_user:
    # Display dashboard based on the user's role
    pages = check_role(authenticated_role)
    pg = st.navigation(pages)
    pg.run()

    with st.sidebar:
        if st.button("Logout", type="primary"):
            st.session_state.clear()
            cookies['authenticated'] = ""
            cookies['role'] = ""
            cookies.save()  # Ensure the deletion is saved immediately
            # Reload the page
            streamlit_js_eval(js_expressions="parent.window.location.reload()")

else:
    # Show login and register tabs when the user is not authenticated
    st.sidebar.empty()  # Hide the sidebar during login
    tab_login, tab_register = st.tabs(["Login", "Register"])

    # Login tab
    with tab_login:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            auth = authenticate_user(conn, login_username, login_password)
            if auth:
                # Set authentication cookies
                cookies['authenticated'] = login_username
                cookies['role'] = auth["role"]  # Default role for the user (can be adjusted based on the role)
                cookies.save()  # Save the cookies
                st.success("Login successful!")
                # Reload the page after login
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
            else:
                st.error("Invalid username or password.")

    # Register tab
    with tab_register:
        st.subheader("Register")
        register_username = st.text_input("Username", key="register_username")
        register_password = st.text_input("Password", type="password", key="register_password")

        if st.button("Register"):
            if register_user(conn, register_username, register_password):
                st.success("Registration successful! You can now log in.")
                # Reload the page after registration
                streamlit_js_eval(js_expressions="parent.window.location.reload()")
            else:
                st.error("Username already exists. Please choose a different one.")
