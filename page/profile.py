import streamlit as st
from utils import authenticate_user, hash_password
import sqlite3

def update_password(conn, username, old_password, new_password):
    # Check if the old password matches the current password
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    stored_password = cursor.fetchone()
    
    if stored_password and stored_password[0] == hash_password(old_password):
        # Update with new password
        cursor.execute("UPDATE users SET password = ? WHERE username = ?", (hash_password(new_password), username))
        conn.commit()
        return True
    return False

# Main Profile Page
# Get authenticated user from cookies
cookies = st.session_state.get("cookies")
authenticated_user = cookies.get('authenticated')

if authenticated_user:
    st.title(f"Account: {authenticated_user}")
    st.subheader("Change Password")

    # Get old and new password inputs
    old_password = st.text_input("Current Password", type="password")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm New Password", type="password")

    # Database connection
    conn = sqlite3.connect('users.db')

    if st.button("Update Password", type="primary"):
        if new_password == confirm_password:
            success = update_password(conn, authenticated_user, old_password, new_password)
            if success:
                st.success("Password updated successfully!")
            else:
                st.error("Incorrect old password. Please try again.")
        else:
            st.error("New passwords do not match. Please confirm your new password.")

    conn.close()
else:
    st.warning("You need to log in first to access your profile.")
