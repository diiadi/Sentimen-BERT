import streamlit as st
import sqlite3
from utils import hash_password
import pandas as pd

# Function to fetch all users from the database
def fetch_users(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT username, role FROM users")
    return cursor.fetchall()

# Function to update the password for a specific user
def update_user_password(conn, username, new_password):
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password = ? WHERE username = ?", (hash_password(new_password), username))
    conn.commit()

# Function to add a new user
def add_user(conn, username, password, role):
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                       (username, hash_password(password), role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# Function to delete a user
def delete_user(conn, username):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()

st.title("User Management")

# Database connection
conn = sqlite3.connect('users.db')

# Fetch all users
users = fetch_users(conn)

if users:

    st.write("### Users Table")
    st.write(f"Total users: {len(users)}")
    # Convert to DataFrame
    users_df = pd.DataFrame(users, columns=["Username", "Role"])
    st.dataframe(users_df, use_container_width=True, hide_index=True)

    # Update Password Form
    st.subheader("Change User Password")
    with st.expander("Change Password"):
        username_to_update = st.selectbox("Select User", [user[0] for user in users])
        new_password = st.text_input("New Password", type="password")
        confirm_new_password = st.text_input("Confirm New Password", type="password")

        if st.button("Update Password", key="update_password"):
            if new_password == confirm_new_password:
                update_user_password(conn, username_to_update, new_password)
                st.success(f"Password for {username_to_update} has been updated successfully!")
            else:
                st.error("New passwords do not match. Please confirm your new password.")

    # Add New User Form
    st.subheader("Add New User")
    with st.expander("Add User"):
        new_username = st.text_input("New Username")
        new_user_password = st.text_input("Password", type="password")
        new_user_confirm_password = st.text_input("Confirm Password", type="password")
        new_user_role = st.selectbox("Role", ["admin", "user"])

        if st.button("Add User", key="add_user"):
            if new_user_password == new_user_confirm_password:
                if add_user(conn, new_username, new_user_password, new_user_role):
                    st.success(f"User {new_username} has been added successfully!")
                else:
                    st.error(f"Failed to add user {new_username}. The username might already exist.")
            else:
                st.error("Passwords do not match. Please confirm the password.")

    # Delete User Form
    st.subheader("Delete User")
    with st.expander("Delete User"):
        username_to_delete = st.selectbox("Select User to Delete", [user[0] for user in users])

        if st.button("Delete User", key="delete_user"):
            delete_user(conn, username_to_delete)
            st.success(f"User {username_to_delete} has been deleted successfully!")

else:
    st.warning("No users found in the database.")

conn.close()
