import sqlite3
import hashlib

# Initialize database connection
def init_db():
    conn = sqlite3.connect("users.db")
    with conn:
        # Buat tabel jika belum ada
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                role TEXT
            )
        """)

        # Periksa apakah admin sudah ada
        result = conn.execute("SELECT 1 FROM users WHERE username = 'admin'").fetchone()
        if not result:
            # Buat admin default jika belum ada
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("admin", hash_password("admin123"), "admin")
            )
            print("Admin user created with default password 'xxxxxxx'.")
        else:
            print("Admin user already exists.")
    return conn

# Password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authenticate user credentials
def authenticate_user(conn, username, password):
    with conn:
        result = conn.execute(
            "SELECT password, role FROM users WHERE username = ?", (username,)
        ).fetchone()
    if result and result[0] == hash_password(password):
        return {"username": username, "role": result[1]}
    return None

# Register a new user
def register_user(conn, username, password, role="user"):
    try:
        with conn:
            conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                         (username, hash_password(password), role))
        return True
    except sqlite3.IntegrityError:
        return False
