import sqlite3

try:
    conn = sqlite3.connect('database.sqlite')
    cursor = conn.cursor()

    # Query to get all table names in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("Tables found in database.sqlite:")
    if tables:
        for table in tables:
            print(f"- {table[0]}")
    else:
        print("No tables found. The database file might be empty or corrupted.")

except sqlite3.Error as e:
    print(f"An SQLite error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if conn:
        conn.close()