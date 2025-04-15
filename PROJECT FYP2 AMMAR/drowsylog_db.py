import sqlite3

# Connect to the database (it will create the database if it doesn't exist)
conn = sqlite3.connect('sleep_log.db')
c = conn.cursor()

# Create the table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS drowsiness_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT,
    system_uptime REAL,
    buzzer_status INTEGER,
    camera_status INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# Commit and close the connection
conn.commit()
conn.close()

print("Database and table created successfully.")
