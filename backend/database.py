import os
import sqlite3
from datetime import datetime

# Setup the local database file path at the root level of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'tickets_history.db')

def init_db():
    # Establishes connection and builds standard database table structures natively
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            description TEXT NOT NULL,
            cleaned_text TEXT NOT NULL,
            predicted_department TEXT NOT NULL,
            predicted_priority TEXT NOT NULL,
            human_override_department TEXT,
            human_override_priority TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_ticket_to_db(tenant_id, subject, description, cleaned_text, predicted_dept, predicted_prio):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = """
        INSERT INTO tickets (
            tenant_id, subject, description, cleaned_text, 
            predicted_department, predicted_priority, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute(query, (
        tenant_id, subject, description, cleaned_text, 
        predicted_dept, predicted_prio, current_time
    ))
    
    # Grab the newly generated database auto-increment row ID
    inserted_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    return inserted_id