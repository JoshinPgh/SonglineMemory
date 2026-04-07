import sqlite3
from datetime import datetime

# Define the name of our local database file
DB_NAME = "landscape.db"

def init_db():
    """
    Initializes the SQLite database. 
    If 'landscape.db' doesn't exist, Python creates it automatically.
    """
    # Connect to the database (this creates the file if it's missing)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create the LANDMARKS table (The data points/facts)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS landmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        concept_label TEXT NOT NULL,
        core_data TEXT NOT NULL,
        creation_date TEXT NOT NULL
    )
    ''')

    # Create the SONGLINES table (The narrative pathways connecting the facts)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS songlines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        origin_id INTEGER,
        destination_id INTEGER,
        narrative_context TEXT NOT NULL,
        FOREIGN KEY(origin_id) REFERENCES landmarks(id),
        FOREIGN KEY(destination_id) REFERENCES landmarks(id)
    )
    ''')

    conn.commit()
    conn.close()
    print("SUCCESS: The Aboriginal Narrative Landscape database has been initialized!")

def add_landmark(concept, data):
    """Adds a new landmark to the landscape."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
    INSERT INTO landmarks (concept_label, core_data, creation_date)
    VALUES (?, ?, ?)
    ''', (concept, data, timestamp))
    
    landmark_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print(f"LANDMARK PLACED: '{concept}' (ID: {landmark_id})")
    return landmark_id

# --- TEST EXECUTION ---
# When you run this file, it will trigger the setup function below.
if __name__ == "__main__":
    print("Booting up the Landscape...")
    init_db()