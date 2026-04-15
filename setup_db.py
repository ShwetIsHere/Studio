import mysql.connector
from mysql.connector import Error

def setup_database():
    """
    Connects to the local MySQL server, creates the 'cctv_security' database 
    and the 'threat_alerts' table if they do not already exist.
    """
    
    # --- CONFIGURATION SECTION ---
    # Update these variables with your local MySQL credentials
    db_config = {
        "host": "localhost",
        "user": "root",       # Replace with your MySQL username
        "password": "shwet" # Replace with your MySQL password
    }
    # -----------------------------

    db_name = "cctv"
    table_name = "threat_alerts"

    connection = None
    try:
        # 1. Connect to MySQL Server
        print(f"Connecting to MySQL server at {db_config['host']}...")
        connection = mysql.connector.connect(**db_config)
        
        if connection.is_connected():
            cursor = connection.cursor()

            # 2. Create Database if it doesn't exist
            print(f"Ensuring database '{db_name}' exists...")
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            
            # 3. Switch to the newly created database
            cursor.execute(f"USE {db_name}")

            # 4. Create Threat Alerts Table
            print(f"Ensuring table '{table_name}' exists...")
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp BIGINT NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                confidence FLOAT NOT NULL,
                image_path VARCHAR(255),
                camera_id VARCHAR(50) DEFAULT 'CAM_01'
            );
            """
            cursor.execute(create_table_query)
            
            print("Database and Table setup completed successfully.")

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        print("\n[TIP] Check if:")
        print("1. Your MySQL server is running locally.")
        print("2. The username and password in the script are correct.")
        print("3. The user has privileges to create databases.")

    finally:
        # Close the connection
        if connection is not None and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")

if __name__ == "__main__":
    setup_database()
