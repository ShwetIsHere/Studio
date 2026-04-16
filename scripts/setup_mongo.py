import pymongo
from pymongo.errors import ConnectionFailure
from pathlib import Path

def setup_mongodb():
    """
    Connects to the local MongoDB server, creates the 'cctv' database 
    and the 'threat_alerts' collection if they do not already exist.
    """
    # Project structure setup
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
    
    # --- CONFIGURATION SECTION ---
    # Update these variables with your local MongoDB connection string
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "cctv"
    collection_name = "threat_alerts"
    # -----------------------------

    client = None
    try:
        # 1. Connect to MongoDB Server
        print(f"Connecting to MongoDB server at {mongo_uri}...")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # Check if server is available
        client.admin.command('ping')
        print("Connected to MongoDB successfully.")

        # 2. Access/Create Database
        db = client[db_name]
        
        # 3. Access/Create Collection
        if collection_name in db.list_collection_names():
            print(f"Collection '{collection_name}' already exists in database '{db_name}'.")
        else:
            print(f"Creating collection '{collection_name}' in database '{db_name}'...")
            # In MongoDB, the collection is created automatically when the first document is inserted.
            # We can also explicitly create it.
            db.create_collection(collection_name)
            print("Collection created successfully.")

        # Optional: Setup an index for timestamp to speed up queries
        print("Setting up index on 'timestamp'...")
        db[collection_name].create_index([("timestamp", pymongo.DESCENDING)])
        
        print("\nMongoDB setup completed successfully.")

    except ConnectionFailure:
        print("Error: Could not connect to MongoDB. Is the server running?")
        print("\n[TIP] Check if:")
        print("1. Your MongoDB service is started.")
        print("2. The uri in the script is correct (default is 'mongodb://localhost:27017/').")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if client:
            client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    setup_mongodb()
