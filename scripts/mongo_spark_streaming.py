import os
import sys

# Force a Spark-compatible JDK on this machine before importing PySpark.
JAVA_17_CANDIDATES = [
    r"C:\jdk-17.0.0.1",
    r"C:\Users\shwet\AppData\Roaming\Code\User\globalStorage\pleiades.java-extension-pack-jdk\java\17",
]

java_home = None
for candidate in JAVA_17_CANDIDATES:
    if os.path.exists(os.path.join(candidate, "bin", "java.exe")):
        java_home = candidate
        break

if java_home:
    java_bin = os.path.join(java_home, "bin")
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = java_bin + os.pathsep + os.environ["PATH"]
    print(f"Using JAVA_HOME for Spark: {java_home}")
else:
    print("WARNING: Java 17 not found in expected paths. Spark may fail with newer Java versions.")

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType
from pyspark.sql.functions import lower, col
from pathlib import Path

# --- PROJECT STRUCTURE SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

# --- WINDOWS SPARK FIX ---
hadoop_home = str(PROJECT_ROOT / "hadoop")
hadoop_bin = os.path.join(hadoop_home, "bin")
os.environ['HADOOP_HOME'] = hadoop_home
os.environ['PATH'] = hadoop_bin + os.pathsep + os.environ['PATH']
sys.path.append(hadoop_bin)
# --------------------------

# --- MONGODB CONFIGURATION ---
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "cctv"
MONGO_COLLECTION = "threat_alerts"
# -----------------------------

# Path to monitor for incoming JSON alert files
ALERTS_DIR = str(PROJECT_ROOT / "alerts" / "logs")
CHECKPOINT_DIR = str(PROJECT_ROOT / "alerts" / "mongo_checkpoints")

# 1. Initialize Spark Session with MongoDB Connector
spark = SparkSession.builder \
    .appName("CCTV_Security_Mongo_Streaming") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.13:11.0.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Define Schema to match the JSON alerts
alert_schema = StructType([
    StructField("timestamp", LongType(), True),
    StructField("event_type", StringType(), True),
    StructField("confidence", FloatType(), True),
    StructField("image_path", StringType(), True),
    StructField("camera_id", StringType(), True)
])

# Ensure the directories exist locally
if not os.path.exists(ALERTS_DIR):
    os.makedirs(ALERTS_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# 3. Create a Streaming DataFrame by reading JSON files
print(f"Monitoring directory: {ALERTS_DIR} for new alerts...")
raw_alerts_df = spark.readStream \
    .schema(alert_schema) \
    .json(ALERTS_DIR)

# 4. Filter the data: confidence > 0.5 and (event_type IS 'fire' OR 'weapon')
filtered_alerts_df = raw_alerts_df.filter(
    (col("confidence") > 0.5) & 
    (lower(col("event_type")).isin("fire", "weapon"))
)

# 5. Define Function to write each batch to MongoDB
def write_to_mongodb(batch_df, batch_id):
    """
    Writes the filtered micro-batch DataFrame to MongoDB.
    """
    if batch_df.count() > 0:
        print(f"PROCESSED THREAT: Batch ID {batch_id} | Saved {batch_df.count()} alerts to Database.")
        batch_df.write \
            .format("mongodb") \
            .option("connection.uri", MONGO_URI) \
            .option("database", MONGO_DB) \
            .option("collection", MONGO_COLLECTION) \
            .mode("append") \
            .save()
    else:
        # No data in this batch after filtering
        pass

# 6. Start the Streaming Query
query = filtered_alerts_df.writeStream \
    .foreachBatch(write_to_mongodb) \
    .option("checkpointLocation", CHECKPOINT_DIR) \
    .start()

print("MongoDB Streaming started. Waiting for logs...")
query.awaitTermination()
