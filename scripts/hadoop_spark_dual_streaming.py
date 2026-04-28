import os
import sys
from pathlib import Path

# --- JDK & HADOOP ENVIRONMENT SETUP ---
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

# --- WINDOWS SPARK FIX ---
hadoop_home = str(PROJECT_ROOT / "hadoop")
hadoop_bin = os.path.join(hadoop_home, "bin")
os.environ['HADOOP_HOME'] = hadoop_home
os.environ['PATH'] = hadoop_bin + os.pathsep + os.environ['PATH']
sys.path.append(hadoop_bin)

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, FloatType
from pyspark.sql.functions import lower, col

# --- DATABASES CONFIGURATION ---
# MongoDB
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "cctv"
MONGO_COLLECTION = "threat_alerts"

# MySQL
MYSQL_USER = "root"
MYSQL_PASSWORD = "shwet"  # Replace with actual
MYSQL_DB_URL = "jdbc:mysql://localhost:3306/cctv"
MYSQL_TABLE = "threat_alerts"

# --- HDFS CONFIGURATION ---
ALERTS_DIR = "webhdfs://localhost:9870/cctv/alerts/logs"
# Checkpoint MUST be local to prevent Spark from crashing on DataNode blocks
CHECKPOINT_DIR = "file:///c:/Users/shwet/Downloads/fire/alerts/checkpoints_dual"

# 1. Initialize Spark Session with Both MySQL and MongoDB Connectors
spark = SparkSession.builder \
    .appName("HDFS_Dual_Streaming_MongoDB_MySQL") \
    .config("spark.sql.streaming.schemaInference", "true") \
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
    .config("spark.jars.packages", "mysql:mysql-connector-java:8.0.33,org.mongodb.spark:mongo-spark-connector_2.13:11.0.0") \
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

# 3. Create a Streaming DataFrame by reading JSON files from HDFS
print(f"Monitoring HDFS directory: {ALERTS_DIR} for new alerts...")
raw_alerts_df = spark.readStream \
    .schema(alert_schema) \
    .json(ALERTS_DIR)

# 4. Filter the data: confidence > 0.5 and (event_type IS 'fire' OR 'weapon')
filtered_alerts_df = raw_alerts_df.filter(
    (col("confidence") > 0.5) & 
    (lower(col("event_type")).isin("fire", "weapon"))
)

# 5. Define Function for DUAL OUTPUT (MongoDB + MySQL)
def write_dual_output(batch_df, batch_id):
    """
    Writes the filtered micro-batch DataFrame to BOTH MongoDB and MySQL.
    Matches the 'Dual Output' architecture diagram.
    """
    count = batch_df.count()
    if count > 0:
        print(f"PROCESSED THREAT: Batch ID {batch_id} | Saving {count} alerts to MongoDB and MySQL...")
        
        # Write to MongoDB (Semi-Structured)
        try:
            batch_df.write \
                .format("mongodb") \
                .option("connection.uri", MONGO_URI) \
                .option("database", MONGO_DB) \
                .option("collection", MONGO_COLLECTION) \
                .mode("append") \
                .save()
            print(" -> Successfully saved to MongoDB")
        except Exception as e:
            print(f" -> Error writing to MongoDB: {e}")
        
        # Write to MySQL (Structured)
        try:
            batch_df.write \
                .format("jdbc") \
                .option("url", MYSQL_DB_URL) \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("dbtable", MYSQL_TABLE) \
                .option("user", MYSQL_USER) \
                .option("password", MYSQL_PASSWORD) \
                .mode("append") \
                .save()
            print(" -> Successfully saved to MySQL")
        except Exception as e:
            print(f" -> Error writing to MySQL: {e}")
    else:
        pass

# 6. Start the Streaming Query
query = filtered_alerts_df.writeStream \
    .foreachBatch(write_dual_output) \
    .option("checkpointLocation", CHECKPOINT_DIR) \
    .start()

print("Dual-Output Streaming to MongoDB & MySQL started. Waiting for logs on HDFS...")
query.awaitTermination()
