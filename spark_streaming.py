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

# --- WINDOWS SPARK FIX ---
# Spark on Windows requires 'winutils.exe' in a 'bin' folder under HADOOP_HOME.
# 1. Create a folder named 'hadoop' in this project directory.
# 2. Inside 'hadoop', create a folder named 'bin'.
# 3. Download 'winutils.exe' from https://github.com/cdarlint/winutils/blob/master/hadoop-3.2.2/bin/winutils.exe
# 4. Place 'winutils.exe' inside the 'hadoop/bin' folder.
# The following code automatically tells Spark where to find it:
hadoop_home = os.path.abspath("hadoop")
hadoop_bin = os.path.join(hadoop_home, "bin")

os.environ['HADOOP_HOME'] = hadoop_home
os.environ['PATH'] = hadoop_bin + os.pathsep + os.environ['PATH']
sys.path.append(hadoop_bin)
# --------------------------

# --- MYSQL CONFIGURATION ---
MYSQL_USER = "root"
MYSQL_PASSWORD = "shwet"  # Replace with your actual password
MYSQL_DB_URL = "jdbc:mysql://localhost:3306/cctv"
MYSQL_TABLE = "threat_alerts"
# ---------------------------

# Path to monitor for incoming JSON alert files
ALERTS_DIR = "alerts/logs/"
CHECKPOINT_DIR = "alerts/checkpoints/"

# 1. Initialize Spark Session with MySQL JDBC Connector
spark = SparkSession.builder \
    .appName("CCTV_Security_Streaming") \
    .config("spark.jars.packages", "mysql:mysql-connector-java:8.0.33") \
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

# Ensure the alerts directory exists
if not os.path.exists(ALERTS_DIR):
    os.makedirs(ALERTS_DIR)
    print(f"Created directory: {ALERTS_DIR}")

# 3. Create a Streaming DataFrame by reading JSON files
print(f"Monitoring directory: {ALERTS_DIR} for new alerts...")
raw_alerts_df = spark.readStream \
    .schema(alert_schema) \
    .json(ALERTS_DIR)

# 4. Filter the data: confidence > 0.5 and (event_type IS 'fire' OR 'Weapon')
# We use lower() to handle case sensitivity safely
from pyspark.sql.functions import lower, col

filtered_alerts_df = raw_alerts_df.filter(
    (col("confidence") > 0.5) & 
    (lower(col("event_type")).isin("fire", "weapon"))
)

# 5. Define Function to write each batch to MySQL
def write_to_mysql(batch_df, batch_id):
    """
    Writes the filtered micro-batch DataFrame to MySQL using JDBC.
    """
    if batch_df.count() > 0:
        print(f"PROCESSED THREAT: Batch ID {batch_id} | Saved {batch_df.count()} alerts to Database.")
        batch_df.write \
            .format("jdbc") \
            .option("url", MYSQL_DB_URL) \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("dbtable", MYSQL_TABLE) \
            .option("user", MYSQL_USER) \
            .option("password", MYSQL_PASSWORD) \
            .mode("append") \
            .save()
    else:
        # This will show up if logs are received but they don't meet the confidence/type criteria
        pass

# 6. Start the Streaming Query
query = filtered_alerts_df.writeStream \
    .foreachBatch(write_to_mysql) \
    .option("checkpointLocation", CHECKPOINT_DIR) \
    .start()

print("Streaming started. Waiting for logs...")
query.awaitTermination()
