# 🛡️ Campus Safety System — Real-Time Weapon Detection

An AI-powered campus surveillance system that monitors multiple CCTV camera feeds to detect weapons and knives in real time, process captured frames, and store detection results using a distributed data-processing and storage architecture.

## 📌 Project Overview

The Campus Safety System integrates computer vision, real-time data streaming, parallel processing, and distributed storage to improve campus security. It processes surveillance data from multiple cameras, detects potential weapon threats, captures relevant images, and stores detection information for monitoring and investigation.

The system uses Apache Kafka for multi-camera streaming, PySpark for parallel processing, MySQL for structured detection records, MongoDB for bounding-box metadata, and Hadoop HDFS for captured image storage.

## ✨ Key Features

- **Multi-Camera Surveillance:** Handles surveillance feeds from multiple cameras.
- **Real-Time Weapon Detection:** Detects weapons and knives in captured video frames using an AI object-detection model.
- **Confidence Score Tracking:** Records the model's confidence score for each detection.
- **Parallel Processing:** Uses PySpark to support distributed processing of image-related workloads.
- **Structured Data Management:** Stores detection details and Hadoop image paths in MySQL.
- **Bounding-Box Metadata Storage:** Stores bounding-box coordinates and related captured-frame metadata in MongoDB.
- **Distributed Image Storage:** Stores captured images in Hadoop HDFS.
- **Detection Traceability:** Links detection records with their corresponding images and frame metadata.

## 🏗️ System Architecture

```text
        Multiple CCTV Cameras
                 |
                 v
          Apache Kafka
       (Data / Event Streaming)
                 |
                 v
       AI Weapon Detection
        (Computer Vision)
                 |
                 v
       Capture Relevant Frames
                 |
                 v
           PySpark
      (Parallel Processing)
                 |
       +---------+---------+
       |         |         |
       v         v         v
     MySQL    MongoDB    Hadoop HDFS
       |         |         |
       |         |         |
       v         v         v
 Detection   Bounding-   Captured
 Records     Box Data    Images
```

*Note: The diagram represents the logical architecture. Actual execution order depends on the implemented pipeline.*

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic and pipeline integration |
| Computer Vision / AI Model | Weapon and knife detection |
| Apache Kafka | Streaming data from multiple camera sources |
| PySpark | Parallel and distributed data processing |
| MySQL | Structured detection records |
| MongoDB | Bounding-box coordinates and captured-frame metadata |
| Hadoop HDFS | Distributed storage of captured images |

## 🗄️ Database and Storage Design

### 1. MySQL — Structured Detection Records

MySQL stores structured information associated with each weapon detection.

**Stored information:**
- Image path pointing to the captured image in Hadoop HDFS
- Confidence score returned by the detection model
- Detected weapon type, such as knife or another supported weapon class

**Example record:**

| Field | Example |
|---|---|
| `image_path` | `/campus/images/frame_001.jpg` |
| `confidence` | `0.94` |
| `weapon_type` | `knife` |

*The field names and example values are illustrative; the actual schema may differ.*

### 2. MongoDB — Bounding-Box Metadata

MongoDB stores bounding-box information associated with captured frames. This information describes where a detected object appears within an image.

**Stored information may include:**
- Bounding-box coordinates (`x`, `y`, `width`, `height`, or equivalent coordinates)
- Captured-frame metadata
- Object or detection identifiers, if implemented

**Example document:**

```json
{
  "frame_id": "frame_001",
  "weapon_type": "knife",
  "bounding_box": {
    "x": 120,
    "y": 85,
    "width": 160,
    "height": 210
  }
}
```

*This is an illustrative document, not a verified copy of the actual MongoDB schema.*

### 3. Hadoop HDFS — Captured Image Storage

Hadoop Distributed File System (HDFS) stores the captured images generated during weapon detection.

**Responsibilities:**
- Stores captured images as files.
- Supports distributed storage across a Hadoop cluster.
- Provides image paths that can be referenced by MySQL records.

MySQL stores the image path, while Hadoop HDFS stores the actual image file.

## 🔄 Processing Workflow

1. **Camera Input:** Multiple CCTV cameras provide surveillance video feeds.
2. **Streaming:** Apache Kafka manages the flow of camera data or frame-related events.
3. **Frame Processing:** The application extracts or receives frames for analysis.
4. **Weapon Detection:** The AI model identifies supported weapon classes and produces confidence scores and bounding boxes.
5. **Frame Capture:** Relevant frames are captured for storage and further investigation.
6. **Parallel Processing:** PySpark processes image-related data in parallel according to the implemented pipeline.
7. **Image Storage:** Captured images are stored in Hadoop HDFS.
8. **Structured Storage:** MySQL records the image path, confidence score, and detected weapon type.
9. **Metadata Storage:** MongoDB stores bounding-box coordinates and captured-frame metadata.

## 📂 Example Project Structure

```text
campus-safety-system/
│
├── README.md
├── requirements.txt
├── config/
│   └── config.example
├── src/
│   ├── camera_stream.py
│   ├── kafka_producer.py
│   ├── kafka_consumer.py
│   ├── weapon_detection.py
│   ├── spark_processing.py
│   ├── mysql_storage.py
│   ├── mongodb_storage.py
│   └── hadoop_storage.py
│
├── models/
│   └── README.md
│
└── tests/
    └── README.md
```

*This is a suggested structure. Rename the files and directories to match your actual repository.*

## ⚙️ Installation and Setup

### Prerequisites

Install or configure the following components:

- Python 3
- Apache Kafka
- Apache Spark with PySpark
- MySQL Server
- MongoDB
- Hadoop HDFS
- The required AI model and its dependencies

### Setup

1. Clone the repository:

   ```bash
   git clone <your-repository-url>
   cd campus-safety-system
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   ```

   Windows:

   ```bash
   venv\Scripts\activate
   ```

   Linux/macOS:

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure Kafka, Spark, MySQL, MongoDB, and Hadoop HDFS using your local environment settings.

5. Configure database credentials, Kafka topics, HDFS directories, camera sources, and model paths.

6. Start the required services and run the application using the actual entry point in your repository.

**Important:** The commands above provide a setup template. Exact service-start and application-run commands depend on your implementation and environment.

## 🔐 Configuration and Security

- Store database credentials and service configuration in environment variables or a protected configuration file.
- Do not commit passwords, API keys, or sensitive camera credentials to GitHub.
- Restrict access to captured surveillance images and detection records.
- Establish appropriate image-retention and access-control policies.
- Validate detections before treating them as confirmed security incidents.

## 🎯 Applications

- Campus and university security
- Educational institution surveillance
- Real-time weapon-threat monitoring
- Centralized multi-camera monitoring
- Security incident recording and investigation

## 🚀 Future Enhancements

- A real-time security monitoring dashboard
- Camera-wise detection statistics
- Automated alerts for high-confidence detections
- Searchable incident history
- Detection performance evaluation and false-positive reduction
- Role-based access control for security personnel

## ⚠️ Disclaimer

This project is intended to support campus security monitoring. AI detections can produce false positives and false negatives. Detection results should be reviewed by authorized security personnel before taking action. The system should be deployed with appropriate privacy safeguards and institutional authorization.

## 👨‍💻 Project Summary

The Campus Safety System combines AI-based weapon detection with Apache Kafka, PySpark, MySQL, MongoDB, and Hadoop HDFS to support multi-camera surveillance, parallel data processing, structured detection logging, bounding-box metadata storage, and distributed image storage.
