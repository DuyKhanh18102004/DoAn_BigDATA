# 📊 PHÂN TÍCH CHI TIẾT DỰ ÁN BIG DATA - DEEPFAKE DETECTION

## 📋 TÓM TẮT CHUNG

**Tên Dự Án**: Deepfake Detection using TensorFlow & Apache Spark  
**Mục Đích**: Xây dựng pipeline End-to-End để phát hiện ảnh deepfake sử dụng Deep Learning + Big Data  
**Công Nghệ Stack**: Apache Spark, TensorFlow MobileNetV2, HDFS, Docker, Python  
**Kiến Trúc**: Data Lake (HDFS) → Feature Extraction (ML) → Model Training (Spark) → Model Serving

---

## 🏗️ KIẾN TRÚC TỔNG QUAN

```
┌─────────────────────────────────────────────────────────────────┐
│                     LOCAL DATASET (data/)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────▼──────────────┐
                │   HDFS CLUSTER           │
                │ (Namenode + 2 Datanodes) │
                │ (/user/data/raw)         │
                └────────────┬──────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼─────┐  ┌──────▼──────┐  ┌──────▼──────┐
    │  Upload     │  │  Feature    │  │   Training  │
    │  (Raw)      │──│ Extraction  │──│   (ML)      │
    │  HDFS       │  │ (TF Mobile  │  │   (LR)      │
    │             │  │  NetV2)     │  │   Spark     │
    └─────────────┘  └──────┬──────┘  └──────┬──────┘
                             │                │
                    ┌────────▼────────┐  ┌───▼─────────┐
                    │ Features        │  │   Model     │
                    │ (/user/data/    │  │ (/user/     │
                    │ features_tf)    │  │ models)     │
                    └─────────────────┘  └─────┬───────┘
                                               │
                                    ┌──────────▼─────────┐
                                    │  Evaluation &      │
                                    │  Load & Predict    │
                                    │  (Metrics)         │
                                    └────────────────────┘
```

---

## 📂 CẤU TRÚC THƯ MỤC DỰ ÁN

```
project/
├── docker-compose.yml          # Định cấu hình các container (HDFS, Spark)
├── Dockerfile                  # Image Spark + TensorFlow
├── PROJECT_ANALYSIS.md         # Tài liệu này
│
├── data/                       # Local dataset (train/test, REAL/FAKE)
│   ├── train/
│   │   ├── REAL/              # Ảnh thực người dùng
│   │   └── FAKE/              # Ảnh deepfake
│   └── test/
│       ├── REAL/
│       └── FAKE/
│
├── spark-config/              # Cấu hình Spark cluster
│   ├── spark-defaults.conf    # Spark tuning, memory, serialization
│   └── history-server.sh      # Event logging
│
├── scripts/
│   └── run_pipeline.ps1       # PowerShell orchestration script
│       (Tự động chạy 5 step pipeline)
│
└── src/                        # Source code Python
    ├── __init__.py
    │
    ├── 1_ingestion/           # Step 1: Data Upload
    │   ├── upload_to_hdfs.py  # Upload local → HDFS
    │   └── __init__.py
    │
    ├── 2_feature_extraction/  # Step 2: Feature Extraction
    │   ├── extract_mobilenetv2_features.py
    │   │   (TensorFlow MobileNetV2 → 1280-dim vectors)
    │   └── __init__.py
    │
    ├── 4_ml_training/         # Step 3: Model Training
    │   ├── ml_training_tf_features.py
    │   │   (LogisticRegression on TF features)
    │   ├── load_and_predict.py
    │   │   (Load model & predict on test data)
    │   ├── model_utils.py
    │   │   (Helper class: ModelManager)
    │   └── __init__.py
    │
    ├── 5_evaluation/          # Step 4: Evaluation
    │   ├── evaluate_tf_model.py
    │   │   (Metrics: Accuracy, Precision, Recall, F1)
    │   └── __init__.py
    │
    ├── config/                # Configuration
    │   ├── hdfs_config.py     # HDFS paths, connection settings
    │   └── __init__.py
    │
    └── utils/                 # Utilities
        ├── logging_utils.py   # Centralized logging
        └── __init__.py
```

---

## 🔄 PIPELINE 5-STEP

### **Step 1: Data Ingestion (Upload to HDFS)**
**File**: `src/1_ingestion/upload_to_hdfs.py`

```
Local Dataset → HDFS (/user/data/raw)
```

**Chức năng**:
- ✅ Upload ảnh từ local (`data/train`, `data/test`) lên HDFS
- ✅ Kiểm tra file tồn tại, folder structure
- ✅ Logging chi tiết quá trình upload
- ✅ Hỗ trợ test mode (max_files)

**Output**:
```
hdfs:///user/data/raw/train/REAL/  → 1000+ ảnh
hdfs:///user/data/raw/train/FAKE/  → 1000+ ảnh deepfake
hdfs:///user/data/raw/test/REAL/   → Test set
hdfs:///user/data/raw/test/FAKE/   → Test set
```

---

### **Step 2: Feature Extraction (TensorFlow MobileNetV2)**
**File**: `src/2_feature_extraction/extract_mobilenetv2_features.py`

```
Raw Images (HDFS) → 1280-dim Feature Vectors → HDFS (/user/data/features_tf)
```

**Chức năng**:
- 🧠 **MobileNetV2** (pre-trained ImageNet)
  - Lightweight model (3.5M parameters)
  - Output: 1280-dimensional dense vector
  - Memory-efficient cho Spark distributed processing
  
- 📊 **Batch Processing** (50 batches)
  - Mỗi batch ~ 100 ảnh
  - Tổng ~5000 ảnh train
  
- 💾 **Memory Optimization**
  - Periodic garbage collection
  - Disk spilling để tránh OOM
  - Clear TensorFlow session
  
- 🔄 **Save to Parquet** (Columnar format)
  - HDFS path: `hdfs:///user/data/features_tf/train/REAL/batch_*`
  - Compressed, efficient storage

**Key Metrics Tracked**:
- Batch processing time
- Memory usage (process + system)
- Feature statistics (mean, std, min, max)

**Output**:
```
Parquet files with schema:
  - label: Int (0=FAKE, 1=REAL)
  - features: Vector (1280 dimensions)
  - batch_id: String
```

---

### **Step 3: ML Model Training**
**File**: `src/4_ml_training/ml_training_tf_features.py`

```
Features (1280-dim) + Labels → Logistic Regression Model → HDFS Model
```

**Chức năng**:
- 📈 **Logistic Regression** (Binary Classification)
  - **Hyperparameters (Tuned)**:
    - `maxIter`: 300 (iterations)
    - `regParam`: 0.001 (L2 regularization)
    - `elasticNetParam`: 0.0 (pure L2, no L1)
    - `tol`: 1e-5 (tolerance)
    - `threshold`: 0.5 (decision boundary)
  
- 📚 **Data Split**:
  - Training: 80% (~4000 samples)
  - Validation: 20% (~1000 samples)
  - Test: 10 batches (separate test set)

- 🎯 **Evaluation Metrics**:
  - **Validation**: Accuracy, Precision, Recall, F1
  - **Test**: Same metrics for final assessment
  - **Confusion Matrix**: TP, TN, FP, FN analysis

- 💾 **Model Persistence**:
  - **Save to HDFS**: `hdfs:///user/models/logistic_regression_tf`
  - **Format**: Spark MLlib SerializableModel
  - **Size**: ~1KB (small model)

- 📊 **Results saved**:
  - `hdfs:///user/data/results_tf/metrics_tuned` (Parquet)
  - `hdfs:///user/data/results_tf/test_predictions_tuned` (Parquet)

**Output Metrics Example**:
```
Validation Accuracy: 92.5%
Validation F1-Score: 91.8%
Test Accuracy: 91.2%
Test F1-Score: 90.5%
```

---

### **Step 4: Load Model & Predict (NEW)**
**Files**: 
- `src/4_ml_training/load_and_predict.py` (Script)
- `src/4_ml_training/model_utils.py` (Helper Class)

```
Saved Model (HDFS) → Load & Predict on Test Data → Predictions
```

**Chức năng**:
- 🔓 **Load Pre-trained Model**
  - Load từ `hdfs:///user/models/logistic_regression_tf`
  - Display model metadata (coefficients dimension, threshold, etc.)
  
- 🔮 **Make Predictions**
  - Transform test features → predictions + probabilities
  - Output: prediction (0/1) + probability (0.0-1.0)
  
- 📈 **Evaluate Predictions**
  - Calculate metrics từ predictions
  - Confusion Matrix
  - Per-class performance
  
- 📋 **Utility Class** (model_utils.py):
  ```python
  class ModelManager:
      @staticmethod
      def load_tf_model(model_path) → LogisticRegressionModel
      @staticmethod
      def predict(model, df_features) → predictions_df
      @staticmethod
      def get_model_info(model) → dict
  ```

**Cách sử dụng trong code khác**:
```python
from src.utils.model_utils import ModelManager

# Load model
model = ModelManager.load_tf_model()

# Predict
predictions = ModelManager.predict(model, df_features)

# Get info
info = ModelManager.get_model_info(model)
```

---

### **Step 5: Evaluation & Analysis**
**File**: `src/5_evaluation/evaluate_tf_model.py`

```
Test Predictions → Comprehensive Evaluation → Metrics Report
```

**Chức năng**:
- 📊 **Load Test Predictions** từ parquet
- 🎯 **Calculate Metrics**:
  - Accuracy, Precision, Recall, F1
  - ROC-AUC
  - Per-class metrics (weighted average)
  
- 🔍 **Error Analysis**:
  - False Positives (FAKE predicted as REAL)
  - False Negatives (REAL predicted as FAKE)
  - Confidence score distribution
  
- 📈 **Visualizations** (optional):
  - Confusion matrix heatmap
  - ROC curve
  - Class distribution charts

- 💾 **Save reports** to HDFS/local

---

## 🛠️ CÔNG NGHỆ STACK CHI TIẾT

| Layer | Công Nghệ | Phiên Bản | Chức Năng |
|-------|-----------|----------|----------|
| **Containerization** | Docker | Latest | Isolate, reproducibility |
| **Data Storage** | HDFS | Hadoop 3.2.1 | Distributed file system |
| **Data Processing** | Apache Spark | 3.3.0 | Distributed computing |
| **ML Framework** | TensorFlow | 2.11.0 | Deep Learning (MobileNetV2) |
| **Image Processing** | Pillow | 9.5.0 | Image loading, resizing |
| **Linear Algebra** | NumPy | 1.23.5 | Feature vectors |
| **ML Algorithms** | Spark MLlib | 3.3.0 | Logistic Regression |
| **Serialization** | Kryo | Built-in | Fast object serialization |
| **Orchestration** | PowerShell | 5.x | Script automation |

---

## 💻 INFRASTRUCTURE SETUP

### **Docker Compose Services**

| Service | Image | CPU/Memory | Ports | Chức Năng |
|---------|-------|-----------|-------|----------|
| **namenode** | hadoop:3.2.1 | 1 CPU, 2GB | 9870 | HDFS Name node, metadata |
| **datanode-1** | hadoop:3.2.1 | 1 CPU, 2GB | N/A | HDFS Data node 1, storage |
| **datanode-2** | hadoop:3.2.1 | 1 CPU, 2GB | N/A | HDFS Data node 2, storage |
| **spark-master** | spark-py:3.3.0 | 4 CPU, 8GB | 8080, 7077 | Spark cluster master |
| **spark-worker-1** | spark-py:3.3.0 | 2 CPU, 6GB | 8081 | Spark worker node 1 |
| **spark-worker-2** | spark-py:3.3.0 | 2 CPU, 6GB | 8082 | Spark worker node 2 |

**Total Resources**: 12 CPU, 28GB RAM

### **Custom Dockerfile**
```dockerfile
FROM apache/spark-py:v3.3.0

# System dependencies
- python3-pip
- libjpeg-dev (Image processing)
- zlib1g-dev (Compression)
- libpng-dev (PNG images)

# Python packages
- tensorflow==2.11.0 (Deep Learning)
- Pillow==9.5.0 (Image operations)
- numpy==1.23.5 (Numerical computing)
- keras==2.11.0 (Neural network API)
- h5py==3.8.0 (HDF5 support)
```

---

## ⚙️ SPARK CONFIGURATION (spark-defaults.conf)

**Memory Management** (Tuned cho GPU-like performance):
```properties
spark.memory.fraction=0.6           # 60% for execution
spark.memory.offHeap.enabled=true   # Off-heap memory
spark.memory.offHeap.size=2g        # 2GB extra memory

spark.driver.memory=4g              # Driver node
spark.executor.memory=4g            # Worker nodes
spark.driver.maxResultSize=2g
```

**Serialization** (Performance):
```properties
spark.serializer=KryoSerializer     # Fast serialization
spark.kryoserializer.buffer.max=512m
```

**Shuffle Optimization**:
```properties
spark.shuffle.spill=true            # Disk spilling
spark.shuffle.file.buffer=64k
spark.reducer.maxSizeInFlight=96m
```

**Event Logging** (Monitoring):
```properties
spark.eventLog.enabled=true
spark.eventLog.dir=hdfs:///spark-logs
spark.history.fs.logDirectory=hdfs:///spark-logs
spark.history.ui.port=18080
```

---

## 📊 DATA PATHS (HDFS)

| Data Type | HDFS Path | Format | Samples |
|-----------|-----------|--------|---------|
| **Raw Images** | `/user/data/raw/{train,test}/{REAL,FAKE}/` | JPG/PNG | ~5000 train, ~1000 test |
| **TF Features** | `/user/data/features_tf/{train,test}/{REAL,FAKE}/batch_*` | Parquet | 50 train batches |
| **Model** | `/user/models/logistic_regression_tf` | MLlib format | ~1KB |
| **Metrics** | `/user/data/results_tf/metrics_tuned` | Parquet | Key-value metrics |
| **Predictions** | `/user/data/results_tf/test_predictions_tuned` | Parquet | Label + Prediction + Prob |
| **Spark Logs** | `/spark-logs/` | Event logs | Monitoring & debugging |

---

## 🚀 CÁCH CHẠY PIPELINE

### **1. Chạy đầy đủ (tất cả 5 steps)**
```powershell
cd scripts
.\run_pipeline.ps1
```

### **2. Chạy từng step riêng**
```powershell
# Skip upload & feature extraction, chỉ chạy training + evaluation
.\run_pipeline.ps1 -SkipUpload $true -SkipFeatureExtraction $true

# Chỉ training
.\run_pipeline.ps1 -SkipUpload $true -SkipFeatureExtraction $true -SkipLoad $true -SkipEvaluation $true

# Training + Load model
.\run_pipeline.ps1 -SkipUpload $true -SkipFeatureExtraction $true -SkipEvaluation $true
```

### **3. Chạy từng script riêng lẻ**
```bash
# Train
spark-submit --master local[2] --driver-memory 3g \
  src/4_ml_training/ml_training_tf_features.py

# Load & Predict
spark-submit --master local[2] --driver-memory 3g \
  src/4_ml_training/load_and_predict.py

# Evaluate
spark-submit --master local[2] --driver-memory 3g \
  src/5_evaluation/evaluate_tf_model.py
```

---

## 🎯 KEY FEATURES & INNOVATIONS

### **1. Memory-Optimized Feature Extraction**
- ✅ MobileNetV2 (lightweight, 1280-dim output)
- ✅ Batch processing dengan periodic garbage collection
- ✅ Disk spilling for large datasets
- ✅ TensorFlow session cleanup

### **2. Distributed ML Training**
- ✅ Spark MLlib LogisticRegression (scalable)
- ✅ Tuned hyperparameters (maxIter=300, regParam=0.001)
- ✅ Train/Val/Test split (80/20/separate)
- ✅ Model persistence to HDFS

### **3. Model Reusability**
- ✅ Save model to HDFS (persistent storage)
- ✅ Load model in separate script (no retraining)
- ✅ ModelManager utility class (DRY principle)
- ✅ Predictions + confidence scores

### **4. Comprehensive Monitoring**
- ✅ Memory tracking (process + system)
- ✅ Spark event logging to HDFS
- ✅ Detailed logging per step
- ✅ Metrics visualization & export

### **5. Automated Orchestration**
- ✅ PowerShell pipeline script (5 steps)
- ✅ Skip parameters for flexibility
- ✅ Error handling & reporting
- ✅ Execution time tracking

---

## 📈 PERFORMANCE METRICS

**Expected Results** (Based on tuning):
- **Validation Accuracy**: 91-93%
- **Test Accuracy**: 90-92%
- **F1-Score**: 90-91%
- **Training Time**: 5-10 minutes (50 batches)
- **Feature Extraction**: 15-20 minutes (5000 images)
- **Model Size**: ~1KB (very compact)

**Memory Usage**:
- Driver: 3-4GB
- Each Executor: 3-4GB
- Total: ~12GB for 3-node setup

---

## 🔐 BEST PRACTICES IMPLEMENTED

✅ **Configuration Management**
- Centralized config (`hdfs_config.py`)
- No hardcoded paths

✅ **Logging & Debugging**
- Structured logging (`logging_utils.py`)
- Memory monitoring
- Event log persistence

✅ **Code Organization**
- Modular structure (1_ingestion → 5_evaluation)
- Clear separation of concerns
- Reusable utilities

✅ **Data Pipeline**
- Immutable data in HDFS
- Versioned results
- Parquet format (columnar, compressed)

✅ **Error Handling**
- Try-catch blocks
- Graceful failures
- Detailed error messages

✅ **Scalability**
- Spark distributed processing
- HDFS for large datasets
- Batch processing for memory efficiency

---

## 📚 PROJECT DEPENDENCIES

```
Python 3.8+
├── spark==3.3.0          (PySpark)
├── tensorflow==2.11.0
│   ├── keras==2.11.0
│   └── h5py==3.8.0
├── pillow==9.5.0
├── numpy==1.23.5
└── logging (standard library)
```

---

## 🎓 LEARNING OUTCOMES

Dự án này demonstrasi:

1. **Big Data Processing**: Distributed HDFS storage, Spark processing
2. **Deep Learning**: TensorFlow feature extraction, pre-trained models
3. **ML Engineering**: Model training, evaluation, persistence
4. **Data Pipeline**: ETL workflow, modular design
5. **DevOps**: Docker containerization, memory optimization
6. **Software Engineering**: Code organization, logging, error handling

---

## 📝 TÓMO TẮT ĐIỂM QUAN TRỌNG

| Điểm | Mô Tả | Tác Động |
|------|-------|---------|
| **End-to-End Pipeline** | Từ raw images → predictions | Full MLOps workflow |
| **Memory Optimization** | Garbage collection, disk spilling | Xử lý large datasets |
| **Model Reusability** | Save/Load mechanism | Production-ready |
| **Distributed Processing** | Spark + HDFS | Scalable to GB/TB data |
| **Comprehensive Metrics** | Accuracy, Precision, Recall, F1 | Full evaluation |
| **Automated Orchestration** | PowerShell pipeline script | Easy execution |
| **Modular Code** | Clear separation of 5 steps | Maintainability |
| **Hyperparameter Tuning** | Optimized LR params | Better accuracy |

---

**Ngày phân tích**: December 19, 2025  
**Phiên bản**: v1.0
