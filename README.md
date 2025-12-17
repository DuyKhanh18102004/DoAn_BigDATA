# Deepfake Detection System - Big Data Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Spark-3.3.0-orange.svg)](https://spark.apache.org/)
[![HDFS](https://img.shields.io/badge/HDFS-3.3.0-green.svg)](https://hadoop.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

Hệ thống phát hiện Deepfake sử dụng **Distributed Big Data Processing** với Apache Spark, HDFS và Deep Learning.

## 📋 Table of Contents

- [Tổng Quan](#tổng-quan)
- [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Cấu Trúc Project](#cấu-trúc-project)
- [Sử Dụng](#sử-dụng)
- [Pipeline Flow](#pipeline-flow)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Documentation](#documentation)

## 🎯 Tổng Quan

### Mục Tiêu
Xây dựng hệ thống **phân tán** để:
- ✅ Xử lý **120,000 images** trên HDFS
- ✅ Extract features sử dụng **ResNet50** (distributed inference)
- ✅ Train ML models với **Spark MLlib**
- ✅ Đạt accuracy > 85% trong việc phát hiện deepfake

### Công Nghệ Sử Dụng
- **Storage**: Hadoop HDFS (distributed file system)
- **Processing**: Apache Spark (distributed computing)
- **ML**: PyTorch (ResNet50) + Spark MLlib (RandomForest, LogisticRegression)
- **Orchestration**: Docker Compose
- **Monitoring**: Spark History Server

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEEPFAKE DETECTION SYSTEM                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Raw Data   │─────▶│     HDFS     │─────▶│    Spark     │
│  (Dataset)   │      │   Storage    │      │  Processing  │
└──────────────┘      └──────────────┘      └──────────────┘
                             │                      │
                             │                      ▼
                             │              ┌──────────────┐
                             │              │   Feature    │
                             │              │ Extraction   │
                             │              │  (ResNet50)  │
                             │              └──────────────┘
                             │                      │
                             │                      ▼
                             │              ┌──────────────┐
                             │              │  ML Training │
                             │              │   (Spark ML) │
                             │              └──────────────┘
                             │                      │
                             ▼                      ▼
                      ┌──────────────┐      ┌──────────────┐
                      │   Results    │◀─────│  Evaluation  │
                      │  (Parquet)   │      │  & Metrics   │
                      └──────────────┘      └──────────────┘
```

## 💻 Yêu Cầu Hệ Thống

### Hardware
- **CPU**: 4+ cores
- **RAM**: 16GB+ (khuyến nghị 32GB)
- **Disk**: 50GB+ free space

### Software
- **Docker Desktop** (Windows/Mac) hoặc **Docker + Docker Compose** (Linux)
- **Python** 3.8+
- **Git**

## 🚀 Cài Đặt

### 1. Clone Repository
```bash
git clone https://github.com/your-username/DoAn_BigDATA.git
cd DoAn_BigDATA
```

### 2. Start Docker Services
```bash
docker-compose up -d
```

Kiểm tra containers:
```bash
docker-compose ps
```

Expected output:
```
NAME                STATUS
namenode            Up
datanode-1          Up
datanode-2          Up
spark-master        Up
spark-worker-1      Up
spark-worker-2      Up
spark-history       Up
```

### 3. Setup HDFS Directories
```bash
bash scripts/setup_hdfs.sh
```

### 4. Verify Services

**HDFS NameNode UI**: http://localhost:9870  
**Spark Master UI**: http://localhost:8080  
**Spark History Server**: http://localhost:18080

## 📁 Cấu Trúc Project

```
DoAn_TH_BIGDATA/
├── src/                              # Source code modules
│   ├── config/                       # Configuration
│   │   ├── hdfs_config.py
│   │   ├── spark_config.py
│   │   └── model_config.py
│   ├── utils/                        # Utilities
│   │   ├── hdfs_utils.py
│   │   ├── spark_utils.py
│   │   ├── image_utils.py
│   │   └── logging_utils.py
│   ├── 1_ingestion/                  # Data upload to HDFS
│   │   ├── upload_to_hdfs.py
│   │   └── verify_upload.py
│   ├── 2_preprocessing/              # Data validation
│   │   ├── load_data.py
│   │   ├── validate_images.py
│   │   └── prepare_dataframe.py
│   ├── 3_feature_extraction/         # ResNet50 features
│   │   ├── model_loader.py
│   │   ├── feature_extractor.py
│   │   └── extract_pipeline.py
│   ├── 4_ml_training/                # Spark ML training
│   │   ├── prepare_vectors.py
│   │   ├── train_classifier.py
│   │   └── save_model.py
│   ├── 5_evaluation/                 # Model evaluation
│   │   ├── evaluate_model.py
│   │   ├── confusion_matrix.py
│   │   └── generate_report.py
│   └── 6_inference/                  # Production inference
│       └── batch_inference.py
├── scripts/                          # Automation scripts
│   ├── setup_hdfs.sh
│   ├── run_full_pipeline.sh
│   ├── run_test_100_images.sh
│   └── check_spark_history.sh
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
├── tests/                            # Unit tests
│   ├── test_config.py
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   └── test_ml_training.py
├── data/                             # Dataset (local)
├── models/                           # Saved models
├── logs/                             # Application logs
├── results/                          # Output results
├── docker-compose.yml
└── README.md
```

## 🎮 Sử Dụng

### Quick Test (100 Images)

**1. Prepare test dataset:**
```bash
# Copy 100 images (50 REAL + 50 FAKE)
mkdir -p Dataset_Test/train/REAL
mkdir -p Dataset_Test/train/FAKE
# Copy files...
```

**2. Run test pipeline:**
```bash
bash scripts/run_test_100_images.sh
```

Expected time: ~30 minutes

### Full Pipeline (120K Images)

**1. Upload data to HDFS:**
```bash
python src/1_ingestion/upload_to_hdfs.py \
    --local_path data/train \
    --hdfs_path /user/data/raw/train
```

**2. Extract features:**
```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --driver-memory 4g \
    --executor-memory 4g \
    src/3_feature_extraction/extract_pipeline.py \
    --input_path /user/data/raw/train \
    --output_path /user/data/features/train
```

Expected time: 6-7 hours

**3. Train ML models:**
```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    src/4_ml_training/train_classifier.py
```

**4. Evaluate:**
```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    src/5_evaluation/evaluate_model.py
```

## 🔄 Pipeline Flow

```
1. INGESTION
   └─> Upload local images to HDFS
       Input:  local Dataset/train/*.jpg
       Output: HDFS /user/data/raw/train/

2. PREPROCESSING
   └─> Load with Spark binaryFile, validate, label
       Input:  HDFS /user/data/raw/train/
       Output: DataFrame[path, content, label]

3. FEATURE EXTRACTION (Distributed)
   └─> ResNet50 inference on Spark Workers
       Input:  DataFrame[content]
       Output: HDFS /user/data/features/ (Parquet)

4. ML TRAINING
   └─> Train RandomForest + LogisticRegression
       Input:  HDFS /user/data/features/
       Output: HDFS /user/models/

5. EVALUATION
   └─> Calculate metrics, generate report
       Input:  Model + test data
       Output: HDFS /user/data/results/
```

## 📊 Monitoring

### Spark History Server
```bash
# Access at http://localhost:18080

# Or use script:
bash scripts/check_spark_history.sh
```

**Screenshots cần capture:**
- Job timeline
- Stage details (parallel tasks)
- Executor metrics
- DAG visualization

### HDFS NameNode UI
```bash
# Access at http://localhost:9870

# Check files:
docker exec namenode hdfs dfs -ls -R /user/data
```

## 🧪 Testing

### Run Unit Tests
```bash
# All tests
python -m pytest tests/

# Specific module
python -m pytest tests/test_config.py

# With coverage
python -m pytest tests/ --cov=src
```

### Test Configuration
```bash
python tests/test_config.py
```

## 📚 Documentation

Xem chi tiết tại:
- **[ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md)** - Kiến trúc chi tiết
- **[API Documentation](docs/API.md)** - API reference
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Xử lý lỗi

## 🎯 Expected Results

### Test (100 images)
- **Accuracy**: ~70-80%
- **Processing time**: ~30 minutes
- **Feature dimension**: 2048 (ResNet50)

### Full (120K images)
- **Accuracy**: > 85%
- **Processing time**: ~6-7 hours
- **Models**: RandomForest + LogisticRegression

## 🐛 Troubleshooting

### Common Issues

**1. Docker containers not starting:**
```bash
docker-compose down
docker-compose up -d
docker-compose logs -f
```

**2. HDFS connection timeout:**
```bash
# Check namenode
docker exec namenode hdfs dfsadmin -report

# Restart HDFS
docker restart namenode datanode-1 datanode-2
```

**3. Spark job fails:**
```bash
# Check Spark logs
docker logs spark-master
docker logs spark-worker-1

# Check Spark History
open http://localhost:18080
```

## 👥 Contributors

- **Team**: BigData Team
- **Project**: Deepfake Detection System
- **Course**: Big Data Processing

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Apache Spark** - Distributed computing
- **Hadoop HDFS** - Distributed storage
- **PyTorch** - Deep learning framework
- **Docker** - Containerization

---

**Last Updated**: December 16, 2025
