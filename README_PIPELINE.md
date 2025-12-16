# Deepfake Detection Pipeline - Big Data Project

## 📌 Tổng quan

Pipeline Big Data phân tán để phát hiện ảnh Deepfake sử dụng:
- **Storage**: HDFS (Hadoop Distributed File System)
- **Processing**: Apache Spark (Distributed Computing)
- **ML**: Spark MLlib + Transfer Learning (ResNet50)

## 🏗️ Kiến trúc Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Data Ingestion (✅ COMPLETED)                      │
│  - 120,000 ảnh JPG uploaded lên HDFS                        │
│  - Cấu trúc: /user/data/raw/{train,test}/{REAL,FAKE}/      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Feature Extraction (DISTRIBUTED)                   │
│  - Load ảnh từ HDFS bằng binaryFiles()                      │
│  - Extract features bằng ResNet50 (UDF trên Workers)        │
│  - Output: 2048-dim vectors                                  │
│  - Lưu Parquet vào /user/data/features/                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: ML Training (Spark MLlib)                          │
│  - Logistic Regression                                       │
│  - Random Forest (100 trees)                                 │
│  - Lưu models + predictions vào HDFS                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Evaluation & Business Insight                      │
│  - Metrics: Accuracy, Precision, Recall, F1, AUC-ROC        │
│  - Confusion Matrix                                          │
│  - Lưu reports vào HDFS                                      │
└─────────────────────────────────────────────────────────────┘
```

## 📂 Cấu trúc Dữ liệu trên HDFS

```
hdfs://namenode:8020/
├── user/data/
│   ├── raw/                          # ✅ COMPLETED - 120,000 images
│   │   ├── train/
│   │   │   ├── REAL/  (50,000 .jpg)
│   │   │   └── FAKE/  (50,000 .jpg)
│   │   └── test/
│   │       ├── REAL/  (10,000 .jpg)
│   │       └── FAKE/  (10,000 .jpg)
│   │
│   ├── features/                     # Output của Step 2
│   │   ├── train/ (Parquet)
│   │   └── test/  (Parquet)
│   │
│   ├── models/                       # Trained models
│   │   ├── logistic_regression/
│   │   └── random_forest/
│   │
│   └── results/                      # Predictions & Metrics
│       ├── lr_predictions/
│       ├── rf_predictions/
│       └── metrics_summary/
│
└── spark-logs/                       # Event logs cho History Server
```

## 🚀 Cách chạy Pipeline

### Bước 1: Copy code vào Spark Master container

```bash
docker cp feature_extraction.py spark-master:/app/
docker cp ml_training.py spark-master:/app/
```

### Bước 2: Chạy Feature Extraction

```bash
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  --executor-memory 2g \
  --total-executor-cores 4 \
  /app/feature_extraction.py
```

**Thời gian dự kiến**: ~20-30 phút với 120,000 ảnh

### Bước 3: Chạy ML Training

```bash
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  --executor-memory 2g \
  --total-executor-cores 4 \
  /app/ml_training.py
```

**Thời gian dự kiến**: ~10-15 phút

### HOẶC: Chạy toàn bộ pipeline tự động

```bash
python run_pipeline.py
```

## 📊 Xem kết quả

### 1. Spark History Server
```
URL: http://localhost:18080
```

**Cần chụp màn hình**:
- Job Overview (số lượng tasks, stages)
- Task Distribution (chứng minh phân tán)
- Timeline (parallelism)

### 2. Xem metrics từ HDFS

```bash
# Xem metrics summary
docker exec namenode hdfs dfs -cat /user/data/results/metrics_summary/*.parquet

# Hoặc dùng Spark shell
docker exec spark-master spark-shell

scala> val metrics = spark.read.parquet("hdfs://namenode:8020/user/data/results/metrics_summary")
scala> metrics.show()
```

### 3. Xem predictions

```bash
docker exec spark-master pyspark

>>> df = spark.read.parquet("hdfs://namenode:8020/user/data/results/lr_predictions")
>>> df.show(20)
>>> df.groupBy("label", "prediction").count().show()
```

## 🎯 Tuân thủ yêu cầu đồ án

### ✅ Checklist

- [x] **Dữ liệu lên HDFS trước**: 120,000 ảnh đã upload vào HDFS
- [x] **Không dùng vòng lặp local**: Dùng `binaryFiles()` và Spark transformations
- [x] **AI phân tán**: ResNet50 chạy trong UDF trên Spark Workers
- [x] **Lưu Parquet**: Features và predictions đều ở định dạng Parquet
- [x] **Spark History Server**: Event logs ghi vào `/spark-logs` trên HDFS
- [x] **Spark MLlib**: Dùng LogisticRegression và RandomForestClassifier

## 📈 Expected Metrics

### Model Performance (dự kiến)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~85-90% | ~85-90% | ~85-90% | ~85-90% |
| Random Forest | ~90-95% | ~90-95% | ~90-95% | ~90-95% |

**Lưu ý**: Metrics thực tế phụ thuộc vào:
- Chất lượng features từ ResNet50
- Hyperparameters của models
- Distribution của train/test data

## 🔧 Troubleshooting

### Issue 1: Out of Memory

**Giải pháp**:
```bash
# Tăng executor memory
--executor-memory 4g
--driver-memory 4g
```

### Issue 2: Feature extraction quá chậm

**Giải pháp**:
```python
# Tăng số partitions
df.repartition(200)

# Hoặc sample một phần data để test
train_sample = train_df.sample(fraction=0.1)
```

### Issue 3: Model không converge

**Giải pháp**:
```python
# Tăng iterations
lr = LogisticRegression(maxIter=200)

# Hoặc điều chỉnh learning rate
lr = LogisticRegression(regParam=0.001)
```

## 📝 Business Insight Report Template

### Câu hỏi 1: Model có trích xuất đủ thông tin không?

**Trả lời**:
- ResNet50 features (2048 dims) capture được:
  - Low-level: edges, textures
  - Mid-level: patterns, shapes  
  - High-level: object parts
  
- Accuracy >85% chứng tỏ features đủ discriminative
- Nếu <80%: cần thử models khác (EfficientNet, ViT)

### Câu hỏi 2: So sánh Logistic Regression vs Random Forest?

**Expected findings**:
- LR: Nhanh hơn, đơn giản hơn, interpretable
- RF: Accuracy cao hơn, handle non-linearity tốt hơn
- Trade-off: Speed vs Performance

### Câu hỏi 3: Scalability?

**Evidence**:
- Spark History UI: số tasks chạy song song
- Processing time: tỷ lệ với số executors
- HDFS: distributed storage → handle TB-scale data

## 📸 Screenshots cần có

1. ✅ **Docker containers running** (docker ps)
2. ✅ **HDFS WebUI** (http://localhost:9870) - showing 120,000 files
3. **Spark Master UI** (http://localhost:8080) - showing workers
4. **Spark History Server** (http://localhost:18080):
   - Application list
   - Job stages
   - Task timeline
   - Executor stats
5. **Terminal output**: Metrics summary

## 🎓 Học điểm chính

### Big Data Concepts
- **Distributed Storage**: HDFS replicas, fault tolerance
- **Distributed Computing**: Spark DAG, lazy evaluation
- **Partitioning**: Data parallelism
- **Serialization**: Parquet columnar format

### ML Engineering
- **Transfer Learning**: Pretrained models
- **Feature Engineering**: Dimensionality reduction
- **Model Selection**: Classical ML on deep features
- **Evaluation**: Comprehensive metrics

## 📧 Support

Nếu gặp vấn đề:
1. Kiểm tra logs: `docker logs spark-master`
2. Kiểm tra HDFS: `docker exec namenode hdfs dfsadmin -report`
3. Kiểm tra Spark UI: http://localhost:8080

Good luck! 🚀
