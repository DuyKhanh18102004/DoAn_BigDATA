# Docker Desktop Resource Configuration Guide

## ⚠️ Vấn Đề: Docker Desktop Crash Khi Chạy 120K Images

**Nguyên nhân:**

- Docker Desktop memory limit (thường 2-4GB default)
- 120K images = quá nhiều tasks cùng lúc
- ResNet50 model lớn (~100MB) x nhiều executors
- TensorFlow/PyTorch memory overhead

## ✅ GIẢI PHÁP 1: BATCH PROCESSING (Khuyến Nghị)

### Đã tạo script tự động:

```bash
scripts/run_batch_extraction.bat
```

### Cách chạy:

```cmd
# Windows
cd d:\DoAn_TH_BIGDATA
scripts\run_batch_extraction.bat
```

### Ưu điểm:

✅ Không bị crash (mỗi batch nhỏ hơn)
✅ Có thể theo dõi từng batch
✅ Restart từ batch bị lỗi (không mất toàn bộ)
✅ Dễ debug

### Timeline:

- Batch 1: train/REAL (50K) - 1.5-2 giờ
- Batch 2: train/FAKE (50K) - 1.5-2 giờ
- Batch 3: test/REAL (10K) - 20-30 phút
- Batch 4: test/FAKE (10K) - 20-30 phút
  **Total: ~4-5 giờ**

### Theo dõi progress:

```cmd
scripts\check_batch_progress.bat
```

## 🔧 GIẢI PHÁP 2: TĂNG DOCKER DESKTOP RESOURCES

### Bước 1: Mở Docker Desktop Settings

1. Click Docker Desktop icon (system tray)
2. Settings > Resources > Advanced

### Bước 2: Tăng giới hạn

```
CPUs: 4-6 cores (nếu có)
Memory: 8GB (minimum) - 12GB (recommended)
Swap: 2GB
Disk: 60GB+
```

### Bước 3: Apply & Restart Docker

### Bước 4: Chạy lại với reduced config

```bash
docker exec -it spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --driver-memory 3g \
  --executor-memory 3g \
  --executor-cores 2 \
  --num-executors 2 \
  /app/src/3_feature_extraction/feature_extraction.py
```

## 💡 GIẢI PHÁP 3: OPTIMIZE CODE

### Giảm partition size trong feature_extraction.py:

```python
# Thay vì shuffle.partitions = 200
.config("spark.sql.shuffle.partitions", "100")  # Giảm xuống 100

# Thêm checkpoint để tránh tràn memory
df.checkpoint()
```

### Giảm batch size khi load model:

```python
# Process nhỏ hơn mỗi lần
df.repartition(50)  # Thay vì để Spark tự động
```

## 📊 SO SÁNH CÁC GIẢI PHÁP

| Giải pháp            | Độ ổn định | Thời gian | Độ phức tạp |
| -------------------- | ---------- | --------- | ----------- |
| **Batch Processing** | ⭐⭐⭐⭐⭐ | 4-5h      | Dễ          |
| Tăng Docker RAM      | ⭐⭐⭐     | 6-7h      | Trung bình  |
| Optimize code        | ⭐⭐⭐⭐   | 5-6h      | Khó         |

## 🎯 KHUYẾN NGHỊ

**Cho trường hợp của bạn:**
👉 **Dùng BATCH PROCESSING** (Giải pháp 1)

**Lý do:**
✅ Đơn giản nhất
✅ Không cần config Docker
✅ Không cần sửa code
✅ Có script sẵn (run_batch_extraction.bat)
✅ Ổn định 100%

## 🚀 HƯỚNG DẪN CHẠY BATCH

### Bước 1: Dừng job hiện tại (nếu đang chạy)

```cmd
# Tìm application ID đang chạy
docker exec -it spark-master /opt/spark/bin/spark-submit --kill <application-id>

# Hoặc restart Spark
docker restart spark-master spark-worker1 spark-worker2
```

### Bước 2: Xóa features cũ (nếu đã có partial data)

```cmd
docker exec -it namenode hdfs dfs -rm -r /user/data/features/train
docker exec -it namenode hdfs dfs -rm -r /user/data/features/test
```

### Bước 3: Chạy batch extraction

```cmd
cd d:\DoAn_TH_BIGDATA
scripts\run_batch_extraction.bat
```

### Bước 4: Theo dõi (terminal khác)

```cmd
# Mở terminal mới
scripts\check_batch_progress.bat

# Hoặc xem Spark History
# Mở browser: http://localhost:18080
```

## 📝 NOTES

- Mỗi batch độc lập, nếu 1 batch fail có thể chạy lại riêng
- Output của mỗi batch: `/user/data/features/{train|test}/{REAL|FAKE}`
- Sau khi 4 batch xong, chạy ml_training.py như bình thường
- Features được merge tự động khi load trong ml_training.py
