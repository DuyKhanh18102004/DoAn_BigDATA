# CRASH TROUBLESHOOTING - COMPLETE GUIDE

## 🔥 Nếu Batch 50K Vẫn Crash

### ✅ GIẢI PHÁP: Super Safe Batch (10K/batch)

Đã tạo sẵn script:
```cmd
scripts\run_super_safe_batch.bat
```

**Quy trình:**
1. Test batch đầu tiên (10K ảnh)
2. Nếu OK → Tiếp tục các batch còn lại
3. Nếu FAIL → Chuyển sang Ultra Mini Batch (5K/batch)

---

## 🆘 Nếu 10K Vẫn Crash

### Chạy test với 5K:
```cmd
scripts\test_ultra_mini_batch.bat
```

### Nếu 5K OK:
- Chia 120K thành **24 batches** (mỗi batch 5K)
- Tổng thời gian: ~8-10 giờ
- Ổn định 100%

### Nếu 5K vẫn FAIL:
**Docker Desktop Settings không đủ!**

---

## ⚙️ KIỂM TRA & FIX DOCKER SETTINGS

### Bước 1: Mở Docker Desktop Settings

```
Docker Desktop → Settings → Resources → Advanced
```

### Bước 2: Minimum Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **Memory** | 6 GB | 8-12 GB |
| **CPUs** | 2 cores | 4 cores |
| **Swap** | 1 GB | 2 GB |
| **Disk** | 40 GB | 60 GB |

### Bước 3: Apply Changes

1. Click "Apply & Restart"
2. Đợi Docker restart (2-3 phút)
3. Verify:
```cmd
docker info | findstr Memory
docker info | findstr CPUs
```

---

## 🔍 DEBUG COMMANDS

### Kiểm tra Docker resources đang dùng:
```cmd
# Windows
docker stats --no-stream

# Check Spark logs
docker logs spark-master --tail 100
docker logs spark-worker1 --tail 100
```

### Kiểm tra memory của containers:
```cmd
docker inspect spark-master | findstr Memory
docker inspect spark-worker1 | findstr Memory
```

### Xem Spark job đang chạy:
```cmd
# Spark Master UI
http://localhost:8080

# Spark History Server
http://localhost:18080
```

---

## 📊 SO SÁNH BATCH SIZES

| Batch Size | Batches Needed | Time/Batch | Total Time | Crash Risk |
|------------|----------------|------------|------------|------------|
| 120K (full) | 1 | 6-7h | 6-7h | ⚠️⚠️⚠️⚠️⚠️ |
| 50K | 4 | 1.5-2h | 6-8h | ⚠️⚠️⚠️ |
| 10K | 12 | 20-30m | 4-6h | ⚠️ |
| 5K | 24 | 10-15m | 4-6h | ✅ Safe |
| 1K | 120 | 2-3m | 4-6h | ✅ Super Safe |

---

## 🎯 DECISION TREE

```
Chạy 50K batch
    ↓
    Crash? 
    ↓
YES → Chạy 10K batch (scripts\run_super_safe_batch.bat)
    ↓
    Crash?
    ↓
YES → Check Docker Settings (cần 6GB+ RAM)
    ↓
    Fixed?
    ↓
YES → Chạy lại 10K batch
    ↓
NO → Chạy 5K batch (test_ultra_mini_batch.bat)
    ↓
    Crash?
    ↓
YES → Máy không đủ cấu hình
    → Giải pháp: Cloud hoặc máy khác
```

---

## 🚨 EMERGENCY OPTIONS

### Option 1: Chạy trên Local (không dùng Docker)
```bash
# Cài PySpark local
pip install pyspark tensorflow

# Chạy trực tiếp
python src/3_feature_extraction/feature_extraction_quick.py
```

### Option 2: Chạy trên Google Colab
- Upload code lên Colab
- Free GPU available (T4/P100)
- No Docker needed
- RAM: 12-25GB

### Option 3: Chạy trên Cloud
```
AWS EMR:
- m5.xlarge: 4 vCPU, 16GB RAM
- Cost: ~$0.20/hour
- Total: ~$1.5 for 6-7 hours

GCP Dataproc:
- n1-standard-4: 4 vCPU, 15GB RAM
- Cost: ~$0.19/hour
- Total: ~$1.4 for 6-7 hours
```

---

## 💡 BEST PRACTICE

### Chiến lược an toàn nhất:

1. **Test nhỏ trước:**
```cmd
scripts\test_ultra_mini_batch.bat
```

2. **Nếu test OK, scale dần:**
- 5K → 10K → 20K → 50K
- Tìm được batch size tối ưu cho máy của bạn

3. **Monitor liên tục:**
```cmd
# Terminal 1: Chạy batch
scripts\run_super_safe_batch.bat

# Terminal 2: Monitor
docker stats

# Browser: Spark History
http://localhost:18080
```

4. **Checkpoint sau mỗi batch:**
- Features đã lưu HDFS
- Nếu crash, chỉ mất batch hiện tại
- Restart từ batch tiếp theo

---

## 📝 CHECKLIST TRƯỚC KHI CHẠY

- [ ] Docker Desktop đã cấp đủ RAM (6GB+)
- [ ] Không có app khác đang dùng nhiều RAM
- [ ] Docker containers đang chạy healthy
- [ ] HDFS đã có đủ không gian (check: `hdfs dfs -df -h`)
- [ ] Spark History Server accessible (http://localhost:18080)
- [ ] Đã restart Docker Desktop gần đây
- [ ] Đã xóa features cũ/corrupt
- [ ] Đã test với batch nhỏ trước

---

## 🎓 SUPPORT

Nếu vẫn gặp vấn đề:

1. Check logs:
```cmd
docker logs spark-master > spark-master.log
docker logs namenode > namenode.log
```

2. Share error messages
3. Check Docker Desktop version (cần >= 4.0)
4. Check Windows version (nên dùng Windows 10/11 Pro)

