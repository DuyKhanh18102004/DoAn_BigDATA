# HƯỚNG DẪN TĂNG DOCKER DESKTOP RESOURCES

## 🎯 Mục Tiêu

Tăng RAM/CPU cho Docker Desktop để xử lý được batch lớn hơn (50K ảnh)

---

## 📋 BƯỚC 1: Kiểm tra RAM hiện tại của máy

### Windows:

```cmd
# Mở Task Manager (Ctrl + Shift + Esc)
# Xem tab "Performance" → Memory

# Hoặc dùng command:
systeminfo | findstr /C:"Total Physical Memory"
```

### Cần có:

- **Minimum:** 16GB RAM tổng (cấp cho Docker 8GB)
- **Recommended:** 32GB RAM tổng (cấp cho Docker 12GB)

---

## 🔧 BƯỚC 2: Mở Docker Desktop Settings

### Cách 1: Từ System Tray

1. Click chuột phải vào **Docker Desktop icon** (góc dưới bên phải)
2. Chọn **"Settings"**

### Cách 2: Từ Docker Desktop App

1. Mở **Docker Desktop**
2. Click icon **⚙️ Settings** (góc trên bên phải)

---

## ⚙️ BƯỚC 3: Điều chỉnh Resources

### Navigate to Resources:

```
Settings → Resources → Advanced
```

### Recommended Settings cho 120K images:

| Setting             | Current (Likely) | Recommended | Max Safe      |
| ------------------- | ---------------- | ----------- | ------------- |
| **CPUs**            | 2                | 4-6         | Tất cả trừ 2  |
| **Memory**          | 2-4 GB           | 8-10 GB     | 75% total RAM |
| **Swap**            | 1 GB             | 2 GB        | 4 GB          |
| **Disk image size** | 60 GB            | 80 GB       | 100 GB        |

### Ví dụ cụ thể:

**Nếu máy có 16GB RAM:**

```
CPUs: 4 cores
Memory: 8 GB (50% của 16GB)
Swap: 2 GB
Disk: 80 GB
```

**Nếu máy có 32GB RAM:**

```
CPUs: 6 cores
Memory: 12 GB (37.5% của 32GB)
Swap: 4 GB
Disk: 100 GB
```

**Nếu máy chỉ có 8GB RAM:**

```
⚠️ KHÔNG NÊN chạy Docker với dataset lớn
→ Dùng Cloud hoặc Google Colab
```

---

## 💾 BƯỚC 4: Apply & Restart

1. Click **"Apply & Restart"** button
2. Đợi Docker Desktop restart (2-3 phút)
3. Verify settings đã áp dụng:

```cmd
docker info | findstr Memory
docker info | findstr CPUs
```

Expected output:

```
Total Memory: 8.589 GiB   (hoặc số bạn đã set)
CPUs: 4                   (hoặc số bạn đã set)
```

---

## ✅ BƯỚC 5: Test với Batch Lớn Hơn

Sau khi tăng resources, test lại:

### Test 1: 10K batch (Safe)

```cmd
scripts\run_super_safe_batch.bat
```

### Test 2: Nếu 10K OK, thử 25K

Tạo file test custom:

```python
# Modify feature_extraction_train_real.py
# Change: .limit(50000) → .limit(25000)
```

### Test 3: Nếu 25K OK, thử 50K full batch

```cmd
scripts\run_batch_extraction.bat
```

---

## 📊 BENCHMARK: Batch Size vs RAM

| RAM Available | Max Batch Size | Risk      | Time/Batch |
| ------------- | -------------- | --------- | ---------- |
| 4 GB          | 5K             | High      | 15m        |
| 6 GB          | 10K            | Medium    | 25m        |
| 8 GB          | 25K            | Low       | 45m        |
| 10 GB         | 50K            | Very Low  | 1.5h       |
| 12 GB         | 100K           | Safe      | 3h         |
| 16 GB         | 120K (full)    | Very Safe | 6-7h       |

---

## 🔍 BƯỚC 6: Monitor Resource Usage

### Trong khi chạy batch:

**Terminal 1: Run batch**

```cmd
scripts\run_batch_extraction.bat
```

**Terminal 2: Monitor Docker**

```cmd
docker stats --no-stream

# Xem memory usage realtime
docker stats
```

**Task Manager:**

- Xem CPU usage của Docker Desktop
- Xem Memory usage của Docker Desktop
- Đảm bảo không quá 90% (để lại buffer)

---

## ⚠️ WARNING SIGNS

Nếu thấy các dấu hiệu này → DỪNG NGAY:

1. **Memory usage > 95%**

```
Docker Desktop using 11.5GB / 12GB
```

→ Giảm batch size hoặc tăng Docker RAM

2. **Swap usage cao (> 50%)**

```
Swap: 1.8GB / 2GB
```

→ Thiếu RAM thật, cần giảm batch

3. **CPU sustained at 100%**

```
CPU: 100% for 10+ minutes
```

→ Có thể OK, nhưng check temperature

4. **Docker Desktop "Not Responding"**
   → Force quit, restart, giảm batch size

---

## 🆘 TROUBLESHOOTING

### Lỗi: "Not enough memory"

```
Solution:
1. Đóng tất cả app đang dùng RAM (Chrome, VS Code, etc.)
2. Tăng Docker memory lên max có thể
3. Giảm batch size xuống
```

### Lỗi: "Docker Desktop crashed"

```
Solution:
1. Restart Docker Desktop
2. Check Event Viewer (Windows):
   eventvwr.msc → Application logs → Docker
3. Nếu thấy OutOfMemory → Tăng Docker RAM
4. Nếu vẫn crash → Dùng batch nhỏ hơn
```

### Lỗi: "No space left on device"

```
Solution:
1. Clean Docker images/containers:
   docker system prune -a --volumes

2. Tăng "Disk image size" trong Docker Settings

3. Check HDFS space:
   docker exec -it namenode hdfs dfs -df -h
```

---

## 📝 CHECKLIST SAU KHI TĂNG RESOURCES

- [ ] Docker Desktop RAM ≥ 8GB
- [ ] Docker Desktop CPUs ≥ 4 cores
- [ ] Swap ≥ 2GB
- [ ] Disk ≥ 80GB
- [ ] Đã restart Docker Desktop
- [ ] Verify bằng `docker info`
- [ ] Test với 10K batch trước
- [ ] Monitor resources trong khi chạy
- [ ] Chuẩn bị plan B nếu vẫn crash (batch nhỏ hơn)

---

## 🎯 NEXT STEPS

Sau khi tăng resources:

1. **Test incremental:**

```
5K → 10K → 25K → 50K → 100K → 120K
```

2. **Tìm sweet spot:**

- Batch size lớn nhất mà không crash
- Balance giữa speed và stability

3. **Run production:**

```cmd
# Nếu 50K stable:
scripts\run_batch_extraction.bat

# Nếu chỉ 10K stable:
scripts\run_super_safe_batch.bat

# Nếu chỉ 5K stable:
scripts\test_ultra_mini_batch.bat
```

---

## 💡 PRO TIPS

1. **Close unnecessary apps** trước khi chạy:

   - Chrome (RAM hog)
   - VS Code (nếu không cần)
   - Other IDEs

2. **Run overnight** để tránh dùng máy:

   - Less interference
   - Can use max resources

3. **Setup monitoring:**

   - Spark History Server: http://localhost:18080
   - Docker stats: Terminal window
   - Task Manager: Background

4. **Have backup plan:**
   - Script cho batch nhỏ sẵn
   - Cloud option (Colab/AWS/GCP)
   - Sample dataset (1K images) để test nhanh
