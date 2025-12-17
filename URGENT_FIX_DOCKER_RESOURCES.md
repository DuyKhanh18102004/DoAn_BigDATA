# 🚨 CRITICAL: Docker Desktop Resources Too Low!

## ❌ Vấn Đề Phát Hiện

```
Container CPU usage: 11.00% / 000% (0 CPUs available)
Container memory usage: 3.08GB / 0B
```

**Nghĩa là:**
- Docker Desktop chưa được cấp CPU cores
- Docker Desktop chưa được cấp RAM
- Đang chạy với resources mặc định (quá thấp)

---

## ✅ GIẢI PHÁP: TĂNG DOCKER DESKTOP RESOURCES (5 PHÚT)

### BƯỚC 1: Mở Docker Desktop Settings

1. **Tìm Docker Desktop icon** (góc dưới bên phải màn hình - system tray)
2. **Click chuột phải** vào icon
3. Chọn **"Settings"** (hoặc "Preferences" trên Mac)

### BƯỚC 2: Navigate to Resources

```
Settings → Resources → Advanced
```

### BƯỚC 3: Cấu Hình Resources (QUAN TRỌNG!)

**Minimum Settings (để chạy được 10K batch):**
```
CPUs: 4 cores
Memory: 6 GB
Swap: 2 GB
Disk image size: 60 GB
```

**Recommended Settings (chạy ổn định 50K batch):**
```
CPUs: 6 cores (hoặc tất cả trừ 2)
Memory: 8-10 GB
Swap: 2 GB
Disk image size: 80 GB
```

**Optimal Settings (nếu máy có 16GB+ RAM):**
```
CPUs: 6-8 cores
Memory: 12 GB
Swap: 4 GB
Disk image size: 100 GB
```

### BƯỚC 4: Apply & Restart

1. Click **"Apply & Restart"** (góc dưới bên phải)
2. Đợi Docker Desktop restart (2-3 phút)
3. Verify:

```cmd
docker info | findstr Memory
docker info | findstr CPUs
```

Expected output:
```
Total Memory: 8 GiB         <- Phải > 6GB
CPUs: 4                     <- Phải >= 4
```

---

## 🎯 SAU KHI TĂNG RESOURCES

### Option A: Nếu cấp được 8GB+ RAM
👉 **Chạy lại Super Safe Batch (10K test)**

```cmd
# Restart Spark
docker restart spark-master spark-worker-1 spark-worker-2

# Đợi 15 giây
timeout /t 15 /nobreak

# Chạy lại
cd d:\DoAn_TH_BIGDATA
scripts\run_super_safe_batch.bat
```

**Kết quả mong đợi:**
- Test batch (10K) sẽ chạy thành công
- Tiếp tục với 3 batches còn lại
- Total: 4-6 giờ

---

### Option B: Nếu chỉ cấp được 4-6GB RAM
👉 **Chạy Ultra Mini Batch (5K)**

```cmd
# Sẽ tạo script 5K/batch
# Total: 24 batches, 6-8 giờ
```

---

## 🔍 KIỂM TRA MÁY CỦA BẠN

### Xem tổng RAM của máy:
```cmd
systeminfo | findstr "Total Physical Memory"
```

### Khuyến nghị theo RAM:

| RAM Máy | Docker RAM | CPU Cores | Batch Size | Risk |
|---------|-----------|-----------|------------|------|
| 8 GB | 4 GB | 2 | 5K | Medium |
| 12 GB | 6 GB | 4 | 10K | Low |
| 16 GB | 8 GB | 4-6 | 25K | Very Low |
| 32 GB | 12 GB | 6-8 | 50K | Safe |

---

## ⚠️ QUAN TRỌNG

**KHÔNG NÊN** chạy Ultra Mini Batch (5K) nếu chưa thử tăng Docker resources!

**LÝ DO:**
1. Tăng resources = GIẢI PHÁP CĂN BẢN
2. Ultra Mini Batch = WORKAROUND, chạy lâu hơn (24 batches vs 4 batches)
3. Sau khi tăng RAM, 10K batch sẽ chạy tốt

---

## 📝 ACTION PLAN CHO BẠN

### ✅ STEP 1: TĂNG DOCKER RESOURCES (5 phút)
```
Docker Desktop → Settings → Resources
→ CPUs: 4-6
→ Memory: 8 GB (hoặc max có thể)
→ Apply & Restart
```

### ✅ STEP 2: VERIFY
```cmd
docker info | findstr Memory
docker info | findstr CPUs
```

### ✅ STEP 3: RESTART SPARK
```cmd
docker restart spark-master spark-worker-1 spark-worker-2
timeout /t 15 /nobreak
```

### ✅ STEP 4: RUN AGAIN
```cmd
cd d:\DoAn_TH_BIGDATA
scripts\run_super_safe_batch.bat
```

### ✅ STEP 5: MONITOR
```cmd
# Terminal mới
docker stats
```

---

## 🆘 NẾU VẪN KHÔNG ĐƯỢC

**Nếu máy không có đủ RAM (< 12GB total):**

1. **Đóng tất cả apps** (Chrome, VS Code, etc.)
2. **Cấp max RAM cho Docker** (75% total RAM)
3. **Chạy Ultra Mini Batch (5K)** - tôi sẽ tạo script

**Nếu máy có đủ RAM nhưng Docker vẫn crash:**

1. **Restart máy hoàn toàn**
2. **Update Docker Desktop** (latest version)
3. **Check Windows version** (cần Windows 10/11 Pro)

---

## 💡 NEXT STEPS

**BÂY GIỜ:**
1. Mở Docker Desktop Settings
2. Tăng Memory lên 8GB (minimum 6GB)
3. Tăng CPUs lên 4 cores (minimum)
4. Apply & Restart
5. Ping tôi khi done!

**SAU ĐÓ:**
Tôi sẽ hướng dẫn chạy lại batch với resources mới! 🚀

