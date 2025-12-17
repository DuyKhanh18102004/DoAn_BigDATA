@echo off
REM ========================================
REM ULTRA MINI BATCH - 5000 images
REM Last resort if 10K still crashes
REM ========================================

echo ========================================
echo ULTRA MINI BATCH TEST
echo 5,000 images only
echo ========================================
echo.

REM Giảm memory xuống mức tối thiểu
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 1g ^
  --executor-memory 1g ^
  --executor-cores 1 ^
  --num-executors 1 ^
  --conf spark.sql.shuffle.partitions=20 ^
  --conf spark.default.parallelism=20 ^
  --py-files /app/src/3_feature_extraction/ultra_mini_batch.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ❌ ULTRA MINI BATCH FAILED!
    echo ========================================
    echo.
    echo Ngay cả 5K ảnh với 1GB RAM cũng không được!
    echo.
    echo PHẢI LÀM:
    echo 1. Kiểm tra Docker Desktop Settings:
    echo    - Memory: Phải có ít nhất 6GB
    echo    - CPUs: Ít nhất 2 cores
    echo    - Swap: 2GB
    echo.
    echo 2. Restart Docker Desktop hoàn toàn
    echo.
    echo 3. Check logs:
    echo    docker logs spark-master
    echo    docker logs spark-worker1
    echo.
    echo 4. Nếu vẫn lỗi: Máy không đủ RAM
    echo    - Cần chạy trên cloud
    echo    - Hoặc giảm xuống 1000 ảnh test thôi
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ ULTRA MINI BATCH SUCCEEDED!
echo ========================================
echo.
echo Docker CÓ THỂ xử lý 5K ảnh/batch.
echo Khuyến nghị: Chạy với 5K ảnh/batch
echo Total batches cần: 24 batches (120K / 5K)
echo.

pause
