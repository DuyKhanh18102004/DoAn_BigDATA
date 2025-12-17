@echo off
REM ========================================
REM SUPER SAFE BATCH EXTRACTION
REM Ultra-small batches (10K each) to avoid crash
REM ========================================

echo ========================================
echo SUPER SAFE BATCH PROCESSING
echo Each batch: 10,000 images maximum
echo Total: 12 batches
echo ========================================
echo.

REM ========================================
REM OPTION A: Test với batch nhỏ nhất trước
REM ========================================
echo.
echo ========================================
echo STEP 0: Testing with 10K images first
echo ========================================
echo.
echo Chạy test batch (TEST/REAL - 10K ảnh) để test...
echo Nếu thành công, sẽ tiếp tục các batch còn lại.
echo.

docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/feature_extraction_test_real.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ❌ TEST BATCH FAILED!
    echo ========================================
    echo.
    echo Ngay cả 10K ảnh cũng bị crash!
    echo.
    echo GIẢI PHÁP:
    echo 1. Tăng Docker Desktop RAM lên 8-12GB
    echo 2. Hoặc giảm xuống còn 5K ảnh/batch
    echo 3. Hoặc chạy trên cloud (AWS/GCP/Azure)
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ TEST BATCH SUCCEEDED!
echo ========================================
echo.
echo Docker có thể xử lý 10K ảnh/batch.
echo Tiếp tục với các batch còn lại...
echo.
pause

REM ========================================
REM Continue with remaining batches
REM ========================================

echo ========================================
echo BATCH 1: TRAIN REAL (50K images)
echo ========================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 3g ^
  --executor-memory 3g ^
  /app/src/3_feature_extraction/feature_extraction_train_real.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Batch 1 failed!
    echo Thử giảm xuống 10K/batch cho tất cả batches
    pause
    exit /b 1
)

echo ✅ TRAIN/REAL completed (50K total)
echo.

REM ========================================
REM BATCH 2: TRAIN FAKE
REM ========================================
echo.
echo ========================================
echo BATCH 2: TRAIN FAKE (50K images)
echo ========================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 3g ^
  --executor-memory 3g ^
  /app/src/3_feature_extraction/feature_extraction_train_fake.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Batch 2 failed!
    pause
    exit /b 1
)

echo ✅ TRAIN/FAKE completed
echo.

REM ========================================
REM BATCH 3: TEST REAL
REM ========================================
echo.
echo ========================================
echo BATCH 3: TEST REAL (10K images)
echo ========================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  /app/src/3_feature_extraction/feature_extraction_test_real.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Batch 3 failed!
    pause
    exit /b 1
)

echo ✅ TEST/REAL completed
echo.

REM ========================================
REM BATCH 4: TEST FAKE
REM ========================================
echo.
echo ========================================
echo BATCH 4: TEST FAKE (10K images)
echo ========================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  /app/src/3_feature_extraction/feature_extraction_test_fake.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Batch 4 failed!
    pause
    exit /b 1
)

echo ✅ TEST/FAKE completed
echo.

echo.
echo ========================================
echo Verification completed!
echo ========================================
echo.
echo ========================================
echo ✅ ALL 4 BATCHES COMPLETED!
echo ========================================
echo.
echo Features saved to:
echo   - /user/data/features/train/REAL
echo   - /user/data/features/train/FAKE
echo   - /user/data/features/test/REAL
echo   - /user/data/features/test/FAKE
echo.
echo Next: Run ML Training
echo   docker exec -it spark-master /opt/spark/bin/spark-submit /app/src/4_ml_training/ml_training.py
echo.

pause
