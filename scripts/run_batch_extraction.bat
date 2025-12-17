@echo off
REM ========================================
REM Run Feature Extraction in 4 Batches
REM Avoid Docker Desktop crash
REM ========================================

echo ========================================
echo BATCH PROCESSING - 120K Images
echo Divided into 4 batches to avoid crash
echo ========================================
echo.

REM ========================================
REM BATCH 1/4: TRAIN REAL (50,000 images)
REM ========================================
echo.
echo ========================================
echo BATCH 1/4: Extracting TRAIN/REAL
echo Expected: ~1.5-2 hours
echo ========================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 4g ^
  --executor-memory 4g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/feature_extraction_train_real.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ BATCH 1 FAILED!
    exit /b 1
)
echo ✅ BATCH 1/4 COMPLETED
echo.

REM ========================================
REM BATCH 2/4: TRAIN FAKE (50,000 images)
REM ========================================
echo.
echo ========================================
echo BATCH 2/4: Extracting TRAIN/FAKE
echo Expected: ~1.5-2 hours
echo ========================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 4g ^
  --executor-memory 4g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/feature_extraction_train_fake.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ BATCH 2 FAILED!
    exit /b 1
)
echo ✅ BATCH 2/4 COMPLETED
echo.

REM ========================================
REM BATCH 3/4: TEST REAL (10,000 images)
REM ========================================
echo.
echo ========================================
echo BATCH 3/4: Extracting TEST/REAL
echo Expected: ~20-30 minutes
echo ========================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 4g ^
  --executor-memory 4g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/feature_extraction_test_real.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ BATCH 3 FAILED!
    exit /b 1
)
echo ✅ BATCH 3/4 COMPLETED
echo.

REM ========================================
REM BATCH 4/4: TEST FAKE (10,000 images)
REM ========================================
echo.
echo ========================================
echo BATCH 4/4: Extracting TEST/FAKE
echo Expected: ~20-30 minutes
echo ========================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 4g ^
  --executor-memory 4g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/feature_extraction_test_fake.py

if %ERRORLEVEL% NEQ 0 (
    echo ❌ BATCH 4 FAILED!
    exit /b 1
)
echo ✅ BATCH 4/4 COMPLETED
echo.

REM ========================================
REM VERIFY ALL FEATURES
REM ========================================
echo.
echo ========================================
echo Verifying all features extracted...
echo ========================================
docker exec -it namenode hdfs dfs -ls /user/data/features/train/REAL
docker exec -it namenode hdfs dfs -ls /user/data/features/train/FAKE
docker exec -it namenode hdfs dfs -ls /user/data/features/test/REAL
docker exec -it namenode hdfs dfs -ls /user/data/features/test/FAKE

echo.
echo ========================================
echo ✅ ALL 4 BATCHES COMPLETED!
echo Total time: ~4-5 hours
echo ========================================
echo.
echo Next step: Run ML Training
echo Command: docker exec -it spark-master /opt/spark/bin/spark-submit /app/src/4_ml_training/ml_training.py
echo.
