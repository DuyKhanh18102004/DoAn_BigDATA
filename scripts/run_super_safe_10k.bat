@echo off
REM ================================================================================
REM SUPER SAFE BATCH PROCESSING - 10K images per batch
REM Total: 12 batches x 10-15 minutes = ~2-3 hours
REM ================================================================================

echo.
echo ================================================================================
echo SUPER SAFE BATCH MODE: 10K images per batch
echo ================================================================================
echo Total batches: 12
echo Estimated time: 2-3 hours
echo.
echo Batch breakdown:
echo   - TRAIN/REAL: 5 batches (50K images)
echo   - TRAIN/FAKE: 5 batches (50K images)
echo   - TEST/REAL:  1 batch  (10K images)
echo   - TEST/FAKE:  1 batch  (10K images)
echo ================================================================================
echo.

set START_TIME=%TIME%

REM ============================================================================
REM TRAIN/REAL - 5 batches (50,000 images total)
REM ============================================================================

echo.
echo [1/12] BATCH 1: TRAIN/REAL (0-9,999)
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_real_1.py

if errorlevel 1 (
    echo ERROR: Batch 1 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 1 completed successfully
echo.

echo [2/12] BATCH 2: TRAIN/REAL (10,000-19,999)
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_real_2.py

if errorlevel 1 (
    echo ERROR: Batch 2 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 2 completed successfully
echo.

echo [3/12] BATCH 3: TRAIN/REAL (20,000-29,999)
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_real_3.py

if errorlevel 1 (
    echo ERROR: Batch 3 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 3 completed successfully
echo.

echo [4/12] BATCH 4: TRAIN/REAL (30,000-39,999)
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_real_4.py

if errorlevel 1 (
    echo ERROR: Batch 4 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 4 completed successfully
echo.

echo [5/12] BATCH 5: TRAIN/REAL (40,000-49,999) - FINAL TRAIN/REAL
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_real_5.py

if errorlevel 1 (
    echo ERROR: Batch 5 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 5 completed - TRAIN/REAL FINISHED (50K images)
echo.

REM ============================================================================
REM TRAIN/FAKE - 5 batches (50,000 images total)
REM ============================================================================

echo [6/12] BATCH 6: TRAIN/FAKE (0-9,999)
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_fake_1.py

if errorlevel 1 (
    echo ERROR: Batch 6 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 6 completed successfully
echo.

echo [7/12] BATCH 7: TRAIN/FAKE (10,000-19,999)
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_fake_2.py

if errorlevel 1 (
    echo ERROR: Batch 7 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 7 completed successfully
echo.

echo [8/12] BATCH 8: TRAIN/FAKE (20,000-29,999)
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_fake_3.py

if errorlevel 1 (
    echo ERROR: Batch 8 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 8 completed successfully
echo.

echo [9/12] BATCH 9: TRAIN/FAKE (30,000-39,999)
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_fake_4.py

if errorlevel 1 (
    echo ERROR: Batch 9 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 9 completed successfully
echo.

echo [10/12] BATCH 10: TRAIN/FAKE (40,000-49,999) - FINAL TRAIN/FAKE
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_train_fake_5.py

if errorlevel 1 (
    echo ERROR: Batch 10 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 10 completed - TRAIN/FAKE FINISHED (50K images)
echo.

REM ============================================================================
REM TEST - 2 batches (20,000 images total)
REM ============================================================================

echo [11/12] BATCH 11: TEST/REAL (10,000 images)
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_test_real.py

if errorlevel 1 (
    echo ERROR: Batch 11 FAILED!
    pause
    exit /b 1
)
echo ✅ Batch 11 completed - TEST/REAL FINISHED (10K images)
echo.

echo [12/12] BATCH 12: TEST/FAKE (10,000 images) - FINAL BATCH!
echo ================================================================================
docker exec -it spark-master /opt/spark/bin/spark-submit ^
  --master spark://spark-master:7077 ^
  --driver-memory 2g ^
  --executor-memory 2g ^
  --executor-cores 2 ^
  --num-executors 2 ^
  /app/src/3_feature_extraction/batch_test_fake.py

if errorlevel 1 (
    echo ERROR: Batch 12 FAILED!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo 🎉 ALL 12 BATCHES COMPLETED SUCCESSFULLY!
echo ================================================================================
echo Total images processed: 120,000
echo   - TRAIN/REAL: 50,000 images (5 batches)
echo   - TRAIN/FAKE: 50,000 images (5 batches)
echo   - TEST/REAL:  10,000 images (1 batch)
echo   - TEST/FAKE:  10,000 images (1 batch)
echo.
echo Start time: %START_TIME%
echo End time:   %TIME%
echo ================================================================================
echo.
echo Next step: Run ML Training
echo   scripts\run_ml_training.bat
echo.

pause
