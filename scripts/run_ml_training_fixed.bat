@echo off
REM ========================================
REM FIXED ML TRAINING - LOAD FULL 100K DATA
REM ========================================

echo.
echo ====================================================================
echo      FIXED BATCH ML TRAINING - FULL 100K TRAINING DATA
echo ====================================================================
echo.
echo FIX: Properly load all 100,000 training samples
echo Previous Issue: Only loaded 64 samples
echo Solution: Count each batch explicitly before union
echo.
echo Training Samples: 100,000 (50K REAL + 50K FAKE)
echo Test Samples: 20,000 (10K REAL + 10K FAKE)
echo Models: Logistic Regression + Random Forest
echo.
echo ====================================================================
echo.

echo [%TIME%] Starting FIXED ML Training Pipeline...
echo.

REM Copy fixed script to container
echo [1/2] Copying FIXED ML training script to spark-master...
docker cp src\4_ml_training\ml_training_fixed.py spark-master:/app/ml_training_fixed.py
if errorlevel 1 (
    echo ERROR: Failed to copy script to container
    pause
    exit /b 1
)
echo      ✓ Fixed script copied successfully
echo.

REM Execute training
echo [2/2] Executing FIXED Batch ML Training on Spark...
echo ====================================================================
echo.

docker exec spark-master /opt/spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --driver-memory 2g ^
    --executor-memory 2g ^
    --executor-cores 2 ^
    --total-executor-cores 4 ^
    --conf spark.sql.shuffle.partitions=100 ^
    --conf spark.default.parallelism=100 ^
    /app/ml_training_fixed.py

if errorlevel 1 (
    echo.
    echo ====================================================================
    echo ❌ FIXED ML TRAINING FAILED
    echo ====================================================================
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo 🎉 FIXED ML TRAINING COMPLETED SUCCESSFULLY!
echo ====================================================================
echo.
echo 📊 Expected Results:
echo    - Training: 100,000 samples (vs previous 64)
echo    - Test: 20,000 samples
echo    - Better accuracy with full dataset
echo.
echo 📦 Saved Artifacts (with _fixed suffix):
echo    ✓ Logistic Regression Model
echo    ✓ Random Forest Model
echo    ✓ LR Predictions
echo    ✓ RF Predictions
echo    ✓ Metrics Summary
echo.
echo 🎯 Next Steps:
echo    1. Compare metrics with previous run
echo    2. Check Spark History Server: http://localhost:18080
echo    3. Take screenshots for project documentation
echo.
echo ====================================================================

pause
