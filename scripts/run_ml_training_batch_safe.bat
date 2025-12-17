@echo off
REM ========================================
REM BATCH ML TRAINING - FULL 100K DATA
REM Train on complete dataset safely
REM ========================================

echo.
echo ====================================================================
echo      BATCH ML TRAINING - FULL 100K TRAINING DATA
echo ====================================================================
echo.
echo Strategy: Load 10 batches incrementally + Union + Train
echo Training Samples: 100,000 (50K REAL + 50K FAKE)
echo Test Samples: 20,000 (10K REAL + 10K FAKE)
echo Models: Logistic Regression + Random Forest
echo.
echo ====================================================================
echo.

echo [%TIME%] Starting ML Training Pipeline...
echo.

REM Copy script to container
echo [1/2] Copying ML training script to spark-master...
docker cp src\4_ml_training\ml_training_batch_safe.py spark-master:/app/ml_training_batch_safe.py
if errorlevel 1 (
    echo ERROR: Failed to copy script to container
    pause
    exit /b 1
)
echo      ✓ Script copied successfully
echo.

REM Execute training
echo [2/2] Executing Batch ML Training on Spark...
echo ====================================================================
echo.

docker exec spark-master /opt/spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --driver-memory 2g ^
    --executor-memory 2g ^
    --executor-cores 2 ^
    --total-executor-cores 4 ^
    --conf spark.sql.shuffle.partitions=100 ^
    /app/ml_training_batch_safe.py

if errorlevel 1 (
    echo.
    echo ====================================================================
    echo ❌ ML TRAINING FAILED
    echo ====================================================================
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo 🎉 ML TRAINING COMPLETED SUCCESSFULLY!
echo ====================================================================
echo.
echo 📊 Results Summary:
echo    - Training completed on 100K samples
echo    - Models tested on 20K samples
echo    - Models saved to HDFS
echo    - Predictions saved to HDFS
echo    - Metrics report generated
echo.
echo 📦 Saved Artifacts:
echo    ✓ Logistic Regression Model
echo    ✓ Random Forest Model
echo    ✓ LR Predictions
echo    ✓ RF Predictions
echo    ✓ Metrics Summary
echo.
echo 🎯 Next Steps:
echo    1. Check Spark History Server: http://localhost:18080
echo    2. Review accuracy metrics and confusion matrices
echo    3. Take screenshots for project documentation
echo.
echo ====================================================================

pause
