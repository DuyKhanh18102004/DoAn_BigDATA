@echo off
REM Test if features can be loaded without OOM

echo.
echo ====================================================================
echo      TESTING FEATURE LOADING (100K samples)
echo ====================================================================
echo.

docker cp src\4_ml_training\test_feature_loading.py spark-master:/app/test_feature_loading.py
if errorlevel 1 (
    echo ERROR: Failed to copy test script
    pause
    exit /b 1
)

echo Running feature load test...
echo.

docker exec spark-master /opt/spark/bin/spark-submit ^
    --master spark://spark-master:7077 ^
    --driver-memory 2g ^
    --executor-memory 2g ^
    /app/test_feature_loading.py

if errorlevel 1 (
    echo.
    echo ❌ FEATURE LOAD TEST FAILED
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo ✅ FEATURE LOAD TEST PASSED!
echo ====================================================================
echo.
echo Next: Run scripts\run_ml_training_batch_safe.bat
echo.

pause
