@echo off
REM ========================================
REM Quick Check: Batch Progress Monitor
REM ========================================

echo ========================================
echo BATCH PROCESSING - Progress Check
echo ========================================
echo.

echo Checking HDFS directories...
echo.

echo [1/4] TRAIN/REAL:
docker exec -it namenode hdfs dfs -count /user/data/features/train/REAL 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    ✅ Batch 1 completed
) else (
    echo    ⏳ Batch 1 not yet completed
)
echo.

echo [2/4] TRAIN/FAKE:
docker exec -it namenode hdfs dfs -count /user/data/features/train/FAKE 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    ✅ Batch 2 completed
) else (
    echo    ⏳ Batch 2 not yet completed
)
echo.

echo [3/4] TEST/REAL:
docker exec -it namenode hdfs dfs -count /user/data/features/test/REAL 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    ✅ Batch 3 completed
) else (
    echo    ⏳ Batch 3 not yet completed
)
echo.

echo [4/4] TEST/FAKE:
docker exec -it namenode hdfs dfs -count /user/data/features/test/FAKE 2>nul
if %ERRORLEVEL% EQU 0 (
    echo    ✅ Batch 4 completed
) else (
    echo    ⏳ Batch 4 not yet completed
)
echo.

echo ========================================
echo Spark History Server: http://localhost:18080
echo ========================================
