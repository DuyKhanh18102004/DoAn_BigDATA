@echo off
REM Monitor Batch Progress
cls
echo ========================================
echo BATCH PROCESSING MONITOR
echo ========================================
echo.
echo Checking Spark jobs and HDFS output...
echo.

:LOOP

echo [%TIME%] Checking progress...
echo.

REM Check if features are being saved
echo Features on HDFS:
docker exec namenode hdfs dfs -du -h /user/data/features/ 2>nul
echo.

REM Check Docker stats
echo Docker Resource Usage:
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" spark-master spark-worker-1 spark-worker-2
echo.

echo ----------------------------------------
echo Press Ctrl+C to stop monitoring
echo Refreshing in 30 seconds...
echo ----------------------------------------
timeout /t 30 /nobreak >nul

goto LOOP
