# Pipeline execution script for Big Data project
# Executes: Data upload -> Feature extraction -> ML training -> Evaluation

param(
    [bool]$SkipUpload = $false,
    [bool]$SkipFeatureExtraction = $false,
    [bool]$SkipTraining = $false,
    [bool]$SkipEvaluation = $false,
    [bool]$CleanFeatures = $false
)

# Configuration
$projectRoot = Split-Path -Parent $PSScriptRoot
$srcPath = Join-Path $projectRoot "src"
$uploadScript = Join-Path $srcPath "1_ingestion\upload_to_hdfs.py"
$extractScript = Join-Path $srcPath "2_feature_extraction\extract_mobilenetv2_optimized.py"
$trainScript = Join-Path $srcPath "4_ml_training\ml_training_tf_features.py"
$evalScript = Join-Path $srcPath "5_evaluation\evaluate_tf_model.py"

$startTime = Get-Date
$successCount = 0
$failureCount = 0

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "=================================================="
    Write-Host $Message
    Write-Host "=================================================="
}

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Message"
}

function Write-Success {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] SUCCESS: $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] ERROR: $Message" -ForegroundColor Red
}

function Execute-Command {
    param([string]$Description, [scriptblock]$Command)
    
    Write-Step $Description
    try {
        & $Command
        if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq $null) {
            Write-Success $Description
            $script:successCount++
            return $true
        } else {
            Write-Error-Custom "$Description (Exit Code: $LASTEXITCODE)"
            $script:failureCount++
            return $false
        }
    } catch {
        Write-Error-Custom "$Description - Exception: $_"
        $script:failureCount++
        return $false
    }
}

# Main execution
Write-Header "Big Data Pipeline Execution Started"
Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Step 1: Upload Data
if (-not $SkipUpload) {
    Execute-Command "Step 1: Uploading data to HDFS" {
        python $uploadScript
    }
} else {
    Write-Step "Step 1: Skipped (Upload)"
}

# Step 2: Feature Extraction
if (-not $SkipFeatureExtraction) {
    Execute-Command "Step 2a: Creating feature extraction directory" {
        docker exec spark-master mkdir -p /app/src/2_feature_extraction
    }
    
    Execute-Command "Step 2b: Copying feature extraction script" {
        docker cp "$extractScript" spark-master:/app/src/2_feature_extraction/
    }
    
    if ($CleanFeatures) {
        Execute-Command "Step 2c: Cleaning old features from HDFS" {
            docker exec namenode hdfs dfs -rm -r -f /user/data/features_tf/
        }
    }
    
    Execute-Command "Step 2d: Running feature extraction" {
        docker exec spark-master /opt/spark/bin/spark-submit --master local[2] --driver-memory 3g /app/src/2_feature_extraction/extract_mobilenetv2_optimized.py
    }
} else {
    Write-Step "Step 2: Skipped (Feature Extraction)"
}

# Step 3: ML Training
if (-not $SkipTraining) {
    Execute-Command "Step 3a: Creating training directory" {
        docker exec spark-master mkdir -p /app/src/4_ml_training
    }
    
    Execute-Command "Step 3b: Copying training script" {
        docker cp "$trainScript" spark-master:/app/src/4_ml_training/
    }
    
    Execute-Command "Step 3c: Running ML training" {
        docker exec spark-master /opt/spark/bin/spark-submit --master local[2] --driver-memory 3g /app/src/4_ml_training/ml_training_tf_features.py
    }
} else {
    Write-Step "Step 3: Skipped (Training)"
}

# Step 4: Evaluation
if (-not $SkipEvaluation) {
    Execute-Command "Step 4a: Creating evaluation directory" {
        docker exec spark-master mkdir -p /app/src/5_evaluation
    }
    
    Execute-Command "Step 4b: Copying evaluation script" {
        docker cp "$evalScript" spark-master:/app/src/5_evaluation/
    }
    
    Execute-Command "Step 4c: Running model evaluation" {
        docker exec spark-master /opt/spark/bin/spark-submit --master local[2] --driver-memory 3g /app/src/5_evaluation/evaluate_tf_model.py
    }
} else {
    Write-Step "Step 4: Skipped (Evaluation)"
}

# Summary
Write-Header "Pipeline Execution Summary"
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "Successful steps: $successCount" -ForegroundColor Green
Write-Host "Failed steps: $failureCount" -ForegroundColor $(if ($failureCount -gt 0) { "Red" } else { "Green" })
Write-Host "Duration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s"
Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

if ($failureCount -eq 0) {
    Write-Host ""
    Write-Success "All pipeline steps completed successfully!"
} else {
    Write-Host ""
    Write-Error-Custom "Pipeline completed with errors. Check logs above."
}

exit $failureCount
