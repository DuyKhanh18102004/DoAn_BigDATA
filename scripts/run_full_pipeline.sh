#!/bin/bash
# Run Full Pipeline
# Execute complete deepfake detection pipeline

echo "🚀 Starting Deepfake Detection Pipeline..."

# Step 1: Upload data to HDFS
echo ""
echo "📤 Step 1: Uploading data to HDFS..."
python src/1_ingestion/upload_to_hdfs.py \
    --local_path data/train \
    --hdfs_path /user/data/raw/train

# Step 2: Extract features
echo ""
echo "🔬 Step 2: Extracting features..."
docker exec -it spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --driver-memory 4g \
    --executor-memory 4g \
    /app/src/3_feature_extraction/extract_pipeline.py \
    --input_path /user/data/raw/train \
    --output_path /user/data/features/train

# Step 3: Train ML models
echo ""
echo "🤖 Step 3: Training ML models..."
docker exec -it spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    /app/src/4_ml_training/train_classifier.py

# Step 4: Evaluate models
echo ""
echo "📊 Step 4: Evaluating models..."
docker exec -it spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    /app/src/5_evaluation/evaluate_model.py

echo ""
echo "✅ Pipeline completed successfully!"
echo "📋 Check results at:"
echo "  - HDFS: hdfs://namenode:8020/user/data/results/"
echo "  - Spark History: http://localhost:18080"
