#!/bin/bash
# Run Test with 100 Images
# Quick validation pipeline

echo "🧪 Running test with 100 images..."

# Upload 100 images
echo "📤 Uploading test data..."
python src/1_ingestion/upload_to_hdfs.py \
    --local_path Dataset_Test/train \
    --hdfs_path /user/data/raw/train_test \
    --max_files 100

# Extract features
echo "🔬 Extracting features..."
docker exec -it spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --driver-memory 2g \
    --executor-memory 2g \
    /app/src/3_feature_extraction/extract_pipeline.py \
    --input_path /user/data/raw/train_test \
    --output_path /user/data/features/train_test \
    --model resnet50

# Train model
echo "🤖 Training model..."
docker exec -it spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    /app/src/4_ml_training/train_classifier.py \
    --features_path /user/data/features/train_test \
    --model_output /user/models/test_rf

echo "✅ Test completed!"
