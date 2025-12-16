#!/usr/bin/env python3
"""
Feature Extraction - Quick Version
Extract features from 1000 sample images for testing
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import ArrayType, FloatType
import sys

print("🚀 Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("DeepfakeDetection-FeatureExtraction-Quick") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

print(f"✅ Spark version: {spark.version}")
print(f"✅ Master: {spark.sparkContext.master}")

# Extract features function (loads model per partition)
def extract_features_udf():
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from PIL import Image
    import numpy as np
    import io
    
    def extract_features(image_binary):
        try:
            # Load model (cached per executor)
            if not hasattr(extract_features.model, 'predict'):
                extract_features.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            
            # Decode image
            img = Image.open(io.BytesIO(image_binary))
            img = img.convert('RGB')
            img = img.resize((224, 224))
            
            # Extract features
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            features = extract_features.model.predict(img_array, verbose=0)
            return features[0].tolist()
        except Exception as e:
            print(f"Error processing image: {e}")
            return [0.0] * 2048
    
    extract_features.model = None
    return extract_features

# Register UDF
extract_udf = udf(extract_features_udf(), ArrayType(FloatType()))

# Read and process each dataset (limit to 250 images each = 1000 total)
print("\n📂 Reading SAMPLE images from HDFS (250 per category)...")

datasets = [
    ("train", "REAL", "hdfs://namenode:8020/user/data/raw/train/REAL"),
    ("train", "FAKE", "hdfs://namenode:8020/user/data/raw/train/FAKE"),
    ("test", "REAL", "hdfs://namenode:8020/user/data/raw/test/REAL"),
    ("test", "FAKE", "hdfs://namenode:8020/user/data/raw/test/FAKE"),
]

for split, label, path in datasets:
    print(f"\n  - Reading {split}/{label} (250 samples)...")
    
    # Read binary files and limit to 250
    df = spark.read.format("binaryFile").load(path).limit(250)
    
    print(f"    📊 Sample count: {df.count()}")
    
    # Add label and extract features
    df = df.withColumn("label", lit(1 if label == "FAKE" else 0)) \
           .withColumn("features", extract_udf(col("content"))) \
           .select("path", "label", "features")
    
    # Save to HDFS
    output_path = f"hdfs://namenode:8020/user/data/features_quick/{split}/{label}"
    print(f"    💾 Saving features to {output_path}...")
    
    df.write.mode("overwrite").parquet(output_path)
    
    print(f"    ✅ Saved {split}/{label} features")

print("\n🎉 Feature extraction completed!")
print("📍 Features saved to: hdfs://namenode:8020/user/data/features_quick/")

spark.stop()
