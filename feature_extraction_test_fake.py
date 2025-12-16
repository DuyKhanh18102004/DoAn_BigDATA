#!/usr/bin/env python3
"""
Feature Extraction - TEST FAKE ONLY (10,000 images)
Batch 4/4 of full pipeline
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

print("=" * 80)
print("🚀 BATCH 4/4: TEST FAKE Feature Extraction")
print("=" * 80)

# Initialize Spark
spark = SparkSession.builder \
    .appName("FeatureExtraction-TestFAKE-Batch4") \
    .getOrCreate()

print(f"✅ Spark Session: {spark.sparkContext.appName}")
print(f"✅ Spark Version: {spark.version}")
print(f"✅ Master: {spark.sparkContext.master}")

# Create ResNet50 UDF
def create_feature_extractor_udf():
    """Create UDF for ResNet50 feature extraction"""
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    def extract_features_with_label(image_bytes, label_str):
        try:
            img = Image.open(BytesIO(image_bytes)).convert('RGB')
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = model.predict(img_array, verbose=0)[0]
            label = 1.0 if label_str == "REAL" else 0.0
            return [label] + features.tolist()
        except Exception as e:
            print(f"Error processing image: {e}")
            return [0.0] + [0.0] * 2048
    
    return udf(extract_features_with_label, ArrayType(FloatType()))

extract_features_udf = create_feature_extractor_udf()

print("\n📂 Loading TEST FAKE images from HDFS...")
test_fake_df = spark.read.format("binaryFile") \
    .load("hdfs://namenode:8020/user/data/raw/test/FAKE/*.jpg") \
    .withColumn("label_str", lit("FAKE"))

count_fake = test_fake_df.count()
print(f"✅ Test FAKE: {count_fake} images")

print("\n🔬 Extracting features using ResNet50 UDF...")
features_fake = test_fake_df.withColumn(
    "features_label",
    extract_features_udf(col("content"), col("label_str"))
).select("path", "features_label")

print("\n💾 Saving features to HDFS...")
output_path = "hdfs://namenode:8020/user/data/features/test/FAKE"
features_fake.write.mode("overwrite").parquet(output_path)

print(f"\n✅ BATCH 4/4 COMPLETED!")
print(f"✅ Output: {output_path}")
print(f"✅ Processed: {count_fake} images")
print("=" * 80)

spark.stop()
