#!/usr/bin/env python3
"""
Feature Extraction với TensorFlow/Keras MobileNetV2 - OPTIMIZED
- Batch prediction thay vì single image
- Xử lý song song với Pandas UDF
- Memory cleanup sau mỗi batch
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, monotonically_increasing_id, pandas_udf
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from PIL import Image
from io import BytesIO
import time
import gc
import os
import sys

print("="*80)
print("🔬 FEATURE EXTRACTION - TENSORFLOW MOBILENETV2 (OPTIMIZED)")
print("="*80)

# ============================================================================
# SPARK SESSION
# ============================================================================

spark = SparkSession.builder \
    .appName("FeatureExtraction_TF_MobileNetV2_Optimized") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.default.parallelism", "4") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .config("spark.memory.fraction", "0.7") \
    .config("spark.memory.storageFraction", "0.3") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("\n✅ Spark Session created")

# ============================================================================
# TENSORFLOW MODEL SETUP
# ============================================================================

print("\n📦 Loading TensorFlow MobileNetV2 model...")
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Load model once on driver
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
print(f"   ✅ Model loaded! Output shape: {base_model.output_shape}")

# Broadcast model weights
model_weights = base_model.get_weights()
model_weights_broadcast = spark.sparkContext.broadcast(model_weights)

# ============================================================================
# FEATURE EXTRACTION UDF
# ============================================================================

def extract_features_single(image_bytes):
    """Extract features from single image bytes"""
    try:
        if image_bytes is None or len(image_bytes) == 0:
            return [0.0] * 1280
        
        # Load model on executor (cached)
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        
        # Reconstruct model
        model = MobileNetV2(weights=None, include_top=False, pooling='avg', input_shape=(224, 224, 3))
        model.set_weights(model_weights_broadcast.value)
        
        # Load and preprocess image
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        features = model.predict(img_array, verbose=0)
        
        return features.flatten().tolist()
        
    except Exception as e:
        print(f"Error: {e}")
        return [0.0] * 1280

# Register UDF that returns array
extract_features_udf = udf(extract_features_single, ArrayType(FloatType()))

# Convert array to Vector
@udf(VectorUDT())
def array_to_vector(arr):
    if arr is None:
        return Vectors.dense([0.0] * 1280)
    return Vectors.dense(arr)

# ============================================================================
# MEMORY CLEANUP
# ============================================================================

def cleanup_memory():
    spark.catalog.clearCache()
    gc.collect()
    gc.collect()
    gc.collect()
    time.sleep(2)

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_images(input_path, output_path, label, batch_size=1000, start_batch=1, total_batches=5):
    """
    Process images in batches
    """
    print(f"\n{'='*60}")
    print(f"📂 Processing: {input_path}")
    print(f"   Label: {label}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Count total images
    df_all = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(input_path)
    
    total_count = df_all.count()
    print(f"   📊 Total images available: {total_count}")
    print(f"   📊 Will process: {batch_size * total_batches} images ({total_batches} batches x {batch_size})")
    
    all_saved = 0
    
    for batch_idx in range(start_batch - 1, total_batches):
        batch_num = batch_idx + 1
        offset = batch_idx * batch_size
        
        print(f"\n   🔄 Batch {batch_num}/{total_batches}")
        batch_start = time.time()
        
        # Get batch using row_number window function
        from pyspark.sql.window import Window
        from pyspark.sql.functions import row_number
        
        df_with_row = df_all.withColumn("row_num", monotonically_increasing_id())
        df_batch = df_with_row.filter(
            (col("row_num") >= offset) & (col("row_num") < offset + batch_size)
        )
        
        batch_count = df_batch.count()
        if batch_count == 0:
            print(f"      ⚠️ No more images to process")
            break
        
        print(f"      Loading {batch_count} images...")
        
        # Extract features
        print(f"      Extracting features...")
        features_df = df_batch.select(
            col("path"),
            extract_features_udf(col("content")).alias("features_array"),
            lit(label).alias("label")
        )
        
        # Convert to vector
        features_df = features_df.select(
            col("path"),
            array_to_vector(col("features_array")).alias("features"),
            col("label")
        )
        
        # Save
        batch_output = f"{output_path}/batch_{batch_num}"
        features_df.write.mode("overwrite").parquet(batch_output)
        
        saved_count = spark.read.parquet(batch_output).count()
        all_saved += saved_count
        
        batch_time = time.time() - batch_start
        print(f"      ✅ Saved {saved_count} features ({batch_time:.1f}s)")
        
        # Memory cleanup
        df_batch.unpersist()
        features_df.unpersist()
        cleanup_memory()
    
    total_time = time.time() - start_time
    print(f"\n   ⏱️  Total time: {total_time:.1f}s")
    print(f"   📊 Total saved: {all_saved}")
    
    return all_saved

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    HDFS_BASE = "hdfs://namenode:8020/user/data"
    RAW_PATH = f"{HDFS_BASE}/raw"
    FEATURES_PATH = f"{HDFS_BASE}/features_tf"
    
    # Configuration
    BATCH_SIZE = 10000  # 10K per batch
    TOTAL_BATCHES = 5   # 5 batches = 50K per class
    
    print("\n" + "="*80)
    print(f"🚀 FEATURE EXTRACTION: {BATCH_SIZE * TOTAL_BATCHES} images per class")
    print("="*80)
    
    # Process REAL
    real_count = process_images(
        input_path=f"{RAW_PATH}/train/REAL",
        output_path=f"{FEATURES_PATH}/train/REAL",
        label=1,
        batch_size=BATCH_SIZE,
        total_batches=TOTAL_BATCHES
    )
    
    # Cleanup between classes
    cleanup_memory()
    
    # Process FAKE
    fake_count = process_images(
        input_path=f"{RAW_PATH}/train/FAKE",
        output_path=f"{FEATURES_PATH}/train/FAKE",
        label=0,
        batch_size=BATCH_SIZE,
        total_batches=TOTAL_BATCHES
    )
    
    # Summary
    print("\n" + "="*80)
    print("📊 EXTRACTION COMPLETE")
    print("="*80)
    print(f"   REAL: {real_count} features")
    print(f"   FAKE: {fake_count} features")
    print(f"   Total: {real_count + fake_count} features")
    print("="*80)
    
    spark.stop()
    print("\n✅ Done!")
