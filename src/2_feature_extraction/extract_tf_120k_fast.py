#!/usr/bin/env python3
"""
Feature Extraction với TensorFlow/Keras MobileNetV2 - OPTIMIZED FOR SPEED
- Extract 120K images với batch processing trong TensorFlow
- Driver-side extraction để tận dụng batch predict
- Lưu vào HDFS theo cấu trúc batch
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from PIL import Image
from io import BytesIO
import time
import gc
import os

print("="*80)
print("🔬 FEATURE EXTRACTION - TENSORFLOW MOBILENETV2 (120K - OPTIMIZED)")
print("="*80)

# ============================================================================
# SPARK SESSION
# ============================================================================

spark = SparkSession.builder \
    .appName("FeatureExtraction_TF_MobileNetV2_120K_Fast") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("\n✅ Spark Session created")

# ============================================================================
# TENSORFLOW MOBILENETV2 MODEL
# ============================================================================

print("\n📦 Loading TensorFlow MobileNetV2 model...")
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Load model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
print(f"   ✅ MobileNetV2 loaded! Output shape: {model.output_shape}")

# ============================================================================
# BATCH FEATURE EXTRACTION
# ============================================================================

def preprocess_image(image_bytes):
    """Preprocess single image bytes to array"""
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        return np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return np.zeros((224, 224, 3), dtype=np.float32)

def extract_batch_features(images_data, batch_size=32):
    """
    Extract features for a list of (path, image_bytes) tuples
    Returns list of (path, features) tuples
    """
    results = []
    total = len(images_data)
    
    for i in range(0, total, batch_size):
        batch_data = images_data[i:i+batch_size]
        
        # Preprocess batch
        batch_arrays = []
        batch_paths = []
        for path, img_bytes in batch_data:
            arr = preprocess_image(img_bytes)
            batch_arrays.append(arr)
            batch_paths.append(path)
        
        # Stack into numpy array
        batch_input = np.stack(batch_arrays)
        batch_input = preprocess_input(batch_input)
        
        # Extract features (batch prediction - much faster!)
        features = model.predict(batch_input, verbose=0, batch_size=len(batch_data))
        
        # Add to results
        for path, feat in zip(batch_paths, features):
            results.append((path, feat.tolist()))
        
        # Progress
        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= total:
            print(f"         Extracted: {min(i + batch_size, total)}/{total}")
    
    return results

def cleanup():
    gc.collect()
    gc.collect()
    time.sleep(1)

# ============================================================================
# PROCESS FUNCTION
# ============================================================================

def process_class(input_path, output_path, label, batch_size_per_file=10000, num_batches=5):
    """
    Process một class (REAL/FAKE) và lưu theo batches
    """
    print(f"\n{'='*60}")
    print(f"📂 Processing: {input_path}")
    print(f"   Label: {label} ({'REAL' if label == 1 else 'FAKE'})")
    print(f"   Files: {num_batches} batches x {batch_size_per_file} images")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Load paths only first
    df_all = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(input_path)
    
    total_available = df_all.count()
    print(f"   📊 Total available: {total_available}")
    
    total_extracted = 0
    
    for batch_num in range(1, num_batches + 1):
        batch_start = time.time()
        
        start_idx = (batch_num - 1) * batch_size_per_file
        end_idx = min(start_idx + batch_size_per_file, total_available)
        
        if start_idx >= total_available:
            print(f"   ⚠️ No more images!")
            break
        
        print(f"\n   🔄 Batch {batch_num}/{num_batches}")
        print(f"      Range: {start_idx + 1} - {end_idx}")
        
        # Collect batch data to driver (path + content)
        print(f"      Loading images to driver...")
        batch_df = df_all.limit(end_idx).subtract(df_all.limit(start_idx)) if start_idx > 0 else df_all.limit(batch_size_per_file)
        
        # Actually collect this batch
        batch_data = batch_df.select("path", "content").rdd \
            .map(lambda row: (row.path, bytes(row.content))) \
            .collect()
        
        print(f"      Loaded {len(batch_data)} images")
        
        # Extract features using batch processing
        print(f"      Extracting features (TF batch mode)...")
        features_list = extract_batch_features(batch_data, batch_size=64)
        
        # Convert to DataFrame and save
        print(f"      Saving to HDFS...")
        
        # Create DataFrame with features
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
        
        schema = StructType([
            StructField("path", StringType(), True),
            StructField("features_array", ArrayType(FloatType()), True),
            StructField("label", IntegerType(), True)
        ])
        
        rows = [(path, feats, label) for path, feats in features_list]
        features_df = spark.createDataFrame(rows, schema)
        
        # Convert to Vector
        from pyspark.sql.functions import udf
        
        @udf(VectorUDT())
        def to_vector(arr):
            return Vectors.dense(arr) if arr else Vectors.dense([0.0] * 1280)
        
        features_df = features_df.select(
            col("path"),
            to_vector(col("features_array")).alias("features"),
            col("label")
        )
        
        # Save
        batch_output = f"{output_path}/batch_{batch_num}"
        features_df.write.mode("overwrite").parquet(batch_output)
        
        saved = spark.read.parquet(batch_output).count()
        total_extracted += saved
        
        batch_time = time.time() - batch_start
        print(f"      ✅ Saved: {saved} features ({batch_time:.1f}s, {len(batch_data)/batch_time:.1f} img/s)")
        
        # Cleanup
        del batch_data, features_list, features_df
        cleanup()
        print(f"      🧹 Memory cleaned")
    
    total_time = time.time() - start_time
    print(f"\n   📊 TOTAL: {total_extracted} features")
    print(f"   ⏱️  Time: {total_time/60:.1f} minutes")
    
    return total_extracted

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    HDFS_BASE = "hdfs://namenode:8020/user/data"
    RAW_PATH = f"{HDFS_BASE}/raw"
    FEATURES_PATH = f"{HDFS_BASE}/features"
    
    print("\n" + "="*80)
    print("🚀 FEATURE EXTRACTION - 120K IMAGES (OPTIMIZED)")
    print("="*80)
    print(f"""
📋 PLAN:
   ├── Train REAL: 50,000 (5 batches x 10,000)
   ├── Train FAKE: 50,000 (5 batches x 10,000)
   ├── Test REAL:  10,000 (1 batch)
   └── Test FAKE:  10,000 (1 batch)
   Total: 120,000 images
   Model: MobileNetV2 (1280-dim features)
""")
    
    results = {}
    
    # Train REAL
    results['train_real'] = process_class(
        f"{RAW_PATH}/train/REAL", f"{FEATURES_PATH}/train/REAL", 1, 10000, 5)
    cleanup()
    
    # Train FAKE
    results['train_fake'] = process_class(
        f"{RAW_PATH}/train/FAKE", f"{FEATURES_PATH}/train/FAKE", 0, 10000, 5)
    cleanup()
    
    # Test REAL
    results['test_real'] = process_class(
        f"{RAW_PATH}/test/REAL", f"{FEATURES_PATH}/test/REAL", 1, 10000, 1)
    cleanup()
    
    # Test FAKE
    results['test_fake'] = process_class(
        f"{RAW_PATH}/test/FAKE", f"{FEATURES_PATH}/test/FAKE", 0, 10000, 1)
    
    # Summary
    print("\n" + "="*80)
    print("📊 EXTRACTION COMPLETE")
    print("="*80)
    print(f"""
   ✅ Train REAL: {results['train_real']:,}
   ✅ Train FAKE: {results['train_fake']:,}
   ✅ Test REAL:  {results['test_real']:,}
   ✅ Test FAKE:  {results['test_fake']:,}
   
   📊 Total: {sum(results.values()):,} features
""")
    
    # Verify sample
    print("🔍 Verifying features...")
    from pyspark.sql.functions import udf
    
    @udf("float")
    def vec_sum(vec):
        return float(sum(vec.toArray())) if vec else 0.0
    
    sample = spark.read.parquet(f"{FEATURES_PATH}/train/REAL/batch_1").limit(3)
    sample.select("path", vec_sum("features").alias("sum")).show(3, truncate=40)
    
    spark.stop()
    print("\n✅ Done!")
