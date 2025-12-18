#!/usr/bin/env python3
"""
Feature Extraction với TensorFlow/Keras MobileNetV2 - MEMORY SAFE
- Extract 120K images với small batch processing
- Batch 2000 images để tránh OOM crash
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
print("🔬 FEATURE EXTRACTION - TENSORFLOW MOBILENETV2 (MEMORY SAFE)")
print("="*80)

# ============================================================================
# SPARK SESSION - Lower memory to be safe
# ============================================================================

spark = SparkSession.builder \
    .appName("FeatureExtraction_TF_MobileNetV2_Safe") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.maxResultSize", "2g") \
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
# CONFIG - SMALLER BATCHES TO AVOID CRASH
# ============================================================================

BATCH_SIZE = 2000  # Images per HDFS batch (smaller = safer)
TF_BATCH_SIZE = 16  # Images per TF inference batch (smaller = less memory)
HDFS_BASE = "hdfs://namenode:8020/user/data"

# Cấu hình datasets - Continue from where we left off
DATASETS = [
    # Train REAL: 50K images -> 25 batches of 2000
    {"split": "train", "class": "REAL", "label": 1, "total": 50000, "batches": 25, "start_batch": 1},
    # Train FAKE: 50K images -> 25 batches of 2000  
    {"split": "train", "class": "FAKE", "label": 0, "total": 50000, "batches": 25, "start_batch": 1},
    # Test REAL: 10K images -> 5 batches of 2000
    {"split": "test", "class": "REAL", "label": 1, "total": 10000, "batches": 5, "start_batch": 1},
    # Test FAKE: 10K images -> 5 batches of 2000
    {"split": "test", "class": "FAKE", "label": 0, "total": 10000, "batches": 5, "start_batch": 1},
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def preprocess_image(image_bytes):
    """Preprocess single image bytes to array"""
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        return np.array(img, dtype=np.float32)
    except Exception as e:
        return np.zeros((224, 224, 3), dtype=np.float32)

def extract_batch_features(images_data, batch_size=TF_BATCH_SIZE):
    """Extract features for a list of (path, image_bytes) tuples"""
    results = []
    total = len(images_data)
    
    for i in range(0, total, batch_size):
        batch_data = images_data[i:i+batch_size]
        
        # Preprocess batch
        batch_arrays = [preprocess_image(data[1]) for data in batch_data]
        batch_input = np.stack(batch_arrays)
        batch_input = preprocess_input(batch_input)
        
        # Extract features
        features = model.predict(batch_input, verbose=0)
        
        # Add to results
        for j, (path, _) in enumerate(batch_data):
            results.append((path, features[j].tolist()))
        
        # Progress
        if (i + batch_size) % 500 == 0 or (i + batch_size) >= total:
            print(f"      TF processed: {min(i+batch_size, total)}/{total}")
        
        # Memory cleanup every few batches
        if (i // batch_size) % 10 == 0:
            gc.collect()
    
    return results

def check_batch_exists(output_path):
    """Check if batch already exists in HDFS"""
    try:
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jvm.java.net.URI(HDFS_BASE),
            hadoop_conf
        )
        path = spark._jvm.org.apache.hadoop.fs.Path(output_path)
        return fs.exists(path)
    except:
        return False

# ============================================================================
# MAIN EXTRACTION
# ============================================================================

print("\n" + "="*80)
print("🚀 FEATURE EXTRACTION - 120K IMAGES (MEMORY SAFE)")
print("="*80)

print(f"""
📋 PLAN:
   ├── Train REAL: 50,000 (25 batches x 2,000)
   ├── Train FAKE: 50,000 (25 batches x 2,000)
   ├── Test REAL:  10,000 (5 batches x 2,000)
   └── Test FAKE:  10,000 (5 batches x 2,000)
   Total: 120,000 images
   Model: MobileNetV2 (1280-dim features)
   Batch size: {BATCH_SIZE} (safe for memory)
""")

start_time = time.time()
total_processed = 0
total_errors = 0

for ds in DATASETS:
    split = ds["split"]
    cls = ds["class"]
    label = ds["label"]
    num_batches = ds["batches"]
    start_batch = ds["start_batch"]
    
    input_path = f"{HDFS_BASE}/raw/{split}/{cls}"
    
    print(f"\n{'='*60}")
    print(f"📂 Processing: {input_path}")
    print(f"   Label: {label} ({cls})")
    print(f"   Batches: {num_batches} x {BATCH_SIZE} images")
    print("="*60)
    
    # List all files
    try:
        files_df = spark.read.format("binaryFile") \
            .option("pathGlobFilter", "*.jpg") \
            .option("recursiveFileLookup", "false") \
            .load(input_path) \
            .select("path", "content")
        
        all_paths = [row.path for row in files_df.select("path").collect()]
        total_available = len(all_paths)
        print(f"   📊 Total available: {total_available}")
        
    except Exception as e:
        print(f"   ❌ Error listing files: {e}")
        continue
    
    # Process each batch
    for batch_num in range(start_batch, num_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = min(batch_num * BATCH_SIZE, total_available)
        
        if start_idx >= total_available:
            print(f"\n   ⚠️ Batch {batch_num}: No more images (only {total_available} available)")
            break
        
        output_path = f"{HDFS_BASE}/features_tf/{split}/{cls}/batch_{batch_num}"
        
        # Check if batch exists
        if check_batch_exists(output_path):
            print(f"\n   ✅ Batch {batch_num}/{num_batches}: Already exists, skipping")
            continue
        
        print(f"\n   🔄 Batch {batch_num}/{num_batches}")
        print(f"      Range: {start_idx + 1} - {end_idx}")
        
        batch_start = time.time()
        
        try:
            # Get batch paths
            batch_paths = all_paths[start_idx:end_idx]
            
            # Load images for this batch only
            print(f"      Loading {len(batch_paths)} images to driver...")
            batch_df = files_df.filter(col("path").isin(batch_paths))
            images_data = [(row.path, bytes(row.content)) for row in batch_df.collect()]
            print(f"      Loaded {len(images_data)} images")
            
            # Extract features
            print(f"      Extracting features (TF batch mode)...")
            features_data = extract_batch_features(images_data)
            
            # Create DataFrame
            rows = [(path, Vectors.dense(feat), label) for path, feat in features_data]
            
            from pyspark.sql.types import StructType, StructField, StringType, IntegerType
            schema = StructType([
                StructField("path", StringType(), False),
                StructField("features", VectorUDT(), False),
                StructField("label", IntegerType(), False)
            ])
            
            features_df = spark.createDataFrame(rows, schema)
            
            # Save to HDFS
            print(f"      Saving to {output_path}...")
            features_df.write.mode("overwrite").parquet(output_path)
            
            batch_time = time.time() - batch_start
            total_processed += len(features_data)
            
            print(f"      ✅ Done! {len(features_data)} images in {batch_time:.1f}s")
            print(f"      📈 Total processed: {total_processed}")
            
            # Cleanup
            del images_data, features_data, rows, features_df, batch_df
            gc.collect()
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            total_errors += 1
            gc.collect()
            continue

# ============================================================================
# SUMMARY
# ============================================================================

total_time = time.time() - start_time

print("\n" + "="*80)
print("📊 EXTRACTION COMPLETE!")
print("="*80)
print(f"""
   ✅ Total processed: {total_processed} images
   ❌ Errors: {total_errors}
   ⏱️  Total time: {total_time/60:.1f} minutes
   📁 Output: {HDFS_BASE}/features_tf/
""")

# Verify output
print("\n📂 Verifying output structure...")
try:
    import subprocess
    result = subprocess.run(
        ["hdfs", "dfs", "-ls", "-R", f"{HDFS_BASE}/features_tf/"],
        capture_output=True, text=True
    )
    print(result.stdout)
except:
    print("   (Run 'hdfs dfs -ls -R /user/data/features_tf/' to verify)")

spark.stop()
print("\n✅ Spark stopped. Feature extraction complete!")
