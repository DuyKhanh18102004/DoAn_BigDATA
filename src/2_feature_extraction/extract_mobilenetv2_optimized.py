#!/usr/bin/env python3
"""
Feature Extraction với TensorFlow MobileNetV2 - MEMORY OPTIMIZED
- Clear memory sau mỗi batch với đo lường chi tiết
- Lưu vào hdfs://namenode:8020/user/data/features_tf/
- Batch size nhỏ để tránh OOM
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import numpy as np
from PIL import Image
from io import BytesIO
import time
import gc
import os
import sys

# ============================================================================
# MEMORY MONITORING UTILITIES
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'rss': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    except ImportError:
        return {'rss': 0, 'vms': 0, 'percent': 0}

def get_system_memory():
    """Get system memory info"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total': mem.total / 1024 / 1024 / 1024,  # GB
            'available': mem.available / 1024 / 1024 / 1024,  # GB
            'used': mem.used / 1024 / 1024 / 1024,  # GB
            'percent': mem.percent
        }
    except ImportError:
        return {'total': 0, 'available': 0, 'used': 0, 'percent': 0}

def print_memory_status(label=""):
    """Print current memory status"""
    proc_mem = get_memory_usage()
    sys_mem = get_system_memory()
    print(f"   💾 Memory [{label}]:")
    print(f"      Process: RSS={proc_mem['rss']:.1f}MB, VMS={proc_mem['vms']:.1f}MB ({proc_mem['percent']:.1f}%)")
    print(f"      System:  Used={sys_mem['used']:.2f}GB / {sys_mem['total']:.2f}GB ({sys_mem['percent']:.1f}%)")
    print(f"      Available: {sys_mem['available']:.2f}GB")
    return proc_mem, sys_mem

def clear_memory_and_report(label=""):
    """Clear memory and report freed amount"""
    print(f"\n   🧹 Clearing memory [{label}]...")
    
    # Get memory before
    before_proc, before_sys = get_memory_usage(), get_system_memory()
    
    # Force garbage collection multiple times
    gc.collect()
    gc.collect()
    gc.collect()
    
    # Clear TensorFlow session if possible
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except:
        pass
    
    # Force garbage collection again
    gc.collect()
    
    # Get memory after
    after_proc, after_sys = get_memory_usage(), get_system_memory()
    
    # Calculate freed memory
    freed_rss = before_proc['rss'] - after_proc['rss']
    freed_sys = after_sys['available'] - before_sys['available']
    
    print(f"   ✅ Memory cleared:")
    print(f"      Process freed: {freed_rss:.1f}MB (RSS: {before_proc['rss']:.1f}MB → {after_proc['rss']:.1f}MB)")
    print(f"      System freed:  {freed_sys*1024:.1f}MB (Available: {before_sys['available']:.2f}GB → {after_sys['available']:.2f}GB)")
    
    return freed_rss, freed_sys

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("="*80)
print("🔬 FEATURE EXTRACTION - TENSORFLOW MOBILENETV2 (MEMORY OPTIMIZED)")
print("="*80)

print_memory_status("STARTUP")

# ============================================================================
# SPARK SESSION - Conservative memory settings
# ============================================================================

spark = SparkSession.builder \
    .appName("FeatureExtraction_MobileNetV2_MemoryOptimized") \
    .config("spark.driver.memory", "3g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.maxResultSize", "1g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("\n✅ Spark Session created")
print_memory_status("AFTER SPARK")

# ============================================================================
# TENSORFLOW MOBILENETV2 MODEL - Lightweight
# ============================================================================

print("\n📦 Loading TensorFlow MobileNetV2 model...")
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Load MobileNetV2 - much smaller than ResNet50
# MobileNetV2: ~14MB, 1280-dim features
# ResNet50: ~98MB, 2048-dim features
model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    pooling='avg',  # Global average pooling -> 1280-dim vector
    input_shape=(224, 224, 3)
)

print(f"   ✅ MobileNetV2 loaded!")
print(f"   📊 Output shape: {model.output_shape} (1280 features)")
print(f"   📦 Parameters: {model.count_params():,}")
print_memory_status("AFTER MODEL LOAD")

# ============================================================================
# CONFIG
# ============================================================================

BATCH_SIZE = 1000  # Images per HDFS save batch (smaller = safer)
TF_BATCH_SIZE = 8  # Images per TF inference (smaller = less memory)
HDFS_BASE = "hdfs://namenode:8020/user/data"
OUTPUT_BASE = f"{HDFS_BASE}/features_tf"

# Dataset configuration
DATASETS = [
    # Train REAL: 50K -> 50 batches of 1000
    {"split": "train", "class": "REAL", "label": 1, "total": 50000, "batch_size": BATCH_SIZE},
    # Train FAKE: 50K -> 50 batches of 1000
    {"split": "train", "class": "FAKE", "label": 0, "total": 50000, "batch_size": BATCH_SIZE},
    # Test REAL: 10K -> 10 batches of 1000
    {"split": "test", "class": "REAL", "label": 1, "total": 10000, "batch_size": BATCH_SIZE},
    # Test FAKE: 10K -> 10 batches of 1000
    {"split": "test", "class": "FAKE", "label": 0, "total": 10000, "batch_size": BATCH_SIZE},
]

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def preprocess_image(image_bytes):
    """Preprocess single image bytes to array"""
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        # Explicitly delete PIL image
        del img
        return arr
    except Exception as e:
        return np.zeros((224, 224, 3), dtype=np.float32)

def extract_features_batch(images_data):
    """
    Extract features for a list of (path, image_bytes) tuples
    With aggressive memory cleanup
    """
    results = []
    total = len(images_data)
    
    for i in range(0, total, TF_BATCH_SIZE):
        batch_data = images_data[i:i+TF_BATCH_SIZE]
        
        # Preprocess batch
        batch_arrays = []
        for path, img_bytes in batch_data:
            arr = preprocess_image(img_bytes)
            batch_arrays.append(arr)
        
        batch_input = np.stack(batch_arrays)
        batch_input = preprocess_input(batch_input)
        
        # Clear intermediate arrays
        del batch_arrays
        
        # Extract features
        features = model.predict(batch_input, verbose=0, batch_size=TF_BATCH_SIZE)
        
        # Clear input
        del batch_input
        
        # Store results
        for j, (path, _) in enumerate(batch_data):
            results.append((path, features[j].tolist()))
        
        # Clear features array
        del features
        
        # Progress every 200 images
        if (i + TF_BATCH_SIZE) % 200 == 0 or (i + TF_BATCH_SIZE) >= total:
            print(f"      Extracted: {min(i+TF_BATCH_SIZE, total)}/{total}")
            # Mini garbage collection
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
        exists = fs.exists(path)
        # Check if it has _SUCCESS file (completed)
        if exists:
            success_path = spark._jvm.org.apache.hadoop.fs.Path(output_path + "/_SUCCESS")
            return fs.exists(success_path)
        return False
    except Exception as e:
        print(f"      Warning checking path: {e}")
        return False

# ============================================================================
# MAIN EXTRACTION LOOP
# ============================================================================

print("\n" + "="*80)
print("🚀 FEATURE EXTRACTION - 120K IMAGES")
print("="*80)

total_batches = sum(ds["total"] // ds["batch_size"] for ds in DATASETS)
print(f"""
📋 CONFIGURATION:
   ├── Output: {OUTPUT_BASE}
   ├── Batch size: {BATCH_SIZE} images per save
   ├── TF batch size: {TF_BATCH_SIZE} images per inference
   ├── Model: MobileNetV2 (1280-dim features, ~14MB)
   └── Total batches: {total_batches}

📋 DATASETS:
   ├── Train REAL: 50,000 → {50000//BATCH_SIZE} batches
   ├── Train FAKE: 50,000 → {50000//BATCH_SIZE} batches
   ├── Test REAL:  10,000 → {10000//BATCH_SIZE} batches
   └── Test FAKE:  10,000 → {10000//BATCH_SIZE} batches
""")

start_time = time.time()
total_processed = 0
total_skipped = 0
total_errors = 0
batch_counter = 0

for ds in DATASETS:
    split = ds["split"]
    cls = ds["class"]
    label = ds["label"]
    batch_size = ds["batch_size"]
    max_images = ds["total"]
    
    input_path = f"{HDFS_BASE}/raw/{split}/{cls}"
    
    print(f"\n{'='*70}")
    print(f"📂 DATASET: {split}/{cls} (Label={label})")
    print(f"   Input: {input_path}")
    print(f"   Output: {OUTPUT_BASE}/{split}/{cls}/")
    print("="*70)
    
    print_memory_status(f"START {split}/{cls}")
    
    # List all files once
    try:
        print(f"\n   📋 Listing files from HDFS...")
        files_df = spark.read.format("binaryFile") \
            .option("pathGlobFilter", "*.jpg") \
            .option("recursiveFileLookup", "false") \
            .load(input_path) \
            .select("path", "content")
        
        # Get paths only (don't load content yet)
        all_paths = [row.path for row in files_df.select("path").collect()]
        total_available = min(len(all_paths), max_images)
        num_batches = (total_available + batch_size - 1) // batch_size
        
        print(f"   ✅ Found {len(all_paths)} files, will process {total_available}")
        print(f"   📊 Will create {num_batches} batches of {batch_size}")
        
    except Exception as e:
        print(f"   ❌ Error listing files: {e}")
        continue
    
    # Process each batch
    for batch_num in range(1, num_batches + 1):
        batch_counter += 1
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(batch_num * batch_size, total_available)
        
        if start_idx >= total_available:
            break
        
        output_path = f"{OUTPUT_BASE}/{split}/{cls}/batch_{batch_num}"
        
        print(f"\n   {'─'*60}")
        print(f"   🔄 Batch {batch_num}/{num_batches} (Global: {batch_counter}/{total_batches})")
        print(f"      Range: images {start_idx + 1} - {end_idx}")
        print(f"      Output: {output_path}")
        
        # Check if already exists
        if check_batch_exists(output_path):
            print(f"      ⏭️  Already exists, skipping")
            total_skipped += (end_idx - start_idx)
            continue
        
        print_memory_status("BEFORE BATCH")
        batch_start = time.time()
        
        try:
            # Get batch paths
            batch_paths = all_paths[start_idx:end_idx]
            actual_batch_size = len(batch_paths)
            
            # Load images for this batch only
            print(f"      📥 Loading {actual_batch_size} images...")
            batch_df = files_df.filter(col("path").isin(batch_paths))
            
            # Collect to driver
            images_data = []
            for row in batch_df.collect():
                images_data.append((row.path, bytes(row.content)))
            
            print(f"      ✅ Loaded {len(images_data)} images to memory")
            print_memory_status("AFTER LOAD")
            
            # Clear batch_df reference
            del batch_df
            gc.collect()
            
            # Extract features
            print(f"      🔬 Extracting features with MobileNetV2...")
            features_data = extract_features_batch(images_data)
            
            # Clear images data immediately
            del images_data
            gc.collect()
            
            print(f"      ✅ Extracted {len(features_data)} feature vectors")
            print_memory_status("AFTER EXTRACT")
            
            # Create DataFrame
            print(f"      💾 Creating DataFrame...")
            schema = StructType([
                StructField("path", StringType(), False),
                StructField("features", VectorUDT(), False),
                StructField("label", IntegerType(), False)
            ])
            
            rows = [(path, Vectors.dense(feat), label) for path, feat in features_data]
            
            # Clear features_data
            del features_data
            gc.collect()
            
            features_df = spark.createDataFrame(rows, schema)
            
            # Clear rows
            del rows
            gc.collect()
            
            # Save to HDFS
            print(f"      📤 Saving to HDFS...")
            features_df.write.mode("overwrite").parquet(output_path)
            
            # Clear DataFrame
            del features_df
            
            batch_time = time.time() - batch_start
            total_processed += actual_batch_size
            
            print(f"      ✅ Batch complete! {actual_batch_size} images in {batch_time:.1f}s")
            print(f"      📈 Progress: {total_processed + total_skipped}/{sum(ds['total'] for ds in DATASETS)}")
            
            # AGGRESSIVE MEMORY CLEANUP after each batch
            freed_rss, freed_sys = clear_memory_and_report(f"AFTER BATCH {batch_num}")
            
            # Extra delay to let system stabilize
            time.sleep(0.5)
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            total_errors += 1
            
            # Still try to clear memory on error
            clear_memory_and_report("AFTER ERROR")
            continue
    
    # Clear after each dataset
    print(f"\n   🧹 Clearing memory after {split}/{cls} dataset...")
    clear_memory_and_report(f"AFTER DATASET {split}/{cls}")
    
    # Clear files_df
    del files_df, all_paths
    gc.collect()

# ============================================================================
# SUMMARY
# ============================================================================

total_time = time.time() - start_time

print("\n" + "="*80)
print("📊 EXTRACTION COMPLETE!")
print("="*80)
print(f"""
   ✅ Total processed: {total_processed:,} images
   ⏭️  Total skipped:   {total_skipped:,} images (already existed)
   ❌ Total errors:    {total_errors}
   ⏱️  Total time:      {total_time/60:.1f} minutes
   📁 Output:          {OUTPUT_BASE}/
""")

print_memory_status("FINAL")

# Verify output structure
print("\n📂 Verifying output structure...")
try:
    verify_df = spark.read.parquet(f"{OUTPUT_BASE}/train/REAL/batch_1")
    sample = verify_df.first()
    if sample:
        features = sample.features.toArray()
        print(f"   ✅ Sample verification:")
        print(f"      Path: {sample.path[:50]}...")
        print(f"      Features dim: {len(features)}")
        print(f"      Features sum: {np.sum(features):.4f}")
        print(f"      Features max: {np.max(features):.4f}")
        print(f"      Features non-zero: {np.count_nonzero(features)}/{len(features)}")
        
        if np.sum(features) > 0:
            print(f"   ✅ Features are VALID (non-zero)!")
        else:
            print(f"   ❌ WARNING: Features are all zeros!")
except Exception as e:
    print(f"   ⚠️  Could not verify: {e}")

spark.stop()
print("\n✅ Spark stopped. Feature extraction complete!")
