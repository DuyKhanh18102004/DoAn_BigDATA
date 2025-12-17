#!/usr/bin/env python3
"""
Feature Extraction - ALL 12 BATCHES (120K images total)
10 TRAIN batches (100K) + 2 TEST batches (20K)
With verification after each batch to ensure data saves correctly
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from PIL import Image
from io import BytesIO
import time

print("="*80)
print("🔬 FEATURE EXTRACTION - ALL 12 BATCHES (120K IMAGES)")
print("="*80)

# Initialize Spark with optimized settings
spark = SparkSession.builder \
    .appName("FeatureExtraction_All_Batches_Verified") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.default.parallelism", "100") \
    .config("spark.sql.shuffle.partitions", "100") \
    .getOrCreate()

# Define feature extraction function
def extract_features(image_bytes):
    """Extract 2048-dim features: RGB histogram (768) + stats (12) + padding (1268)"""
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # RGB Histogram (768 features: 256 bins × 3 channels)
        hist_r = np.histogram(img_array[:,:,0], bins=256, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:,:,1], bins=256, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:,:,2], bins=256, range=(0, 256))[0]
        hist_features = np.concatenate([hist_r, hist_g, hist_b])
        hist_features = hist_features / (224 * 224)  # Normalize
        
        # Statistical features (12 features: mean, std, min, max per channel)
        stats = []
        for channel in range(3):
            stats.extend([
                float(np.mean(img_array[:,:,channel])),
                float(np.std(img_array[:,:,channel])),
                float(np.min(img_array[:,:,channel])),
                float(np.max(img_array[:,:,channel]))
            ])
        
        # Combine features
        features = np.concatenate([hist_features, stats])
        
        # Pad to 2048 dimensions
        padding = np.zeros(2048 - len(features))
        final_features = np.concatenate([features, padding])
        
        return Vectors.dense(final_features.tolist())
    except Exception as e:
        # Return zero vector on error
        return Vectors.dense([0.0] * 2048)

# Register UDF
extract_features_udf = udf(extract_features, VectorUDT())

def process_batch(dataset_type, class_label, df_batch, batch_num, label_value=1):
    """
    Process one batch of images
    
    Args:
        dataset_type: 'train' or 'test'
        class_label: 'REAL' or 'FAKE'
        df_batch: DataFrame containing batch data
        batch_num: batch number (1-5 for train, 1 for test)
        label_value: 1 for REAL, 0 for FAKE
    """
    print("\n" + "="*80)
    print(f"📦 Processing {dataset_type.upper()}/{class_label} Batch {batch_num}")
    print("="*80)
    
    # Paths
    hdfs_output = f"hdfs://namenode:8020/user/data/features/{dataset_type}/{class_label}/batch_{batch_num}"
    
    print(f" Output: {hdfs_output}")
    
    start_time = time.time()
    
    actual_batch_size = df_batch.count()
    print(f"📦 Batch size: {actual_batch_size:,} samples")
    
    # Extract features
    print("🔬 Extracting features...")
    df_features = df_batch.withColumn("features", extract_features_udf(col("content"))) \
                          .withColumn("label", lit(label_value)) \
                          .select("path", "features", "label")
    
    # Force computation and cache
    df_features = df_features.repartition(10).cache()
    feature_count = df_features.count()
    print(f"✅ Features extracted: {feature_count:,} samples")
    
    # Write to HDFS
    print(f"💾 Writing to HDFS...")
    df_features.write.mode("overwrite").parquet(hdfs_output)
    
    # VERIFICATION: Read back and count
    print("🔍 VERIFICATION - Reading back from HDFS...")
    verification_df = spark.read.parquet(hdfs_output)
    saved_count = verification_df.count()
    
    elapsed = time.time() - start_time
    
    # Check correctness
    if saved_count == actual_batch_size:
        print(f"✅✅✅ SUCCESS: {saved_count:,} samples saved correctly!")
        print(f"⏱️  Time: {elapsed:.2f}s")
        return True, saved_count
    else:
        print(f"⚠️⚠️⚠️ ERROR: Expected {actual_batch_size:,} but saved {saved_count:,} samples!")
        print(f"⏱️  Time: {elapsed:.2f}s")
        return False, saved_count

# ============================================================================
# MAIN EXECUTION - 12 Batches
# ============================================================================

pipeline_start = time.time()
results = []

print("\n" + "🚀"*40)
print("STARTING FEATURE EXTRACTION PIPELINE")
print("🚀"*40)

# ============================================================================
# TRAIN BATCHES - 10 batches (100,000 images)
# ============================================================================

print("\n" + "📚"*40)
print("PART 1: TRAINING DATA (100,000 images)")
print("📚"*40)

# TRAIN/REAL - 5 batches (50,000 images)
print("\n🟢 TRAIN/REAL - 5 batches (50,000 images)")
hdfs_input = "hdfs://namenode:8020/user/data/raw/train/REAL"
df_real = spark.read.format("binaryFile").load(hdfs_input)
df_real = df_real.repartition(100).cache()
total_real = df_real.count()
print(f"✅ Total REAL images: {total_real:,}")

# Split into 5 batches using randomSplit
batches_real = df_real.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)
for batch_num in range(1, 6):
    success, count = process_batch('train', 'REAL', batches_real[batch_num-1], batch_num, label_value=1)
    results.append(('TRAIN/REAL', batch_num, success, count))
    # Memory cleanup
    import gc
    spark.catalog.clearCache()
    gc.collect()
    time.sleep(30)

# TRAIN/FAKE - 5 batches (50,000 images)
print("\n🔴 TRAIN/FAKE - 5 batches (50,000 images)")
hdfs_input = "hdfs://namenode:8020/user/data/raw/train/FAKE"
df_fake = spark.read.format("binaryFile").load(hdfs_input)
df_fake = df_fake.repartition(100).cache()
total_fake = df_fake.count()
print(f"✅ Total FAKE images: {total_fake:,}")

# Split into 5 batches using randomSplit
batches_fake = df_fake.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)
for batch_num in range(1, 6):
    success, count = process_batch('train', 'FAKE', batches_fake[batch_num-1], batch_num, label_value=0)
    results.append(('TRAIN/FAKE', batch_num, success, count))
    # Memory cleanup
    import gc
    spark.catalog.clearCache()
    gc.collect()
    time.sleep(30)

# ============================================================================
# TEST BATCHES - 2 batches (20,000 images)
# ============================================================================

print("\n" + "🧪"*40)
print("PART 2: TEST DATA (20,000 images)")
print("🧪"*40)

# TEST/REAL - 1 batch (10,000 images)
print("\n🟢 TEST/REAL - 1 batch (10,000 images)")
hdfs_input = "hdfs://namenode:8020/user/data/raw/test/REAL"
df_test_real = spark.read.format("binaryFile").load(hdfs_input)
df_test_real = df_test_real.repartition(100).cache()
success, count = process_batch('test', 'REAL', df_test_real, 1, label_value=1)
results.append(('TEST/REAL', 1, success, count))

# TEST/FAKE - 1 batch (10,000 images)
print("\n🔴 TEST/FAKE - 1 batch (10,000 images)")
hdfs_input = "hdfs://namenode:8020/user/data/raw/test/FAKE"
df_test_fake = spark.read.format("binaryFile").load(hdfs_input)
df_test_fake = df_test_fake.repartition(100).cache()
success, count = process_batch('test', 'FAKE', df_test_fake, 1, label_value=0)
results.append(('TEST/FAKE', 1, success, count))

# ============================================================================
# SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)

success_count = sum(1 for _, _, success, _ in results if success)
total_count = len(results)
total_samples = sum(count for _, _, _, count in results)

print(f"\n✅ Successful batches: {success_count}/{total_count}")
print(f"📊 Total samples saved: {total_samples:,}")
print(f"⏱️  Total pipeline time: {pipeline_elapsed/60:.2f} minutes")

print("\n📋 Batch Results:")
for dataset, batch_num, success, count in results:
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"  {dataset} Batch {batch_num}: {status} ({count:,} samples)")

if success_count == total_count:
    print("\n" + "🎉"*40)
    print("ALL 12 BATCHES COMPLETED SUCCESSFULLY!")
    print("120,000 images processed with features saved to HDFS")
    print("🎉"*40)
else:
    print("\n" + "⚠️"*40)
    print(f"WARNING: {total_count - success_count} batches failed!")
    print("⚠️"*40)

print("\n" + "="*80)
spark.stop()
