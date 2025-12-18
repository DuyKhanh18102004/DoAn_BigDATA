#!/usr/bin/env python3
"""
Feature Extraction với ResNet50 Pre-trained (ImageNet)
- Sử dụng ResNet50 để trích xuất 2048-dim features
- Deep learning features cho accuracy cao hơn
- Xử lý theo batch để tránh OOM
- Tích hợp Spark History Server
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, monotonically_increasing_id
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from PIL import Image
from io import BytesIO
import time
import gc
import os

print("="*80)
print("🔬 FEATURE EXTRACTION - RESNET50 (Deep Learning Features)")
print("="*80)

# ============================================================================
# SPARK SESSION với History Server
# ============================================================================

spark = SparkSession.builder \
    .appName("FeatureExtraction_ResNet50") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.default.parallelism", "20") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.driver.maxResultSize", "512m") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .config("spark.history.fs.logDirectory", "hdfs://namenode:8020/spark-logs") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.cleaner.periodicGC.interval", "5min") \
    .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ============================================================================
# RESNET50 FEATURE EXTRACTOR
# ============================================================================

# Global model (loaded once per executor)
_resnet_model = None

def get_resnet_model():
    """Load ResNet50 model (lazy loading, once per executor)"""
    global _resnet_model
    if _resnet_model is None:
        import torch
        import torchvision.models as models
        from torchvision.models import ResNet50_Weights
        
        # Load pre-trained ResNet50 với weights mới (PyTorch 2.x compatible)
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove final classification layer to get features
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        
        # Move to CPU (không dùng GPU trong Spark)
        model = model.cpu()
        
        _resnet_model = model
        print("✅ ResNet50 model loaded!")
    
    return _resnet_model

def extract_resnet_features(image_bytes):
    """
    Extract 2048-dim features using ResNet50
    
    Input: Raw image bytes
    Output: 2048-dimensional feature vector
    """
    try:
        import torch
        import torchvision.transforms as transforms
        
        # Load and preprocess image
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # ResNet50 preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        
        # Extract features
        model = get_resnet_model()
        with torch.no_grad():
            features = model(img_tensor)
        
        # Flatten to 2048-dim vector
        features = features.squeeze().numpy()
        
        return Vectors.dense(features.tolist())
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Return zero vector on error
        return Vectors.dense([0.0] * 2048)

# Register UDF
extract_features_udf = udf(extract_resnet_features, VectorUDT())

# ============================================================================
# BATCH PROCESSING FUNCTION với Memory Management
# ============================================================================

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    import gc
    # Clear all Spark caches
    spark.catalog.clearCache()
    # Force Python garbage collection multiple times
    for _ in range(3):
        gc.collect()
    # Sleep để cho GC hoàn thành
    time.sleep(2)
    print("🧹 Memory cleanup completed")

def process_batch(dataset_type, class_label, df_batch, batch_num, label_value):
    """Process one batch of images with ResNet50 - Memory Optimized"""
    
    print(f"\n{'='*80}")
    print(f"📦 Processing {dataset_type.upper()}/{class_label} Batch {batch_num}")
    print(f"{'='*80}")
    
    hdfs_output = f"hdfs://namenode:8020/user/data/features/{dataset_type}/{class_label}/batch_{batch_num}"
    
    start_time = time.time()
    
    # Count
    batch_size = df_batch.count()
    print(f"📊 Batch size: {batch_size:,} images")
    
    # Extract features - KHÔNG cache để tiết kiệm memory
    print("🔬 Extracting ResNet50 features...")
    df_features = df_batch.withColumn("features", extract_features_udf(col("content"))) \
                          .withColumn("label", lit(label_value)) \
                          .select("path", "features", "label")
    
    # Repartition nhỏ hơn để giảm memory pressure
    df_features = df_features.repartition(5)
    
    # Save trực tiếp không cache
    print(f"💾 Saving to HDFS: {hdfs_output}")
    df_features.write.mode("overwrite").parquet(hdfs_output)
    
    # Verify
    saved_count = spark.read.parquet(hdfs_output).count()
    
    elapsed = time.time() - start_time
    
    # AGGRESSIVE CLEANUP
    del df_features
    force_memory_cleanup()
    
    # Cooldown sau mỗi batch
    print(f"⏳ Cooldown 15s để hệ thống ổn định...")
    time.sleep(15)
    
    if saved_count == batch_size:
        print(f"✅ SUCCESS: {saved_count:,} samples saved! Time: {elapsed:.1f}s")
        return True, saved_count
    else:
        print(f"⚠️ WARNING: Expected {batch_size:,} but saved {saved_count:,}")
        return saved_count > 0, saved_count

# ============================================================================
# MAIN PIPELINE
# ============================================================================

pipeline_start = time.time()
results = []

print("\n" + "🚀"*40)
print("STARTING RESNET50 FEATURE EXTRACTION")
print("🚀"*40)

# ============================================================================
# TRAIN DATA - 100,000 images (50K REAL + 50K FAKE)
# ============================================================================

print("\n" + "📚"*40)
print("PART 1: TRAINING DATA (100,000 images)")
print("📚"*40)

# TRAIN/REAL - 5 batches
print("\n🟢 TRAIN/REAL - Loading 50,000 images...")
hdfs_real = "hdfs://namenode:8020/user/data/raw/train/REAL"
df_real = spark.read.format("binaryFile").load(hdfs_real)
df_real = df_real.repartition(25)  # Giảm partitions
total_real = df_real.count()
print(f"✅ Total REAL: {total_real:,}")

# Split into 5 batches - KHÔNG cache df_real
batches_real = df_real.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)

for i in range(5):
    print(f"\n{'🔄'*20} BATCH {i+1}/5 {'🔄'*20}")
    success, count = process_batch('train', 'REAL', batches_real[i], i+1, label_value=1)
    results.append(('TRAIN/REAL', i+1, success, count))
    # Force cleanup sau mỗi batch
    batches_real[i] = None  # Release reference
    force_memory_cleanup()

# Release tất cả references
del batches_real
del df_real
force_memory_cleanup()
print("\n✅ TRAIN/REAL completed - Memory released")

# TRAIN/FAKE - 5 batches
print("\n🔴 TRAIN/FAKE - Loading 50,000 images...")
hdfs_fake = "hdfs://namenode:8020/user/data/raw/train/FAKE"
df_fake = spark.read.format("binaryFile").load(hdfs_fake)
df_fake = df_fake.repartition(25)  # Giảm partitions
total_fake = df_fake.count()
print(f"✅ Total FAKE: {total_fake:,}")

# Split into 5 batches - KHÔNG cache df_fake
batches_fake = df_fake.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)

for i in range(5):
    print(f"\n{'🔄'*20} BATCH {i+1}/5 {'🔄'*20}")
    success, count = process_batch('train', 'FAKE', batches_fake[i], i+1, label_value=0)
    results.append(('TRAIN/FAKE', i+1, success, count))
    # Force cleanup sau mỗi batch
    batches_fake[i] = None  # Release reference
    force_memory_cleanup()

# Release tất cả references
del batches_fake
del df_fake
force_memory_cleanup()
print("\n✅ TRAIN/FAKE completed - Memory released")

# ============================================================================
# TEST DATA - 20,000 images (10K REAL + 10K FAKE)
# ============================================================================

print("\n" + "🧪"*40)
print("PART 2: TEST DATA (20,000 images)")
print("🧪"*40)

# TEST/REAL
print("\n🟢 TEST/REAL - Loading 10,000 images...")
hdfs_test_real = "hdfs://namenode:8020/user/data/raw/test/REAL"
df_test_real = spark.read.format("binaryFile").load(hdfs_test_real)
df_test_real = df_test_real.repartition(10)  # Giảm partitions
success, count = process_batch('test', 'REAL', df_test_real, 1, label_value=1)
results.append(('TEST/REAL', 1, success, count))
del df_test_real
force_memory_cleanup()

# TEST/FAKE
print("\n🔴 TEST/FAKE - Loading 10,000 images...")
hdfs_test_fake = "hdfs://namenode:8020/user/data/raw/test/FAKE"
df_test_fake = spark.read.format("binaryFile").load(hdfs_test_fake)
df_test_fake = df_test_fake.repartition(10)  # Giảm partitions
success, count = process_batch('test', 'FAKE', df_test_fake, 1, label_value=0)
results.append(('TEST/FAKE', 1, success, count))
del df_test_fake
force_memory_cleanup()

# ============================================================================
# SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("📊 FINAL SUMMARY - RESNET50 FEATURE EXTRACTION")
print("="*80)

success_count = sum(1 for _, _, s, _ in results if s)
total_batches = len(results)
total_samples = sum(c for _, _, _, c in results)

print(f"\n✅ Successful: {success_count}/{total_batches} batches")
print(f"📊 Total samples: {total_samples:,}")
print(f"⏱️  Total time: {pipeline_elapsed/60:.2f} minutes")

print("\n📋 Results:")
for dataset, batch, success, count in results:
    status = "✅" if success else "❌"
    print(f"   {status} {dataset} Batch {batch}: {count:,} samples")

if success_count == total_batches:
    print("\n" + "🎉"*40)
    print("ALL BATCHES COMPLETED WITH RESNET50 FEATURES!")
    print(f"120,000 images → 2048-dim deep learning features")
    print("🎉"*40)

spark.stop()
print("\n✅ Done!")
