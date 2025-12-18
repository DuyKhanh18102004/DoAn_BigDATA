#!/usr/bin/env python3
"""
Feature Extraction với MobileNetV2 Pre-trained (ImageNet) - Memory Optimized
- Sử dụng MobileNetV2 (nhẹ hơn ResNet50 ~7x) để trích xuất 1280-dim features
- Memory tracking và cleanup sau mỗi batch
- Đo lường và in ra resource đã giải phóng
- Tích hợp Spark History Server
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from PIL import Image
from io import BytesIO
import time
import gc
import os
import sys
import psutil

print("="*80)
print("🔬 FEATURE EXTRACTION - MOBILENETV2 (Lightweight Deep Learning)")
print("="*80)

# ============================================================================
# MEMORY TRACKING UTILITIES
# ============================================================================

def get_process_memory_mb():
    """Get current process memory in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_system_memory_info():
    """Get system memory info"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / 1024**3,
        'available_gb': mem.available / 1024**3,
        'used_gb': mem.used / 1024**3,
        'percent': mem.percent
    }

def print_memory_status(prefix=""):
    """Print current memory status"""
    proc_mem = get_process_memory_mb()
    sys_mem = get_system_memory_info()
    print(f"{prefix}📊 Process Memory: {proc_mem:.1f} MB")
    print(f"{prefix}📊 System Memory: {sys_mem['used_gb']:.2f}/{sys_mem['total_gb']:.2f} GB ({sys_mem['percent']:.1f}% used)")
    print(f"{prefix}📊 Available: {sys_mem['available_gb']:.2f} GB")
    return proc_mem, sys_mem

# ============================================================================
# SPARK SESSION với History Server - Memory Optimized
# ============================================================================

spark = SparkSession.builder \
    .appName("FeatureExtraction_MobileNetV2") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.default.parallelism", "16") \
    .config("spark.sql.shuffle.partitions", "16") \
    .config("spark.driver.maxResultSize", "512m") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .config("spark.history.fs.logDirectory", "hdfs://namenode:8020/spark-logs") \
    .config("spark.memory.fraction", "0.5") \
    .config("spark.memory.storageFraction", "0.2") \
    .config("spark.cleaner.periodicGC.interval", "1min") \
    .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.rpc.message.maxSize", "256") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n✅ Spark Session created")
print_memory_status("   ")

# ============================================================================
# MOBILENETV2 FEATURE EXTRACTOR
# ============================================================================

# Global model (loaded once per executor, cleared after batch)
_mobilenet_model = None
_model_loaded_time = None

def get_mobilenet_model():
    """Load MobileNetV2 model (lazy loading, once per executor)"""
    global _mobilenet_model, _model_loaded_time
    if _mobilenet_model is None:
        import torch
        import torchvision.models as models
        from torchvision.models import MobileNet_V2_Weights
        
        print("   🔄 Loading MobileNetV2 model...")
        mem_before = get_process_memory_mb()
        
        # Load pre-trained MobileNetV2 (~14MB vs ResNet50 ~98MB)
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Remove final classification layer to get features
        # MobileNetV2 outputs 1280-dim features (vs ResNet50 2048-dim)
        model.classifier = torch.nn.Identity()
        model.eval()
        
        # Move to CPU
        model = model.cpu()
        
        _mobilenet_model = model
        _model_loaded_time = time.time()
        
        mem_after = get_process_memory_mb()
        print(f"   ✅ MobileNetV2 loaded! Memory used: {mem_after - mem_before:.1f} MB")
    
    return _mobilenet_model

def clear_model():
    """Clear the model from memory"""
    global _mobilenet_model, _model_loaded_time
    if _mobilenet_model is not None:
        import torch
        mem_before = get_process_memory_mb()
        
        del _mobilenet_model
        _mobilenet_model = None
        _model_loaded_time = None
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        mem_after = get_process_memory_mb()
        print(f"   🗑️ Model cleared! Memory freed: {mem_before - mem_after:.1f} MB")

def extract_mobilenet_features(image_bytes):
    """
    Extract 1280-dim features using MobileNetV2
    
    Input: Raw image bytes
    Output: 1280-dimensional feature vector
    """
    try:
        import torch
        import torchvision.transforms as transforms
        
        # Load and preprocess image
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # MobileNetV2 preprocessing (same as ImageNet)
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
        model = get_mobilenet_model()
        with torch.no_grad():
            features = model(img_tensor)
        
        # Flatten to 1280-dim vector
        features = features.squeeze().numpy()
        
        # Cleanup image tensor
        del img_tensor, img
        
        return Vectors.dense(features.tolist())
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Return zero vector on error (1280 dims for MobileNetV2)
        return Vectors.dense([0.0] * 1280)

# Register UDF
extract_features_udf = udf(extract_mobilenet_features, VectorUDT())

# ============================================================================
# AGGRESSIVE MEMORY CLEANUP
# ============================================================================

def force_memory_cleanup(clear_model_flag=False):
    """
    Force aggressive memory cleanup with measurement
    
    Args:
        clear_model_flag: If True, also clear the loaded model
    
    Returns:
        Tuple of (memory_before, memory_after, memory_freed)
    """
    print("\n   🧹 Starting memory cleanup...")
    mem_before = get_process_memory_mb()
    sys_before = get_system_memory_info()
    
    # 1. Clear Spark caches
    spark.catalog.clearCache()
    
    # 2. Clear model if requested
    if clear_model_flag:
        clear_model()
    
    # 3. Force Python garbage collection multiple times
    for i in range(5):
        collected = gc.collect()
        if collected == 0:
            break
    
    # 4. Clear any lingering PyTorch tensors
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Clear CPU tensor cache by creating and deleting small tensor
        _ = torch.zeros(1)
        del _
    except:
        pass
    
    # 5. Sleep to let GC complete
    time.sleep(3)
    
    # Final GC
    gc.collect()
    
    mem_after = get_process_memory_mb()
    sys_after = get_system_memory_info()
    
    memory_freed = mem_before - mem_after
    system_freed = sys_before['used_gb'] - sys_after['used_gb']
    
    print(f"   ✅ Memory cleanup completed!")
    print(f"   📉 Process Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (freed: {memory_freed:.1f} MB)")
    print(f"   📉 System Memory: {sys_before['used_gb']:.2f} GB → {sys_after['used_gb']:.2f} GB (freed: {system_freed*1024:.1f} MB)")
    
    return mem_before, mem_after, memory_freed

# ============================================================================
# BATCH PROCESSING FUNCTION với Memory Management
# ============================================================================

def process_batch(dataset_type, class_label, df_batch, batch_num, label_value, total_batches):
    """Process one batch of images with MobileNetV2 - Memory Optimized"""
    
    print(f"\n{'='*80}")
    print(f"📦 Processing {dataset_type.upper()}/{class_label} Batch {batch_num}/{total_batches}")
    print(f"{'='*80}")
    
    # Memory status before processing
    print("\n📊 MEMORY STATUS BEFORE BATCH:")
    mem_start, sys_start = print_memory_status("   ")
    
    hdfs_output = f"hdfs://namenode:8020/user/data/features/{dataset_type}/{class_label}/batch_{batch_num}"
    
    start_time = time.time()
    
    try:
        # Count
        batch_size = df_batch.count()
        print(f"\n📊 Batch size: {batch_size:,} images")
        
        # Extract features - KHÔNG cache để tiết kiệm memory
        print("🔬 Extracting MobileNetV2 features (1280-dim)...")
        df_features = df_batch.withColumn("features", extract_features_udf(col("content"))) \
                              .withColumn("label", lit(label_value)) \
                              .select("path", "features", "label")
        
        # Repartition nhỏ để giảm memory pressure
        df_features = df_features.repartition(4)
        
        # Save trực tiếp không cache
        print(f"💾 Saving to HDFS: {hdfs_output}")
        df_features.write.mode("overwrite").parquet(hdfs_output)
        
        # Verify
        saved_count = spark.read.parquet(hdfs_output).count()
        
        elapsed = time.time() - start_time
        
        # Release DataFrame references
        df_features.unpersist()
        del df_features
        
        # Memory status after processing
        print("\n📊 MEMORY STATUS AFTER BATCH:")
        mem_end, sys_end = print_memory_status("   ")
        
        # Calculate memory used during batch
        mem_used = mem_end - mem_start
        print(f"\n📈 Memory used for this batch: {mem_used:.1f} MB")
        
        # AGGRESSIVE CLEANUP - Clear model every batch to release memory
        print("\n🧹 PERFORMING AGGRESSIVE CLEANUP...")
        mem_before_cleanup, mem_after_cleanup, mem_freed = force_memory_cleanup(clear_model_flag=True)
        
        # Cooldown với adaptive time based on memory freed
        cooldown_time = max(10, min(30, int(mem_used / 100) * 5))
        print(f"\n⏳ Cooldown {cooldown_time}s để hệ thống ổn định...")
        time.sleep(cooldown_time)
        
        # Final memory check
        print("\n📊 FINAL MEMORY STATUS:")
        print_memory_status("   ")
        
        if saved_count == batch_size:
            print(f"\n✅ SUCCESS: {saved_count:,} samples saved! Time: {elapsed:.1f}s")
            return True, saved_count
        else:
            print(f"\n⚠️ WARNING: Expected {batch_size:,} but saved {saved_count:,}")
            return saved_count > 0, saved_count
            
    except Exception as e:
        print(f"\n❌ ERROR processing batch: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        force_memory_cleanup(clear_model_flag=True)
        return False, 0

# ============================================================================
# MAIN PIPELINE
# ============================================================================

pipeline_start = time.time()
results = []

print("\n" + "🚀"*40)
print("STARTING MOBILENETV2 FEATURE EXTRACTION")
print("Model: MobileNetV2 (~14MB) - Output: 1280-dim features")
print("🚀"*40)

print("\n📊 INITIAL SYSTEM STATUS:")
print_memory_status("   ")

# ============================================================================
# TRAIN DATA - 100,000 images (50K REAL + 50K FAKE)
# ============================================================================

print("\n" + "📚"*40)
print("PART 1: TRAINING DATA (100,000 images)")
print("📚"*40)

# TRAIN/REAL - 5 batches
print("\n🟢 TRAIN/REAL - Loading 50,000 images...")
hdfs_real = "hdfs://namenode:8020/user/data/raw/train/REAL"

try:
    df_real = spark.read.format("binaryFile").load(hdfs_real)
    df_real = df_real.repartition(20)  # Giảm partitions
    total_real = df_real.count()
    print(f"✅ Total REAL: {total_real:,}")
    
    # Split into 5 batches - KHÔNG cache df_real
    batches_real = df_real.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)
    
    for i in range(5):
        print(f"\n{'🔄'*20} TRAIN/REAL BATCH {i+1}/5 {'🔄'*20}")
        success, count = process_batch('train', 'REAL', batches_real[i], i+1, label_value=1, total_batches=5)
        results.append(('TRAIN/REAL', i+1, success, count))
        
        # Release reference immediately
        batches_real[i] = None
        gc.collect()
    
    # Release all references
    del batches_real
    del df_real
    force_memory_cleanup(clear_model_flag=True)
    print("\n✅ TRAIN/REAL completed - Memory released")
    
except Exception as e:
    print(f"❌ Error processing TRAIN/REAL: {e}")
    import traceback
    traceback.print_exc()

# TRAIN/FAKE - 5 batches
print("\n🔴 TRAIN/FAKE - Loading 50,000 images...")
hdfs_fake = "hdfs://namenode:8020/user/data/raw/train/FAKE"

try:
    df_fake = spark.read.format("binaryFile").load(hdfs_fake)
    df_fake = df_fake.repartition(20)  # Giảm partitions
    total_fake = df_fake.count()
    print(f"✅ Total FAKE: {total_fake:,}")
    
    # Split into 5 batches - KHÔNG cache df_fake
    batches_fake = df_fake.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)
    
    for i in range(5):
        print(f"\n{'🔄'*20} TRAIN/FAKE BATCH {i+1}/5 {'🔄'*20}")
        success, count = process_batch('train', 'FAKE', batches_fake[i], i+1, label_value=0, total_batches=5)
        results.append(('TRAIN/FAKE', i+1, success, count))
        
        # Release reference immediately
        batches_fake[i] = None
        gc.collect()
    
    # Release all references
    del batches_fake
    del df_fake
    force_memory_cleanup(clear_model_flag=True)
    print("\n✅ TRAIN/FAKE completed - Memory released")
    
except Exception as e:
    print(f"❌ Error processing TRAIN/FAKE: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST DATA - 20,000 images (10K REAL + 10K FAKE)
# ============================================================================

print("\n" + "🧪"*40)
print("PART 2: TEST DATA (20,000 images)")
print("🧪"*40)

# TEST/REAL
print("\n🟢 TEST/REAL - Loading 10,000 images...")
hdfs_test_real = "hdfs://namenode:8020/user/data/raw/test/REAL"

try:
    df_test_real = spark.read.format("binaryFile").load(hdfs_test_real)
    df_test_real = df_test_real.repartition(8)  # Giảm partitions
    success, count = process_batch('test', 'REAL', df_test_real, 1, label_value=1, total_batches=1)
    results.append(('TEST/REAL', 1, success, count))
    del df_test_real
    force_memory_cleanup(clear_model_flag=True)
except Exception as e:
    print(f"❌ Error processing TEST/REAL: {e}")
    results.append(('TEST/REAL', 1, False, 0))

# TEST/FAKE
print("\n🔴 TEST/FAKE - Loading 10,000 images...")
hdfs_test_fake = "hdfs://namenode:8020/user/data/raw/test/FAKE"

try:
    df_test_fake = spark.read.format("binaryFile").load(hdfs_test_fake)
    df_test_fake = df_test_fake.repartition(8)  # Giảm partitions
    success, count = process_batch('test', 'FAKE', df_test_fake, 1, label_value=0, total_batches=1)
    results.append(('TEST/FAKE', 1, success, count))
    del df_test_fake
    force_memory_cleanup(clear_model_flag=True)
except Exception as e:
    print(f"❌ Error processing TEST/FAKE: {e}")
    results.append(('TEST/FAKE', 1, False, 0))

# ============================================================================
# SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("📊 FINAL SUMMARY - MOBILENETV2 FEATURE EXTRACTION")
print("="*80)

success_count = sum(1 for _, _, s, _ in results if s)
total_batches = len(results)
total_samples = sum(c for _, _, _, c in results)

print(f"\n✅ Successful: {success_count}/{total_batches} batches")
print(f"📊 Total samples: {total_samples:,}")
print(f"⏱️  Total time: {pipeline_elapsed/60:.2f} minutes")
print(f"📊 Feature dimension: 1280 (MobileNetV2)")

print("\n📋 Results by batch:")
for dataset, batch, success, count in results:
    status = "✅" if success else "❌"
    print(f"   {status} {dataset} Batch {batch}: {count:,} samples")

# Final memory status
print("\n📊 FINAL MEMORY STATUS:")
print_memory_status("   ")

if success_count == total_batches:
    print("\n" + "🎉"*40)
    print("ALL BATCHES COMPLETED WITH MOBILENETV2 FEATURES!")
    print(f"Total: {total_samples:,} images → 1280-dim features")
    print("🎉"*40)
else:
    print("\n" + "⚠️"*40)
    print(f"COMPLETED WITH {total_batches - success_count} FAILED BATCHES")
    print("⚠️"*40)

spark.stop()
print("\n✅ Spark stopped. Done!")
