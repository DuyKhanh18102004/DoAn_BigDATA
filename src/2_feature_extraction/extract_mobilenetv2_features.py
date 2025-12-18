#!/usr/bin/env python3
"""Feature Extraction with MobileNetV2 Pre-trained Model.

Extracts 1280-dim features from images using MobileNetV2 model.
Memory-optimized for low RAM environments with periodic cleanup.
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
import psutil

# Memory tracking utilities

def get_process_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_system_memory_info():
    """Get system memory information."""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / 1024**3,
        'available_gb': mem.available / 1024**3,
        'used_gb': mem.used / 1024**3,
        'percent': mem.percent
    }

def print_memory_status(prefix=""):
    """Print current memory status."""
    proc_mem = get_process_memory_mb()
    sys_mem = get_system_memory_info()
    print(f"{prefix}Process Memory: {proc_mem:.1f} MB")
    print(f"{prefix}System Memory: {sys_mem['used_gb']:.2f}/{sys_mem['total_gb']:.2f} GB ({sys_mem['percent']:.1f}% used)")
    print(f"{prefix}Available: {sys_mem['available_gb']:.2f} GB")
    return proc_mem, sys_mem

# Spark Session with memory optimization

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
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.rpc.message.maxSize", "256") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark Session created successfully")
print_memory_status("   ")

# Global model loading
_mobilenet_model = None

def get_mobilenet_model():
    """Load MobileNetV2 model (lazy loading per executor)."""
    global _mobilenet_model
    if _mobilenet_model is None:
        import torch
        import torchvision.models as models
        from torchvision.models import MobileNet_V2_Weights

        print("   Loading MobileNetV2 model...")
        mem_before = get_process_memory_mb()

        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        model.eval()
        model = model.cpu()

        _mobilenet_model = model

        mem_after = get_process_memory_mb()
        print(f"   MobileNetV2 loaded. Memory used: {mem_after - mem_before:.1f} MB")

    return _mobilenet_model

def clear_model():
    """Clear model from memory."""
    global _mobilenet_model
    if _mobilenet_model is not None:
        import torch
        mem_before = get_process_memory_mb()

        del _mobilenet_model
        _mobilenet_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        mem_after = get_process_memory_mb()
        print(f"   Model cleared. Memory freed: {mem_before - mem_after:.1f} MB")

def extract_mobilenet_features(image_bytes):
    """Extract 1280-dim features using MobileNetV2.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        DenseVector: 1280-dimensional feature vector
    """
    try:
        import torch
        import torchvision.transforms as transforms

        img = Image.open(BytesIO(image_bytes)).convert('RGB')

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        img_tensor = preprocess(img).unsqueeze(0)

        model = get_mobilenet_model()
        with torch.no_grad():
            features = model(img_tensor)

        features = features.squeeze().numpy()

        del img_tensor, img

        return Vectors.dense(features.tolist())

    except Exception as e:
        print(f"Error extracting features: {e}")
        return Vectors.dense([0.0] * 1280)

extract_features_udf = udf(extract_mobilenet_features, VectorUDT())

def force_memory_cleanup(clear_model_flag=False):
    """Aggressive memory cleanup.
    
    Args:
        clear_model_flag: If True, clear the loaded model
        
    Returns:
        tuple: (memory_before, memory_after, memory_freed)
    """
    print("\n   Performing memory cleanup...")
    mem_before = get_process_memory_mb()
    sys_before = get_system_memory_info()

    spark.catalog.clearCache()

    if clear_model_flag:
        clear_model()

    for i in range(5):
        collected = gc.collect()
        if collected == 0:
            break

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _ = torch.zeros(1)
        del _
    except:
        pass

    time.sleep(3)
    gc.collect()

    mem_after = get_process_memory_mb()
    sys_after = get_system_memory_info()

    memory_freed = mem_before - mem_after
    system_freed = sys_before['used_gb'] - sys_after['used_gb']

    print(f"   Memory cleanup completed.")
    print(f"   Process Memory: {mem_before:.1f} MB -> {mem_after:.1f} MB (freed: {memory_freed:.1f} MB)")
    print(f"   System Memory: {sys_before['used_gb']:.2f} GB -> {sys_after['used_gb']:.2f} GB")

    return mem_before, mem_after, memory_freed

def process_batch(dataset_type, class_label, df_batch, batch_num, label_value, total_batches):
    """Process one batch of images.
    
    Args:
        dataset_type: 'train' or 'test'
        class_label: 'REAL' or 'FAKE'
        df_batch: Spark DataFrame
        batch_num: Batch number
        label_value: Label value (0 or 1)
        total_batches: Total batches
        
    Returns:
        tuple: (success: bool, count: int)
    """
    print(f"\nProcessing {dataset_type.upper()}/{class_label} Batch {batch_num}/{total_batches}")

    print("MEMORY STATUS BEFORE BATCH:")
    mem_start, sys_start = print_memory_status("   ")

    hdfs_output = f"hdfs://namenode:8020/user/data/features/{dataset_type}/{class_label}/batch_{batch_num}"

    start_time = time.time()

    try:
        batch_size = df_batch.count()
        print(f"Batch size: {batch_size:,} images")

        print("Extracting MobileNetV2 features (1280-dim)...")
        df_features = df_batch.withColumn("features", extract_features_udf(col("content"))) \
                              .withColumn("label", lit(label_value)) \
                              .select("path", "features", "label")

        df_features = df_features.repartition(4)

        print(f"Saving to HDFS: {hdfs_output}")
        df_features.write.mode("overwrite").parquet(hdfs_output)

        saved_count = spark.read.parquet(hdfs_output).count()

        elapsed = time.time() - start_time

        df_features.unpersist()
        del df_features

        print("MEMORY STATUS AFTER BATCH:")
        mem_end, sys_end = print_memory_status("   ")

        mem_used = mem_end - mem_start
        print(f"Memory used for batch: {mem_used:.1f} MB")

        print("Performing aggressive cleanup...")
        force_memory_cleanup(clear_model_flag=True)

        cooldown_time = max(10, min(30, int(mem_used / 100) * 5))
        print(f"Cooldown {cooldown_time}s for system stabilization...")
        time.sleep(cooldown_time)

        print("FINAL MEMORY STATUS:")
        print_memory_status("   ")

        if saved_count == batch_size:
            print(f"SUCCESS: {saved_count:,} samples saved. Time: {elapsed:.1f}s")
            return True, saved_count
        else:
            print(f"WARNING: Expected {batch_size:,} but saved {saved_count:,}")
            return saved_count > 0, saved_count

    except Exception as e:
        print(f"ERROR processing batch: {e}")
        import traceback
        traceback.print_exc()
        force_memory_cleanup(clear_model_flag=True)
        return False, 0


def process_batch(dataset_type, class_label, df_batch, batch_num, label_value, total_batches):
    """Process one batch of images.
    
    Args:
        dataset_type: 'train' or 'test'
        class_label: 'REAL' or 'FAKE'
        df_batch: Spark DataFrame
        batch_num: Batch number
        label_value: Label value (0 or 1)
        total_batches: Total batches
        
    Returns:
        tuple: (success: bool, count: int)
    """
    print(f"\nProcessing {dataset_type.upper()}/{class_label} Batch {batch_num}/{total_batches}")

    print("MEMORY STATUS BEFORE BATCH:")
    mem_start, sys_start = print_memory_status("   ")

    hdfs_output = f"hdfs://namenode:8020/user/data/features/{dataset_type}/{class_label}/batch_{batch_num}"

    start_time = time.time()

    try:
        batch_size = df_batch.count()
        print(f"Batch size: {batch_size:,} images")

        print("Extracting MobileNetV2 features (1280-dim)...")
        df_features = df_batch.withColumn("features", extract_features_udf(col("content"))) \
                              .withColumn("label", lit(label_value)) \
                              .select("path", "features", "label")

        df_features = df_features.repartition(4)

        print(f"Saving to HDFS: {hdfs_output}")
        df_features.write.mode("overwrite").parquet(hdfs_output)

        saved_count = spark.read.parquet(hdfs_output).count()

        elapsed = time.time() - start_time

        df_features.unpersist()
        del df_features

        print("MEMORY STATUS AFTER BATCH:")
        mem_end, sys_end = print_memory_status("   ")

        mem_used = mem_end - mem_start
        print(f"Memory used for batch: {mem_used:.1f} MB")

        print("Performing aggressive cleanup...")
        force_memory_cleanup(clear_model_flag=True)

        cooldown_time = max(10, min(30, int(mem_used / 100) * 5))
        print(f"Cooldown {cooldown_time}s for system stabilization...")
        time.sleep(cooldown_time)

        print("FINAL MEMORY STATUS:")
        print_memory_status("   ")

        if saved_count == batch_size:
            print(f"SUCCESS: {saved_count:,} samples saved. Time: {elapsed:.1f}s")
            return True, saved_count
        else:
            print(f"WARNING: Expected {batch_size:,} but saved {saved_count:,}")
            return saved_count > 0, saved_count

    except Exception as e:
        print(f"ERROR processing batch: {e}")
        import traceback
        traceback.print_exc()
        force_memory_cleanup(clear_model_flag=True)
        return False, 0


# Main pipeline
pipeline_start = time.time()
results = []

print("\n" + "="*80)
print("STARTING MOBILENETV2 FEATURE EXTRACTION")
print("Model: MobileNetV2 (14MB) | Output: 1280-dim features")
print("="*80)

print("\nINITIAL SYSTEM STATUS:")
print_memory_status("   ")

# TRAIN DATA
print("\n" + "="*80)
print("PART 1: TRAINING DATA (100,000 images)")
print("="*80)

print("\nTRAIN/REAL - Loading 50,000 images...")
hdfs_real = "hdfs://namenode:8020/user/data/raw/train/REAL"

try:
    df_real = spark.read.format("binaryFile").load(hdfs_real)
    df_real = df_real.repartition(20)
    total_real = df_real.count()
    print(f"Total REAL: {total_real:,}")

    batches_real = df_real.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)

    for i in range(5):
        print(f"\nTRAIN/REAL BATCH {i+1}/5")
        success, count = process_batch('train', 'REAL', batches_real[i], i+1, label_value=1, total_batches=5)
        results.append(('TRAIN/REAL', i+1, success, count))
        batches_real[i] = None
        gc.collect()

    del batches_real
    del df_real
    force_memory_cleanup(clear_model_flag=True)
    print("\nTRAIN/REAL completed - Memory released")

except Exception as e:
    print(f"Error processing TRAIN/REAL: {e}")
    import traceback
    traceback.print_exc()

print("\nTRAIN/FAKE - Loading 50,000 images...")
hdfs_fake = "hdfs://namenode:8020/user/data/raw/train/FAKE"

try:
    df_fake = spark.read.format("binaryFile").load(hdfs_fake)
    df_fake = df_fake.repartition(20)
    total_fake = df_fake.count()
    print(f"Total FAKE: {total_fake:,}")

    batches_fake = df_fake.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)

    for i in range(5):
        print(f"\nTRAIN/FAKE BATCH {i+1}/5")
        success, count = process_batch('train', 'FAKE', batches_fake[i], i+1, label_value=0, total_batches=5)
        results.append(('TRAIN/FAKE', i+1, success, count))
        batches_fake[i] = None
        gc.collect()

    del batches_fake
    del df_fake
    force_memory_cleanup(clear_model_flag=True)
    print("\nTRAIN/FAKE completed - Memory released")

except Exception as e:
    print(f"Error processing TRAIN/FAKE: {e}")
    import traceback
    traceback.print_exc()

# TEST DATA
print("\n" + "="*80)
print("PART 2: TEST DATA (20,000 images)")
print("="*80)

print("\nTEST/REAL - Loading 10,000 images...")
hdfs_test_real = "hdfs://namenode:8020/user/data/raw/test/REAL"

try:
    df_test_real = spark.read.format("binaryFile").load(hdfs_test_real)
    df_test_real = df_test_real.repartition(8)
    success, count = process_batch('test', 'REAL', df_test_real, 1, label_value=1, total_batches=1)
    results.append(('TEST/REAL', 1, success, count))
    del df_test_real
    force_memory_cleanup(clear_model_flag=True)
except Exception as e:
    print(f"Error processing TEST/REAL: {e}")
    results.append(('TEST/REAL', 1, False, 0))

print("\nTEST/FAKE - Loading 10,000 images...")
hdfs_test_fake = "hdfs://namenode:8020/user/data/raw/test/FAKE"

try:
    df_test_fake = spark.read.format("binaryFile").load(hdfs_test_fake)
    df_test_fake = df_test_fake.repartition(8)
    success, count = process_batch('test', 'FAKE', df_test_fake, 1, label_value=0, total_batches=1)
    results.append(('TEST/FAKE', 1, success, count))
    del df_test_fake
    force_memory_cleanup(clear_model_flag=True)
except Exception as e:
    print(f"Error processing TEST/FAKE: {e}")
    results.append(('TEST/FAKE', 1, False, 0))

# Summary
pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("FINAL SUMMARY - MOBILENETV2 FEATURE EXTRACTION")
print("="*80)

success_count = sum(1 for _, _, s, _ in results if s)
total_batches = len(results)
total_samples = sum(c for _, _, _, c in results)

print(f"\nSuccessful: {success_count}/{total_batches} batches")
print(f"Total samples: {total_samples:,}")
print(f"Total time: {pipeline_elapsed/60:.2f} minutes")
print(f"Feature dimension: 1280 (MobileNetV2)")

print("\nResults by batch:")
for dataset, batch, success, count in results:
    status = "OK" if success else "FAILED"
    print(f"   [{status}] {dataset} Batch {batch}: {count:,} samples")

print("\nFINAL MEMORY STATUS:")
print_memory_status("   ")

if success_count == total_batches:
    print("\n" + "="*80)
    print("ALL BATCHES COMPLETED - MOBILENETV2 FEATURES EXTRACTED")
    print(f"Total: {total_samples:,} images -> 1280-dim features")
    print("="*80)
else:
    print("\n" + "="*80)
    print(f"COMPLETED WITH {total_batches - success_count} FAILED BATCHES")
    print("="*80)

spark.stop()
print("\nSpark stopped. Done!")

