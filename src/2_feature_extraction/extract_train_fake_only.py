#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔴 EXTRACT TRAIN/FAKE ONLY - FIX MISSING DATA
Extract features for TRAIN/FAKE: 50,000 images split into 5 batches of 10K each
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit
from pyspark.sql.types import ArrayType, FloatType
from PIL import Image
import numpy as np
import io
import time
import gc

# =============================================================================
# 1. FEATURE EXTRACTION FUNCTION
# =============================================================================

def extract_features(image_bytes):
    """Extract features từ image bytes"""
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        
        # RGB histogram (256 bins per channel)
        hist_r = np.histogram(np.array(img)[:,:,0], bins=256, range=(0,256))[0]
        hist_g = np.histogram(np.array(img)[:,:,1], bins=256, range=(0,256))[0]
        hist_b = np.histogram(np.array(img)[:,:,2], bins=256, range=(0,256))[0]
        
        # Statistical features
        img_array = np.array(img).flatten()
        stats = [
            float(np.mean(img_array)),
            float(np.std(img_array)),
            float(np.min(img_array)),
            float(np.max(img_array)),
            float(np.median(img_array)),
            float(np.percentile(img_array, 25)),
            float(np.percentile(img_array, 75)),
            float(np.var(img_array)),
            float(np.ptp(img_array)),
            float(img_array.sum()),
            float(len(img_array)),
            float(np.count_nonzero(img_array))
        ]
        
        # Combine features
        features = (hist_r.tolist() + hist_g.tolist() + hist_b.tolist() + stats)
        
        # Pad to 2048 dimensions
        features = features + [0.0] * (2048 - len(features))
        
        return features[:2048]
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return [0.0] * 2048

# =============================================================================
# 2. MAIN EXTRACTION
# =============================================================================

if __name__ == "__main__":
    print("🔴 TRAIN/FAKE FEATURE EXTRACTION")
    print("=" * 70)
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Extract_TRAIN_FAKE_Only") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Register UDF
    extract_features_udf = udf(extract_features, ArrayType(FloatType()))
    
    # HDFS paths
    input_path = "hdfs://namenode:8020/user/data/raw/train/FAKE"
    
    try:
        # 1. LOAD ALL FAKE IMAGES
        print(f"\n📂 Loading all FAKE images from: {input_path}")
        df_all = spark.read.format("binaryFile").load(input_path)
        df_all = df_all.repartition(100).cache()
        
        total_count = df_all.count()
        print(f"✅ Total FAKE images loaded: {total_count:,}")
        
        if total_count == 0:
            print("❌ ERROR: No FAKE images found!")
            spark.stop()
            exit(1)
        
        # 2. SPLIT INTO 5 EQUAL BATCHES USING randomSplit
        print(f"\n🔪 Splitting into 5 batches...")
        
        # Split weights: [0.2, 0.2, 0.2, 0.2, 0.2] for equal distribution
        seed = 42
        df_batches = df_all.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=seed)
        
        print("✅ Split completed!")
        print(f"📊 Batch counts:")
        for i, df_batch in enumerate(df_batches, 1):
            count = df_batch.count()
            print(f"   Batch {i}: {count:,} images")
        
        # 3. PROCESS EACH BATCH
        for batch_num in range(1, 6):
            print(f"\n{'='*70}")
            print(f"📦 Processing TRAIN/FAKE Batch {batch_num}")
            print(f"{'='*70}")
            
            df_batch = df_batches[batch_num - 1]
            output_path = f"hdfs://namenode:8020/user/data/features/train/FAKE/batch_{batch_num}"
            
            print(f"💾 Output: {output_path}")
            
            start_time = time.time()
            
            # Extract features
            print("🔬 Extracting features...")
            df_features = df_batch.withColumn("features", extract_features_udf(col("content"))) \
                                  .withColumn("label", lit(0)) \
                                  .select("path", "features", "label")
            
            # Repartition and cache
            df_features = df_features.repartition(10).cache()
            feature_count = df_features.count()
            print(f"✅ Features extracted: {feature_count:,} samples")
            
            # Write to HDFS
            print("💾 Writing to HDFS...")
            df_features.write.mode("overwrite").parquet(output_path)
            
            # VERIFICATION
            print("🔍 VERIFICATION - Reading back from HDFS...")
            verification_df = spark.read.parquet(output_path)
            saved_count = verification_df.count()
            
            elapsed = time.time() - start_time
            
            if saved_count == feature_count:
                print(f"✅✅✅ SUCCESS: {saved_count:,} samples saved correctly!")
            else:
                print(f"❌ WARNING: Mismatch! Extracted={feature_count}, Saved={saved_count}")
            
            print(f"⏱️  Time: {elapsed:.2f}s")
            
            # MEMORY CLEANUP
            print("🧹 Cleaning up memory...")
            df_features.unpersist()
            spark.catalog.clearCache()
            gc.collect()
            
            time.sleep(30)  # Pause 30s between batches
        
        # 4. FINAL SUMMARY
        print(f"\n{'='*70}")
        print("🎉 TRAIN/FAKE EXTRACTION COMPLETE!")
        print(f"{'='*70}")
        
        print("\n📊 FINAL VERIFICATION:")
        total_saved = 0
        for i in range(1, 6):
            path = f"hdfs://namenode:8020/user/data/features/train/FAKE/batch_{i}"
            df_verify = spark.read.parquet(path)
            count = df_verify.count()
            total_saved += count
            print(f"  batch_{i}: {count:,} samples")
        
        print(f"\n✅ TOTAL TRAIN/FAKE SAVED: {total_saved:,} samples")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Final cleanup...")
        spark.catalog.clearCache()
        spark.stop()
        print("✅ Spark stopped")
