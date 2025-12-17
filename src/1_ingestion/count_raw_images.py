#!/usr/bin/env python3
"""
Count raw images uploaded to HDFS
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name
import sys

spark = SparkSession.builder \
    .appName("Count_Raw_Images") \
    .getOrCreate()

print("="*80)
print("🔍 COUNTING RAW IMAGES IN HDFS")
print("="*80)

base_path = "hdfs://namenode:8020/user/data/raw/train"

try:
    # Try to read all images
    print(f"\n📂 Checking path: {base_path}")
    
    # Check REAL images
    real_path = f"{base_path}/REAL"
    print(f"\n📊 Counting REAL images...")
    try:
        df_real = spark.read.format("binaryFile").load(real_path)
        real_count = df_real.count()
        print(f"   ✅ REAL: {real_count:,} images")
    except Exception as e:
        print(f"   ❌ REAL: Path not found or error: {e}")
        real_count = 0
    
    # Check FAKE images
    fake_path = f"{base_path}/FAKE"
    print(f"\n📊 Counting FAKE images...")
    try:
        df_fake = spark.read.format("binaryFile").load(fake_path)
        fake_count = df_fake.count()
        print(f"   ✅ FAKE: {fake_count:,} images")
    except Exception as e:
        print(f"   ❌ FAKE: Path not found or error: {e}")
        fake_count = 0
    
    total_count = real_count + fake_count
    
    print("\n" + "="*80)
    print("📊 SUMMARY - RAW IMAGES")
    print("="*80)
    print(f"REAL images:  {real_count:,}")
    print(f"FAKE images:  {fake_count:,}")
    print(f"TOTAL:        {total_count:,}")
    
    if total_count >= 100000:
        print("\n✅✅✅ SUCCESS: Found at least 100,000 training images!")
    elif total_count > 0:
        print(f"\n⚠️ WARNING: Only found {total_count:,} images (expected ~100,000)")
    else:
        print("\n❌ ERROR: No images found! Need to upload data to HDFS first.")
        print("\n💡 Run this command to upload:")
        print("   python src/1_ingestion/upload_to_hdfs.py --local_path data/train --hdfs_path /user/data/raw/train")
    
    print("="*80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n💡 This might mean:")
    print("   1. Images haven't been uploaded to HDFS yet")
    print("   2. Path /user/data/raw/train doesn't exist")
    print("   3. HDFS connection issue")
    sys.exit(1)

spark.stop()
