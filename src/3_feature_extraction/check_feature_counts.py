#!/usr/bin/env python3
"""
Quick verification of extracted feature counts
"""
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Verify_Feature_Counts") \
    .getOrCreate()

print("="*80)
print("🔍 VERIFYING EXTRACTED FEATURES")
print("="*80)

total_train = 0
total_test = 0

# Check TRAIN REAL batches
print("\n📊 TRAIN/REAL Batches:")
for i in range(1, 6):
    path = f"hdfs://namenode:8020/user/data/features/train/REAL/batch_{i}"
    df = spark.read.parquet(path)
    count = df.count()
    total_train += count
    print(f"   Batch {i}: {count:,} samples")

# Check TRAIN FAKE batches
print("\n📊 TRAIN/FAKE Batches:")
for i in range(1, 6):
    path = f"hdfs://namenode:8020/user/data/features/train/FAKE/batch_{i}"
    df = spark.read.parquet(path)
    count = df.count()
    total_train += count
    print(f"   Batch {i}: {count:,} samples")

# Check TEST REAL
print("\n📊 TEST/REAL:")
path = "hdfs://namenode:8020/user/data/features/test/REAL/batch_1"
df = spark.read.parquet(path)
count = df.count()
total_test += count
print(f"   Batch 1: {count:,} samples")

# Check TEST FAKE
print("\n📊 TEST/FAKE:")
path = "hdfs://namenode:8020/user/data/features/test/FAKE/batch_1"
df = spark.read.parquet(path)
count = df.count()
total_test += count
print(f"   Batch 1: {count:,} samples")

print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"Total TRAIN: {total_train:,} samples")
print(f"Total TEST:  {total_test:,} samples")
print(f"GRAND TOTAL: {total_train + total_test:,} samples")

if total_train + total_test == 120000:
    print("\n✅✅✅ SUCCESS: Extracted exactly 120,000 features!")
else:
    expected = 120000
    actual = total_train + total_test
    diff = actual - expected
    print(f"\n⚠️ WARNING: Expected 120,000 but got {actual:,} (diff: {diff:+,})")

print("="*80)
spark.stop()
