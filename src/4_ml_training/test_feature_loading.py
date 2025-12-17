"""
Quick Test: Verify features can be loaded batch-by-batch
Test script to ensure no OOM when loading 100K features
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark
print("🚀 Initializing Spark for Feature Loading Test...")
spark = SparkSession.builder \
    .appName("FeatureLoadTest") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Define batch paths
train_real_batches = [
    "hdfs://namenode:8020/user/data/features/train/REAL/batch_1",
    "hdfs://namenode:8020/user/data/features/train/REAL/batch_2",
    "hdfs://namenode:8020/user/data/features/train/REAL/batch_3",
    "hdfs://namenode:8020/user/data/features/train/REAL/batch_4",
    "hdfs://namenode:8020/user/data/features/train/REAL/batch_5"
]

train_fake_batches = [
    "hdfs://namenode:8020/user/data/features/train/FAKE/batch_1",
    "hdfs://namenode:8020/user/data/features/train/FAKE/batch_2",
    "hdfs://namenode:8020/user/data/features/train/FAKE/batch_3",
    "hdfs://namenode:8020/user/data/features/train/FAKE/batch_4",
    "hdfs://namenode:8020/user/data/features/train/FAKE/batch_5"
]

test_paths = [
    "hdfs://namenode:8020/user/data/features/test/REAL",
    "hdfs://namenode:8020/user/data/features/test/FAKE"
]

print("\n📦 Testing Feature Loading...")
print("=" * 60)

# Test loading each batch
print("\n🔍 Loading TRAIN/REAL batches:")
for i, path in enumerate(train_real_batches, 1):
    df = spark.read.parquet(path)
    count = df.count()
    print(f"  [{i}/5] batch_{i}: {count:,} samples ✓")

print("\n🔍 Loading TRAIN/FAKE batches:")
for i, path in enumerate(train_fake_batches, 1):
    df = spark.read.parquet(path)
    count = df.count()
    print(f"  [{i}/5] batch_{i}: {count:,} samples ✓")

print("\n🔍 Loading TEST batches:")
for i, path in enumerate(test_paths, 1):
    df = spark.read.parquet(path)
    count = df.count()
    label_type = "REAL" if "REAL" in path else "FAKE"
    print(f"  [{i}/2] {label_type}: {count:,} samples ✓")

# Test union operation
print("\n🔗 Testing Union Operation...")
all_train_batches = train_real_batches + train_fake_batches

print("  - Loading first batch...")
train_df = spark.read.parquet(all_train_batches[0])

print("  - Unioning remaining batches...")
for path in all_train_batches[1:]:
    batch_df = spark.read.parquet(path)
    train_df = train_df.union(batch_df)

total_count = train_df.count()
print(f"\n✅ Total training samples after union: {total_count:,}")

# Check label distribution
print("\n📊 Label Distribution:")
train_df.groupBy("label").count().orderBy("label").show()

# Test test data union
print("\n🔗 Testing Test Data Union...")
test_df = spark.read.parquet(test_paths[0]).union(spark.read.parquet(test_paths[1]))
test_count = test_df.count()
print(f"✅ Total test samples: {test_count:,}")

print("\n📊 Test Label Distribution:")
test_df.groupBy("label").count().orderBy("label").show()

print("\n" + "=" * 60)
print("🎉 FEATURE LOAD TEST SUCCESSFUL!")
print("=" * 60)
print(f"\n📊 Summary:")
print(f"   ✓ Training samples: {total_count:,}")
print(f"   ✓ Test samples:     {test_count:,}")
print(f"   ✓ No OOM errors")
print(f"   ✓ Ready for ML Training")
print("\n✅ You can proceed with ml_training_batch_safe.py")

spark.stop()
