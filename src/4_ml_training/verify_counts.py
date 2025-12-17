"""
DEBUG: Verify actual sample counts in HDFS feature batches
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("VerifyFeatureCounts") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

print("=" * 70)
print("VERIFYING FEATURE COUNTS IN HDFS")
print("=" * 70)

# Check train REAL batches
print("\n📊 TRAIN/REAL Batches:")
for i in range(1, 6):
    path = f"hdfs://namenode:8020/user/data/features/train/REAL/batch_{i}"
    try:
        df = spark.read.parquet(path)
        count = df.count()
        print(f"  batch_{i}: {count:,} samples")
    except Exception as e:
        print(f"  batch_{i}: ERROR - {e}")

# Check train FAKE batches  
print("\n📊 TRAIN/FAKE Batches:")
for i in range(1, 6):
    path = f"hdfs://namenode:8020/user/data/features/train/FAKE/batch_{i}"
    try:
        df = spark.read.parquet(path)
        count = df.count()
        print(f"  batch_{i}: {count:,} samples")
    except Exception as e:
        print(f"  batch_{i}: ERROR - {e}")

# Check test
print("\n📊 TEST Batches:")
for label in ["REAL", "FAKE"]:
    path = f"hdfs://namenode:8020/user/data/features/test/{label}"
    try:
        df = spark.read.parquet(path)
        count = df.count()
        print(f"  {label}: {count:,} samples")
    except Exception as e:
        print(f"  {label}: ERROR - {e}")

print("\n" + "=" * 70)
spark.stop()
