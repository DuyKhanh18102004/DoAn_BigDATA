"""
Check features structure and statistics
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, mean, stddev
from pyspark.sql.types import FloatType, ArrayType
from pyspark.ml.linalg import VectorUDT

spark = SparkSession.builder \
    .appName("CheckFeatures") \
    .master("local[2]") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("FEATURE INSPECTION")
print("=" * 60)

# Load sample REAL
print("\n📂 Loading REAL batch_1...")
df_real = spark.read.parquet("hdfs://namenode:8020/user/data/features/train/REAL/batch_1")
print(f"   Count: {df_real.count()}")

# Load sample FAKE
print("\n📂 Loading FAKE batch_1...")
df_fake = spark.read.parquet("hdfs://namenode:8020/user/data/features/train/FAKE/batch_1")
print(f"   Count: {df_fake.count()}")

print("\n📋 Schema:")
df_real.printSchema()

print("\n📋 Sample REAL:")
df_real.select("path", "label").show(5, truncate=40)

print("\n📋 Sample FAKE:")
df_fake.select("path", "label").show(5, truncate=40)

# Check feature type
@udf(FloatType())
def get_feature_mean(vec):
    if vec is None:
        return 0.0
    return float(sum(vec.toArray()) / len(vec.toArray()))

@udf(FloatType())
def get_feature_std(vec):
    if vec is None:
        return 0.0
    arr = vec.toArray()
    m = sum(arr) / len(arr)
    variance = sum((x - m) ** 2 for x in arr) / len(arr)
    return float(variance ** 0.5)

@udf(FloatType())
def get_feature_max(vec):
    if vec is None:
        return 0.0
    return float(max(vec.toArray()))

@udf(FloatType())
def get_feature_min(vec):
    if vec is None:
        return 0.0
    return float(min(vec.toArray()))

print("\n📊 Feature Statistics REAL (sample 100):")
df_real_sample = df_real.limit(100)
df_real_stats = df_real_sample.select(
    get_feature_mean("features").alias("mean"),
    get_feature_std("features").alias("std"),
    get_feature_max("features").alias("max"),
    get_feature_min("features").alias("min")
)
print("   Mean of feature means:", df_real_stats.agg(mean("mean")).first()[0])
print("   Mean of feature stds:", df_real_stats.agg(mean("std")).first()[0])
print("   Mean of feature max:", df_real_stats.agg(mean("max")).first()[0])
print("   Mean of feature min:", df_real_stats.agg(mean("min")).first()[0])

print("\n📊 Feature Statistics FAKE (sample 100):")
df_fake_sample = df_fake.limit(100)
df_fake_stats = df_fake_sample.select(
    get_feature_mean("features").alias("mean"),
    get_feature_std("features").alias("std"),
    get_feature_max("features").alias("max"),
    get_feature_min("features").alias("min")
)
print("   Mean of feature means:", df_fake_stats.agg(mean("mean")).first()[0])
print("   Mean of feature stds:", df_fake_stats.agg(mean("std")).first()[0])
print("   Mean of feature max:", df_fake_stats.agg(mean("max")).first()[0])
print("   Mean of feature min:", df_fake_stats.agg(mean("min")).first()[0])

spark.stop()
print("\n✅ Done!")
