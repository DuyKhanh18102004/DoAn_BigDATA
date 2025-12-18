"""
Check existing extracted features - no PyTorch needed
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, mean, stddev
from pyspark.sql.types import FloatType, ArrayType
from pyspark.ml.linalg import VectorUDT

print("=" * 60)
print("🔍 CHECKING EXISTING FEATURES")
print("=" * 60)

spark = SparkSession.builder \
    .appName("CheckExistingFeatures") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# UDFs to analyze vectors
@udf(FloatType())
def vec_sum(vec):
    if vec is None:
        return 0.0
    try:
        return float(sum(vec.toArray()))
    except:
        return 0.0

@udf(FloatType())
def vec_mean(vec):
    if vec is None:
        return 0.0
    try:
        arr = vec.toArray()
        return float(sum(arr) / len(arr)) if len(arr) > 0 else 0.0
    except:
        return 0.0

@udf(FloatType())
def vec_max(vec):
    if vec is None:
        return 0.0
    try:
        return float(max(vec.toArray()))
    except:
        return 0.0

@udf(FloatType())
def vec_min(vec):
    if vec is None:
        return 0.0
    try:
        return float(min(vec.toArray()))
    except:
        return 0.0

@udf("int")
def vec_len(vec):
    if vec is None:
        return 0
    try:
        return len(vec.toArray())
    except:
        return 0

@udf("int")
def count_non_zero(vec):
    if vec is None:
        return 0
    try:
        arr = vec.toArray()
        return sum(1 for x in arr if abs(x) > 1e-10)
    except:
        return 0

# Check REAL batch 1
print("\n📂 Loading REAL batch_1...")
df_real = spark.read.parquet("hdfs://namenode:8020/user/data/features/train/REAL/batch_1")
real_count = df_real.count()
print(f"   Count: {real_count}")

print("\n📊 Feature Analysis (REAL - first 20 samples):")
df_real_sample = df_real.limit(20).select(
    "path",
    vec_len("features").alias("dim"),
    vec_sum("features").alias("sum"),
    vec_mean("features").alias("mean"),
    vec_max("features").alias("max"),
    vec_min("features").alias("min"),
    count_non_zero("features").alias("non_zero_count")
)
df_real_sample.show(20, truncate=30)

# Check FAKE batch 1
print("\n📂 Loading FAKE batch_1...")
df_fake = spark.read.parquet("hdfs://namenode:8020/user/data/features/train/FAKE/batch_1")
fake_count = df_fake.count()
print(f"   Count: {fake_count}")

print("\n📊 Feature Analysis (FAKE - first 20 samples):")
df_fake_sample = df_fake.limit(20).select(
    "path",
    vec_len("features").alias("dim"),
    vec_sum("features").alias("sum"),
    vec_mean("features").alias("mean"),
    vec_max("features").alias("max"),
    vec_min("features").alias("min"),
    count_non_zero("features").alias("non_zero_count")
)
df_fake_sample.show(20, truncate=30)

# Overall statistics
print("\n📊 Overall Statistics:")

# Count zero vectors
@udf("boolean")
def is_zero_vector(vec):
    if vec is None:
        return True
    try:
        arr = vec.toArray()
        return all(abs(x) < 1e-10 for x in arr)
    except:
        return True

# REAL
real_zero = df_real.filter(is_zero_vector("features")).count()
print(f"   REAL: {real_zero}/{real_count} zero vectors ({100*real_zero/real_count:.1f}%)")

# FAKE
fake_zero = df_fake.filter(is_zero_vector("features")).count()
print(f"   FAKE: {fake_zero}/{fake_count} zero vectors ({100*fake_zero/fake_count:.1f}%)")

spark.stop()
print("\n✅ Done!")
