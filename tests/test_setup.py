"""
Test Script - Verify Spark Setup
Kiểm tra xem Spark có thể đọc data từ HDFS không
"""

from pyspark.sql import SparkSession

print("🧪 Testing Spark Setup...")

# Create Spark session
spark = SparkSession.builder \
    .appName("TestSetup") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .getOrCreate()

print(f"✅ Spark version: {spark.version}")
print(f"✅ Master: {spark.sparkContext.master}")

# Test HDFS read
print("\n📂 Testing HDFS read...")
try:
    # Read a small sample
    df = spark.read.format("binaryFile") \
        .load("hdfs://namenode:8020/user/data/raw/train/REAL/*.jpg") \
        .limit(10)
    
    count = df.count()
    print(f"✅ Successfully read {count} files from HDFS")
    
    # Show schema
    print("\n📋 Schema:")
    df.printSchema()
    
    # Show sample
    print("\n📊 Sample:")
    df.select("path", "length").show(5, truncate=False)
    
except Exception as e:
    print(f"❌ Error reading from HDFS: {e}")

# Test write to HDFS
print("\n💾 Testing HDFS write...")
try:
    test_data = [(1, "test"), (2, "data")]
    test_df = spark.createDataFrame(test_data, ["id", "value"])
    
    output_path = "hdfs://namenode:8020/user/data/test_output"
    test_df.write.mode("overwrite").parquet(output_path)
    
    print(f"✅ Successfully wrote to {output_path}")
    
    # Read back
    read_back = spark.read.parquet(output_path)
    print(f"✅ Successfully read back {read_back.count()} rows")
    read_back.show()
    
except Exception as e:
    print(f"❌ Error writing to HDFS: {e}")

print("\n🎉 Setup test completed!")
spark.stop()
