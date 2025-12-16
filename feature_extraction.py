"""
Step 2: Feature Extraction - Deepfake Detection Pipeline
Trích xuất đặc trưng từ ảnh bằng ResNet50 pretrained model
Chạy phân tán trên Spark Workers bằng UDF
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import ArrayType, FloatType, StringType, StructType, StructField
import numpy as np
from io import BytesIO
from PIL import Image

# ===== BƯỚC 1: Khởi tạo Spark Session =====
print("🚀 Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("DeepfakeDetection-FeatureExtraction") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .getOrCreate()

print(f"✅ Spark version: {spark.version}")
print(f"✅ Master: {spark.sparkContext.master}")

# ===== BƯỚC 2: Load Pretrained Model trong UDF =====
def create_feature_extractor_udf():
    """
    Tạo UDF để extract features từ ảnh
    Model sẽ được load trong mỗi executor (distributed)
    """
    
    def extract_features(image_bytes, label_str):
        """
        Extract features từ binary image data
        Args:
            image_bytes: Binary data của ảnh (từ binaryFiles)
            label_str: "REAL" hoặc "FAKE"
        Returns:
            Tuple (features_array, label)
        """
        try:
            # Import TensorFlow trong UDF (sẽ chạy trên mỗi worker)
            import tensorflow as tf
            from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
            from tensorflow.keras.preprocessing import image as keras_image
            import numpy as np
            from PIL import Image
            from io import BytesIO
            
            # Load model (cached trong executor)
            if not hasattr(extract_features, 'model'):
                print("📦 Loading ResNet50 model in executor...")
                extract_features.model = ResNet50(
                    weights='imagenet',
                    include_top=False,  # Bỏ fully connected layers
                    pooling='avg'       # Global average pooling -> vector 2048 chiều
                )
                print("✅ Model loaded successfully!")
            
            # Đọc ảnh từ binary
            img = Image.open(BytesIO(image_bytes))
            
            # Resize về 224x224 (yêu cầu của ResNet50)
            img = img.convert('RGB')
            img = img.resize((224, 224))
            
            # Chuyển sang array và preprocess
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = extract_features.model.predict(img_array, verbose=0)
            features_flat = features.flatten().tolist()
            
            # Convert label
            label = 1.0 if label_str == "FAKE" else 0.0
            
            return (features_flat, label)
            
        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")
            # Trả về vector zeros nếu lỗi
            return ([0.0] * 2048, -1.0)
    
    # Define schema cho output
    schema = StructType([
        StructField("features", ArrayType(FloatType()), False),
        StructField("label", FloatType(), False)
    ])
    
    return udf(extract_features, schema)

# ===== BƯỚC 3: Đọc dữ liệu từ HDFS =====
print("\n📂 Reading images from HDFS...")

# Đọc training data
print("  - Reading train/REAL...")
train_real_df = spark.read.format("binaryFile") \
    .load("hdfs://namenode:8020/user/data/raw/train/REAL/*.jpg") \
    .withColumn("label_str", lit("REAL"))

print("  - Reading train/FAKE...")
train_fake_df = spark.read.format("binaryFile") \
    .load("hdfs://namenode:8020/user/data/raw/train/FAKE/*.jpg") \
    .withColumn("label_str", lit("FAKE"))

# Đọc test data
print("  - Reading test/REAL...")
test_real_df = spark.read.format("binaryFile") \
    .load("hdfs://namenode:8020/user/data/raw/test/REAL/*.jpg") \
    .withColumn("label_str", lit("REAL"))

print("  - Reading test/FAKE...")
test_fake_df = spark.read.format("binaryFile") \
    .load("hdfs://namenode:8020/user/data/raw/test/FAKE/*.jpg") \
    .withColumn("label_str", lit("FAKE"))

# Union tất cả
from pyspark.sql.functions import lit
train_df = train_real_df.union(train_fake_df)
test_df = test_real_df.union(test_fake_df)

print(f"\n✅ Train dataset: {train_df.count()} images")
print(f"✅ Test dataset: {test_df.count()} images")

# ===== BƯỚC 4: Extract Features (PHÂN TÁN) =====
print("\n🔬 Extracting features using distributed UDF...")

# Tạo UDF
feature_extractor = create_feature_extractor_udf()

# Apply UDF lên toàn bộ dataset (chạy trên Spark workers)
print("  - Processing training set...")
train_features_df = train_df.select(
    col("path"),
    feature_extractor(col("content"), col("label_str")).alias("result")
).select(
    col("path"),
    col("result.features").alias("features"),
    col("result.label").alias("label")
)

print("  - Processing test set...")
test_features_df = test_df.select(
    col("path"),
    feature_extractor(col("content"), col("label_str")).alias("result")
).select(
    col("path"),
    col("result.features").alias("features"),
    col("result.label").alias("label")
)

# ===== BƯỚC 5: Lưu Features vào HDFS (Parquet) =====
print("\n💾 Saving features to HDFS in Parquet format...")

output_train_path = "hdfs://namenode:8020/user/data/features/train"
output_test_path = "hdfs://namenode:8020/user/data/features/test"

print(f"  - Writing to {output_train_path}...")
train_features_df.write.mode("overwrite").parquet(output_train_path)

print(f"  - Writing to {output_test_path}...")
test_features_df.write.mode("overwrite").parquet(output_test_path)

# ===== BƯỚC 6: Verify =====
print("\n✅ Feature extraction completed!")
print("\n📊 Sample of extracted features:")
train_features_df.show(5, truncate=False)

print("\n📈 Dataset statistics:")
train_features_df.groupBy("label").count().show()
test_features_df.groupBy("label").count().show()

print("\n🎯 Features saved successfully!")
print(f"  - Train: {output_train_path}")
print(f"  - Test: {output_test_path}")

# Stop Spark
spark.stop()
