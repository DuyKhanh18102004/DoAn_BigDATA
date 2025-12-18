"""
Debug script to test MobileNetV2 feature extraction locally
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit, length
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from PIL import Image
from io import BytesIO
import gc

print("=" * 60)
print("🔍 DEBUG: Testing Feature Extraction")
print("=" * 60)

spark = SparkSession.builder \
    .appName("DebugFeatures") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load MobileNetV2 model
print("\n📦 Loading MobileNetV2 model...")
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V2_Weights

model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier = torch.nn.Identity()
model.eval()
print("   ✅ Model loaded!")

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Broadcast model to executors
model_broadcast = spark.sparkContext.broadcast(model)
preprocess_broadcast = spark.sparkContext.broadcast(preprocess)

def extract_features_from_bytes(image_bytes):
    """Extract features from image bytes"""
    try:
        if image_bytes is None or len(image_bytes) == 0:
            print("Warning: Empty image bytes!")
            return Vectors.dense([0.0] * 1280)
        
        # Load image
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess
        img_tensor = preprocess_broadcast.value(img).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = model_broadcast.value(img_tensor)
        
        feature_array = features.squeeze().numpy()
        
        # Check for NaN
        if np.isnan(feature_array).any():
            print("Warning: NaN in features!")
            feature_array = np.nan_to_num(feature_array, 0.0)
        
        return Vectors.dense(feature_array.tolist())
    
    except Exception as e:
        print(f"Error: {e}")
        return Vectors.dense([0.0] * 1280)

extract_udf = udf(extract_features_from_bytes, VectorUDT())

# Load a few test images
print("\n📂 Loading test images from HDFS...")
test_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.jpg") \
    .load("hdfs://namenode:8020/user/data/raw/train/REAL").limit(5)

print(f"   Loaded: {test_df.count()} images")

# Check content column
print("\n📋 Image content info:")
test_df.select("path", length("content").alias("content_length")).show(5, truncate=40)

# Extract features
print("\n🔄 Extracting features...")
features_df = test_df.withColumn("features", extract_udf(col("content")))

# Check features
print("\n📋 Feature check:")
features_df.select("path", "features").show(5, truncate=60)

# Feature statistics
@udf("float")
def get_vec_mean(vec):
    if vec is None:
        return 0.0
    arr = list(vec.toArray())
    return float(sum(arr) / len(arr))

@udf("float") 
def get_vec_max(vec):
    if vec is None:
        return 0.0
    return float(max(vec.toArray()))

@udf("float")
def get_vec_min(vec):
    if vec is None:
        return 0.0
    return float(min(vec.toArray()))

@udf("float")
def get_vec_std(vec):
    if vec is None:
        return 0.0
    arr = list(vec.toArray())
    m = sum(arr) / len(arr)
    return float((sum((x - m) ** 2 for x in arr) / len(arr)) ** 0.5)

stats_df = features_df.select(
    "path",
    get_vec_mean("features").alias("mean"),
    get_vec_std("features").alias("std"),
    get_vec_max("features").alias("max"),
    get_vec_min("features").alias("min")
)

print("\n📊 Feature Statistics:")
stats_df.show(5, truncate=30)

# Overall stats
print("\n📊 Overall Statistics:")
from pyspark.sql.functions import avg
stats_df.select(
    avg("mean").alias("avg_mean"),
    avg("std").alias("avg_std"),
    avg("max").alias("avg_max"),
    avg("min").alias("avg_min")
).show()

spark.stop()
print("\n✅ Done!")
