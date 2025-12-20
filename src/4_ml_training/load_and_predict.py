#!/usr/bin/env python3
"""Load TensorFlow MobileNetV2 Trained Model and Make Predictions.

This script demonstrates how to load a pre-trained LogisticRegression model
and use it for predictions on new data.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
import sys

print("="*80)
print("LOAD & PREDICT - TENSORFLOW MOBILENETV2 MODEL")
print("="*80)

# Initialize Spark
spark = SparkSession.builder \
    .appName("Load_TF_Model_Predict") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "1") \
    .config("spark.default.parallelism", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ============================================================================
# CONFIGURATION
# ============================================================================

HDFS_BASE = "hdfs://namenode:8020/user/data"
FEATURES_BASE = f"{HDFS_BASE}/features_tf"
MODELS_BASE = "hdfs://namenode:8020/user/models"
MODEL_PATH = f"{MODELS_BASE}/logistic_regression_tf"

# ============================================================================
# STEP 1: Load Pre-trained Model
# ============================================================================

print("\n" + "="*80)
print("📂 STEP 1: Loading Pre-trained Model")
print("="*80)

try:
    model = LogisticRegressionModel.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    print(f"\n📊 Model Info:")
    print(f"   - Features column: {model.featuresCol}")
    print(f"   - Label column: {model.labelCol}")
    print(f"   - Prediction column: {model.predictionCol}")
    print(f"   - Model coefficients dimension: {len(model.coefficients)}")
    print(f"   - Intercept: {model.intercept:.6f}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n⚠️ Make sure to run ml_training_tf_features.py first to train and save the model.")
    sys.exit(1)

# ============================================================================
# STEP 2: Load Test Data
# ============================================================================

print("\n" + "="*80)
print("📂 STEP 2: Loading Test Data")
print("="*80)

test_data = []
print(f"📦 Loading 10 test batches...")

for batch_id in range(1, 11):
    try:
        test_real_path = f"{FEATURES_BASE}/test/REAL/batch_{batch_id}"
        df_test_real = spark.read.parquet(test_real_path)
        
        test_fake_path = f"{FEATURES_BASE}/test/FAKE/batch_{batch_id}"
        df_test_fake = spark.read.parquet(test_fake_path)
        
        test_data.append(df_test_real)
        test_data.append(df_test_fake)
        
    except Exception as e:
        print(f"   ⚠️ Error loading batch {batch_id}: {e}")

if not test_data:
    print("❌ No test data found!")
    sys.exit(1)

df_test = test_data[0]
for df in test_data[1:]:
    df_test = df_test.union(df)

df_test = df_test.repartition(8).cache()
test_count = df_test.count()
print(f"✅ Test data loaded: {test_count:,} samples")

# ============================================================================
# STEP 3: Make Predictions
# ============================================================================

print("\n" + "="*80)
print("🔮 STEP 3: Making Predictions")
print("="*80)

predictions = model.transform(df_test)
predictions = predictions.cache()

# Count predictions
real_pred = predictions.filter(col("prediction") == 1).count()
fake_pred = predictions.filter(col("prediction") == 0).count()

print(f"\n📊 Prediction Results:")
print(f"   - Predicted REAL: {real_pred:,}")
print(f"   - Predicted FAKE: {fake_pred:,}")
print(f"   - Total: {real_pred + fake_pred:,}")

# ============================================================================
# STEP 4: Evaluate Predictions (if labels available)
# ============================================================================

print("\n" + "="*80)
print("📈 STEP 4: Evaluating Predictions")
print("="*80)

try:
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_prec = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_rec = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")
    
    accuracy = evaluator_acc.evaluate(predictions)
    precision = evaluator_prec.evaluate(predictions)
    recall = evaluator_rec.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    
    print(f"\n🎯 Test Metrics:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1-Score:  {f1*100:.2f}%")
    
    # Confusion Matrix
    tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   TP (REAL→REAL): {tp:,}")
    print(f"   TN (FAKE→FAKE): {tn:,}")
    print(f"   FP (FAKE→REAL): {fp:,}")
    print(f"   FN (REAL→FAKE): {fn:,}")
    
except Exception as e:
    print(f"⚠️ Could not evaluate (labels might not be available): {e}")

# ============================================================================
# STEP 5: Show Sample Predictions
# ============================================================================

print("\n" + "="*80)
print("🔍 STEP 5: Sample Predictions")
print("="*80)

sample_pred = predictions.select(
    "label",
    "prediction",
    vector_to_array("probability")[1].alias("prob_real")
).limit(10).collect()

print("\nLabel | Prediction | Prob(REAL)")
print("------|------------|----------")
for row in sample_pred:
    label_name = "REAL" if row.label == 1 else "FAKE"
    pred_name = "REAL" if row.prediction == 1 else "FAKE"
    prob = row.prob_real
    print(f" {label_name}   |    {pred_name}    | {prob:.4f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ LOAD & PREDICT COMPLETE")
print("="*80)

print(f"\n📂 Model Location: {MODEL_PATH}")
print(f"📊 Test Samples: {test_count:,}")
print(f"🎯 Predictions: REAL={real_pred:,}, FAKE={fake_pred:,}")

print("\n" + "="*80)
print("💡 How to use this in your code:")
print("="*80)
print("""
from pyspark.ml.classification import LogisticRegressionModel

# Load model
model = LogisticRegressionModel.load("hdfs://namenode:8020/user/models/logistic_regression_tf")

# Make predictions on new data
predictions = model.transform(df_new_features)

# Get prediction results
predictions.select("features", "prediction", "probability").show()
""")

predictions.unpersist()
df_test.unpersist()

spark.stop()
print("\n✅ Spark session stopped.")
