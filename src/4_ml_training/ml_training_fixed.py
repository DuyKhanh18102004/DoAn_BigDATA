"""
Step 3: FIXED BATCH ML Training - Load FULL 100K Data
Fixed version that properly loads all training samples
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import time

# ===== BƯỚC 1: Khởi tạo Spark Session =====
print("🚀 Initializing Spark Session for FIXED BATCH ML Training...")
spark = SparkSession.builder \
    .appName("DeepfakeDetection-FixedBatchTraining") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.default.parallelism", "100") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .getOrCreate()

print(f"✅ Spark version: {spark.version}")
print(f"✅ Spark master: {spark.sparkContext.master}")

# ===== BƯỚC 2: Load ALL Features - FIXED APPROACH =====
print("\n" + "="*70)
print("📂 LOADING FEATURES WITH FIXED BATCH STRATEGY")
print("="*70)

# Define paths
train_real_batches = [f"hdfs://namenode:8020/user/data/features/train/REAL/batch_{i}" for i in range(1, 6)]
train_fake_batches = [f"hdfs://namenode:8020/user/data/features/train/FAKE/batch_{i}" for i in range(1, 6)]
test_real_path = "hdfs://namenode:8020/user/data/features/test/REAL"
test_fake_path = "hdfs://namenode:8020/user/data/features/test/FAKE"

print(f"\n📦 Training Batch Configuration:")
print(f"   - REAL batches: {len(train_real_batches)}")
print(f"   - FAKE batches: {len(train_fake_batches)}")
print(f"   - Total batches: {len(train_real_batches) + len(train_fake_batches)}")

# ===== LOAD TRAINING DATA - USE unionAll INSTEAD =====
print("\n🔄 Loading REAL training batches...")
train_real_df = None
for i, path in enumerate(train_real_batches, 1):
    print(f"  [{i}/{len(train_real_batches)}] Loading REAL batch_{i}...")
    df = spark.read.parquet(path)
    count = df.count()
    print(f"      → Loaded {count:,} samples")
    if train_real_df is None:
        train_real_df = df
    else:
        train_real_df = train_real_df.union(df)

real_total = train_real_df.count()
print(f"✅ Total REAL samples: {real_total:,}")

print("\n🔄 Loading FAKE training batches...")
train_fake_df = None
for i, path in enumerate(train_fake_batches, 1):
    print(f"  [{i}/{len(train_fake_batches)}] Loading FAKE batch_{i}...")
    df = spark.read.parquet(path)
    count = df.count()
    print(f"      → Loaded {count:,} samples")
    if train_fake_df is None:
        train_fake_df = df
    else:
        train_fake_df = train_fake_df.union(df)

fake_total = train_fake_df.count()
print(f"✅ Total FAKE samples: {fake_total:,}")

# Combine REAL + FAKE
print("\n🔗 Combining REAL + FAKE training data...")
train_df = train_real_df.union(train_fake_df)
train_count = train_df.count()
print(f"✅ TOTAL TRAINING SAMPLES: {train_count:,}")

# Load test data
print("\n📂 Loading test data...")
print("  - Loading test REAL...")
test_real_df = spark.read.parquet(test_real_path)
real_test_count = test_real_df.count()
print(f"      → {real_test_count:,} samples")

print("  - Loading test FAKE...")
test_fake_df = spark.read.parquet(test_fake_path)
fake_test_count = test_fake_df.count()
print(f"      → {fake_test_count:,} samples")

test_df = test_real_df.union(test_fake_df)
test_count = test_df.count()
print(f"✅ TOTAL TEST SAMPLES: {test_count:,}")

# Show label distribution
print("\n📊 Training Label Distribution:")
train_df.groupBy("label").count().orderBy("label").show()

print("\n📊 Test Label Distribution:")
test_df.groupBy("label").count().orderBy("label").show()

# ===== BƯỚC 3: Convert Features to Vector =====
print("\n" + "="*70)
print("🔄 CONVERTING FEATURES TO VECTOR TYPE")
print("="*70)

def list_to_vector(features_list):
    """Convert list of floats to DenseVector"""
    return Vectors.dense(features_list)

# Create UDF
list_to_vector_udf = udf(list_to_vector, VectorUDT())

# Apply transformation
print("\n  - Converting training features...")
train_ml = train_df.select(
    col("path"),
    list_to_vector_udf(col("features")).alias("features"),
    col("label")
)

print("  - Converting test features...")
test_ml = test_df.select(
    col("path"),
    list_to_vector_udf(col("features")).alias("features"),
    col("label")
)

print("✅ Features converted to Vector type")

# Repartition and cache
print("\n⚡ Optimizing data distribution...")
train_ml = train_ml.repartition(100).cache()
test_ml = test_ml.repartition(50).cache()

# Force cache with count
print("💾 Caching training data...")
cached_train_count = train_ml.count()
print(f"  ✅ {cached_train_count:,} training samples cached")

print("💾 Caching test data...")
cached_test_count = test_ml.count()
print(f"  ✅ {cached_test_count:,} test samples cached")

print("\n✅ Data preparation completed!")
train_ml.show(5, truncate=False)

# ===== BƯỚC 4: Train Logistic Regression =====
print("\n" + "="*70)
print("🤖 TRAINING LOGISTIC REGRESSION MODEL")
print("="*70)

lr_start = time.time()

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.8,
    aggregationDepth=5
)

print(f"\n📋 Model Configuration:")
print(f"   - Training Samples: {cached_train_count:,}")
print(f"   - Max Iterations: {lr.getMaxIter()}")
print(f"   - Regularization: {lr.getRegParam()}")
print(f"   - ElasticNet Param: {lr.getElasticNetParam()}")

print(f"\n🏋️ Fitting Logistic Regression on {cached_train_count:,} samples...")
lr_model = lr.fit(train_ml)

lr_duration = time.time() - lr_start
print(f"✅ Logistic Regression trained in {lr_duration:.2f} seconds")

# Predict
print("\n🔮 Making predictions on test set...")
lr_predictions = lr_model.transform(test_ml)

# Save predictions
lr_output_path = "hdfs://namenode:8020/user/data/results/lr_predictions_fixed"
print(f"💾 Saving predictions to {lr_output_path}")
lr_predictions.select("path", "label", "prediction", "probability").write.mode("overwrite").parquet(lr_output_path)
print("✅ LR Predictions saved!")

# ===== BƯỚC 5: Train Random Forest =====
print("\n" + "="*70)
print("🌲 TRAINING RANDOM FOREST MODEL")
print("="*70)

rf_start = time.time()

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=10,
    maxBins=32,
    seed=42,
    subsamplingRate=0.8
)

print(f"\n📋 Model Configuration:")
print(f"   - Training Samples: {cached_train_count:,}")
print(f"   - Number of Trees: {rf.getNumTrees()}")
print(f"   - Max Depth: {rf.getMaxDepth()}")
print(f"   - Max Bins: {rf.getMaxBins()}")
print(f"   - Subsampling Rate: {rf.getSubsamplingRate()}")

print(f"\n🏋️ Fitting Random Forest on {cached_train_count:,} samples...")
rf_model = rf.fit(train_ml)

rf_duration = time.time() - rf_start
print(f"✅ Random Forest trained in {rf_duration:.2f} seconds")

# Predict
print("\n🔮 Making predictions on test set...")
rf_predictions = rf_model.transform(test_ml)

# Save predictions
rf_output_path = "hdfs://namenode:8020/user/data/results/rf_predictions_fixed"
print(f"💾 Saving predictions to {rf_output_path}")
rf_predictions.select("path", "label", "prediction", "probability").write.mode("overwrite").parquet(rf_output_path)
print("✅ RF Predictions saved!")

# ===== BƯỚC 6: Model Evaluation =====
print("\n" + "="*70)
print("📊 MODEL EVALUATION")
print("="*70)

# Evaluators
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Logistic Regression Metrics
print("\n🔹 LOGISTIC REGRESSION RESULTS:")
print("-" * 70)
lr_auc = binary_evaluator.evaluate(lr_predictions)
lr_accuracy = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "accuracy"})
lr_precision = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
lr_recall = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
lr_f1 = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "f1"})

print(f"  📈 AUC-ROC:        {lr_auc:.4f}")
print(f"  🎯 Accuracy:       {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"  🎯 Precision:      {lr_precision:.4f}")
print(f"  🎯 Recall:         {lr_recall:.4f}")
print(f"  🎯 F1-Score:       {lr_f1:.4f}")
print(f"  ⏱️  Training Time:  {lr_duration:.2f} seconds")

# Random Forest Metrics
print("\n🔹 RANDOM FOREST RESULTS:")
print("-" * 70)
rf_auc = binary_evaluator.evaluate(rf_predictions)
rf_accuracy = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "accuracy"})
rf_precision = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
rf_recall = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
rf_f1 = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "f1"})

print(f"  📈 AUC-ROC:        {rf_auc:.4f}")
print(f"  🎯 Accuracy:       {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"  🎯 Precision:      {rf_precision:.4f}")
print(f"  🎯 Recall:         {rf_recall:.4f}")
print(f"  🎯 F1-Score:       {rf_f1:.4f}")
print(f"  ⏱️  Training Time:  {rf_duration:.2f} seconds")

# Confusion Matrix
print("\n📈 CONFUSION MATRIX - Logistic Regression:")
print("-" * 70)
lr_cm = lr_predictions.groupBy("label", "prediction").count().orderBy("label", "prediction")
lr_cm.show()

print("\n📈 CONFUSION MATRIX - Random Forest:")
print("-" * 70)
rf_cm = rf_predictions.groupBy("label", "prediction").count().orderBy("label", "prediction")
rf_cm.show()

# ===== BƯỚC 7: Save Models =====
print("\n" + "="*70)
print("💾 SAVING TRAINED MODELS")
print("="*70)

lr_model_path = "hdfs://namenode:8020/user/data/models/logistic_regression_fixed"
rf_model_path = "hdfs://namenode:8020/user/data/models/random_forest_fixed"

print(f"\n📦 Saving Logistic Regression model...")
print(f"   → {lr_model_path}")
lr_model.write().overwrite().save(lr_model_path)
print("   ✅ LR Model saved!")

print(f"\n📦 Saving Random Forest model...")
print(f"   → {rf_model_path}")
rf_model.write().overwrite().save(rf_model_path)
print("   ✅ RF Model saved!")

# ===== BƯỚC 8: Save Metrics Report =====
print("\n" + "="*70)
print("📝 GENERATING METRICS REPORT")
print("="*70)

metrics_data = [
    ("Logistic Regression", lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc, lr_duration, cached_train_count),
    ("Random Forest", rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc, rf_duration, cached_train_count)
]

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

metrics_schema = StructType([
    StructField("model", StringType(), False),
    StructField("accuracy", DoubleType(), False),
    StructField("precision", DoubleType(), False),
    StructField("recall", DoubleType(), False),
    StructField("f1_score", DoubleType(), False),
    StructField("auc_roc", DoubleType(), False),
    StructField("training_time_seconds", DoubleType(), False),
    StructField("training_samples", LongType(), False)
])

metrics_df = spark.createDataFrame(metrics_data, schema=metrics_schema)
metrics_path = "hdfs://namenode:8020/user/data/results/metrics_summary_fixed"

print(f"\n💾 Saving metrics to {metrics_path}")
metrics_df.write.mode("overwrite").parquet(metrics_path)

print("\n✅ Metrics saved!")
print("\n📊 FINAL METRICS SUMMARY:")
print("-" * 70)
metrics_df.show(truncate=False)

# ===== HOÀN TẤT =====
print("\n" + "="*70)
print("🎉 FIXED BATCH ML TRAINING COMPLETED!")
print("="*70)

print(f"\n📊 TRAINING SUMMARY:")
print(f"   ✓ Training Samples: {cached_train_count:,}")
print(f"   ✓ Test Samples:     {cached_test_count:,}")
print(f"   ✓ Total Runtime:    {lr_duration + rf_duration:.2f} seconds")

print(f"\n📦 SAVED ARTIFACTS:")
print(f"   ✓ LR Model:         {lr_model_path}")
print(f"   ✓ RF Model:         {rf_model_path}")
print(f"   ✓ LR Predictions:   {lr_output_path}")
print(f"   ✓ RF Predictions:   {rf_output_path}")
print(f"   ✓ Metrics Summary:  {metrics_path}")

print(f"\n🎯 NEXT STEPS:")
print(f"   1. Check Spark History Server: http://localhost:18080")
print(f"   2. Review confusion matrices and accuracy metrics")
print(f"   3. Compare with previous run (64 samples vs {cached_train_count:,} samples)")
print(f"   4. Take screenshots for project report")

print("\n✨ Thank you for using FIXED Distributed ML Training Pipeline!")
print("="*70)

spark.stop()
