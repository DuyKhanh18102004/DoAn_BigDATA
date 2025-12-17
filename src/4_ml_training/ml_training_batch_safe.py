"""
Step 3: BATCH ML Training - Safe Incremental Approach
Train models on FULL 100K data by loading batches incrementally
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import time

# ===== BƯỚC 1: Khởi tạo Spark Session =====
print("🚀 Initializing Spark Session for BATCH ML Training...")
spark = SparkSession.builder \
    .appName("DeepfakeDetection-BatchMLTraining") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .getOrCreate()

print(f"✅ Spark version: {spark.version}")

# ===== BƯỚC 2: Load ALL Features từ HDFS (BATCH BY BATCH) =====
print("\n📂 Loading features from HDFS in SAFE BATCHES...")
print("=" * 70)

# Define all batch paths
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

# Combine all training batch paths
all_train_batches = train_real_batches + train_fake_batches

print(f"📦 Total training batches: {len(all_train_batches)}")
print(f"   - REAL batches: {len(train_real_batches)}")
print(f"   - FAKE batches: {len(train_fake_batches)}")

# ===== STRATEGY: Load batches incrementally and union =====
print("\n🔄 Loading training batches incrementally...")

train_dfs = []
for i, batch_path in enumerate(all_train_batches, 1):
    print(f"  [{i}/{len(all_train_batches)}] Loading {batch_path.split('/')[-2]}/{batch_path.split('/')[-1]}...")
    batch_df = spark.read.parquet(batch_path)
    train_dfs.append(batch_df)

print("\n🔗 Combining all training batches...")
train_df = train_dfs[0]
for df in train_dfs[1:]:
    train_df = train_df.union(df)

train_count = train_df.count()
print(f"✅ Total training samples: {train_count:,}")

# Load test data
print("\n📂 Loading test data...")
test_dfs = []
for test_path in test_paths:
    print(f"  - Loading {test_path.split('/')[-1]}...")
    test_df = spark.read.parquet(test_path)
    test_dfs.append(test_df)

test_df = test_dfs[0].union(test_dfs[1])
test_count = test_df.count()
print(f"✅ Total test samples: {test_count:,}")

# Show label distribution
print("\n📊 Training Label Distribution:")
train_df.groupBy("label").count().orderBy("label").show()

print("\n📊 Test Label Distribution:")
test_df.groupBy("label").count().orderBy("label").show()

# ===== BƯỚC 3: Chuyển đổi Features sang Vector =====
print("\n🔄 Converting features to Vector type...")

def list_to_vector(features_list):
    """Convert list of floats to DenseVector"""
    return Vectors.dense(features_list)

# Create UDF
list_to_vector_udf = udf(list_to_vector, VectorUDT())

# Apply transformation
print("  - Converting training features...")
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

# Repartition for better performance
print("\n⚡ Repartitioning data for optimal training...")
train_ml = train_ml.repartition(100)
test_ml = test_ml.repartition(50)

# Cache data cho training nhanh hơn
print("💾 Caching data for faster training...")
train_ml.cache()
test_ml.cache()

# Force cache by triggering action
print(f"  - Training samples cached: {train_ml.count():,}")
print(f"  - Test samples cached: {test_ml.count():,}")

print("\n✅ Data preparation completed!")
train_ml.show(5)

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
    aggregationDepth=5  # Help with deep trees in distributed training
)

print(f"📋 Model Configuration:")
print(f"   - Max Iterations: {lr.getMaxIter()}")
print(f"   - Regularization: {lr.getRegParam()}")
print(f"   - ElasticNet Param: {lr.getElasticNetParam()}")

print("\n🏋️ Fitting Logistic Regression on 100K training samples...")
lr_model = lr.fit(train_ml)

lr_duration = time.time() - lr_start
print(f"✅ Logistic Regression trained in {lr_duration:.2f} seconds")

# Predict on test set
print("\n🔮 Making predictions on test set...")
lr_predictions = lr_model.transform(test_ml)

# Save predictions
lr_output_path = "hdfs://namenode:8020/user/data/results/lr_predictions"
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
    subsamplingRate=0.8  # Use 80% of data per tree for faster training
)

print(f"📋 Model Configuration:")
print(f"   - Number of Trees: {rf.getNumTrees()}")
print(f"   - Max Depth: {rf.getMaxDepth()}")
print(f"   - Max Bins: {rf.getMaxBins()}")
print(f"   - Subsampling Rate: {rf.getSubsamplingRate()}")

print("\n🏋️ Fitting Random Forest on 100K training samples...")
rf_model = rf.fit(train_ml)

rf_duration = time.time() - rf_start
print(f"✅ Random Forest trained in {rf_duration:.2f} seconds")

# Predict on test set
print("\n🔮 Making predictions on test set...")
rf_predictions = rf_model.transform(test_ml)

# Save predictions
rf_output_path = "hdfs://namenode:8020/user/data/results/rf_predictions"
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

lr_model_path = "hdfs://namenode:8020/user/data/models/logistic_regression"
rf_model_path = "hdfs://namenode:8020/user/data/models/random_forest"

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
    ("Logistic Regression", lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc, lr_duration),
    ("Random Forest", rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc, rf_duration)
]

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

metrics_schema = StructType([
    StructField("model", StringType(), False),
    StructField("accuracy", DoubleType(), False),
    StructField("precision", DoubleType(), False),
    StructField("recall", DoubleType(), False),
    StructField("f1_score", DoubleType(), False),
    StructField("auc_roc", DoubleType(), False),
    StructField("training_time_seconds", DoubleType(), False)
])

metrics_df = spark.createDataFrame(metrics_data, schema=metrics_schema)
metrics_path = "hdfs://namenode:8020/user/data/results/metrics_summary"

print(f"\n💾 Saving metrics to {metrics_path}")
metrics_df.write.mode("overwrite").parquet(metrics_path)

print("\n✅ Metrics saved!")
print("\n📊 FINAL METRICS SUMMARY:")
print("-" * 70)
metrics_df.show(truncate=False)

# ===== HOÀN TẤT =====
print("\n" + "="*70)
print("🎉 BATCH ML TRAINING PIPELINE COMPLETED!")
print("="*70)

print("\n📊 TRAINING SUMMARY:")
print(f"   ✓ Training Samples: {train_count:,}")
print(f"   ✓ Test Samples:     {test_count:,}")
print(f"   ✓ Total Runtime:    {lr_duration + rf_duration:.2f} seconds")

print("\n📦 SAVED ARTIFACTS:")
print(f"   ✓ LR Model:         {lr_model_path}")
print(f"   ✓ RF Model:         {rf_model_path}")
print(f"   ✓ LR Predictions:   {lr_output_path}")
print(f"   ✓ RF Predictions:   {rf_output_path}")
print(f"   ✓ Metrics Summary:  {metrics_path}")

print("\n🎯 NEXT STEPS:")
print("   1. Check Spark History Server: http://localhost:18080")
print("   2. Review confusion matrices and accuracy metrics")
print("   3. Take screenshots for project report")
print("   4. Prepare business insights presentation")

print("\n✨ Thank you for using Distributed ML Training Pipeline!")
print("="*70)

spark.stop()
