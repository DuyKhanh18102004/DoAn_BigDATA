"""
Step 3: Distributed ML Training - Deepfake Detection Pipeline
Huấn luyện model phân loại bằng Spark MLlib
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import time

# ===== BƯỚC 1: Khởi tạo Spark Session =====
print("🚀 Initializing Spark Session for ML Training...")
spark = SparkSession.builder \
    .appName("DeepfakeDetection-MLTraining") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .getOrCreate()

print(f"✅ Spark version: {spark.version}")

# ===== BƯỚC 2: Load Features từ HDFS =====
print("\n📂 Loading features from HDFS (Parquet)...")

train_path = "hdfs://namenode:8020/user/data/features/train"
test_path = "hdfs://namenode:8020/user/data/features/test"

print(f"  - Reading train features from {train_path}")
train_df = spark.read.parquet(train_path)

print(f"  - Reading test features from {test_path}")
test_df = spark.read.parquet(test_path)

print(f"\n✅ Loaded {train_df.count()} training samples")
print(f"✅ Loaded {test_df.count()} test samples")

# Show schema
print("\n📋 Schema:")
train_df.printSchema()

# ===== BƯỚC 3: Chuyển đổi Features sang Vector =====
print("\n🔄 Converting features to Vector type...")

# VectorAssembler để chuyển array thành Vector
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

def list_to_vector(features_list):
    """Convert list of floats to DenseVector"""
    return Vectors.dense(features_list)

# Create UDF
list_to_vector_udf = udf(list_to_vector, VectorUDT())

# Apply transformation
train_ml = train_df.select(
    col("path"),
    list_to_vector_udf(col("features")).alias("features"),
    col("label")
)

test_ml = test_df.select(
    col("path"),
    list_to_vector_udf(col("features")).alias("features"),
    col("label")
)

print("✅ Features converted to Vector type")
train_ml.show(5)

# Cache data cho training nhanh hơn
train_ml.cache()
test_ml.cache()

# ===== BƯỚC 4: Train Logistic Regression =====
print("\n🤖 Training Logistic Regression Model...")
print("=" * 60)

lr_start = time.time()

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.8
)

print("  - Fitting model on training data...")
lr_model = lr.fit(train_ml)

lr_duration = time.time() - lr_start
print(f"✅ Logistic Regression trained in {lr_duration:.2f} seconds")

# Predict on test set
print("  - Making predictions on test set...")
lr_predictions = lr_model.transform(test_ml)

# Save predictions
lr_output_path = "hdfs://namenode:8020/user/data/results/lr_predictions"
print(f"  - Saving predictions to {lr_output_path}")
lr_predictions.select("path", "label", "prediction", "probability").write.mode("overwrite").parquet(lr_output_path)

# ===== BƯỚC 5: Train Random Forest =====
print("\n🌲 Training Random Forest Model...")
print("=" * 60)

rf_start = time.time()

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=10,
    seed=42
)

print("  - Fitting model on training data...")
rf_model = rf.fit(train_ml)

rf_duration = time.time() - rf_start
print(f"✅ Random Forest trained in {rf_duration:.2f} seconds")

# Predict on test set
print("  - Making predictions on test set...")
rf_predictions = rf_model.transform(test_ml)

# Save predictions
rf_output_path = "hdfs://namenode:8020/user/data/results/rf_predictions"
print(f"  - Saving predictions to {rf_output_path}")
rf_predictions.select("path", "label", "prediction", "probability").write.mode("overwrite").parquet(rf_output_path)

# ===== BƯỚC 6: Model Evaluation =====
print("\n📊 MODEL EVALUATION")
print("=" * 60)

# Evaluators
binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

# Logistic Regression Metrics
print("\n🔹 LOGISTIC REGRESSION:")
lr_auc = binary_evaluator.evaluate(lr_predictions)
lr_accuracy = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "accuracy"})
lr_precision = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
lr_recall = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
lr_f1 = multiclass_evaluator.evaluate(lr_predictions, {multiclass_evaluator.metricName: "f1"})

print(f"  - AUC-ROC:        {lr_auc:.4f}")
print(f"  - Accuracy:       {lr_accuracy:.4f}")
print(f"  - Precision:      {lr_precision:.4f}")
print(f"  - Recall:         {lr_recall:.4f}")
print(f"  - F1-Score:       {lr_f1:.4f}")
print(f"  - Training Time:  {lr_duration:.2f}s")

# Random Forest Metrics
print("\n🔹 RANDOM FOREST:")
rf_auc = binary_evaluator.evaluate(rf_predictions)
rf_accuracy = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "accuracy"})
rf_precision = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
rf_recall = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
rf_f1 = multiclass_evaluator.evaluate(rf_predictions, {multiclass_evaluator.metricName: "f1"})

print(f"  - AUC-ROC:        {rf_auc:.4f}")
print(f"  - Accuracy:       {rf_accuracy:.4f}")
print(f"  - Precision:      {rf_precision:.4f}")
print(f"  - Recall:         {rf_recall:.4f}")
print(f"  - F1-Score:       {rf_f1:.4f}")
print(f"  - Training Time:  {rf_duration:.2f}s")

# Confusion Matrix
print("\n📈 CONFUSION MATRIX (Logistic Regression):")
lr_predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

print("\n📈 CONFUSION MATRIX (Random Forest):")
rf_predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

# ===== BƯỚC 7: Save Models =====
print("\n💾 Saving trained models to HDFS...")

lr_model_path = "hdfs://namenode:8020/user/data/models/logistic_regression"
rf_model_path = "hdfs://namenode:8020/user/data/models/random_forest"

print(f"  - Saving LR model to {lr_model_path}")
lr_model.write().overwrite().save(lr_model_path)

print(f"  - Saving RF model to {rf_model_path}")
rf_model.write().overwrite().save(rf_model_path)

# ===== BƯỚC 8: Save Metrics Report =====
print("\n📝 Generating metrics report...")

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

print(f"  - Saving metrics to {metrics_path}")
metrics_df.write.mode("overwrite").parquet(metrics_path)

print("\n✅ Metrics saved!")
metrics_df.show(truncate=False)

# ===== HOÀN TẤT =====
print("\n" + "="*60)
print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)

print("\n📦 Saved Artifacts:")
print(f"  ✓ LR Model:        {lr_model_path}")
print(f"  ✓ RF Model:        {rf_model_path}")
print(f"  ✓ LR Predictions:  {lr_output_path}")
print(f"  ✓ RF Predictions:  {rf_output_path}")
print(f"  ✓ Metrics Summary: {metrics_path}")

print("\n🎯 Next Steps:")
print("  1. Check Spark History Server at http://localhost:18080")
print("  2. Review metrics and confusion matrices")
print("  3. Prepare business insight report")

spark.stop()
