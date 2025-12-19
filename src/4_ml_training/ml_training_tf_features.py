#!/usr/bin/env python3
"""ML Training with TensorFlow MobileNetV2 Features.

Tuned hyperparameters for improved accuracy.
"""

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import (
    col, lit, when, hash as spark_hash, concat_ws
)
from pyspark.sql.types import StringType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
import time
import gc

print("="*80)
print("ML TRAINING - TENSORFLOW MOBILENETV2 FEATURES")
print("="*80)

# Initialize Spark
spark = SparkSession.builder \
    .appName("ML_Training_TF_Features") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "1") \
    .config("spark.default.parallelism", "4") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.cleaner.periodicGC.interval", "30s") \
    .config("spark.rdd.compress", "true") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .config("spark.history.fs.logDirectory", "hdfs://namenode:8020/spark-logs") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
pipeline_start = time.time()

# ============================================================================
# CONFIGURATION
# ============================================================================

HDFS_BASE = "hdfs://namenode:8020/user/data"
FEATURES_BASE = f"{HDFS_BASE}/features_tf"  # TensorFlow MobileNetV2 features
RESULTS_BASE = f"{HDFS_BASE}/results_tf"

NUM_TRAIN_BATCHES = 50
VALIDATION_RATIO = 0.2
SEED = 42

# LR Hyperparameters
LR_PARAMS = {
    "maxIter": 300,
    "regParam": 0.001,
    "elasticNetParam": 0.0,
    "tol": 1e-5,
    "fitIntercept": True,
    "standardization": True,
    "threshold": 0.5
}

def force_cleanup():
    """Aggressive memory cleanup."""
    spark.catalog.clearCache()
    for _ in range(3):
        gc.collect()
    time.sleep(1)

def create_image_id(df, source_type):
    """Create unique image ID.
    
    Args:
        df: DataFrame with features column
        source_type: Source type identifier
        
    Returns:
        DataFrame with added image_id column
    """
    return df.withColumn(
        "image_id",
        concat_ws("_", 
            lit(source_type),
            spark_hash(col("features").cast("string"))
        ).cast(StringType())
    )

print("\n" + "="*80)
print(f"STEP 1: Loading Training Data ({NUM_TRAIN_BATCHES} batches)")
print("="*80)

all_data = []
total_samples = 0

for batch_id in range(1, NUM_TRAIN_BATCHES + 1):
    print(f"\nLoading Batch {batch_id}/{NUM_TRAIN_BATCHES}...")
    
    try:
        real_path = f"{FEATURES_BASE}/train/REAL/batch_{batch_id}"
        df_real = spark.read.parquet(real_path)
        df_real = create_image_id(df_real, f"REAL_b{batch_id}")
        real_count = df_real.count()
        
        fake_path = f"{FEATURES_BASE}/train/FAKE/batch_{batch_id}"
        df_fake = spark.read.parquet(fake_path)
        df_fake = create_image_id(df_fake, f"FAKE_b{batch_id}")
        fake_count = df_fake.count()
        
        df_batch = df_real.union(df_fake)
        all_data.append(df_batch)
        
        batch_total = real_count + fake_count
        total_samples += batch_total
        print(f"   REAL: {real_count:,} | FAKE: {fake_count:,} | Total: {batch_total:,}")
        
    except Exception as e:
        print(f"   Error loading batch {batch_id}: {e}")

print(f"\nCombining {len(all_data)} batches...")
df_full = all_data[0]
for df in all_data[1:]:
    df_full = df_full.union(df)

df_full = df_full.repartition(16).cache()
actual_count = df_full.count()

print(f"\nTotal loaded: {actual_count:,} samples")

# Verify feature quality
print("\nVerifying feature quality...")
sample = df_full.select("features", "label").take(3)
for i, row in enumerate(sample):
    feat_arr = row.features.toArray()
    import numpy as np
    feat_sum = np.sum(np.abs(feat_arr))
    feat_dim = len(feat_arr)
    print(f"   Sample {i+1}: dim={feat_dim}, sum={feat_sum:.2f}, label={row.label}")

# ============================================================================
# STEP 2: Train/Validation Split
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Train/Validation Split")
print("="*80)

df_train, df_val = df_full.randomSplit([1-VALIDATION_RATIO, VALIDATION_RATIO], seed=SEED)
df_train = df_train.cache()
df_val = df_val.cache()

train_count = df_train.count()
val_count = df_val.count()

print(f"Training: {train_count:,} samples")
print(f"Validation: {val_count:,} samples")

train_dist = df_train.groupBy("label").count().collect()
print("\nTraining class distribution:")
for row in train_dist:
    label_name = "REAL" if row["label"] == 1 else "FAKE"
    pct = 100 * row["count"] / train_count
    print(f"   - {label_name}: {row['count']:,} ({pct:.1f}%)")

# Free memory
df_full.unpersist()
force_cleanup()

# ============================================================================
# STEP 3: Train Logistic Regression
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Training Logistic Regression (Tuned Hyperparameters)")
print("="*80)

print("\nHyperparameters:")
for k, v in LR_PARAMS.items():
    print(f"   - {k}: {v}")

train_start = time.time()

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    **LR_PARAMS
)

print("\nTraining model...")
model = lr.fit(df_train)

train_time = time.time() - train_start
print(f"Training completed in {train_time:.1f}s")

# Model info
print(f"\nModel Info:")
print(f"   - Iterations: {model.summary.totalIterations}")
print(f"   - Objective History (last 5): {model.summary.objectiveHistory[-5:]}")

# ============================================================================
# STEP 4: Evaluate on Validation Set
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Validation Evaluation")
print("="*80)

predictions = model.transform(df_val)
predictions = predictions.cache()

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

print(f"\nValidation Metrics:")
print(f"   Accuracy:  {accuracy*100:.2f}%")
print(f"   Precision: {precision*100:.2f}%")
print(f"   Recall:    {recall*100:.2f}%")
print(f"   F1-Score:  {f1*100:.2f}%")

# Confusion Matrix
tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
tn = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

print(f"\nConfusion Matrix:")
print(f"   TP (REAL→REAL): {tp:,}")
print(f"   TN (FAKE→FAKE): {tn:,}")
print(f"   FP (FAKE→REAL): {fp:,}")
print(f"   FN (REAL→FAKE): {fn:,}")

# ============================================================================
# STEP 5: Test on Test Set
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Test Set Evaluation")
print("="*80)

print("Loading test data...")
test_data = []

for batch_id in range(1, 11):
    try:
        test_real_path = f"{FEATURES_BASE}/test/REAL/batch_{batch_id}"
        df_test_real = spark.read.parquet(test_real_path)
        
        test_fake_path = f"{FEATURES_BASE}/test/FAKE/batch_{batch_id}"
        df_test_fake = spark.read.parquet(test_fake_path)
        
        test_data.append(df_test_real)
        test_data.append(df_test_fake)
        
    except Exception as e:
        print(f"   Error loading test batch {batch_id}: {e}")

df_test = test_data[0]
for df in test_data[1:]:
    df_test = df_test.union(df)

df_test = df_test.repartition(8).cache()
test_count = df_test.count()
print(f"Test data loaded: {test_count:,} samples")

test_predictions = model.transform(df_test)
test_predictions = test_predictions.cache()

test_accuracy = evaluator_acc.evaluate(test_predictions)
test_precision = evaluator_prec.evaluate(test_predictions)
test_recall = evaluator_rec.evaluate(test_predictions)
test_f1 = evaluator_f1.evaluate(test_predictions)

print(f"\nTest Metrics:")
print(f"   Accuracy:  {test_accuracy*100:.2f}%")
print(f"   Precision: {test_precision*100:.2f}%")
print(f"   Recall:    {test_recall*100:.2f}%")
print(f"   F1-Score:  {test_f1*100:.2f}%")

# Test Confusion Matrix
test_tp = test_predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
test_tn = test_predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
test_fp = test_predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
test_fn = test_predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

print(f"\nTest Confusion Matrix:")
print(f"   TP (REAL→REAL): {test_tp:,}")
print(f"   TN (FAKE→FAKE): {test_tn:,}")
print(f"   FP (FAKE→REAL): {test_fp:,}")
print(f"   FN (REAL→FAKE): {test_fn:,}")

# ============================================================================
# STEP 6: Save Results
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Saving Results")
print("="*80)

metrics_data = [
    Row(metric="val_accuracy", value=float(accuracy)),
    Row(metric="val_precision", value=float(precision)),
    Row(metric="val_recall", value=float(recall)),
    Row(metric="val_f1", value=float(f1)),
    Row(metric="test_accuracy", value=float(test_accuracy)),
    Row(metric="test_precision", value=float(test_precision)),
    Row(metric="test_recall", value=float(test_recall)),
    Row(metric="test_f1", value=float(test_f1)),
    Row(metric="train_samples", value=float(train_count)),
    Row(metric="val_samples", value=float(val_count)),
    Row(metric="test_samples", value=float(test_count)),
    Row(metric="training_time_seconds", value=float(train_time)),
]

df_metrics = spark.createDataFrame(metrics_data)
metrics_path = f"{RESULTS_BASE}/metrics_tuned"
df_metrics.write.mode("overwrite").parquet(metrics_path)
print(f"Metrics saved to: {metrics_path}")

test_pred_path = f"{RESULTS_BASE}/test_predictions_tuned"
test_predictions.select(
    "label", "prediction",
    vector_to_array("probability")[1].alias("prob_real")
).write.mode("overwrite").parquet(test_pred_path)
print(f"Test predictions saved to: {test_pred_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

total_time = time.time() - pipeline_start

print("\n" + "="*80)
print("TRAINING COMPLETE (TUNED HYPERPARAM)")
print("="*80)

print(f"\nTotal time: {total_time/60:.2f} minutes")
print(f"\nDataset:")
print(f"   - Training: {train_count:,} samples ({NUM_TRAIN_BATCHES} batches)")
print(f"   - Validation: {val_count:,} samples")
print(f"   - Test: {test_count:,} samples")

print(f"\nVALIDATION RESULTS:")
print(f"   Accuracy:  {accuracy*100:.2f}%")
print(f"   F1-Score:  {f1*100:.2f}%")

print(f"\nTEST RESULTS:")
print(f"   Accuracy:  {test_accuracy*100:.2f}%")
print(f"   F1-Score:  {test_f1*100:.2f}%")

print("\n" + "="*80)
if test_accuracy > 0.90:
    print("EXCELLENT! Đạt >90%")
elif test_accuracy > 0.88:
    print("VERY GOOD! Cải thiện so với baseline")
elif test_accuracy > 0.7:
    print("Model performs WELL!")
else:
    print("Có thể thử regParam nhỏ hơn nữa")
print("="*80)

# Cleanup
predictions.unpersist()
test_predictions.unpersist()
df_train.unpersist()
df_val.unpersist()
df_test.unpersist()

spark.stop()
print("\nSpark session stopped.")