#!/usr/bin/env python3
"""
ML Training - OPTIMIZED Logistic Regression with BATCH PROCESSING
Train on 100K images using 10K batch processing to avoid OOM
- Xử lý 10K hình mỗi batch, release memory sau mỗi batch
- Sử dụng Logistic Regression (nhẹ hơn Random Forest)
- Memory-safe: Load → Train → Predict → Save → Release
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, avg as spark_avg, when,
    hash as spark_hash, concat_ws, udf
)
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
import time
import gc

print("="*80)
print("📈 ML TRAINING - LOGISTIC REGRESSION (10K BATCH PROCESSING)")
print("="*80)

# Initialize Spark with ULTRA MEMORY-SAFE config
spark = SparkSession.builder \
    .appName("ML_Training_LR_BatchProcessing") \
    .config("spark.driver.memory", "1200m") \
    .config("spark.executor.memory", "1200m") \
    .config("spark.executor.cores", "1") \
    .config("spark.default.parallelism", "10") \
    .config("spark.sql.shuffle.partitions", "10") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.memory.fraction", "0.5") \
    .config("spark.memory.storageFraction", "0.2") \
    .config("spark.cleaner.periodicGC.interval", "30s") \
    .config("spark.rdd.compress", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

# Set log level để giảm noise
spark.sparkContext.setLogLevel("WARN")

pipeline_start = time.time()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_image_id(df, source_type):
    """
    Tạo image_id duy nhất từ features hash + source info.
    Đảm bảo ID nhất quán khi join giữa các model predictions.
    """
    # Tạo hash từ features vector để có ID duy nhất cho mỗi ảnh
    # Kết hợp với source_type để tránh collision giữa REAL/FAKE
    return df.withColumn(
        "image_id",
        concat_ws("_", 
            lit(source_type),
            spark_hash(col("features").cast("string"))
        ).cast(StringType())
    )

def get_probability_at_index(prob_col, index):
    """Extract probability at specific index from probability vector."""
    return vector_to_array(prob_col)[index]

# ============================================================================
# CONFIGURATION
# ============================================================================

HDFS_BASE = "hdfs://namenode:8020/user/data"
FEATURES_BASE = f"{HDFS_BASE}/features_tf"  # NEW: TensorFlow MobileNetV2 features
NUM_BATCHES = 50  # 50 batches per category (50K REAL + 50K FAKE = 100K train)
VALIDATION_RATIO = 0.15  # 15% for validation
SEED = 42
BATCH_SIZE = 10000  # Xử lý 10K hình mỗi batch

# Logistic Regression Hyperparameters - Tối ưu cho deep features
LR_PARAMS = {
    "maxIter": 50,            # Giảm để nhanh hơn
    "regParam": 0.01,         # L2 regularization
    "elasticNetParam": 0.0,   # 0 = L2
    "tol": 1e-5,              # Tăng tolerance để converge nhanh hơn
    "fitIntercept": True,
    "standardization": True,
    "threshold": 0.5
}

def force_cleanup():
    """Aggressive memory cleanup"""
    spark.catalog.clearCache()
    for _ in range(5):
        gc.collect()
    time.sleep(2)

# ============================================================================
# STEP 1: Load và Prepare Training Data từ 5 Batches
# ============================================================================

print("\n" + "="*80)
print("📂 STEP 1: Loading Training Data (5 Mixed Batches)")
print("="*80)

all_train_data = []
total_train_samples = 0

for batch_id in range(1, NUM_BATCHES + 1):
    print(f"\n📦 Loading Batch {batch_id}/{NUM_BATCHES}...")
    
    # Load REAL
    real_path = f"{FEATURES_BASE}/train/REAL/batch_{batch_id}"
    df_real = spark.read.parquet(real_path)
    df_real = create_image_id(df_real, f"REAL_batch{batch_id}")
    real_count = df_real.count()
    
    # Load FAKE  
    fake_path = f"{FEATURES_BASE}/train/FAKE/batch_{batch_id}"
    df_fake = spark.read.parquet(fake_path)
    df_fake = create_image_id(df_fake, f"FAKE_batch{batch_id}")
    fake_count = df_fake.count()
    
    # Union và thêm batch_id
    df_batch = df_real.union(df_fake).withColumn("batch_id", lit(batch_id))
    all_train_data.append(df_batch)
    
    batch_count = real_count + fake_count
    total_train_samples += batch_count
    print(f"   ✅ REAL: {real_count:,} | FAKE: {fake_count:,} | Total: {batch_count:,}")

# Union tất cả batches
print(f"\n🔗 Combining all {NUM_BATCHES} batches...")
df_train_full = all_train_data[0]
for df in all_train_data[1:]:
    df_train_full = df_train_full.union(df)

# Repartition nhỏ hơn để tối ưu memory
df_train_full = df_train_full.repartition(20, "image_id").cache()
actual_count = df_train_full.count()

print(f"✅ Total training data: {actual_count:,} samples")
print(f"   - Expected: {total_train_samples:,}")

# ============================================================================
# STEP 2: Split Train/Validation (80/20)
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 2: Creating Train/Validation Split")
print("="*80)

# Split với stratified sampling để giữ balance giữa classes
df_train, df_val = df_train_full.randomSplit([1-VALIDATION_RATIO, VALIDATION_RATIO], seed=SEED)
df_train = df_train.cache()
df_val = df_val.cache()

train_count = df_train.count()
val_count = df_val.count()

# Kiểm tra class distribution
train_label_dist = df_train.groupBy("label").count().collect()
val_label_dist = df_val.groupBy("label").count().collect()

print(f"✅ Training set: {train_count:,} samples")
for row in train_label_dist:
    label_name = "REAL" if row["label"] == 1 else "FAKE"
    print(f"   - {label_name} (label={row['label']}): {row['count']:,}")

print(f"\n✅ Validation set: {val_count:,} samples")
for row in val_label_dist:
    label_name = "REAL" if row["label"] == 1 else "FAKE"
    print(f"   - {label_name} (label={row['label']}): {row['count']:,}")

# Unpersist full data để giải phóng memory
df_train_full.unpersist()

# ============================================================================
# STEP 3: Train Random Forest Models với Cross-Validation trên Batches
# ============================================================================

print("\n" + "="*80)
print("📈 STEP 3: Training Logistic Regression Models")
print("="*80)
print("\n🔧 HYPERPARAMETERS:")
for key, value in LR_PARAMS.items():
    print(f"   - {key}: {value}")

# Lấy unique batches để train riêng biệt (ensemble approach)
batch_ids = [row["batch_id"] for row in df_train.select("batch_id").distinct().collect()]
batch_ids.sort()

print(f"\n📦 Training {len(batch_ids)} models (one per batch)...")

model_predictions = []
individual_accuracies = []

for batch_id in batch_ids:
    print(f"\n{'='*80}")
    print(f"📦 Model {batch_id}/{len(batch_ids)}: Training on Batch {batch_id}")
    print(f"{'='*80}")
    
    batch_start = time.time()
    
    # Filter data cho batch này
    df_batch_train = df_train.filter(col("batch_id") == batch_id).cache()
    batch_count = df_batch_train.count()
    print(f"   📊 Training samples: {batch_count:,}")
    
    # Step 1: Feature Standardization (QUAN TRỌNG: tăng accuracy cho LR)
    print("   📏 Standardizing features...")
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withStd=True,
        withMean=True  # Center features around 0
    )
    scaler_model = scaler.fit(df_batch_train)
    df_batch_scaled = scaler_model.transform(df_batch_train)
    
    # Step 2: Train Logistic Regression (nhẹ hơn RF rất nhiều!)
    print("   📈 Training Logistic Regression...")
    lr = LogisticRegression(
        featuresCol="scaledFeatures",  # Dùng scaled features
        labelCol="label",
        predictionCol="prediction",
        probabilityCol="probability",
        **LR_PARAMS
    )
    
    lr_model = lr.fit(df_batch_scaled)
    print("   ✅ Training completed!")
    
    # Predict trên VALIDATION set - cần scale validation data với cùng scaler
    print("   🔮 Predicting on validation set...")
    df_val_scaled = scaler_model.transform(df_val)
    predictions = lr_model.transform(df_val_scaled)
    
    # Lưu predictions với image_id thực (QUAN TRỌNG: không dùng monotonically_increasing_id)
    # Extract probability cho class 1 (REAL) để ensemble averaging
    predictions = predictions.select(
        col("image_id"),
        col("label"),
        col("prediction").alias(f"pred_{batch_id}"),
        vector_to_array(col("probability"))[1].alias(f"prob_{batch_id}")
    )
    
    # Cache để dùng cho ensemble
    predictions = predictions.cache()
    pred_count = predictions.count()
    
    # Evaluate single model accuracy
    correct = predictions.filter(col("label") == col(f"pred_{batch_id}")).count()
    single_acc = correct / pred_count
    individual_accuracies.append(single_acc)
    print(f"   📊 Single model accuracy: {single_acc*100:.2f}%")
    
    # Lưu predictions vào HDFS
    pred_path = f"{HDFS_BASE}/predictions/rf_model_{batch_id}"
    print(f"   💾 Saving predictions to: {pred_path}")
    predictions.write.mode("overwrite").parquet(pred_path)
    
    model_predictions.append({
        'batch_id': batch_id,
        'pred_path': pred_path,
        'accuracy': single_acc
    })
    
    batch_elapsed = time.time() - batch_start
    print(f"   ✅ Model {batch_id} completed in {batch_elapsed:.2f}s")
    
    # AGGRESSIVE Memory cleanup
    df_batch_train.unpersist()
    df_batch_scaled.unpersist() if 'df_batch_scaled' in dir() else None
    predictions.unpersist()
    del lr_model
    del scaler_model
    del predictions
    del df_batch_train
    spark.catalog.clearCache()
    
    # Force garbage collection
    for _ in range(3):
        gc.collect()
    
    # Cooldown ngắn hơn vì LR nhẹ hơn RF
    print("   ⏳ Cooldown 5s...")
    time.sleep(5)

print("\n" + "="*80)
print(f"✅ All {len(batch_ids)} Logistic Regression models trained!")
print("="*80)

# Show individual accuracies
print("\n📊 Individual Model Accuracies:")
avg_individual = sum(individual_accuracies) / len(individual_accuracies)
for mp in model_predictions:
    print(f"   Model {mp['batch_id']}: {mp['accuracy']*100:.2f}%")
print(f"\n   📈 Average: {avg_individual*100:.2f}%")

# ============================================================================
# STEP 4: Ensemble Predictions - Probability Averaging (Chính xác hơn Majority Voting)
# ============================================================================

print("\n" + "="*80)
print("🔮 STEP 4: Ensemble Predictions (Probability Averaging)")
print("="*80)

# Load và join predictions sử dụng image_id thực (FIX cho bug monotonically_increasing_id)
print("📂 Loading prediction batch 1 as base...")
df_ensemble = spark.read.parquet(model_predictions[0]['pred_path'])
print(f"   ✅ Base loaded: {df_ensemble.count():,} rows")

# Join incrementally với các predictions còn lại
for mp in model_predictions[1:]:
    batch_id = mp['batch_id']
    print(f"\n📂 Loading and joining prediction batch {batch_id}...")
    
    df_pred = spark.read.parquet(mp['pred_path'])
    
    # Join bằng image_id (CHÍNH XÁC - không bị lẫn lộn giữa các ảnh)
    df_ensemble = df_ensemble.join(
        df_pred.select("image_id", f"pred_{batch_id}", f"prob_{batch_id}"),
        on="image_id",
        how="inner"
    )
    
    count = df_ensemble.count()
    print(f"   ✅ Joined: {count:,} rows")

df_ensemble = df_ensemble.repartition(10).cache()
ensemble_count = df_ensemble.count()

print(f"\n✅ Ensemble data ready: {ensemble_count:,} samples")

# ============================================================================
# STEP 5: Calculate Ensemble Prediction với Probability Averaging
# ============================================================================

print("\n" + "="*80)
print("🎯 STEP 5: Calculating Ensemble Prediction")
print("="*80)

# Tính trung bình probability từ tất cả models
# Đây là cách ensemble chính xác hơn majority voting
prob_cols = [f"prob_{mp['batch_id']}" for mp in model_predictions]
print(f"📊 Averaging probabilities from {len(prob_cols)} models...")

# Tính average probability cho class REAL (label=1)
avg_prob_expr = sum([col(c) for c in prob_cols]) / len(prob_cols)

df_ensemble = df_ensemble.withColumn("avg_probability", avg_prob_expr)

# Predict: nếu avg_prob >= 0.5 thì predict REAL (1), ngược lại FAKE (0)
df_ensemble = df_ensemble.withColumn(
    "ensemble_prediction",
    when(col("avg_probability") >= 0.5, 1.0).otherwise(0.0)
)

df_ensemble = df_ensemble.cache()

print("✅ Ensemble predictions calculated!")

# Show sample
print("\n📊 Sample predictions:")
sample_cols = ["image_id", "label"] + prob_cols[:3] + ["avg_probability", "ensemble_prediction"]
df_ensemble.select(*sample_cols).show(5, truncate=False)

# ============================================================================
# STEP 6: Evaluate Ensemble Performance trên Validation Set
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 6: Evaluating Ensemble Performance (Validation Set)")
print("="*80)

# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="ensemble_prediction",
    metricName="accuracy"
)
ensemble_acc = evaluator_acc.evaluate(df_ensemble)

# Precision
evaluator_prec = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="ensemble_prediction",
    metricName="weightedPrecision"
)
ensemble_prec = evaluator_prec.evaluate(df_ensemble)

# Recall
evaluator_rec = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="ensemble_prediction",
    metricName="weightedRecall"
)
ensemble_rec = evaluator_rec.evaluate(df_ensemble)

# F1-Score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="ensemble_prediction",
    metricName="f1"
)
ensemble_f1 = evaluator_f1.evaluate(df_ensemble)

# Tính Confusion Matrix metrics thủ công
tp = df_ensemble.filter((col("label") == 1) & (col("ensemble_prediction") == 1)).count()
tn = df_ensemble.filter((col("label") == 0) & (col("ensemble_prediction") == 0)).count()
fp = df_ensemble.filter((col("label") == 0) & (col("ensemble_prediction") == 1)).count()
fn = df_ensemble.filter((col("label") == 1) & (col("ensemble_prediction") == 0)).count()

print("\n📊 Confusion Matrix:")
print(f"   TP (REAL predicted as REAL): {tp:,}")
print(f"   TN (FAKE predicted as FAKE): {tn:,}")
print(f"   FP (FAKE predicted as REAL): {fp:,}")
print(f"   FN (REAL predicted as FAKE): {fn:,}")

# ============================================================================
# STEP 7: Save Training Results to HDFS
# ============================================================================

print("\n" + "="*80)
print("💾 STEP 7: Saving Training Results to HDFS")
print("="*80)

# Save validation predictions (với probability để evaluation step có thể tính AUC đúng)
val_predictions_path = f"{HDFS_BASE}/results/validation_predictions"
print(f"\n📊 Saving validation predictions to: {val_predictions_path}")
df_ensemble.select(
    "image_id", "label", "ensemble_prediction", "avg_probability"
).write.mode("overwrite").parquet(val_predictions_path)
print("✅ Validation predictions saved!")

# Save detailed predictions với tất cả model votes
detailed_path = f"{HDFS_BASE}/results/validation_detailed"
print(f"\n📊 Saving detailed predictions to: {detailed_path}")
df_ensemble.write.mode("overwrite").parquet(detailed_path)
print("✅ Detailed predictions saved!")

# Save training metrics
from pyspark.sql import Row
metrics_data = [
    Row(metric="Accuracy", value=float(ensemble_acc)),
    Row(metric="Precision", value=float(ensemble_prec)),
    Row(metric="Recall", value=float(ensemble_rec)),
    Row(metric="F1_Score", value=float(ensemble_f1)),
    Row(metric="Validation_Samples", value=float(ensemble_count)),
    Row(metric="Training_Samples", value=float(train_count)),
    Row(metric="Num_Models", value=float(len(model_predictions))),
    Row(metric="MaxIter", value=float(LR_PARAMS["maxIter"])),
    Row(metric="RegParam", value=float(LR_PARAMS["regParam"])),
    Row(metric="TP", value=float(tp)),
    Row(metric="TN", value=float(tn)),
    Row(metric="FP", value=float(fp)),
    Row(metric="FN", value=float(fn)),
    Row(metric="Avg_Individual_Accuracy", value=float(avg_individual))
]
df_metrics = spark.createDataFrame(metrics_data)
metrics_path = f"{HDFS_BASE}/results/training_metrics"
print(f"\n📊 Saving training metrics to: {metrics_path}")
df_metrics.write.mode("overwrite").parquet(metrics_path)
print("✅ Training metrics saved!")

# Save model info cho evaluation step
model_info_path = f"{HDFS_BASE}/results/model_info"
model_info_data = [
    Row(
        batch_id=mp['batch_id'],
        pred_path=mp['pred_path'],
        accuracy=float(mp['accuracy'])
    ) for mp in model_predictions
]
df_model_info = spark.createDataFrame(model_info_data)
print(f"\n📊 Saving model info to: {model_info_path}")
df_model_info.write.mode("overwrite").parquet(model_info_path)
print("✅ Model info saved!")

print("\n" + "="*80)
print("✅ All training results saved to HDFS!")
print("="*80)
print(f"   - Validation predictions: {val_predictions_path}")
print(f"   - Detailed predictions: {detailed_path}")
print(f"   - Training metrics: {metrics_path}")
print(f"   - Model info: {model_info_path}")

print("\n" + "="*80)
print("📈 LOGISTIC REGRESSION ENSEMBLE - TRAINING RESULTS")
print("="*80)
print(f"📊 Training samples: {train_count:,}")
print(f"📊 Validation samples: {ensemble_count:,}")
print(f"\n🎯 Validation Metrics:")
print(f"   Accuracy:  {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
print(f"   Precision: {ensemble_prec:.4f} ({ensemble_prec*100:.2f}%)")
print(f"   Recall:    {ensemble_rec:.4f} ({ensemble_rec*100:.2f}%)")
print(f"   F1-Score:  {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)")

print(f"\n📈 Ensemble Improvement:")
print(f"   Average Individual Model: {avg_individual*100:.2f}%")
print(f"   Ensemble (Prob Averaging): {ensemble_acc*100:.2f}%")
print(f"   Improvement: {(ensemble_acc - avg_individual)*100:+.2f}%")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("🏁 TRAINING PIPELINE COMPLETED")
print("="*80)
print(f"\n⏱️  Total training time: {pipeline_elapsed/60:.2f} minutes")
print(f"\n📊 Models trained: {len(model_predictions)} Logistic Regression models")
print(f"📊 Training strategy: Ensemble with Probability Averaging")
print(f"📊 Validation split: {VALIDATION_RATIO*100:.0f}%")

print("\n" + "="*80)
print("🔧 HYPERPARAMETERS")
print("="*80)
for key, value in LR_PARAMS.items():
    print(f"   {key}: {value}")

print("\n" + "="*80)
print("📊 FINAL VALIDATION METRICS")
print("="*80)
print(f"   Accuracy:  {ensemble_acc*100:.2f}%")
print(f"   Precision: {ensemble_prec*100:.2f}%")
print(f"   Recall:    {ensemble_rec*100:.2f}%")
print(f"   F1-Score:  {ensemble_f1*100:.2f}%")

print("\n" + "="*80)
print("📝 NOTES")
print("="*80)
print("   ✅ Test data NOT used in training phase")
print("   ✅ Use evaluation step to run final test with test data")
print("   ✅ Probability saved for proper AUC calculation in evaluation")
print("   ✅ Image IDs used for accurate ensemble join")

print("\n" + "="*80)
print("🚀 Next Step: Run evaluation script with test data")
print("="*80)

# Cleanup
df_train.unpersist()
df_val.unpersist()
df_ensemble.unpersist()
spark.catalog.clearCache()

spark.stop()
print("\n✅ Spark session stopped. Training complete!")
