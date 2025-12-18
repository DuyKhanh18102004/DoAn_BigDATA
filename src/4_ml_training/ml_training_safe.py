#!/usr/bin/env python3
"""
ML Training - SAFE VERSION with Checkpointing
- Lưu kết quả sau MỖI model (không mất dữ liệu khi crash)
- Giảm resource consumption để tránh OOM
- Có thể resume từ checkpoint nếu bị ngắt
"""

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import (
    col, lit, when, hash as spark_hash, concat_ws
)
from pyspark.sql.types import StringType
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
import time
import gc
import os

print("="*80)
print("🌲 ML TRAINING - SAFE VERSION WITH CHECKPOINTING")
print("="*80)

# ============================================================================
# SPARK SESSION - Optimized for stability
# ============================================================================

spark = SparkSession.builder \
    .appName("ML_Training_Safe") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.default.parallelism", "20") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.cleaner.periodicGC.interval", "5min") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

pipeline_start = time.time()

# ============================================================================
# CONFIGURATION
# ============================================================================

HDFS_BASE = "hdfs://namenode:8020/user/data"
CHECKPOINT_PATH = f"{HDFS_BASE}/checkpoints/training"
RESULTS_PATH = f"{HDFS_BASE}/results/final"
NUM_BATCHES = 5
VALIDATION_RATIO = 0.2
SEED = 42

# Hyperparameters (giảm để ổn định hơn)
RF_PARAMS = {
    "numTrees": 50,      # Giảm từ 100 để tiết kiệm memory
    "maxDepth": 12,      # Giảm từ 15
    "minInstancesPerNode": 2,
    "featureSubsetStrategy": "sqrt",
    "seed": SEED
}

print(f"\n🔧 Configuration:")
print(f"   - Checkpoint path: {CHECKPOINT_PATH}")
print(f"   - Results path: {RESULTS_PATH}")
print(f"   - Num batches: {NUM_BATCHES}")
print(f"   - RF Trees: {RF_PARAMS['numTrees']}")
print(f"   - RF Depth: {RF_PARAMS['maxDepth']}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_image_id(df, source_type):
    """Tạo unique ID cho mỗi ảnh."""
    return df.withColumn(
        "image_id",
        concat_ws("_", 
            lit(source_type),
            spark_hash(col("features").cast("string"))
        ).cast(StringType())
    )

def save_checkpoint(data, path, description):
    """Lưu checkpoint với verification."""
    print(f"   💾 Saving checkpoint: {description}")
    try:
        data.write.mode("overwrite").parquet(path)
        # Verify
        count = spark.read.parquet(path).count()
        print(f"   ✅ Checkpoint saved: {count:,} rows at {path}")
        return True
    except Exception as e:
        print(f"   ❌ Checkpoint failed: {e}")
        return False

def load_checkpoint(path):
    """Load checkpoint nếu tồn tại."""
    try:
        df = spark.read.parquet(path)
        count = df.count()
        print(f"   📂 Loaded checkpoint: {count:,} rows from {path}")
        return df
    except:
        return None

def cleanup_memory():
    """Dọn dẹp memory."""
    spark.catalog.clearCache()
    gc.collect()
    time.sleep(2)

# ============================================================================
# STEP 1: Load Training Data (từng batch một)
# ============================================================================

print("\n" + "="*80)
print("📂 STEP 1: Loading Training Data")
print("="*80)

all_data = []
total_samples = 0

for batch_id in range(1, NUM_BATCHES + 1):
    print(f"\n📦 Batch {batch_id}/{NUM_BATCHES}:")
    
    # Load REAL
    real_path = f"{HDFS_BASE}/features/train/REAL/batch_{batch_id}"
    df_real = spark.read.parquet(real_path)
    df_real = create_image_id(df_real, f"R{batch_id}")
    real_count = df_real.count()
    
    # Load FAKE
    fake_path = f"{HDFS_BASE}/features/train/FAKE/batch_{batch_id}"
    df_fake = spark.read.parquet(fake_path)
    df_fake = create_image_id(df_fake, f"F{batch_id}")
    fake_count = df_fake.count()
    
    # Union
    df_batch = df_real.union(df_fake).withColumn("batch_id", lit(batch_id))
    all_data.append(df_batch)
    
    batch_total = real_count + fake_count
    total_samples += batch_total
    print(f"   REAL: {real_count:,} | FAKE: {fake_count:,} | Total: {batch_total:,}")

# Combine all
print(f"\n🔗 Combining {NUM_BATCHES} batches...")
df_train_full = all_data[0]
for df in all_data[1:]:
    df_train_full = df_train_full.union(df)

df_train_full = df_train_full.repartition(20).cache()
actual_count = df_train_full.count()
print(f"✅ Total: {actual_count:,} samples")

# ============================================================================
# STEP 2: Split Train/Validation
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 2: Train/Validation Split")
print("="*80)

df_train, df_val = df_train_full.randomSplit([1-VALIDATION_RATIO, VALIDATION_RATIO], seed=SEED)
df_train = df_train.cache()
df_val = df_val.cache()

train_count = df_train.count()
val_count = df_val.count()

print(f"✅ Training: {train_count:,} samples")
print(f"✅ Validation: {val_count:,} samples")

# Save validation set cho reuse
val_checkpoint = f"{CHECKPOINT_PATH}/validation_set"
save_checkpoint(df_val.select("image_id", "label", "features", "batch_id"), 
                val_checkpoint, "Validation set")

# Free memory
df_train_full.unpersist()
cleanup_memory()

# ============================================================================
# STEP 3: Train Models với Checkpoint sau mỗi model
# ============================================================================

print("\n" + "="*80)
print("🌲 STEP 3: Training Models (with Checkpointing)")
print("="*80)

print(f"\n🔧 Hyperparameters: {RF_PARAMS}")

batch_ids = sorted(df_train.select("batch_id").distinct().rdd.flatMap(lambda x: x).collect())
print(f"\n📦 Training {len(batch_ids)} models...")

trained_models = []
all_results = []

for i, batch_id in enumerate(batch_ids):
    print(f"\n{'='*80}")
    print(f"📦 MODEL {i+1}/{len(batch_ids)}: Batch {batch_id}")
    print(f"{'='*80}")
    
    model_start = time.time()
    
    # Check if model already trained (resume support)
    model_checkpoint = f"{CHECKPOINT_PATH}/model_{batch_id}"
    pred_checkpoint = f"{CHECKPOINT_PATH}/predictions_{batch_id}"
    
    existing_pred = load_checkpoint(pred_checkpoint)
    if existing_pred is not None:
        print(f"   ⏭️  Model {batch_id} already trained, skipping...")
        trained_models.append({
            'batch_id': batch_id,
            'pred_path': pred_checkpoint,
            'status': 'loaded_from_checkpoint'
        })
        continue
    
    try:
        # Filter batch data
        df_batch = df_train.filter(col("batch_id") == batch_id).cache()
        batch_count = df_batch.count()
        print(f"   📊 Training samples: {batch_count:,}")
        
        # Train
        print(f"   🌲 Training Random Forest...")
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            **RF_PARAMS
        )
        
        model = rf.fit(df_batch)
        print(f"   ✅ Model trained!")
        
        # Save model
        print(f"   💾 Saving model...")
        model.write().overwrite().save(model_checkpoint)
        
        # Predict on validation
        print(f"   🔮 Predicting on validation...")
        predictions = model.transform(df_val)
        
        # Extract results
        results = predictions.select(
            col("image_id"),
            col("label"),
            col("prediction").alias(f"pred_{batch_id}"),
            vector_to_array(col("probability"))[1].alias(f"prob_{batch_id}")
        )
        
        # Calculate accuracy
        correct = results.filter(col("label") == col(f"pred_{batch_id}")).count()
        accuracy = correct / val_count
        print(f"   📊 Accuracy: {accuracy*100:.2f}%")
        
        # ⚡ CRITICAL: Save predictions checkpoint IMMEDIATELY
        save_checkpoint(results, pred_checkpoint, f"Model {batch_id} predictions")
        
        trained_models.append({
            'batch_id': batch_id,
            'pred_path': pred_checkpoint,
            'accuracy': accuracy,
            'status': 'trained'
        })
        
        all_results.append({
            'batch_id': batch_id,
            'accuracy': accuracy,
            'samples': batch_count
        })
        
        # Cleanup
        df_batch.unpersist()
        del model
        cleanup_memory()
        
        elapsed = time.time() - model_start
        print(f"   ✅ Model {batch_id} completed in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"   ❌ ERROR training model {batch_id}: {e}")
        print(f"   ⚠️  Continuing with next model...")
        continue

print("\n" + "="*80)
print(f"✅ Training completed: {len(trained_models)}/{len(batch_ids)} models")
print("="*80)

# ============================================================================
# STEP 4: Ensemble Predictions
# ============================================================================

print("\n" + "="*80)
print("🔮 STEP 4: Ensemble Predictions")
print("="*80)

if len(trained_models) == 0:
    print("❌ No models trained! Exiting...")
    spark.stop()
    exit(1)

# Load first predictions
print(f"\n📂 Loading predictions from {len(trained_models)} models...")
df_ensemble = spark.read.parquet(trained_models[0]['pred_path'])
base_batch = trained_models[0]['batch_id']

# Join remaining predictions
for model_info in trained_models[1:]:
    batch_id = model_info['batch_id']
    pred_path = model_info['pred_path']
    
    print(f"   🔗 Joining model {batch_id}...")
    df_pred = spark.read.parquet(pred_path)
    
    df_ensemble = df_ensemble.join(
        df_pred.select("image_id", f"pred_{batch_id}", f"prob_{batch_id}"),
        on="image_id",
        how="inner"
    )

df_ensemble = df_ensemble.cache()
ensemble_count = df_ensemble.count()
print(f"✅ Ensemble data: {ensemble_count:,} samples")

# ============================================================================
# STEP 5: Calculate Ensemble Prediction
# ============================================================================

print("\n" + "="*80)
print("🎯 STEP 5: Ensemble Calculation")
print("="*80)

# Average probability
prob_cols = [f"prob_{m['batch_id']}" for m in trained_models]
avg_expr = sum([col(c) for c in prob_cols]) / len(prob_cols)

df_ensemble = df_ensemble.withColumn("avg_probability", avg_expr)
df_ensemble = df_ensemble.withColumn(
    "ensemble_prediction",
    when(col("avg_probability") >= 0.5, 1.0).otherwise(0.0)
)

df_ensemble = df_ensemble.cache()
print("✅ Ensemble predictions calculated!")

# ============================================================================
# STEP 6: Evaluate & Save Final Results
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 6: Evaluation & Saving Results")
print("="*80)

# Calculate metrics
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="ensemble_prediction", metricName="accuracy")
evaluator_prec = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="ensemble_prediction", metricName="weightedPrecision")
evaluator_rec = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="ensemble_prediction", metricName="weightedRecall")
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="ensemble_prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(df_ensemble)
precision = evaluator_prec.evaluate(df_ensemble)
recall = evaluator_rec.evaluate(df_ensemble)
f1 = evaluator_f1.evaluate(df_ensemble)

# Confusion matrix
tp = df_ensemble.filter((col("label") == 1) & (col("ensemble_prediction") == 1)).count()
tn = df_ensemble.filter((col("label") == 0) & (col("ensemble_prediction") == 0)).count()
fp = df_ensemble.filter((col("label") == 0) & (col("ensemble_prediction") == 1)).count()
fn = df_ensemble.filter((col("label") == 1) & (col("ensemble_prediction") == 0)).count()

print(f"\n📊 ENSEMBLE METRICS:")
print(f"   Accuracy:  {accuracy*100:.2f}%")
print(f"   Precision: {precision*100:.2f}%")
print(f"   Recall:    {recall*100:.2f}%")
print(f"   F1-Score:  {f1*100:.2f}%")

print(f"\n📊 Confusion Matrix:")
print(f"   TP: {tp:,} | FP: {fp:,}")
print(f"   FN: {fn:,} | TN: {tn:,}")

# ⚡ CRITICAL: Save final results IMMEDIATELY
print("\n💾 Saving final results...")

# Save predictions
final_pred_path = f"{RESULTS_PATH}/predictions"
save_checkpoint(
    df_ensemble.select("image_id", "label", "ensemble_prediction", "avg_probability"),
    final_pred_path, 
    "Final predictions"
)

# Save metrics
metrics_data = [
    Row(metric="Accuracy", value=float(accuracy)),
    Row(metric="Precision", value=float(precision)),
    Row(metric="Recall", value=float(recall)),
    Row(metric="F1_Score", value=float(f1)),
    Row(metric="TP", value=float(tp)),
    Row(metric="TN", value=float(tn)),
    Row(metric="FP", value=float(fp)),
    Row(metric="FN", value=float(fn)),
    Row(metric="Validation_Samples", value=float(ensemble_count)),
    Row(metric="Training_Samples", value=float(train_count)),
    Row(metric="Num_Models", value=float(len(trained_models))),
    Row(metric="NumTrees", value=float(RF_PARAMS["numTrees"])),
    Row(metric="MaxDepth", value=float(RF_PARAMS["maxDepth"]))
]

df_metrics = spark.createDataFrame(metrics_data)
metrics_path = f"{RESULTS_PATH}/metrics"
save_checkpoint(df_metrics, metrics_path, "Final metrics")

# Save model info
model_info_data = [
    Row(
        batch_id=m['batch_id'],
        pred_path=m['pred_path'],
        accuracy=float(m.get('accuracy', 0)),
        status=m['status']
    ) for m in trained_models
]
df_model_info = spark.createDataFrame(model_info_data)
model_info_path = f"{RESULTS_PATH}/model_info"
save_checkpoint(df_model_info, model_info_path, "Model info")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("🏁 TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\n⏱️  Total time: {pipeline_elapsed/60:.2f} minutes")

print(f"\n📊 FINAL RESULTS:")
print(f"   🎯 Accuracy:  {accuracy*100:.2f}%")
print(f"   🎯 Precision: {precision*100:.2f}%")
print(f"   🎯 Recall:    {recall*100:.2f}%")
print(f"   🎯 F1-Score:  {f1*100:.2f}%")

print(f"\n💾 RESULTS SAVED TO HDFS:")
print(f"   - Predictions: {final_pred_path}")
print(f"   - Metrics: {metrics_path}")
print(f"   - Model info: {model_info_path}")
print(f"   - Checkpoints: {CHECKPOINT_PATH}")

if accuracy >= 0.85:
    print("\n🎉🎉🎉 TARGET ACHIEVED: >= 85% Accuracy!")
else:
    print(f"\n⚠️  Below target: {accuracy*100:.2f}% < 85%")
    print("   Consider: Increase numTrees or maxDepth")

print("\n" + "="*80)

# Cleanup
df_train.unpersist()
df_val.unpersist()
df_ensemble.unpersist()

spark.stop()
print("✅ Spark stopped. All results saved safely!")
