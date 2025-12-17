#!/usr/bin/env python3
"""
ML Training - 10 BATCH ENSEMBLE APPROACH
Train 10 separate models (one per batch of 10K samples)
Then ENSEMBLE predictions by majority voting
This avoids OOM by loading only 10K samples at a time
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, lit, avg as spark_avg
import time
import gc

print("="*80)
print("🤖 ML TRAINING - 10 BATCH ENSEMBLE APPROACH")
print("Train 10 models separately (10K samples each) then ENSEMBLE")
print("="*80)

# Initialize Spark with conservative memory settings
spark = SparkSession.builder \
    .appName("ML_Training_10Batch_Ensemble") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "20") \
    .getOrCreate()

pipeline_start = time.time()

# ============================================================================
# STEP 1: DEFINE BATCH PATHS (10 batches = 100K samples)
# ============================================================================

print("\n" + "📚"*40)
print("STEP 1: DEFINING TRAINING BATCHES")
print("📚"*40)

# List all 10 training batches
batch_configs = [
    ("REAL", 1, "hdfs://namenode:8020/user/data/features/train/REAL/batch_1"),
    ("REAL", 2, "hdfs://namenode:8020/user/data/features/train/REAL/batch_2"),
    ("REAL", 3, "hdfs://namenode:8020/user/data/features/train/REAL/batch_3"),
    ("REAL", 4, "hdfs://namenode:8020/user/data/features/train/REAL/batch_4"),
    ("REAL", 5, "hdfs://namenode:8020/user/data/features/train/REAL/batch_5"),
    ("FAKE", 1, "hdfs://namenode:8020/user/data/features/train/FAKE/batch_1"),
    ("FAKE", 2, "hdfs://namenode:8020/user/data/features/train/FAKE/batch_2"),
    ("FAKE", 3, "hdfs://namenode:8020/user/data/features/train/FAKE/batch_3"),
    ("FAKE", 4, "hdfs://namenode:8020/user/data/features/train/FAKE/batch_4"),
    ("FAKE", 5, "hdfs://namenode:8020/user/data/features/train/FAKE/batch_5"),
]

print(f"📋 Total batches: {len(batch_configs)}")
for class_label, batch_num, path in batch_configs:
    print(f"  • {class_label}/batch_{batch_num}: {path}")

# ============================================================================
# STEP 2: LOAD TEST DATA (20K samples)
# ============================================================================

print("\n" + "🧪"*40)
print("STEP 2: LOADING TEST DATA")
print("🧪"*40)

df_test_real = spark.read.parquet("hdfs://namenode:8020/user/data/features/test/REAL/batch_1")
df_test_fake = spark.read.parquet("hdfs://namenode:8020/user/data/features/test/FAKE/batch_1")
df_test = df_test_real.union(df_test_fake).repartition(20).cache()
test_count = df_test.count()
print(f"✅ Test data loaded: {test_count:,} samples")

# ============================================================================
# STEP 3: TRAIN 10 LOGISTIC REGRESSION MODELS (ONE PER BATCH)
# ============================================================================

print("\n" + "="*80)
print("🚀 TRAINING 10 LOGISTIC REGRESSION MODELS")
print("="*80)

lr_models = []
lr_start = time.time()

for i, (class_label, batch_num, batch_path) in enumerate(batch_configs):
    print(f"\n{'='*80}")
    print(f"📦 BATCH {i+1}/10: {class_label}/batch_{batch_num}")
    print(f"{'='*80}")
    
    # Load ONLY this batch (10K samples)
    print(f"📂 Loading {batch_path}")
    df_batch = spark.read.parquet(batch_path).repartition(10).cache()
    batch_count = df_batch.count()
    print(f"✅ Loaded {batch_count:,} samples")
    
    # Train Logistic Regression on this batch
    print(f"🔧 Training Logistic Regression model #{i+1}...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=10,
        regParam=0.01
    )
    
    model_start = time.time()
    lr_model = lr.fit(df_batch)
    model_elapsed = time.time() - model_start
    print(f"✅ Model #{i+1} trained in {model_elapsed:.2f}s")
    
    # Save model
    model_path = f"hdfs://namenode:8020/user/data/models/lr_batch_{i+1}"
    print(f"💾 Saving model to {model_path}")
    lr_model.write().overwrite().save(model_path)
    lr_models.append((i+1, model_path, lr_model))
    
    # CRITICAL: Free memory immediately
    print("🗑️  Freeing memory...")
    df_batch.unpersist()
    del df_batch
    spark.catalog.clearCache()
    gc.collect()
    
    print(f"✅ Batch {i+1}/10 completed and memory freed")
    time.sleep(3)  # Give Spark time to cleanup

lr_elapsed = time.time() - lr_start
print(f"\n✅ ALL 10 LOGISTIC REGRESSION MODELS TRAINED in {lr_elapsed/60:.2f} minutes")

# ============================================================================
# STEP 4: ENSEMBLE EVALUATION - LOGISTIC REGRESSION
# ============================================================================

print("\n" + "="*80)
print("📊 ENSEMBLE EVALUATION - LOGISTIC REGRESSION")
print("="*80)

print("🔮 Making predictions with all 10 models...")

# Collect predictions from all 10 models
all_predictions = []

for model_id, model_path, lr_model in lr_models:
    print(f"  🔍 Model #{model_id} predicting on test data...")
    predictions = lr_model.transform(df_test)
    predictions = predictions.select(
        col("label"),
        col("prediction").alias(f"pred_{model_id}"),
        col("probability").alias(f"prob_{model_id}")
    )
    all_predictions.append(predictions)

# Combine all predictions
print("🔗 Combining predictions from all 10 models...")
df_ensemble = all_predictions[0]
for pred_df in all_predictions[1:]:
    # Join on row number (they should all have same rows)
    df_ensemble = df_ensemble.join(pred_df, on="label", how="inner")

df_ensemble = df_ensemble.cache()
ensemble_count = df_ensemble.count()
print(f"✅ Ensemble predictions: {ensemble_count:,} samples")

# Calculate ensemble prediction by MAJORITY VOTING
print("🗳️  Calculating ensemble prediction by majority voting...")

# Average all predictions (0.0 or 1.0)
from pyspark.sql.functions import round as spark_round

ensemble_pred_expr = spark_round(
    (col("pred_1") + col("pred_2") + col("pred_3") + col("pred_4") + col("pred_5") +
     col("pred_6") + col("pred_7") + col("pred_8") + col("pred_9") + col("pred_10")) / 10.0
)

df_ensemble = df_ensemble.withColumn("ensemble_prediction", ensemble_pred_expr)

# Evaluate ensemble
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label", 
    predictionCol="ensemble_prediction", 
    metricName="accuracy"
)

lr_ensemble_acc = evaluator_acc.evaluate(df_ensemble)

print(f"\n📈 LOGISTIC REGRESSION ENSEMBLE RESULTS:")
print(f"   ⭐ Ensemble Accuracy: {lr_ensemble_acc*100:.2f}%")
print(f"   📊 Based on voting from 10 models")

# Show individual model accuracies
print(f"\n📊 Individual Model Accuracies:")
for model_id in range(1, 11):
    evaluator_ind = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol=f"pred_{model_id}",
        metricName="accuracy"
    )
    ind_acc = evaluator_ind.evaluate(df_ensemble)
    print(f"   Model #{model_id}: {ind_acc*100:.2f}%")

# Cleanup
df_ensemble.unpersist()
for _, _, lr_model in lr_models:
    del lr_model
lr_models.clear()
spark.catalog.clearCache()
gc.collect()
time.sleep(5)

# ============================================================================
# STEP 5: TRAIN 10 RANDOM FOREST MODELS (ONE PER BATCH)
# ============================================================================

print("\n" + "="*80)
print("🌲 TRAINING 10 RANDOM FOREST MODELS")
print("="*80)

rf_models = []
rf_start = time.time()

for i, (class_label, batch_num, batch_path) in enumerate(batch_configs):
    print(f"\n{'='*80}")
    print(f"📦 BATCH {i+1}/10: {class_label}/batch_{batch_num}")
    print(f"{'='*80}")
    
    # Load ONLY this batch (10K samples)
    print(f"📂 Loading {batch_path}")
    df_batch = spark.read.parquet(batch_path).repartition(10).cache()
    batch_count = df_batch.count()
    print(f"✅ Loaded {batch_count:,} samples")
    
    # Train Random Forest on this batch
    print(f"🌲 Training Random Forest model #{i+1}...")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=20,  # Reduced from 30 to speed up
        maxDepth=8,
        seed=42
    )
    
    model_start = time.time()
    rf_model = rf.fit(df_batch)
    model_elapsed = time.time() - model_start
    print(f"✅ Model #{i+1} trained in {model_elapsed:.2f}s")
    
    # Save model
    model_path = f"hdfs://namenode:8020/user/data/models/rf_batch_{i+1}"
    print(f"💾 Saving model to {model_path}")
    rf_model.write().overwrite().save(model_path)
    rf_models.append((i+1, model_path, rf_model))
    
    # CRITICAL: Free memory immediately
    print("🗑️  Freeing memory...")
    df_batch.unpersist()
    del df_batch
    spark.catalog.clearCache()
    gc.collect()
    
    print(f"✅ Batch {i+1}/10 completed and memory freed")
    time.sleep(3)

rf_elapsed = time.time() - rf_start
print(f"\n✅ ALL 10 RANDOM FOREST MODELS TRAINED in {rf_elapsed/60:.2f} minutes")

# ============================================================================
# STEP 6: ENSEMBLE EVALUATION - RANDOM FOREST
# ============================================================================

print("\n" + "="*80)
print("📊 ENSEMBLE EVALUATION - RANDOM FOREST")
print("="*80)

print("🔮 Making predictions with all 10 Random Forest models...")

# Collect predictions from all 10 models
all_rf_predictions = []

for model_id, model_path, rf_model in rf_models:
    print(f"  🔍 Model #{model_id} predicting on test data...")
    predictions = rf_model.transform(df_test)
    predictions = predictions.select(
        col("label"),
        col("prediction").alias(f"rf_pred_{model_id}"),
        col("probability").alias(f"rf_prob_{model_id}")
    )
    all_rf_predictions.append(predictions)

# Combine all predictions
print("🔗 Combining predictions from all 10 Random Forest models...")
df_rf_ensemble = all_rf_predictions[0]
for pred_df in all_rf_predictions[1:]:
    df_rf_ensemble = df_rf_ensemble.join(pred_df, on="label", how="inner")

df_rf_ensemble = df_rf_ensemble.cache()
rf_ensemble_count = df_rf_ensemble.count()
print(f"✅ Random Forest ensemble predictions: {rf_ensemble_count:,} samples")

# Calculate ensemble prediction by MAJORITY VOTING
print("🗳️  Calculating Random Forest ensemble prediction by majority voting...")

rf_ensemble_pred_expr = spark_round(
    (col("rf_pred_1") + col("rf_pred_2") + col("rf_pred_3") + col("rf_pred_4") + col("rf_pred_5") +
     col("rf_pred_6") + col("rf_pred_7") + col("rf_pred_8") + col("rf_pred_9") + col("rf_pred_10")) / 10.0
)

df_rf_ensemble = df_rf_ensemble.withColumn("rf_ensemble_prediction", rf_ensemble_pred_expr)

# Evaluate ensemble
evaluator_rf_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="rf_ensemble_prediction",
    metricName="accuracy"
)

rf_ensemble_acc = evaluator_rf_acc.evaluate(df_rf_ensemble)

print(f"\n📈 RANDOM FOREST ENSEMBLE RESULTS:")
print(f"   ⭐ Ensemble Accuracy: {rf_ensemble_acc*100:.2f}%")
print(f"   📊 Based on voting from 10 models")

# Show individual model accuracies
print(f"\n📊 Individual Random Forest Model Accuracies:")
for model_id in range(1, 11):
    evaluator_rf_ind = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol=f"rf_pred_{model_id}",
        metricName="accuracy"
    )
    rf_ind_acc = evaluator_rf_ind.evaluate(df_rf_ensemble)
    print(f"   Model #{model_id}: {rf_ind_acc*100:.2f}%")

# ============================================================================
# STEP 7: FINAL SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)

print(f"\n📚 Training Strategy:")
print(f"   • 10 batches × ~10K samples = ~100K total")
print(f"   • Each batch trained separately to avoid OOM")
print(f"   • Ensemble by majority voting")

print(f"\n🧪 Test Data: {test_count:,} samples")
print(f"⏱️  Total Time: {pipeline_elapsed/60:.2f} minutes")

print("\n🏆 MODEL COMPARISON:")

print(f"\n📈 Logistic Regression Ensemble (10 models):")
print(f"   Accuracy: {lr_ensemble_acc*100:.2f}%")
print(f"   Time:     {lr_elapsed/60:.2f} min")

print(f"\n📈 Random Forest Ensemble (10 models):")
print(f"   Accuracy: {rf_ensemble_acc*100:.2f}%")
print(f"   Time:     {rf_elapsed/60:.2f} min")

best_acc = max(lr_ensemble_acc, rf_ensemble_acc)
if rf_ensemble_acc > lr_ensemble_acc:
    print(f"\n🥇 BEST MODEL: Random Forest Ensemble ({rf_ensemble_acc*100:.2f}% accuracy)")
else:
    print(f"\n🥇 BEST MODEL: Logistic Regression Ensemble ({lr_ensemble_acc*100:.2f}% accuracy)")

# Check target
target = 0.85
if best_acc >= target:
    print("\n" + "🎉"*40)
    print(f"✅ TARGET ACHIEVED! {best_acc*100:.2f}% >= {target*100:.0f}%")
    print("🎉"*40)
else:
    gap = target - best_acc
    print(f"\n⚠️  Current best: {best_acc*100:.2f}%")
    print(f"   Target: {target*100:.0f}%")
    print(f"   Gap: {gap*100:.2f}%")

print("\n💾 Models saved to HDFS:")
print("   Logistic Regression: /user/data/models/lr_batch_1 to lr_batch_10")
print("   Random Forest:       /user/data/models/rf_batch_1 to rf_batch_10")

print("\n" + "="*80)
print("🎯 ENSEMBLE APPROACH COMPLETED SUCCESSFULLY!")
print("   All 100K samples used for training without OOM crashes")
print("="*80)

spark.stop()
