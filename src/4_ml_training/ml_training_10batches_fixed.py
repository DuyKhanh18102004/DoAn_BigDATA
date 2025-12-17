#!/usr/bin/env python3
"""
ML Training - 10 Batches FIXED Strategy
Train 10 models, each on MIXED REAL+FAKE data (5K REAL + 5K FAKE = 10K per batch)
Then ensemble predictions via majority voting
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time
import gc

print("="*80)
print("🎯 ML TRAINING - 10 BATCHES FIXED (MIXED REAL+FAKE)")
print("="*80)

# Initialize Spark
spark = SparkSession.builder \
    .appName("ML_Training_10Batches_Fixed") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.default.parallelism", "50") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

pipeline_start = time.time()

# ============================================================================
# STEP 1: Load Test Data ONCE (20K samples)
# ============================================================================

print("\n" + "="*80)
print("📂 STEP 1: Loading Test Data")
print("="*80)

test_real_path = "hdfs://namenode:8020/user/data/features/test/REAL/batch_1"
test_fake_path = "hdfs://namenode:8020/user/data/features/test/FAKE/batch_1"

df_test_real = spark.read.parquet(test_real_path)
df_test_fake = spark.read.parquet(test_fake_path)

df_test = df_test_real.union(df_test_fake).repartition(10).cache()
test_count = df_test.count()
print(f"✅ Test data loaded: {test_count:,} samples")
print(f"   - REAL: 10,000 samples (label=1)")
print(f"   - FAKE: 10,000 samples (label=0)")

# ============================================================================
# STEP 2: Define 10 Mixed Training Batches (5K REAL + 5K FAKE each)
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 2: Preparing 10 Mixed Batches")
print("="*80)

batch_configs = []
for i in range(1, 6):
    batch_configs.append({
        'batch_id': i,
        'real_path': f"hdfs://namenode:8020/user/data/features/train/REAL/batch_{i}",
        'fake_path': f"hdfs://namenode:8020/user/data/features/train/FAKE/batch_{i}"
    })

print(f"✅ Total mixed batches: {len(batch_configs)}")
print("   Each batch = ~5K REAL + ~5K FAKE = ~10K samples")

# ============================================================================
# STEP 3: Train 10 Logistic Regression Models
# ============================================================================

print("\n" + "="*80)
print("🤖 STEP 3: Training 10 Logistic Regression Models")
print("="*80)

lr_predictions_list = []

for config in batch_configs:
    batch_id = config['batch_id']
    print(f"\n{'='*80}")
    print(f"📦 Batch {batch_id}/5: MIXED Training")
    print(f"{'='*80}")
    
    batch_start = time.time()
    
    # Load REAL batch
    print(f"📂 Loading REAL: {config['real_path']}")
    df_real = spark.read.parquet(config['real_path'])
    real_count = df_real.count()
    print(f"   ✅ REAL: {real_count:,} samples")
    
    # Load FAKE batch
    print(f"📂 Loading FAKE: {config['fake_path']}")
    df_fake = spark.read.parquet(config['fake_path'])
    fake_count = df_fake.count()
    print(f"   ✅ FAKE: {fake_count:,} samples")
    
    # CRITICAL: Mix REAL + FAKE
    df_batch = df_real.union(df_fake).repartition(10).cache()
    batch_count = df_batch.count()
    print(f"   ✅ MIXED: {batch_count:,} samples total")
    
    # Train Logistic Regression
    print("🤖 Training Logistic Regression...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=20,  # Increased from 10
        regParam=0.01
    )
    lr_model = lr.fit(df_batch)
    
    # Save model
    model_path = f"hdfs://namenode:8020/user/data/models/lr_mixed_batch_{batch_id}"
    print(f"💾 Saving model to: {model_path}")
    lr_model.write().overwrite().save(model_path)
    
    # Predict on test data
    print("🔮 Predicting on test data...")
    predictions = lr_model.transform(df_test)
    
    # Evaluate this single model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    single_acc = evaluator.evaluate(predictions)
    print(f"   📊 Single model accuracy: {single_acc*100:.2f}%")
    
    # Save predictions with row index
    from pyspark.sql.functions import monotonically_increasing_id
    predictions = predictions.select(
        monotonically_increasing_id().alias("row_id"),
        col("label"),
        col("prediction").alias(f"pred_{batch_id}")
    )
    
    pred_path = f"hdfs://namenode:8020/user/data/predictions/lr_mixed_batch_{batch_id}"
    print(f"💾 Saving predictions to: {pred_path}")
    predictions.write.mode("overwrite").parquet(pred_path)
    
    lr_predictions_list.append((batch_id, pred_path, single_acc))
    
    batch_elapsed = time.time() - batch_start
    print(f"✅ Batch {batch_id} completed in {batch_elapsed:.2f}s")
    
    # CRITICAL MEMORY CLEANUP
    df_real.unpersist()
    df_fake.unpersist()
    df_batch.unpersist()
    del df_real, df_fake, df_batch, lr_model, predictions
    spark.catalog.clearCache()
    gc.collect()
    time.sleep(3)

print("\n" + "="*80)
print("✅ All 5 Logistic Regression models trained!")
print("="*80)

# Show individual model accuracies
print("\n📊 Individual Model Accuracies:")
for batch_id, _, acc in lr_predictions_list:
    print(f"   Model {batch_id}: {acc*100:.2f}%")

# ============================================================================
# STEP 4: Ensemble Predictions - Majority Voting
# ============================================================================

print("\n" + "="*80)
print("🔮 STEP 4: Ensemble Predictions (Majority Voting)")
print("="*80)

print("📂 Loading first prediction batch...")
df_ensemble = spark.read.parquet(lr_predictions_list[0][1])
print(f"✅ Base predictions loaded: {df_ensemble.count():,} rows")

# Join incrementally
for batch_id, pred_path, _ in lr_predictions_list[1:]:
    print(f"\n📂 Loading prediction batch {batch_id}...")
    df_pred = spark.read.parquet(pred_path)
    
    print(f"🔗 Joining predictions...")
    df_ensemble = df_ensemble.join(
        df_pred.select("row_id", f"pred_{batch_id}"),
        on="row_id",
        how="inner"
    )
    
    count = df_ensemble.count()
    print(f"✅ Joined {batch_id} predictions: {count:,} rows")
    df_ensemble = df_ensemble.cache()

print("\n🎯 Calculating ensemble prediction (majority voting)...")

# Calculate average and round
ensemble_expr = spark_round((
    col("pred_1") + col("pred_2") + col("pred_3") + col("pred_4") + col("pred_5")
) / 5.0)

df_ensemble = df_ensemble.withColumn("ensemble_prediction", ensemble_expr)
df_ensemble = df_ensemble.cache()
ensemble_count = df_ensemble.count()
print(f"✅ Ensemble predictions ready: {ensemble_count:,} samples")

# Show sample
print("\n📊 Sample predictions:")
df_ensemble.select("label", "pred_1", "pred_2", "pred_3", "pred_4", "pred_5", "ensemble_prediction").show(10)

# ============================================================================
# STEP 5: Evaluate Ensemble Performance
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 5: Evaluating Ensemble Performance")
print("="*80)

# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="ensemble_prediction",
    metricName="accuracy"
)
lr_ensemble_acc = evaluator_acc.evaluate(df_ensemble)

# Precision
evaluator_prec = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="ensemble_prediction",
    metricName="weightedPrecision"
)
lr_ensemble_prec = evaluator_prec.evaluate(df_ensemble)

# Recall
evaluator_rec = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="ensemble_prediction",
    metricName="weightedRecall"
)
lr_ensemble_rec = evaluator_rec.evaluate(df_ensemble)

# F1-Score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="ensemble_prediction",
    metricName="f1"
)
lr_ensemble_f1 = evaluator_f1.evaluate(df_ensemble)

# AUC
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="ensemble_prediction",
    metricName="areaUnderROC"
)
lr_ensemble_auc = evaluator_auc.evaluate(df_ensemble)

print("\n" + "="*80)
print("🏆 LOGISTIC REGRESSION ENSEMBLE RESULTS")
print("="*80)
print(f"📊 Test samples: {ensemble_count:,}")
print(f"🎯 Accuracy:  {lr_ensemble_acc:.4f} ({lr_ensemble_acc*100:.2f}%)")
print(f"🎯 Precision: {lr_ensemble_prec:.4f} ({lr_ensemble_prec*100:.2f}%)")
print(f"🎯 Recall:    {lr_ensemble_rec:.4f} ({lr_ensemble_rec*100:.2f}%)")
print(f"🎯 F1-Score:  {lr_ensemble_f1:.4f} ({lr_ensemble_f1*100:.2f}%)")
print(f"🎯 AUC:       {lr_ensemble_auc:.4f} ({lr_ensemble_auc*100:.2f}%)")

if lr_ensemble_acc >= 0.85:
    print("\n" + "🎉"*40)
    print("✅ TARGET ACHIEVED: Accuracy >= 85%!")
    print("🎉"*40)
else:
    print("\n" + "⚠️"*40)
    print(f"⚠️ Target not met: {lr_ensemble_acc*100:.2f}% < 85%")
    print("⚠️ Will proceed with Random Forest...")
    print("⚠️"*40)

# ============================================================================
# STEP 6: Train 5 Random Forest Models
# ============================================================================

print("\n" + "="*80)
print("🌲 STEP 6: Training 5 Random Forest Models")
print("="*80)

rf_predictions_list = []

for config in batch_configs:
    batch_id = config['batch_id']
    print(f"\n{'='*80}")
    print(f"📦 Batch {batch_id}/5: MIXED Training")
    print(f"{'='*80}")
    
    batch_start = time.time()
    
    # Load REAL + FAKE
    df_real = spark.read.parquet(config['real_path'])
    df_fake = spark.read.parquet(config['fake_path'])
    df_batch = df_real.union(df_fake).repartition(10).cache()
    batch_count = df_batch.count()
    print(f"✅ MIXED batch loaded: {batch_count:,} samples")
    
    # Train Random Forest
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=30,  # Increased from 20
        maxDepth=10,  # Increased from 8
        seed=42
    )
    rf_model = rf.fit(df_batch)
    
    # Save model
    model_path = f"hdfs://namenode:8020/user/data/models/rf_mixed_batch_{batch_id}"
    print(f"💾 Saving model to: {model_path}")
    rf_model.write().overwrite().save(model_path)
    
    # Predict on test data
    print("🔮 Predicting on test data...")
    predictions = rf_model.transform(df_test)
    
    # Evaluate single model
    single_acc = evaluator.evaluate(predictions)
    print(f"   📊 Single model accuracy: {single_acc*100:.2f}%")
    
    # Save predictions
    from pyspark.sql.functions import monotonically_increasing_id
    predictions = predictions.select(
        monotonically_increasing_id().alias("row_id"),
        col("label"),
        col("prediction").alias(f"pred_{batch_id}")
    )
    
    pred_path = f"hdfs://namenode:8020/user/data/predictions/rf_mixed_batch_{batch_id}"
    print(f"💾 Saving predictions to: {pred_path}")
    predictions.write.mode("overwrite").parquet(pred_path)
    
    rf_predictions_list.append((batch_id, pred_path, single_acc))
    
    batch_elapsed = time.time() - batch_start
    print(f"✅ Batch {batch_id} completed in {batch_elapsed:.2f}s")
    
    # Memory cleanup
    df_real.unpersist()
    df_fake.unpersist()
    df_batch.unpersist()
    del df_real, df_fake, df_batch, rf_model, predictions
    spark.catalog.clearCache()
    gc.collect()
    time.sleep(3)

print("\n" + "="*80)
print("✅ All 5 Random Forest models trained!")
print("="*80)

print("\n📊 Individual RF Model Accuracies:")
for batch_id, _, acc in rf_predictions_list:
    print(f"   Model {batch_id}: {acc*100:.2f}%")

# ============================================================================
# STEP 7: Random Forest Ensemble
# ============================================================================

print("\n" + "="*80)
print("🔮 STEP 7: Random Forest Ensemble Predictions")
print("="*80)

df_rf_ensemble = spark.read.parquet(rf_predictions_list[0][1])
print(f"✅ Base predictions loaded: {df_rf_ensemble.count():,} rows")

for batch_id, pred_path, _ in rf_predictions_list[1:]:
    df_pred = spark.read.parquet(pred_path)
    df_rf_ensemble = df_rf_ensemble.join(
        df_pred.select("row_id", f"pred_{batch_id}"),
        on="row_id",
        how="inner"
    )
    df_rf_ensemble = df_rf_ensemble.cache()

rf_ensemble_expr = spark_round((
    col("pred_1") + col("pred_2") + col("pred_3") + col("pred_4") + col("pred_5")
) / 5.0)

df_rf_ensemble = df_rf_ensemble.withColumn("ensemble_prediction", rf_ensemble_expr)
df_rf_ensemble = df_rf_ensemble.cache()

# ============================================================================
# STEP 8: Evaluate Random Forest Ensemble
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 8: Evaluating Random Forest Ensemble")
print("="*80)

rf_acc = evaluator_acc.evaluate(df_rf_ensemble)
rf_prec = evaluator_prec.evaluate(df_rf_ensemble)
rf_rec = evaluator_rec.evaluate(df_rf_ensemble)
rf_f1 = evaluator_f1.evaluate(df_rf_ensemble)
rf_auc = evaluator_auc.evaluate(df_rf_ensemble)

print("\n" + "="*80)
print("🌲 RANDOM FOREST ENSEMBLE RESULTS")
print("="*80)
print(f"📊 Test samples: {df_rf_ensemble.count():,}")
print(f"🎯 Accuracy:  {rf_acc:.4f} ({rf_acc*100:.2f}%)")
print(f"🎯 Precision: {rf_prec:.4f} ({rf_prec*100:.2f}%)")
print(f"🎯 Recall:    {rf_rec:.4f} ({rf_rec*100:.2f}%)")
print(f"🎯 F1-Score:  {rf_f1:.4f} ({rf_f1*100:.2f}%)")
print(f"🎯 AUC:       {rf_auc:.4f} ({rf_auc*100:.2f}%)")

if rf_acc >= 0.85:
    print("\n" + "🎉"*40)
    print("✅ TARGET ACHIEVED: Accuracy >= 85%!")
    print("🎉"*40)
else:
    print("\n" + "⚠️"*40)
    print(f"⚠️ Target not met: {rf_acc*100:.2f}% < 85%")
    print("⚠️"*40)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("🏁 FINAL SUMMARY")
print("="*80)
print(f"\n⏱️  Total pipeline time: {pipeline_elapsed/60:.2f} minutes")
print(f"\n📊 Models trained: 10 models (5 LR + 5 RF)")
print(f"📊 Training strategy: MIXED batches (50% REAL + 50% FAKE)")
print(f"📊 Test samples: {test_count:,}")

print("\n" + "="*80)
print("🏆 PERFORMANCE COMPARISON")
print("="*80)
print(f"{'Metric':<15} {'LR Ensemble':<15} {'RF Ensemble':<15}")
print("-" * 45)
print(f"{'Accuracy':<15} {lr_ensemble_acc*100:>6.2f}%        {rf_acc*100:>6.2f}%")
print(f"{'Precision':<15} {lr_ensemble_prec*100:>6.2f}%        {rf_prec*100:>6.2f}%")
print(f"{'Recall':<15} {lr_ensemble_rec*100:>6.2f}%        {rf_rec*100:>6.2f}%")
print(f"{'F1-Score':<15} {lr_ensemble_f1*100:>6.2f}%        {rf_f1*100:>6.2f}%")
print(f"{'AUC':<15} {lr_ensemble_auc*100:>6.2f}%        {rf_auc*100:>6.2f}%")

best_model = "Logistic Regression" if lr_ensemble_acc > rf_acc else "Random Forest"
best_acc = max(lr_ensemble_acc, rf_acc)

print("\n" + "="*80)
print(f"🏆 BEST MODEL: {best_model}")
print(f"🏆 BEST ACCURACY: {best_acc*100:.2f}%")
print("="*80)

if best_acc >= 0.85:
    print("\n" + "🎉"*40)
    print("✅✅✅ SUCCESS: Target accuracy >= 85% achieved!")
    print("🎉"*40)
else:
    print("\n" + "⚠️"*40)
    print(f"⚠️ Target not met: {best_acc*100:.2f}% < 85%")
    print("⚠️ Possible reasons:")
    print("   1. Features not discriminative enough")
    print("   2. Need more training iterations")
    print("   3. Need better hyperparameter tuning")
    print("⚠️"*40)

print("\n" + "="*80)
spark.stop()
