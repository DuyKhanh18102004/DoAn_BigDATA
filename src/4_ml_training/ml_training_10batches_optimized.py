#!/usr/bin/env python3
"""
ML Training - 10 Batches Optimized Strategy
Train 10 separate models, collect predictions, ensemble via voting
WITHOUT creating 170M row joins!
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round, avg, when
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time
import gc

print("="*80)
print("🎯 ML TRAINING - 10 BATCHES OPTIMIZED ENSEMBLE")
print("="*80)

# Initialize Spark
spark = SparkSession.builder \
    .appName("ML_Training_10Batches_Optimized") \
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

# ============================================================================
# STEP 2: Define 10 Training Batches
# ============================================================================

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

print(f"\n📊 Total batches: {len(batch_configs)}")

# ============================================================================
# STEP 3: Train 10 Logistic Regression Models + Collect Predictions
# ============================================================================

print("\n" + "="*80)
print("🤖 STEP 3: Training 10 Logistic Regression Models")
print("="*80)

lr_predictions_list = []

for i, (class_label, batch_num, batch_path) in enumerate(batch_configs, start=1):
    print(f"\n{'='*80}")
    print(f"📦 Batch {i}/10: {class_label} Batch {batch_num}")
    print(f"{'='*80}")
    
    batch_start = time.time()
    
    # Load batch
    print(f"📂 Loading from: {batch_path}")
    df_batch = spark.read.parquet(batch_path).repartition(10).cache()
    batch_count = df_batch.count()
    print(f"✅ Loaded: {batch_count:,} samples")
    
    # Train Logistic Regression
    print("🤖 Training Logistic Regression...")
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=10,
        regParam=0.01
    )
    lr_model = lr.fit(df_batch)
    
    # Save model
    model_path = f"hdfs://namenode:8020/user/data/models/lr_batch_{i}"
    print(f"💾 Saving model to: {model_path}")
    lr_model.write().overwrite().save(model_path)
    
    # CRITICAL: Predict on test data and ADD INDEX
    print("🔮 Predicting on test data...")
    predictions = lr_model.transform(df_test)
    
    # Select only needed columns and add row index
    from pyspark.sql.functions import monotonically_increasing_id
    predictions = predictions.select(
        monotonically_increasing_id().alias("row_id"),
        col("label"),
        col("prediction").alias(f"pred_{i}")
    )
    
    # SAVE predictions to HDFS to avoid memory issues
    pred_path = f"hdfs://namenode:8020/user/data/predictions/lr_batch_{i}"
    print(f"💾 Saving predictions to: {pred_path}")
    predictions.write.mode("overwrite").parquet(pred_path)
    
    lr_predictions_list.append((i, pred_path))
    
    batch_elapsed = time.time() - batch_start
    print(f"✅ Batch {i} completed in {batch_elapsed:.2f}s")
    
    # CRITICAL MEMORY CLEANUP
    df_batch.unpersist()
    del df_batch, lr_model, predictions
    spark.catalog.clearCache()
    gc.collect()
    time.sleep(3)

print("\n" + "="*80)
print("✅ All 10 Logistic Regression models trained!")
print("="*80)

# ============================================================================
# STEP 4: Ensemble Predictions - LOAD AND JOIN INCREMENTALLY
# ============================================================================

print("\n" + "="*80)
print("🔮 STEP 4: Ensemble Predictions (Majority Voting)")
print("="*80)

print("📂 Loading first prediction batch...")
df_ensemble = spark.read.parquet(lr_predictions_list[0][1])
print(f"✅ Base predictions loaded: {df_ensemble.count():,} rows")

# Join incrementally (only on row_id, much smaller)
for i, pred_path in lr_predictions_list[1:]:
    print(f"\n📂 Loading prediction batch {i}...")
    df_pred = spark.read.parquet(pred_path)
    
    print(f"🔗 Joining predictions...")
    df_ensemble = df_ensemble.join(
        df_pred.select("row_id", f"pred_{i}"),
        on="row_id",
        how="inner"
    )
    
    count = df_ensemble.count()
    print(f"✅ Joined {i} predictions: {count:,} rows")
    
    # Cache after each join
    df_ensemble = df_ensemble.cache()

print("\n🎯 Calculating ensemble prediction (majority voting)...")

# Calculate average of all predictions and round
ensemble_expr = spark_round((
    col("pred_1") + col("pred_2") + col("pred_3") + col("pred_4") + col("pred_5") +
    col("pred_6") + col("pred_7") + col("pred_8") + col("pred_9") + col("pred_10")
) / 10.0)

df_ensemble = df_ensemble.withColumn("ensemble_prediction", ensemble_expr)

# Cache final ensemble
df_ensemble = df_ensemble.cache()
ensemble_count = df_ensemble.count()
print(f"✅ Ensemble predictions ready: {ensemble_count:,} samples")

# Show sample
print("\n📊 Sample predictions:")
df_ensemble.select("label", "pred_1", "pred_2", "pred_3", "ensemble_prediction").show(10)

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
    print("⚠️"*40)

# ============================================================================
# STEP 6: Train 10 Random Forest Models (Optional - if time permits)
# ============================================================================

print("\n" + "="*80)
print("🌲 STEP 6: Training 10 Random Forest Models")
print("="*80)

rf_predictions_list = []

for i, (class_label, batch_num, batch_path) in enumerate(batch_configs, start=1):
    print(f"\n{'='*80}")
    print(f"📦 Batch {i}/10: {class_label} Batch {batch_num}")
    print(f"{'='*80}")
    
    batch_start = time.time()
    
    # Load batch
    print(f"📂 Loading from: {batch_path}")
    df_batch = spark.read.parquet(batch_path).repartition(10).cache()
    batch_count = df_batch.count()
    print(f"✅ Loaded: {batch_count:,} samples")
    
    # Train Random Forest
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=20,
        maxDepth=8,
        seed=42
    )
    rf_model = rf.fit(df_batch)
    
    # Save model
    model_path = f"hdfs://namenode:8020/user/data/models/rf_batch_{i}"
    print(f"💾 Saving model to: {model_path}")
    rf_model.write().overwrite().save(model_path)
    
    # Predict on test data
    print("🔮 Predicting on test data...")
    predictions = rf_model.transform(df_test)
    
    # Select only needed columns and add row index
    from pyspark.sql.functions import monotonically_increasing_id
    predictions = predictions.select(
        monotonically_increasing_id().alias("row_id"),
        col("label"),
        col("prediction").alias(f"pred_{i}")
    )
    
    # SAVE predictions to HDFS
    pred_path = f"hdfs://namenode:8020/user/data/predictions/rf_batch_{i}"
    print(f"💾 Saving predictions to: {pred_path}")
    predictions.write.mode("overwrite").parquet(pred_path)
    
    rf_predictions_list.append((i, pred_path))
    
    batch_elapsed = time.time() - batch_start
    print(f"✅ Batch {i} completed in {batch_elapsed:.2f}s")
    
    # CRITICAL MEMORY CLEANUP
    df_batch.unpersist()
    del df_batch, rf_model, predictions
    spark.catalog.clearCache()
    gc.collect()
    time.sleep(3)

print("\n" + "="*80)
print("✅ All 10 Random Forest models trained!")
print("="*80)

# ============================================================================
# STEP 7: Random Forest Ensemble
# ============================================================================

print("\n" + "="*80)
print("🔮 STEP 7: Random Forest Ensemble Predictions")
print("="*80)

print("📂 Loading first RF prediction batch...")
df_rf_ensemble = spark.read.parquet(rf_predictions_list[0][1])
print(f"✅ Base predictions loaded: {df_rf_ensemble.count():,} rows")

# Join incrementally
for i, pred_path in rf_predictions_list[1:]:
    print(f"\n📂 Loading RF prediction batch {i}...")
    df_pred = spark.read.parquet(pred_path)
    
    print(f"🔗 Joining predictions...")
    df_rf_ensemble = df_rf_ensemble.join(
        df_pred.select("row_id", f"pred_{i}"),
        on="row_id",
        how="inner"
    )
    
    count = df_rf_ensemble.count()
    print(f"✅ Joined {i} predictions: {count:,} rows")
    
    df_rf_ensemble = df_rf_ensemble.cache()

print("\n🎯 Calculating RF ensemble prediction...")

# Calculate average and round
rf_ensemble_expr = spark_round((
    col("pred_1") + col("pred_2") + col("pred_3") + col("pred_4") + col("pred_5") +
    col("pred_6") + col("pred_7") + col("pred_8") + col("pred_9") + col("pred_10")
) / 10.0)

df_rf_ensemble = df_rf_ensemble.withColumn("ensemble_prediction", rf_ensemble_expr)
df_rf_ensemble = df_rf_ensemble.cache()
rf_ensemble_count = df_rf_ensemble.count()
print(f"✅ RF ensemble predictions ready: {rf_ensemble_count:,} samples")

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
print(f"📊 Test samples: {rf_ensemble_count:,}")
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
print(f"\n📊 Models trained: 20 models (10 LR + 10 RF)")
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

# Determine best model
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
    print("Consider tuning hyperparameters or increasing training iterations")
    print("⚠️"*40)

print("\n" + "="*80)
spark.stop()
