#!/usr/bin/env python3
"""
ML Training - OPTIMIZED Random Forest with Enhanced Hyperparameters
Train on 5 MIXED batches with powerful RF configuration to achieve 85%+ accuracy
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time
import gc

print("="*80)
print("🌲 ML TRAINING - OPTIMIZED RANDOM FOREST")
print("="*80)

# Initialize Spark
spark = SparkSession.builder \
    .appName("ML_Training_Optimized_RF") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.default.parallelism", "50") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

pipeline_start = time.time()

# ============================================================================
# STEP 1: Load Test Data (20K samples)
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
# STEP 2: Prepare 5 Mixed Training Batches
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 2: Preparing 5 Mixed Batches")
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
# STEP 3: Train 5 Random Forest Models with OPTIMIZED Hyperparameters
# ============================================================================

print("\n" + "="*80)
print("🌲 STEP 3: Training 5 OPTIMIZED Random Forest Models")
print("="*80)
print("\n🔧 HYPERPARAMETERS (OPTIMIZED):")
print("   - numTrees: 100 (increased from 30)")
print("   - maxDepth: 15 (increased from 10)")
print("   - minInstancesPerNode: 1 (allow detailed learning)")
print("   - featureSubsetStrategy: 'sqrt' (optimal for classification)")
print("   - seed: 42 (reproducibility)")

rf_predictions_list = []
individual_accuracies = []

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
    
    # Mix REAL + FAKE
    df_batch = df_real.union(df_fake).repartition(10).cache()
    batch_count = df_batch.count()
    print(f"   ✅ MIXED: {batch_count:,} samples total")
    
    # Train OPTIMIZED Random Forest
    print("🌲 Training OPTIMIZED Random Forest...")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,                    # Increased from 30
        maxDepth=15,                     # Increased from 10
        minInstancesPerNode=1,           # Allow detailed splits
        featureSubsetStrategy="sqrt",    # Optimal for classification
        seed=42
    )
    
    print("   Training in progress...")
    rf_model = rf.fit(df_batch)
    print("   ✅ Training completed!")
    
    # Note: Not saving individual models to save HDFS space
    # Will save only final ensemble predictions
    
    # Predict on test data
    print("🔮 Predicting on test data...")
    predictions = rf_model.transform(df_test)
    
    # Evaluate single model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    single_acc = evaluator.evaluate(predictions)
    individual_accuracies.append(single_acc)
    print(f"   📊 Single model accuracy: {single_acc*100:.2f}%")
    
    # Save predictions with row index
    from pyspark.sql.functions import monotonically_increasing_id
    predictions = predictions.select(
        monotonically_increasing_id().alias("row_id"),
        col("label"),
        col("prediction").alias(f"pred_{batch_id}")
    )
    
    pred_path = f"hdfs://namenode:8020/user/data/predictions/rf_optimized_batch_{batch_id}"
    print(f"💾 Saving predictions to: {pred_path}")
    predictions.write.mode("overwrite").parquet(pred_path)
    
    rf_predictions_list.append((batch_id, pred_path, single_acc))
    
    batch_elapsed = time.time() - batch_start
    print(f"✅ Batch {batch_id} completed in {batch_elapsed:.2f}s")
    
    # CRITICAL MEMORY CLEANUP
    df_real.unpersist()
    df_fake.unpersist()
    df_batch.unpersist()
    del df_real, df_fake, df_batch, rf_model, predictions
    spark.catalog.clearCache()
    gc.collect()
    time.sleep(3)

print("\n" + "="*80)
print("✅ All 5 OPTIMIZED Random Forest models trained!")
print("="*80)

# Show individual model accuracies
print("\n📊 Individual Model Accuracies:")
avg_individual = sum(individual_accuracies) / len(individual_accuracies)
for batch_id, _, acc in rf_predictions_list:
    print(f"   Model {batch_id}: {acc*100:.2f}%")
print(f"\n   Average: {avg_individual*100:.2f}%")

# ============================================================================
# STEP 4: Ensemble Predictions - Majority Voting
# ============================================================================

print("\n" + "="*80)
print("🔮 STEP 4: Ensemble Predictions (Majority Voting)")
print("="*80)

print("📂 Loading first prediction batch...")
df_ensemble = spark.read.parquet(rf_predictions_list[0][1])
print(f"✅ Base predictions loaded: {df_ensemble.count():,} rows")

# Join incrementally
for batch_id, pred_path, _ in rf_predictions_list[1:]:
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

# AUC
evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="ensemble_prediction",
    metricName="areaUnderROC"
)
ensemble_auc = evaluator_auc.evaluate(df_ensemble)

# ============================================================================
# STEP 6: Save Final Results to HDFS
# ============================================================================

print("\n" + "="*80)
print("💾 STEP 6: Saving Final Results to HDFS")
print("="*80)

# Save ensemble predictions (label + final prediction)
final_predictions_path = "hdfs://namenode:8020/user/data/results/final_predictions"
print(f"\n📊 Saving final predictions to: {final_predictions_path}")
df_ensemble.select("label", "ensemble_prediction").write.mode("overwrite").parquet(final_predictions_path)
print("✅ Final predictions saved!")

# Save detailed predictions with all 5 model votes
detailed_predictions_path = "hdfs://namenode:8020/user/data/results/detailed_predictions"
print(f"\n📊 Saving detailed predictions (with all votes) to: {detailed_predictions_path}")
df_ensemble.write.mode("overwrite").parquet(detailed_predictions_path)
print("✅ Detailed predictions saved!")

# Create and save metrics summary
from pyspark.sql import Row
metrics_data = [
    Row(metric="Accuracy", value=float(ensemble_acc)),
    Row(metric="Precision", value=float(ensemble_prec)),
    Row(metric="Recall", value=float(ensemble_rec)),
    Row(metric="F1_Score", value=float(ensemble_f1)),
    Row(metric="AUC", value=float(ensemble_auc)),
    Row(metric="Test_Samples", value=float(ensemble_count)),
    Row(metric="Training_Batches", value=5.0),
    Row(metric="NumTrees", value=100.0),
    Row(metric="MaxDepth", value=15.0)
]
df_metrics = spark.createDataFrame(metrics_data)
metrics_path = "hdfs://namenode:8020/user/data/results/metrics_summary"
print(f"\n📊 Saving metrics summary to: {metrics_path}")
df_metrics.write.mode("overwrite").parquet(metrics_path)
print("✅ Metrics summary saved!")

print("\n" + "="*80)
print("✅ All results saved to HDFS!")
print("="*80)
print(f"   - Final predictions: {final_predictions_path}")
print(f"   - Detailed predictions: {detailed_predictions_path}")
print(f"   - Metrics summary: {metrics_path}")

print("\n" + "="*80)
print("🌲 OPTIMIZED RANDOM FOREST ENSEMBLE RESULTS")
print("="*80)
print(f"📊 Test samples: {ensemble_count:,}")
print(f"🎯 Accuracy:  {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
print(f"🎯 Precision: {ensemble_prec:.4f} ({ensemble_prec*100:.2f}%)")
print(f"🎯 Recall:    {ensemble_rec:.4f} ({ensemble_rec*100:.2f}%)")
print(f"🎯 F1-Score:  {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)")
print(f"🎯 AUC:       {ensemble_auc:.4f} ({ensemble_auc*100:.2f}%)")

# Calculate improvement
previous_acc = 0.7671  # From previous run
improvement = (ensemble_acc - previous_acc) * 100
print(f"\n📈 Improvement from baseline: {improvement:+.2f}%")
print(f"   Previous (30 trees, depth 10): {previous_acc*100:.2f}%")
print(f"   Current (100 trees, depth 15): {ensemble_acc*100:.2f}%")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("🏁 FINAL SUMMARY")
print("="*80)
print(f"\n⏱️  Total pipeline time: {pipeline_elapsed/60:.2f} minutes")
print(f"\n📊 Models trained: 5 OPTIMIZED Random Forest models")
print(f"📊 Training strategy: MIXED batches (50% REAL + 50% FAKE)")
print(f"📊 Test samples: {test_count:,}")

print("\n" + "="*80)
print("🔧 HYPERPARAMETERS USED")
print("="*80)
print("   numTrees: 100")
print("   maxDepth: 15")
print("   minInstancesPerNode: 1")
print("   featureSubsetStrategy: 'sqrt'")

print("\n" + "="*80)
print("📊 PERFORMANCE METRICS")
print("="*80)
print(f"   Average Individual Model: {avg_individual*100:.2f}%")
print(f"   Ensemble (Majority Voting): {ensemble_acc*100:.2f}%")
print(f"   Improvement via Ensemble: {(ensemble_acc - avg_individual)*100:+.2f}%")

print("\n" + "="*80)
if ensemble_acc >= 0.85:
    print("🎉"*40)
    print("✅✅✅ SUCCESS: Target accuracy >= 85% achieved!")
    print(f"🏆 Final Accuracy: {ensemble_acc*100:.2f}%")
    print("🎉"*40)
else:
    gap = (0.85 - ensemble_acc) * 100
    print("⚠️"*40)
    print(f"⚠️ Target not met: {ensemble_acc*100:.2f}% < 85%")
    print(f"⚠️ Gap remaining: {gap:.2f}%")
    print("\n💡 Next steps to improve:")
    print("   1. Increase numTrees to 150-200")
    print("   2. Try different feature engineering")
    print("   3. Add more diverse training data")
    print("⚠️"*40)

print("\n" + "="*80)
spark.stop()
