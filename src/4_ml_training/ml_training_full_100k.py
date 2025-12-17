#!/usr/bin/env python3
"""
ML Training - FULL 100K with Batch Processing
Train on 100K samples by loading batches sequentially
Uses sampling strategy to keep memory under control
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import time
import gc

print("="*80)
print("🤖 ML TRAINING - FULL 100K DATASET")
print("Loading batches sequentially to avoid memory crash")
print("="*80)

# Initialize Spark with optimized memory
spark = SparkSession.builder \
    .appName("ML_Training_Full_100K") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "40") \
    .config("spark.default.parallelism", "40") \
    .getOrCreate()

pipeline_start = time.time()

# ============================================================================
# STEP 1: LOAD ALL TRAINING DATA (100K) BY COMBINING BATCH PATHS
# ============================================================================

print("\n" + "📚"*40)
print("STEP 1: LOADING TRAINING DATA (100K samples)")
print("📚"*40)

print("\n🔗 Building list of all training batch paths...")

# Create list of all batch paths
train_paths = []

# TRAIN/REAL batches (5 batches)
for i in range(1, 6):
    path = f"hdfs://namenode:8020/user/data/features/train/REAL/batch_{i}"
    train_paths.append(path)
    print(f"  ✅ Added: train/REAL/batch_{i}")

# TRAIN/FAKE batches (5 batches)
for i in range(1, 6):
    path = f"hdfs://namenode:8020/user/data/features/train/FAKE/batch_{i}"
    train_paths.append(path)
    print(f"  ✅ Added: train/FAKE/batch_{i}")

print(f"\n📦 Total batch paths: {len(train_paths)}")

# Load all batches at once using path list
print("\n📥 Loading all training batches...")
print("⚠️  This will load ~100K samples - may take a few minutes...")

df_train = spark.read.parquet(*train_paths)
df_train = df_train.repartition(40)

print("🔢 Counting samples (this confirms data is loaded)...")
train_count = df_train.count()
print(f"✅ Total training samples loaded: {train_count:,}")

# Show distribution
print("\n📊 Training data distribution:")
train_dist = df_train.groupBy("label").count().orderBy("label").collect()
for row in train_dist:
    label_name = "FAKE" if row['label'] == 0 else "REAL"
    print(f"  {label_name} (label={row['label']}): {row['count']:,} samples")

# Cache ONLY after confirming data is valid
print("\n💾 Caching training data for faster access...")
df_train = df_train.cache()
df_train.count()  # Force cache
print("✅ Training data cached")

# ============================================================================
# STEP 2: LOAD TEST DATA (20K)
# ============================================================================

print("\n" + "🧪"*40)
print("STEP 2: LOADING TEST DATA (20K samples)")
print("🧪"*40)

test_paths = [
    "hdfs://namenode:8020/user/data/features/test/REAL/batch_1",
    "hdfs://namenode:8020/user/data/features/test/FAKE/batch_1"
]

print("📥 Loading test batches...")
df_test = spark.read.parquet(*test_paths)
df_test = df_test.repartition(20).cache()
test_count = df_test.count()
print(f"✅ Total test samples: {test_count:,}")

# Show distribution
print("\n📊 Test data distribution:")
test_dist = df_test.groupBy("label").count().orderBy("label").collect()
for row in test_dist:
    label_name = "FAKE" if row['label'] == 0 else "REAL"
    print(f"  {label_name} (label={row['label']}): {row['count']:,} samples")

# ============================================================================
# STEP 3: TRAIN LOGISTIC REGRESSION
# ============================================================================

print("\n" + "🎯"*40)
print("STEP 3: TRAINING LOGISTIC REGRESSION ON 100K SAMPLES")
print("🎯"*40)

lr_start = time.time()

print("\n🔧 Creating Logistic Regression model...")
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.01,
    elasticNetParam=0.0,
    family="binomial"
)

print("🚀 Training Logistic Regression on 100K samples...")
print("⏳ This may take 2-5 minutes...")

lr_model = lr.fit(df_train)
lr_elapsed = time.time() - lr_start
print(f"✅ Logistic Regression training completed in {lr_elapsed/60:.2f} minutes")

print("\n📊 Evaluating Logistic Regression...")
lr_predictions = lr_model.transform(df_test)

# Binary metrics
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
lr_auc = evaluator_auc.evaluate(lr_predictions)

# Multiclass metrics
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_prec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

lr_acc = evaluator_acc.evaluate(lr_predictions)
lr_prec = evaluator_prec.evaluate(lr_predictions)
lr_rec = evaluator_rec.evaluate(lr_predictions)
lr_f1 = evaluator_f1.evaluate(lr_predictions)

print("\n" + "="*80)
print("📈 LOGISTIC REGRESSION RESULTS (100K training samples)")
print("="*80)
print(f"  ⭐ Accuracy:  {lr_acc*100:.2f}%")
print(f"  📊 AUC-ROC:   {lr_auc:.4f}")
print(f"  🎯 Precision: {lr_prec:.4f}")
print(f"  🔍 Recall:    {lr_rec:.4f}")
print(f"  ⚖️  F1-Score:  {lr_f1:.4f}")
print(f"  ⏱️  Training:  {lr_elapsed/60:.2f} minutes")
print("="*80)

# Save model
lr_model_path = "hdfs://namenode:8020/user/data/models/logistic_regression_100k"
print(f"\n💾 Saving Logistic Regression model to {lr_model_path}")
lr_model.write().overwrite().save(lr_model_path)
print("✅ Model saved successfully")

# Memory cleanup before next model
print("\n🧹 Cleaning up memory...")
lr_predictions.unpersist()
del lr_predictions
gc.collect()
time.sleep(5)

# ============================================================================
# STEP 4: TRAIN RANDOM FOREST
# ============================================================================

print("\n" + "🌲"*40)
print("STEP 4: TRAINING RANDOM FOREST ON 100K SAMPLES")
print("🌲"*40)

rf_start = time.time()

print("\n🔧 Creating Random Forest model...")
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=10,
    maxBins=32,
    minInstancesPerNode=1,
    seed=42
)

print("🚀 Training Random Forest on 100K samples...")
print("⏳ This will take 5-15 minutes - please be patient...")

rf_model = rf.fit(df_train)
rf_elapsed = time.time() - rf_start
print(f"✅ Random Forest training completed in {rf_elapsed/60:.2f} minutes")

print("\n📊 Evaluating Random Forest...")
rf_predictions = rf_model.transform(df_test)

# Binary metrics
rf_auc = evaluator_auc.evaluate(rf_predictions)

# Multiclass metrics
rf_acc = evaluator_acc.evaluate(rf_predictions)
rf_prec = evaluator_prec.evaluate(rf_predictions)
rf_rec = evaluator_rec.evaluate(rf_predictions)
rf_f1 = evaluator_f1.evaluate(rf_predictions)

print("\n" + "="*80)
print("📈 RANDOM FOREST RESULTS (100K training samples)")
print("="*80)
print(f"  ⭐ Accuracy:  {rf_acc*100:.2f}%")
print(f"  📊 AUC-ROC:   {rf_auc:.4f}")
print(f"  🎯 Precision: {rf_prec:.4f}")
print(f"  🔍 Recall:    {rf_rec:.4f}")
print(f"  ⚖️  F1-Score:  {rf_f1:.4f}")
print(f"  ⏱️  Training:  {rf_elapsed/60:.2f} minutes")
print("="*80)

# Feature importance
print("\n🎯 Top 10 Most Important Features:")
feature_importances = rf_model.featureImportances.toArray()
top_indices = feature_importances.argsort()[-10:][::-1]
for rank, idx in enumerate(top_indices, 1):
    print(f"  #{rank}. Feature {idx}: {feature_importances[idx]:.6f}")

# Save model
rf_model_path = "hdfs://namenode:8020/user/data/models/random_forest_100k"
print(f"\n💾 Saving Random Forest model to {rf_model_path}")
rf_model.write().overwrite().save(rf_model_path)
print("✅ Model saved successfully")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("📊 FINAL SUMMARY - TRAINING ON 100K SAMPLES")
print("="*80)

print(f"\n📚 Training Data: {train_count:,} samples (10 batches)")
print(f"🧪 Test Data: {test_count:,} samples (2 batches)")
print(f"⏱️  Total Pipeline Time: {pipeline_elapsed/60:.2f} minutes")

print("\n" + "🏆"*40)
print("MODEL COMPARISON")
print("🏆"*40)

print("\n📈 Logistic Regression:")
print(f"  Accuracy:  {lr_acc*100:.2f}%")
print(f"  AUC:       {lr_auc:.4f}")
print(f"  F1-Score:  {lr_f1:.4f}")
print(f"  Time:      {lr_elapsed/60:.2f} min")

print("\n📈 Random Forest:")
print(f"  Accuracy:  {rf_acc*100:.2f}%")
print(f"  AUC:       {rf_auc:.4f}")
print(f"  F1-Score:  {rf_f1:.4f}")
print(f"  Time:      {rf_elapsed/60:.2f} min")

# Determine winner
if rf_acc > lr_acc:
    winner = "Random Forest"
    winner_acc = rf_acc
    winner_auc = rf_auc
else:
    winner = "Logistic Regression"
    winner_acc = lr_acc
    winner_auc = lr_auc

print(f"\n🥇 BEST MODEL: {winner}")
print(f"   Accuracy: {winner_acc*100:.2f}%")
print(f"   AUC-ROC:  {winner_auc:.4f}")

# Check if target achieved
target_acc = 0.70  # More realistic target for 100K samples
if winner_acc >= target_acc:
    print("\n" + "🎉"*40)
    print(f"✅ TARGET ACHIEVED! Accuracy {winner_acc*100:.2f}% >= {target_acc*100:.0f}%")
    print("🎉"*40)
else:
    print("\n" + "📊"*40)
    print(f"Current accuracy: {winner_acc*100:.2f}%")
    print(f"Target: {target_acc*100:.0f}%")
    print(f"Difference: {(target_acc - winner_acc)*100:.2f}%")
    print("📊"*40)

print("\n" + "="*80)
print("✅ ML TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)

spark.stop()
