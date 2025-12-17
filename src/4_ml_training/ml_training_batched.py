#!/usr/bin/env python3
"""
ML Training with Batched Data Loading
Train models on 100K samples by loading 10K at a time to avoid memory issues
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import time
import gc

print("="*80)
print("🤖 ML TRAINING - BATCHED APPROACH")
print("Training on 100K samples by loading 10K batches")
print("="*80)

# Initialize Spark
spark = SparkSession.builder \
    .appName("ML_Training_Batched") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

pipeline_start = time.time()

# ============================================================================
# STEP 1: LOAD TRAINING DATA IN BATCHES
# ============================================================================

print("\n" + "📚"*40)
print("STEP 1: LOADING TRAINING DATA (100K samples in batches)")
print("📚"*40)

train_batches = []

# Load TRAIN/REAL batches (5 batches × ~10K = 50K)
print("\n🟢 Loading TRAIN/REAL batches...")
for batch_num in range(1, 6):
    path = f"hdfs://namenode:8020/user/data/features/train/REAL/batch_{batch_num}"
    print(f"  Loading batch {batch_num}...")
    df_batch = spark.read.parquet(path)
    count = df_batch.count()
    print(f"  ✅ Batch {batch_num}: {count:,} samples")
    train_batches.append(df_batch)
    time.sleep(2)

# Load TRAIN/FAKE batches (5 batches × ~10K = 50K)
print("\n🔴 Loading TRAIN/FAKE batches...")
for batch_num in range(1, 6):
    path = f"hdfs://namenode:8020/user/data/features/train/FAKE/batch_{batch_num}"
    print(f"  Loading batch {batch_num}...")
    df_batch = spark.read.parquet(path)
    count = df_batch.count()
    print(f"  ✅ Batch {batch_num}: {count:,} samples")
    train_batches.append(df_batch)
    time.sleep(2)

# Combine all training batches
print("\n🔗 Combining all training batches...")
df_train = train_batches[0]
for batch in train_batches[1:]:
    df_train = df_train.union(batch)

# Cache and count
df_train = df_train.repartition(50).cache()
train_count = df_train.count()
print(f"✅ Total training samples: {train_count:,}")

# Show distribution
print("\n📊 Training data distribution:")
train_dist = df_train.groupBy("label").count().collect()
for row in train_dist:
    label_name = "REAL" if row['label'] == 1 else "FAKE"
    print(f"  {label_name}: {row['count']:,} samples")

# Memory cleanup
for batch in train_batches:
    batch.unpersist()
train_batches.clear()
gc.collect()
time.sleep(5)

# ============================================================================
# STEP 2: LOAD TEST DATA
# ============================================================================

print("\n" + "🧪"*40)
print("STEP 2: LOADING TEST DATA (20K samples)")
print("🧪"*40)

print("\n🟢 Loading TEST/REAL...")
df_test_real = spark.read.parquet("hdfs://namenode:8020/user/data/features/test/REAL/batch_1")
real_count = df_test_real.count()
print(f"✅ TEST/REAL: {real_count:,} samples")

print("\n🔴 Loading TEST/FAKE...")
df_test_fake = spark.read.parquet("hdfs://namenode:8020/user/data/features/test/FAKE/batch_1")
fake_count = df_test_fake.count()
print(f"✅ TEST/FAKE: {fake_count:,} samples")

# Combine test data
df_test = df_test_real.union(df_test_fake)
df_test = df_test.repartition(20).cache()
test_count = df_test.count()
print(f"\n✅ Total test samples: {test_count:,}")

# Show distribution
print("\n📊 Test data distribution:")
test_dist = df_test.groupBy("label").count().collect()
for row in test_dist:
    label_name = "REAL" if row['label'] == 1 else "FAKE"
    print(f"  {label_name}: {row['count']:,} samples")

# ============================================================================
# STEP 3: TRAIN LOGISTIC REGRESSION
# ============================================================================

print("\n" + "🎯"*40)
print("STEP 3: TRAINING LOGISTIC REGRESSION")
print("🎯"*40)

lr_start = time.time()

print("\n🔧 Creating Logistic Regression model...")
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.01,
    elasticNetParam=0.0
)

print("🚀 Training Logistic Regression...")
lr_model = lr.fit(df_train)
lr_elapsed = time.time() - lr_start
print(f"✅ Training completed in {lr_elapsed:.2f}s")

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

print("\n📈 LOGISTIC REGRESSION RESULTS:")
print(f"  ⭐ Accuracy:  {lr_acc*100:.2f}%")
print(f"  📊 AUC:       {lr_auc:.4f}")
print(f"  🎯 Precision: {lr_prec:.4f}")
print(f"  🔍 Recall:    {lr_rec:.4f}")
print(f"  ⚖️  F1-Score:  {lr_f1:.4f}")

# Save model
lr_model_path = "hdfs://namenode:8020/user/data/models/logistic_regression_batched"
print(f"\n💾 Saving model to {lr_model_path}")
lr_model.write().overwrite().save(lr_model_path)
print("✅ Model saved successfully")

# Memory cleanup
lr_predictions.unpersist()
del lr_predictions
gc.collect()
time.sleep(5)

# ============================================================================
# STEP 4: TRAIN RANDOM FOREST
# ============================================================================

print("\n" + "🌲"*40)
print("STEP 4: TRAINING RANDOM FOREST")
print("🌲"*40)

rf_start = time.time()

print("\n🔧 Creating Random Forest model...")
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=10,
    maxBins=32,
    seed=42
)

print("🚀 Training Random Forest (this may take several minutes)...")
rf_model = rf.fit(df_train)
rf_elapsed = time.time() - rf_start
print(f"✅ Training completed in {rf_elapsed/60:.2f} minutes")

print("\n📊 Evaluating Random Forest...")
rf_predictions = rf_model.transform(df_test)

# Binary metrics
rf_auc = evaluator_auc.evaluate(rf_predictions)

# Multiclass metrics
rf_acc = evaluator_acc.evaluate(rf_predictions)
rf_prec = evaluator_prec.evaluate(rf_predictions)
rf_rec = evaluator_rec.evaluate(rf_predictions)
rf_f1 = evaluator_f1.evaluate(rf_predictions)

print("\n📈 RANDOM FOREST RESULTS:")
print(f"  ⭐ Accuracy:  {rf_acc*100:.2f}%")
print(f"  📊 AUC:       {rf_auc:.4f}")
print(f"  🎯 Precision: {rf_prec:.4f}")
print(f"  🔍 Recall:    {rf_rec:.4f}")
print(f"  ⚖️  F1-Score:  {rf_f1:.4f}")

# Feature importance
print("\n🎯 Top 10 Feature Importances:")
feature_importances = rf_model.featureImportances.toArray()
top_indices = feature_importances.argsort()[-10:][::-1]
for idx in top_indices:
    print(f"  Feature {idx}: {feature_importances[idx]:.6f}")

# Save model
rf_model_path = "hdfs://namenode:8020/user/data/models/random_forest_batched"
print(f"\n💾 Saving model to {rf_model_path}")
rf_model.write().overwrite().save(rf_model_path)
print("✅ Model saved successfully")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)

print(f"\n📚 Training Data: {train_count:,} samples")
print(f"🧪 Test Data: {test_count:,} samples")
print(f"⏱️  Total Pipeline Time: {pipeline_elapsed/60:.2f} minutes")

print("\n" + "🏆"*40)
print("MODEL COMPARISON")
print("🏆"*40)

print("\n📈 Logistic Regression:")
print(f"  Accuracy:  {lr_acc*100:.2f}%")
print(f"  F1-Score:  {lr_f1:.4f}")
print(f"  Time:      {lr_elapsed:.2f}s")

print("\n📈 Random Forest:")
print(f"  Accuracy:  {rf_acc*100:.2f}%")
print(f"  F1-Score:  {rf_f1:.4f}")
print(f"  Time:      {rf_elapsed/60:.2f} min")

# Determine winner
if rf_acc > lr_acc:
    winner = "Random Forest"
    winner_acc = rf_acc
else:
    winner = "Logistic Regression"
    winner_acc = lr_acc

print(f"\n🥇 BEST MODEL: {winner} ({winner_acc*100:.2f}% accuracy)")

# Check if target achieved
target_acc = 0.85
if winner_acc >= target_acc:
    print("\n" + "🎉"*40)
    print(f"✅ TARGET ACHIEVED! Accuracy {winner_acc*100:.2f}% >= {target_acc*100:.0f}%")
    print("🎉"*40)
else:
    print("\n" + "⚠️"*40)
    print(f"⚠️  Target not reached. Current: {winner_acc*100:.2f}%, Target: {target_acc*100:.0f}%")
    print("⚠️"*40)

print("\n" + "="*80)
print("✅ ML TRAINING PIPELINE COMPLETED")
print("="*80)

spark.stop()
