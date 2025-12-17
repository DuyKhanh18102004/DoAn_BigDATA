#!/usr/bin/env python3
"""
ML Training - SMALL SAMPLE (20K training data)
Train on 2 batches (20K samples) to avoid memory crash
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import time

print("="*80)
print("🤖 ML TRAINING - SMALL SAMPLE (20K)")
print("="*80)

# Initialize Spark with conservative memory
spark = SparkSession.builder \
    .appName("ML_Training_Small") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "20") \
    .getOrCreate()

pipeline_start = time.time()

# ============================================================================
# LOAD TRAINING DATA - Only 2 batches (20K samples)
# ============================================================================

print("\n📚 LOADING TRAINING DATA (20K samples)")
print("Loading only first batch from each class to save memory...")

# TRAIN/REAL batch 1 (~10K)
print("\n🟢 Loading TRAIN/REAL batch 1...")
df_real = spark.read.parquet("hdfs://namenode:8020/user/data/features/train/REAL/batch_1")
real_count = df_real.count()
print(f"✅ Loaded: {real_count:,} REAL samples")

# TRAIN/FAKE batch 1 (~10K)
print("\n🔴 Loading TRAIN/FAKE batch 1...")
df_fake = spark.read.parquet("hdfs://namenode:8020/user/data/features/train/FAKE/batch_1")
fake_count = df_fake.count()
print(f"✅ Loaded: {fake_count:,} FAKE samples")

# Combine
df_train = df_real.union(df_fake).repartition(20).cache()
train_count = df_train.count()
print(f"\n✅ Total training samples: {train_count:,}")

# ============================================================================
# LOAD TEST DATA
# ============================================================================

print("\n🧪 LOADING TEST DATA (20K samples)")

df_test_real = spark.read.parquet("hdfs://namenode:8020/user/data/features/test/REAL/batch_1")
df_test_fake = spark.read.parquet("hdfs://namenode:8020/user/data/features/test/FAKE/batch_1")
df_test = df_test_real.union(df_test_fake).repartition(20).cache()
test_count = df_test.count()
print(f"✅ Total test samples: {test_count:,}")

# ============================================================================
# TRAIN LOGISTIC REGRESSION
# ============================================================================

print("\n" + "🎯"*40)
print("TRAINING LOGISTIC REGRESSION")
print("🎯"*40)

lr_start = time.time()

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=10,
    regParam=0.01
)

print("🚀 Training Logistic Regression...")
lr_model = lr.fit(df_train)
lr_elapsed = time.time() - lr_start
print(f"✅ Training completed in {lr_elapsed:.2f}s")

print("\n📊 Evaluating Logistic Regression...")
lr_predictions = lr_model.transform(df_test)

# Evaluate
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

lr_auc = evaluator_auc.evaluate(lr_predictions)
lr_acc = evaluator_acc.evaluate(lr_predictions)

print("\n📈 LOGISTIC REGRESSION RESULTS:")
print(f"  ⭐ Accuracy: {lr_acc*100:.2f}%")
print(f"  📊 AUC:      {lr_auc:.4f}")

# Save model
lr_model_path = "hdfs://namenode:8020/user/data/models/logistic_regression_small"
print(f"\n💾 Saving model to {lr_model_path}")
lr_model.write().overwrite().save(lr_model_path)
print("✅ Model saved")

# Cleanup
lr_predictions.unpersist()

# ============================================================================
# TRAIN RANDOM FOREST
# ============================================================================

print("\n" + "🌲"*40)
print("TRAINING RANDOM FOREST")
print("🌲"*40)

rf_start = time.time()

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=30,
    maxDepth=8,
    seed=42
)

print("🚀 Training Random Forest...")
rf_model = rf.fit(df_train)
rf_elapsed = time.time() - rf_start
print(f"✅ Training completed in {rf_elapsed:.2f}s")

print("\n📊 Evaluating Random Forest...")
rf_predictions = rf_model.transform(df_test)

rf_auc = evaluator_auc.evaluate(rf_predictions)
rf_acc = evaluator_acc.evaluate(rf_predictions)

print("\n📈 RANDOM FOREST RESULTS:")
print(f"  ⭐ Accuracy: {rf_acc*100:.2f}%")
print(f"  📊 AUC:      {rf_auc:.4f}")

# Save model
rf_model_path = "hdfs://namenode:8020/user/data/models/random_forest_small"
print(f"\n💾 Saving model to {rf_model_path}")
rf_model.write().overwrite().save(rf_model_path)
print("✅ Model saved")

# ============================================================================
# SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)

print(f"\n📚 Training Data: {train_count:,} samples (2 batches)")
print(f"🧪 Test Data: {test_count:,} samples")
print(f"⏱️  Total Time: {pipeline_elapsed/60:.2f} minutes")

print("\n🏆 MODEL COMPARISON:")
print(f"\n📈 Logistic Regression: {lr_acc*100:.2f}% accuracy ({lr_elapsed:.1f}s)")
print(f"📈 Random Forest:       {rf_acc*100:.2f}% accuracy ({rf_elapsed:.1f}s)")

if rf_acc > lr_acc:
    print(f"\n🥇 WINNER: Random Forest ({rf_acc*100:.2f}%)")
else:
    print(f"\n🥇 WINNER: Logistic Regression ({lr_acc*100:.2f}%)")

print("\n" + "="*80)
print("✅ TRAINING COMPLETED")
print("="*80)

spark.stop()
