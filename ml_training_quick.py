#!/usr/bin/env python3
"""
STEP 3: ML Training for Deepfake Detection (QUICK VERSION)
- Train Logistic Regression and Random Forest on extracted features
- Working with features_quick dataset (1000 samples)
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

# ===== BƯỚC 1: Initialize Spark Session =====
print("\n🚀 Initializing Spark Session for ML Training...")
spark = SparkSession.builder \
    .appName("DeepfakeDetection-MLTraining-Quick") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

print(f"✅ Spark version: {spark.version}\n")

# ===== BƯỚC 2: Load Features từ HDFS =====
print("\n📂 Loading features from HDFS (Parquet)...")

train_path = "hdfs://namenode:8020/user/data/features_quick/train/*"
test_path = "hdfs://namenode:8020/user/data/features_quick/test/*"

print(f"  - Reading train features from {train_path}")
train_df = spark.read.parquet(train_path)

print(f"  - Reading test features from {test_path}")
test_df = spark.read.parquet(test_path)

print(f"\n✅ Loaded {train_df.count()} training samples")
print(f"✅ Loaded {test_df.count()} test samples")

# Show schema
print("\n📋 Schema:")
train_df.printSchema()

# ===== BƯỚC 3: Chuyển đổi Features sang Vector =====
print("\n🔧 Preparing features for ML...")

# UDF to convert array to Vector
array_to_vector = udf(lambda arr: Vectors.dense(arr), VectorUDT())

# Apply conversion to both train and test
train_assembled = train_df.withColumn("features_vec", array_to_vector("features"))
test_assembled = test_df.withColumn("features_vec", array_to_vector("features"))

print("✅ Features converted to vector format")

# Cache data for multiple model training
train_assembled.cache()
test_assembled.cache()

# ===== BƯỚC 4: Train Logistic Regression =====
print("\n" + "="*60)
print("📊 Training Logistic Regression Model...")
print("="*60)

lr = LogisticRegression(
    featuresCol="features_vec",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

lr_model = lr.fit(train_assembled)
print("✅ Logistic Regression training completed")

# Predict on test set
lr_predictions = lr_model.transform(test_assembled)

# Evaluate LR model
print("\n📈 Logistic Regression Metrics:")

# Binary metrics (AUC-ROC)
binary_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
lr_auc = binary_eval.evaluate(lr_predictions)
print(f"  AUC-ROC: {lr_auc:.4f}")

# Multiclass metrics
multi_eval_acc = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
multi_eval_prec = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
multi_eval_rec = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)
multi_eval_f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

lr_accuracy = multi_eval_acc.evaluate(lr_predictions)
lr_precision = multi_eval_prec.evaluate(lr_predictions)
lr_recall = multi_eval_rec.evaluate(lr_predictions)
lr_f1 = multi_eval_f1.evaluate(lr_predictions)

print(f"  Accuracy: {lr_accuracy:.4f}")
print(f"  Precision: {lr_precision:.4f}")
print(f"  Recall: {lr_recall:.4f}")
print(f"  F1-Score: {lr_f1:.4f}")

# Save LR model
lr_model_path = "hdfs://namenode:8020/user/data/models_quick/logistic_regression"
lr_model.write().overwrite().save(lr_model_path)
print(f"\n✅ LR Model saved to: {lr_model_path}")

# Save predictions
lr_pred_path = "hdfs://namenode:8020/user/data/results_quick/lr_predictions"
lr_predictions.select("label", "prediction", "probability").write.mode("overwrite").parquet(lr_pred_path)
print(f"✅ LR Predictions saved to: {lr_pred_path}")

# ===== BƯỚC 5: Train Random Forest =====
print("\n" + "="*60)
print("🌲 Training Random Forest Model...")
print("="*60)

rf = RandomForestClassifier(
    featuresCol="features_vec",
    labelCol="label",
    numTrees=100,
    maxDepth=10,
    seed=42
)

rf_model = rf.fit(train_assembled)
print("✅ Random Forest training completed")

# Predict on test set
rf_predictions = rf_model.transform(test_assembled)

# Evaluate RF model
print("\n📈 Random Forest Metrics:")

rf_auc = binary_eval.evaluate(rf_predictions)
print(f"  AUC-ROC: {rf_auc:.4f}")

rf_accuracy = multi_eval_acc.evaluate(rf_predictions)
rf_precision = multi_eval_prec.evaluate(rf_predictions)
rf_recall = multi_eval_rec.evaluate(rf_predictions)
rf_f1 = multi_eval_f1.evaluate(rf_predictions)

print(f"  Accuracy: {rf_accuracy:.4f}")
print(f"  Precision: {rf_precision:.4f}")
print(f"  Recall: {rf_recall:.4f}")
print(f"  F1-Score: {rf_f1:.4f}")

# Feature importance
feature_importance = rf_model.featureImportances
print(f"\n🔍 Feature Importance (top 10):")
print(f"  {feature_importance}")

# Save RF model
rf_model_path = "hdfs://namenode:8020/user/data/models_quick/random_forest"
rf_model.write().overwrite().save(rf_model_path)
print(f"\n✅ RF Model saved to: {rf_model_path}")

# Save predictions
rf_pred_path = "hdfs://namenode:8020/user/data/results_quick/rf_predictions"
rf_predictions.select("label", "prediction", "probability").write.mode("overwrite").parquet(rf_pred_path)
print(f"✅ RF Predictions saved to: {rf_pred_path}")

# ===== BƯỚC 6: So sánh Models =====
print("\n" + "="*60)
print("🏆 MODEL COMPARISON")
print("="*60)
print(f"\n{'Metric':<20} {'Logistic Regression':<25} {'Random Forest':<25}")
print("-" * 70)
print(f"{'Accuracy':<20} {lr_accuracy:<25.4f} {rf_accuracy:<25.4f}")
print(f"{'Precision':<20} {lr_precision:<25.4f} {rf_precision:<25.4f}")
print(f"{'Recall':<20} {lr_recall:<25.4f} {rf_recall:<25.4f}")
print(f"{'F1-Score':<20} {lr_f1:<25.4f} {rf_f1:<25.4f}")
print(f"{'AUC-ROC':<20} {lr_auc:<25.4f} {rf_auc:<25.4f}")
print("-" * 70)

best_model = "Random Forest" if rf_accuracy > lr_accuracy else "Logistic Regression"
print(f"\n🎯 Best Model: {best_model}")

# Save metrics summary
metrics_data = [
    ("Logistic Regression", lr_accuracy, lr_precision, lr_recall, lr_f1, lr_auc),
    ("Random Forest", rf_accuracy, rf_precision, rf_recall, rf_f1, rf_auc)
]
metrics_df = spark.createDataFrame(metrics_data, 
    ["model", "accuracy", "precision", "recall", "f1_score", "auc_roc"])

metrics_path = "hdfs://namenode:8020/user/data/results_quick/metrics_summary"
metrics_df.write.mode("overwrite").parquet(metrics_path)
print(f"\n✅ Metrics summary saved to: {metrics_path}")

# ===== BƯỚC 7: Business Insights =====
print("\n" + "="*60)
print("💡 BUSINESS INSIGHTS")
print("="*60)

print("\n1. Can ResNet50 features detect deepfakes?")
if lr_accuracy > 0.75 or rf_accuracy > 0.75:
    print("   ✅ YES! Models achieve >75% accuracy, indicating ResNet50 features")
    print("      are effective for deepfake detection.")
else:
    print("   ⚠️ Moderate performance. May need feature engineering or deeper models.")

print("\n2. Which model performs better?")
print(f"   🏆 {best_model} outperforms with {max(lr_accuracy, rf_accuracy):.2%} accuracy")

print("\n3. Key Performance Indicators:")
print(f"   - False Positive Rate: {1 - lr_precision:.2%} (LR) vs {1 - rf_precision:.2%} (RF)")
print(f"   - False Negative Rate: {1 - lr_recall:.2%} (LR) vs {1 - rf_recall:.2%} (RF)")

print("\n4. Scalability Evidence:")
print("   ✅ Distributed training on Spark cluster with 2 executors")
print("   ✅ HDFS-based storage for reproducibility")
print("   ✅ Event logs available in Spark History Server")

print("\n" + "="*60)
print("🎉 ML TRAINING PIPELINE COMPLETED!")
print("="*60)

# Cleanup
train_assembled.unpersist()
test_assembled.unpersist()

spark.stop()
