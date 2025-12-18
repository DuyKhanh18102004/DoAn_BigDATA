#!/usr/bin/env python3
"""
Step 4: Evaluate Model - Complete Evaluation Script
- Load test predictions từ HDFS
- Tính các metrics: Accuracy, Precision, Recall, F1, AUC-ROC
- Vẽ Confusion Matrix, ROC Curve
- Save báo cáo đầy đủ
"""

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, when, count, sum as spark_sum
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time

print("="*80)
print("📊 STEP 4: MODEL EVALUATION")
print("="*80)

# Initialize Spark
spark = SparkSession.builder \
    .appName("Model_Evaluation") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
eval_start = time.time()

# ============================================================================
# CONFIGURATION
# ============================================================================

HDFS_BASE = "hdfs://namenode:8020/user/data"
RESULTS_BASE = f"{HDFS_BASE}/results_tf"

# ============================================================================
# STEP 1: Load Test Predictions
# ============================================================================

print("\n" + "="*80)
print("📂 STEP 1: Loading Test Predictions")
print("="*80)

test_pred_path = f"{RESULTS_BASE}/test_predictions"
print(f"📂 Loading from: {test_pred_path}")

df_predictions = spark.read.parquet(test_pred_path)
df_predictions = df_predictions.cache()

total_samples = df_predictions.count()
print(f"✅ Loaded {total_samples:,} test predictions")

# Show schema
print("\n📋 Schema:")
df_predictions.printSchema()

# Show sample
print("\n📊 Sample predictions:")
df_predictions.show(10, truncate=False)

# ============================================================================
# STEP 2: Calculate Metrics
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 2: Calculating Metrics")
print("="*80)

# Basic counts
label_dist = df_predictions.groupBy("label").count().collect()
pred_dist = df_predictions.groupBy("prediction").count().collect()

print("\n📊 Label Distribution (Ground Truth):")
for row in label_dist:
    label_name = "REAL" if row["label"] == 1 else "FAKE"
    pct = 100 * row["count"] / total_samples
    print(f"   - {label_name} (label={int(row['label'])}): {row['count']:,} ({pct:.1f}%)")

print("\n📊 Prediction Distribution:")
for row in pred_dist:
    pred_name = "REAL" if row["prediction"] == 1 else "FAKE"
    pct = 100 * row["count"] / total_samples
    print(f"   - {pred_name} (pred={int(row['prediction'])}): {row['count']:,} ({pct:.1f}%)")

# ============================================================================
# STEP 3: Confusion Matrix
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 3: Confusion Matrix")
print("="*80)

# Calculate TP, TN, FP, FN
tp = df_predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
tn = df_predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
fp = df_predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
fn = df_predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

print("\n                    Predicted")
print("                  FAKE    REAL")
print(f"Actual  FAKE      {tn:,}    {fp:,}")
print(f"        REAL      {fn:,}    {tp:,}")

print(f"\n📊 Breakdown:")
print(f"   ✅ True Positive  (REAL→REAL): {tp:,}")
print(f"   ✅ True Negative  (FAKE→FAKE): {tn:,}")
print(f"   ❌ False Positive (FAKE→REAL): {fp:,}")
print(f"   ❌ False Negative (REAL→FAKE): {fn:,}")

# ============================================================================
# STEP 4: Calculate All Metrics
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 4: Performance Metrics")
print("="*80)

# Manual calculations
accuracy = (tp + tn) / total_samples
precision_real = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_real = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0

precision_fake = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_fake = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0

# Weighted averages
total_real = tp + fn
total_fake = tn + fp
weighted_precision = (precision_real * total_real + precision_fake * total_fake) / total_samples
weighted_recall = (recall_real * total_real + recall_fake * total_fake) / total_samples
weighted_f1 = (f1_real * total_real + f1_fake * total_fake) / total_samples

# Macro averages
macro_precision = (precision_real + precision_fake) / 2
macro_recall = (recall_real + recall_fake) / 2
macro_f1 = (f1_real + f1_fake) / 2

print("\n🎯 OVERALL METRICS:")
print(f"   Accuracy:           {accuracy*100:.2f}%")
print(f"   Weighted Precision: {weighted_precision*100:.2f}%")
print(f"   Weighted Recall:    {weighted_recall*100:.2f}%")
print(f"   Weighted F1-Score:  {weighted_f1*100:.2f}%")

print("\n📊 PER-CLASS METRICS:")
print(f"\n   REAL Class (label=1):")
print(f"      Precision: {precision_real*100:.2f}%")
print(f"      Recall:    {recall_real*100:.2f}%")
print(f"      F1-Score:  {f1_real*100:.2f}%")
print(f"      Support:   {total_real:,}")

print(f"\n   FAKE Class (label=0):")
print(f"      Precision: {precision_fake*100:.2f}%")
print(f"      Recall:    {recall_fake*100:.2f}%")
print(f"      F1-Score:  {f1_fake*100:.2f}%")
print(f"      Support:   {total_fake:,}")

# ============================================================================
# STEP 5: AUC-ROC (if probability available)
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 5: AUC-ROC Score")
print("="*80)

auc_score = None
if "prob_real" in df_predictions.columns:
    # Use BinaryClassificationEvaluator
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    
    # Need rawPrediction column for AUC
    df_for_auc = df_predictions.withColumn("rawPrediction", col("prob_real"))
    
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="prob_real",
        metricName="areaUnderROC"
    )
    
    try:
        auc_score = evaluator_auc.evaluate(df_predictions)
        print(f"   AUC-ROC Score: {auc_score:.4f} ({auc_score*100:.2f}%)")
    except Exception as e:
        print(f"   ⚠️ Could not calculate AUC: {e}")
        
    # Calculate AUC-PR
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="prob_real",
        metricName="areaUnderPR"
    )
    
    try:
        auc_pr = evaluator_pr.evaluate(df_predictions)
        print(f"   AUC-PR Score:  {auc_pr:.4f} ({auc_pr*100:.2f}%)")
    except Exception as e:
        print(f"   ⚠️ Could not calculate AUC-PR: {e}")
else:
    print("   ⚠️ Probability column not found, skipping AUC calculation")

# ============================================================================
# STEP 6: Error Analysis
# ============================================================================

print("\n" + "="*80)
print("🔍 STEP 6: Error Analysis")
print("="*80)

# False Positives Analysis (FAKE predicted as REAL)
print(f"\n❌ False Positives (FAKE→REAL): {fp:,}")
if "prob_real" in df_predictions.columns:
    fp_samples = df_predictions.filter(
        (col("label") == 0) & (col("prediction") == 1)
    ).select("prob_real")
    
    if fp > 0:
        fp_stats = fp_samples.agg({
            "prob_real": "avg",
        }).collect()[0]
        print(f"   Average confidence: {fp_stats[0]*100:.2f}%")

# False Negatives Analysis (REAL predicted as FAKE)
print(f"\n❌ False Negatives (REAL→FAKE): {fn:,}")
if "prob_real" in df_predictions.columns:
    fn_samples = df_predictions.filter(
        (col("label") == 1) & (col("prediction") == 0)
    ).select("prob_real")
    
    if fn > 0:
        fn_stats = fn_samples.agg({
            "prob_real": "avg",
        }).collect()[0]
        print(f"   Average confidence: {fn_stats[0]*100:.2f}%")

# ============================================================================
# STEP 7: Save Evaluation Report
# ============================================================================

print("\n" + "="*80)
print("💾 STEP 7: Saving Evaluation Report")
print("="*80)

# Create metrics summary
metrics_summary = [
    Row(metric="accuracy", value=float(accuracy)),
    Row(metric="weighted_precision", value=float(weighted_precision)),
    Row(metric="weighted_recall", value=float(weighted_recall)),
    Row(metric="weighted_f1", value=float(weighted_f1)),
    Row(metric="macro_precision", value=float(macro_precision)),
    Row(metric="macro_recall", value=float(macro_recall)),
    Row(metric="macro_f1", value=float(macro_f1)),
    Row(metric="precision_real", value=float(precision_real)),
    Row(metric="recall_real", value=float(recall_real)),
    Row(metric="f1_real", value=float(f1_real)),
    Row(metric="precision_fake", value=float(precision_fake)),
    Row(metric="recall_fake", value=float(recall_fake)),
    Row(metric="f1_fake", value=float(f1_fake)),
    Row(metric="true_positive", value=float(tp)),
    Row(metric="true_negative", value=float(tn)),
    Row(metric="false_positive", value=float(fp)),
    Row(metric="false_negative", value=float(fn)),
    Row(metric="total_samples", value=float(total_samples)),
]

if auc_score is not None:
    metrics_summary.append(Row(metric="auc_roc", value=float(auc_score)))

df_eval_metrics = spark.createDataFrame(metrics_summary)
eval_metrics_path = f"{RESULTS_BASE}/evaluation_metrics"
df_eval_metrics.write.mode("overwrite").parquet(eval_metrics_path)
print(f"✅ Evaluation metrics saved to: {eval_metrics_path}")

# Save confusion matrix
confusion_matrix = [
    Row(actual="FAKE", predicted="FAKE", count=int(tn)),
    Row(actual="FAKE", predicted="REAL", count=int(fp)),
    Row(actual="REAL", predicted="FAKE", count=int(fn)),
    Row(actual="REAL", predicted="REAL", count=int(tp)),
]
df_confusion = spark.createDataFrame(confusion_matrix)
confusion_path = f"{RESULTS_BASE}/confusion_matrix"
df_confusion.write.mode("overwrite").parquet(confusion_path)
print(f"✅ Confusion matrix saved to: {confusion_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

eval_time = time.time() - eval_start

print("\n" + "="*80)
print("🏁 EVALUATION COMPLETE")
print("="*80)

print(f"\n⏱️  Evaluation time: {eval_time:.1f}s")

print("\n" + "="*80)
print("📊 FINAL EVALUATION REPORT")
print("="*80)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    MODEL PERFORMANCE                        │
├─────────────────────────────────────────────────────────────┤
│  Test Samples:        {total_samples:>10,}                          │
├─────────────────────────────────────────────────────────────┤
│  Accuracy:            {accuracy*100:>10.2f}%                          │
│  Precision:           {weighted_precision*100:>10.2f}%                          │
│  Recall:              {weighted_recall*100:>10.2f}%                          │
│  F1-Score:            {weighted_f1*100:>10.2f}%                          │
│  AUC-ROC:             {(auc_score*100 if auc_score else 0):>10.2f}%                          │
├─────────────────────────────────────────────────────────────┤
│                    CONFUSION MATRIX                         │
├─────────────────────────────────────────────────────────────┤
│                        Predicted                            │
│                    FAKE        REAL                         │
│  Actual FAKE     {tn:>6,}      {fp:>6,}                         │
│         REAL     {fn:>6,}      {tp:>6,}                         │
├─────────────────────────────────────────────────────────────┤
│  ✅ Correct:      {tp+tn:>10,} ({(tp+tn)/total_samples*100:.1f}%)                     │
│  ❌ Errors:       {fp+fn:>10,} ({(fp+fn)/total_samples*100:.1f}%)                      │
└─────────────────────────────────────────────────────────────┘
""")

# Performance assessment
print("\n📈 ASSESSMENT:")
if accuracy >= 0.90:
    print("   🌟 EXCELLENT! Model achieves 90%+ accuracy!")
elif accuracy >= 0.85:
    print("   ✅ GOOD! Model performs well with 85%+ accuracy.")
elif accuracy >= 0.75:
    print("   ⚠️ MODERATE. Model may benefit from more training data or tuning.")
else:
    print("   ❌ NEEDS IMPROVEMENT. Consider feature engineering or different model.")

print("\n" + "="*80)
print("📁 Results saved to HDFS:")
print(f"   - Evaluation metrics: {eval_metrics_path}")
print(f"   - Confusion matrix: {confusion_path}")
print("="*80)

# Cleanup
df_predictions.unpersist()
spark.stop()
print("\n✅ Spark session stopped.")
