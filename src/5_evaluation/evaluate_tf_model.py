
#!/usr/bin/env python3
"""
Model Evaluation Script.

Script này thực hiện đánh giá mô hình: tính toán các chỉ số (accuracy, precision, recall, F1, AUC...), phân tích lỗi, lưu kết quả ra HDFS.
"""


# Import các thư viện Spark và Python cần thiết
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, when, count, sum as spark_sum
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time


# =============================
# BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH
# =============================
print("="*80)
print("MODEL EVALUATION")
print("="*80)


# Khởi tạo SparkSession với cấu hình bộ nhớ phù hợp
spark = SparkSession.builder \
    .appName("Model_Evaluation") \
    .config("spark.driver.memory", "2g") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://namenode:8020/spark-logs") \
    .config("spark.history.fs.logDirectory", "hdfs://namenode:8020/spark-logs") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
eval_start = time.time()  # Đánh dấu thời gian bắt đầu đánh giá


# Đường dẫn HDFS cho dữ liệu và kết quả
HDFS_BASE = "hdfs://namenode:8020/user/data"
RESULTS_BASE = f"{HDFS_BASE}/results_tf"


# =============================
# BƯỚC 1: Load dự đoán từ file parquet
# =============================
print("\n" + "="*80)
print("STEP 1: Loading Test Predictions")
print("="*80)

test_pred_path = f"{RESULTS_BASE}/test_predictions_tuned"
print(f"Loading from: {test_pred_path}")

# Đọc file parquet chứa dự đoán test
df_predictions = spark.read.parquet(test_pred_path)
df_predictions = df_predictions.cache()

total_samples = df_predictions.count()
print(f"Loaded {total_samples:,} test predictions")

print("\nSchema:")
df_predictions.printSchema()

print("\nSample predictions:")
df_predictions.show(10, truncate=False)


# =============================
# BƯỚC 2: Thống kê phân phối nhãn và dự đoán
# =============================
print("\n" + "="*80)
print("STEP 2: Calculating Metrics")
print("="*80)

label_dist = df_predictions.groupBy("label").count().collect()
pred_dist = df_predictions.groupBy("prediction").count().collect()

print("\nLabel Distribution (Ground Truth):")
for row in label_dist:
    label_name = "REAL" if row["label"] == 1 else "FAKE"
    pct = 100 * row["count"] / total_samples
    print(f"   {label_name} (label={int(row['label'])}): {row['count']:,} ({pct:.1f}%)")

print("\nPrediction Distribution:")
for row in pred_dist:
    pred_name = "REAL" if row["prediction"] == 1 else "FAKE"
    pct = 100 * row["count"] / total_samples
    print(f"   {pred_name} (pred={int(row['prediction'])}): {row['count']:,} ({pct:.1f}%)")


# =============================
# BƯỚC 3: Tính confusion matrix (TP, TN, FP, FN)
# =============================
print("\n" + "="*80)
print("STEP 3: Confusion Matrix")
print("="*80)

tp = df_predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()  # True Positive
tn = df_predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()  # True Negative
fp = df_predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()  # False Positive
fn = df_predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()  # False Negative

print("\n                    Predicted")
print("                  FAKE    REAL")
print(f"Actual  FAKE      {tn:,}    {fp:,}")
print(f"        REAL      {fn:,}    {tp:,}")

print(f"\nBreakdown:")
print(f"   True Positive (REAL->REAL):   {tp:,}")
print(f"   True Negative (FAKE->FAKE):   {tn:,}")
print(f"   False Positive (FAKE->REAL):  {fp:,}")
print(f"   False Negative (REAL->FAKE):  {fn:,}")


# =============================
# BƯỚC 4: Tính các chỉ số đánh giá (accuracy, precision, recall, F1, macro/weighted...)
# =============================
print("\n" + "="*80)
print("STEP 4: Performance Metrics")
print("="*80)

accuracy = (tp + tn) / total_samples
precision_real = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_real = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_real = 2 * precision_real * recall_real / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0

precision_fake = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_fake = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_fake = 2 * precision_fake * recall_fake / (precision_fake + recall_fake) if (precision_fake + recall_fake) > 0 else 0

total_real = tp + fn
total_fake = tn + fp
weighted_precision = (precision_real * total_real + precision_fake * total_fake) / total_samples
weighted_recall = (recall_real * total_real + recall_fake * total_fake) / total_samples
weighted_f1 = (f1_real * total_real + f1_fake * total_fake) / total_samples

macro_precision = (precision_real + precision_fake) / 2
macro_recall = (recall_real + recall_fake) / 2
macro_f1 = (f1_real + f1_fake) / 2

print("\nOVERALL METRICS:")
print(f"   Accuracy:           {accuracy*100:.2f}%")
print(f"   Weighted Precision: {weighted_precision*100:.2f}%")
print(f"   Weighted Recall:    {weighted_recall*100:.2f}%")
print(f"   Weighted F1-Score:  {weighted_f1*100:.2f}%")

print("\nPER-CLASS METRICS:")
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



# =============================
# BƯỚC 5: Tính AUC-ROC và AUC-PR nếu có xác suất dự đoán
# =============================
print("\n" + "="*80)
print("STEP 5: AUC-ROC Score")
print("="*80)

auc_score = None
if "prob_real" in df_predictions.columns:
    # Nếu có cột xác suất dự đoán, tính AUC-ROC và AUC-PR
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
        print(f"   Could not calculate AUC: {e}")

    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="prob_real",
        metricName="areaUnderPR"
    )

    try:
        auc_pr = evaluator_pr.evaluate(df_predictions)
        print(f"   AUC-PR Score:  {auc_pr:.4f} ({auc_pr*100:.2f}%)")
    except Exception as e:
        print(f"   Could not calculate AUC-PR: {e}")
else:
    print("   Probability column not found, skipping AUC calculation")


# =============================
# BƯỚC 6: Phân tích lỗi (Error Analysis)
# =============================
print("\n" + "="*80)
print("STEP 6: Error Analysis")
print("="*80)

print(f"\nFalse Positives (FAKE->REAL): {fp:,}")
if "prob_real" in df_predictions.columns:
    # Phân tích confidence của các mẫu FP
    fp_samples = df_predictions.filter(
        (col("label") == 0) & (col("prediction") == 1)
    ).select("prob_real")

    if fp > 0:
        fp_stats = fp_samples.agg({
            "prob_real": "avg",
        }).collect()[0]
        print(f"   Average confidence: {fp_stats[0]*100:.2f}%")

print(f"\nFalse Negatives (REAL->FAKE): {fn:,}")
if "prob_real" in df_predictions.columns:
    # Phân tích confidence của các mẫu FN
    fn_samples = df_predictions.filter(
        (col("label") == 1) & (col("prediction") == 0)
    ).select("prob_real")

    if fn > 0:
        fn_stats = fn_samples.agg({
            "prob_real": "avg",
        }).collect()[0]
        print(f"   Average confidence: {fn_stats[0]*100:.2f}%")


# =============================
# BƯỚC 7: Lưu kết quả đánh giá ra HDFS (parquet)
# =============================
print("\n" + "="*80)
print("STEP 7: Saving Evaluation Report")
print("="*80)

metrics_summary = [
    Row(metric="accuracy", value=float(accuracy)),
    Row(metric="weighted_precision", value=float(weighted_precision)),
    Row(metric="weighted_recall", value=float(weighted_recall)),
    Row(metric="weighted_f1", value=float(weighted_f1)),
    Row(metric="macro_precision", value=float(macro_precision)),
    Row(metric="macro_recall", value=float(macro_recall)),
    Row(metric="macro_f1", value=float(macro_f1)),
    Row(metric="tp", value=float(tp)),
    Row(metric="tn", value=float(tn)),
    Row(metric="fp", value=float(fp)),
    Row(metric="fn", value=float(fn)),
]

if auc_score is not None:
    metrics_summary.append(Row(metric="auc_roc", value=float(auc_score)))

df_metrics = spark.createDataFrame(metrics_summary)
metrics_path = f"{RESULTS_BASE}/evaluation_metrics"
df_metrics.write.mode("overwrite").parquet(metrics_path)
print(f"Metrics saved to: {metrics_path}")

spark.stop()

# =============================
# KẾT THÚC: In tổng kết và dừng Spark
# =============================
eval_time = time.time() - eval_start

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)

print(f"\nTotal samples evaluated: {total_samples:,}")
print(f"Evaluation time: {eval_time:.1f}s")

print(f"\nFinal Accuracy: {accuracy*100:.2f}%")
print(f"F1-Score: {weighted_f1*100:.2f}%")

print("\n" + "="*80)

spark.stop()
print("Spark session stopped.")
