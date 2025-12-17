"""
Full Pipeline Runner - Deepfake Detection
Chạy toàn bộ pipeline từ Feature Extraction đến ML Training
"""

import subprocess
import sys
import time

def run_spark_submit(script_name, description):
    """
    Chạy PySpark script bằng spark-submit
    """
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)
    
    cmd = [
        "docker", "exec", "spark-master",
        "spark-submit",
        "--master", "spark://spark-master:7077",
        "--deploy-mode", "client",
        "--executor-memory", "2g",
        "--total-executor-cores", "4",
        f"/app/{script_name}"
    ]
    
    print(f"\n📝 Command: {' '.join(cmd)}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        duration = time.time() - start_time
        
        print(f"\n✅ {description} completed successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ {description} failed!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"Error: {e}")
        return False

def main():
    """
    Main pipeline execution
    """
    print("="*70)
    print("🎯 DEEPFAKE DETECTION PIPELINE - FULL EXECUTION")
    print("="*70)
    
    print("\n📋 Pipeline Steps:")
    print("  1. Feature Extraction (ResNet50 on Spark Workers)")
    print("  2. ML Training (Logistic Regression + Random Forest)")
    print("  3. Model Evaluation & Metrics Generation")
    
    total_start = time.time()
    
    # Step 1: Feature Extraction
    if not run_spark_submit("feature_extraction.py", "Step 1: Feature Extraction"):
        print("\n❌ Pipeline failed at Step 1")
        sys.exit(1)
    
    # Step 2: ML Training & Evaluation
    if not run_spark_submit("ml_training.py", "Step 2: ML Training & Evaluation"):
        print("\n❌ Pipeline failed at Step 2")
        sys.exit(1)
    
    # Pipeline completed
    total_duration = time.time() - total_start
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\n⏱️  Total execution time: {total_duration/60:.2f} minutes")
    
    print("\n📦 Output Locations (HDFS):")
    print("  ✓ Features:        hdfs://namenode:8020/user/data/features/")
    print("  ✓ Models:          hdfs://namenode:8020/user/data/models/")
    print("  ✓ Predictions:     hdfs://namenode:8020/user/data/results/")
    print("  ✓ Metrics:         hdfs://namenode:8020/user/data/results/metrics_summary/")
    
    print("\n🌐 Spark History Server:")
    print("  URL: http://localhost:18080")
    print("  📸 Please capture screenshots for your report!")
    
    print("\n📊 Next Steps:")
    print("  1. Access Spark History Server to view job execution")
    print("  2. Review metrics in metrics_summary Parquet file")
    print("  3. Prepare business insight report")
    print("  4. Capture evidence (screenshots, logs)")

if __name__ == "__main__":
    main()
