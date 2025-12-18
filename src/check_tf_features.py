"""
Script to verify TensorFlow MobileNetV2 features are NOT zeros
"""
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
import numpy as np

def main():
    spark = SparkSession.builder \
        .appName("Check TF Features") \
        .master("local[1]") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("=" * 60)
    print("KIỂM TRA FEATURES TỪ TENSORFLOW MOBILENETV2")
    print("=" * 60)
    
    # Check one batch from each category
    categories = [
        ("train/REAL/batch_1", "Train REAL"),
        ("train/FAKE/batch_1", "Train FAKE"),
        ("test/REAL/batch_1", "Test REAL"),
        ("test/FAKE/batch_1", "Test FAKE")
    ]
    
    base_path = "hdfs://namenode:8020/user/data/features_tf"
    
    total_checked = 0
    total_non_zero = 0
    
    for path_suffix, name in categories:
        full_path = f"{base_path}/{path_suffix}"
        print(f"\n📁 Checking {name}:")
        print(f"   Path: {full_path}")
        
        try:
            df = spark.read.parquet(full_path)
            count = df.count()
            print(f"   Records: {count}")
            
            # Check sample features
            sample_rows = df.take(5)
            non_zero_count = 0
            
            for i, row in enumerate(sample_rows):
                features = row.features.toArray()
                feature_sum = np.sum(np.abs(features))
                feature_mean = np.mean(features)
                feature_std = np.std(features)
                feature_max = np.max(features)
                feature_min = np.min(features)
                
                is_zero = feature_sum < 1e-6
                
                if not is_zero:
                    non_zero_count += 1
                
                status = "❌ ZERO!" if is_zero else "✅ OK"
                print(f"   Sample {i+1}: sum={feature_sum:.4f}, mean={feature_mean:.4f}, std={feature_std:.4f}, min={feature_min:.4f}, max={feature_max:.4f} {status}")
            
            total_checked += len(sample_rows)
            total_non_zero += non_zero_count
            
            print(f"   Non-zero features: {non_zero_count}/{len(sample_rows)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("TỔNG KẾT:")
    print(f"  - Tổng mẫu kiểm tra: {total_checked}")
    print(f"  - Mẫu có features thực: {total_non_zero}")
    print(f"  - Tỷ lệ: {100*total_non_zero/total_checked:.1f}%")
    
    if total_non_zero == total_checked:
        print("\n✅ TẤT CẢ FEATURES ĐỀU CÓ GIÁ TRỊ THỰC - SẴN SÀNG TRAINING!")
    else:
        print(f"\n⚠️ CÓ {total_checked - total_non_zero} MẪU BỊ ZERO!")
    
    print("=" * 60)
    
    spark.stop()

if __name__ == "__main__":
    main()
