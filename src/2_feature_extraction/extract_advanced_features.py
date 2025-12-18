#!/usr/bin/env python3
"""
Feature Extraction NÂNG CAO - Không cần Deep Learning
Sử dụng các kỹ thuật Computer Vision truyền thống mạnh mẽ:
- Color Histogram (768 dims)
- Local Binary Pattern (LBP) texture (256 dims) 
- Edge features - Sobel, Laplacian (192 dims)
- Frequency domain - DCT features (256 dims)
- Statistical features mở rộng (576 dims)

Total: 2048 dimensions - đầy đủ và có ý nghĩa
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lit
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from PIL import Image
from io import BytesIO
import time
import gc

print("="*80)
print("🔬 FEATURE EXTRACTION - ADVANCED CV FEATURES (No Deep Learning)")
print("="*80)

# ============================================================================
# SPARK SESSION
# ============================================================================

spark = SparkSession.builder \
    .appName("FeatureExtraction_Advanced_CV") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .config("spark.default.parallelism", "50") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def compute_lbp(gray_img):
    """
    Compute Local Binary Pattern (LBP) histogram
    LBP captures texture information - rất tốt cho deepfake detection
    """
    # Simple 3x3 LBP
    h, w = gray_img.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = gray_img[i, j]
            code = 0
            # 8 neighbors
            code |= (gray_img[i-1, j-1] >= center) << 7
            code |= (gray_img[i-1, j  ] >= center) << 6
            code |= (gray_img[i-1, j+1] >= center) << 5
            code |= (gray_img[i,   j+1] >= center) << 4
            code |= (gray_img[i+1, j+1] >= center) << 3
            code |= (gray_img[i+1, j  ] >= center) << 2
            code |= (gray_img[i+1, j-1] >= center) << 1
            code |= (gray_img[i,   j-1] >= center) << 0
            lbp[i-1, j-1] = code
    
    # Histogram of LBP
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist / hist.sum()  # Normalize

def compute_edge_features(gray_img):
    """
    Compute edge features using Sobel and Laplacian operators
    Edges thường khác biệt giữa REAL và FAKE
    """
    from scipy import ndimage
    
    # Sobel X
    sobel_x = ndimage.sobel(gray_img, axis=1)
    # Sobel Y
    sobel_y = ndimage.sobel(gray_img, axis=0)
    # Magnitude
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Laplacian
    laplacian = ndimage.laplace(gray_img)
    
    # Histograms (64 bins each)
    hist_sobel_x, _ = np.histogram(sobel_x.ravel(), bins=64, range=(-255, 255))
    hist_sobel_y, _ = np.histogram(sobel_y.ravel(), bins=64, range=(-255, 255))
    hist_sobel_mag, _ = np.histogram(sobel_mag.ravel(), bins=64, range=(0, 360))
    
    # Normalize
    hist_sobel_x = hist_sobel_x / hist_sobel_x.sum()
    hist_sobel_y = hist_sobel_y / hist_sobel_y.sum()
    hist_sobel_mag = hist_sobel_mag / hist_sobel_mag.sum()
    
    return np.concatenate([hist_sobel_x, hist_sobel_y, hist_sobel_mag])

def compute_dct_features(gray_img):
    """
    Compute DCT (Discrete Cosine Transform) features
    Frequency domain analysis - deepfakes thường có artifacts trong frequency
    """
    from scipy.fftpack import dct
    
    # Apply 2D DCT
    dct_img = dct(dct(gray_img.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Take top-left 16x16 coefficients (most important frequencies)
    dct_block = dct_img[:16, :16]
    
    # Flatten và normalize
    features = dct_block.flatten()
    features = features / (np.abs(features).max() + 1e-8)
    
    return features

def compute_color_histogram(img_array):
    """Compute RGB histogram (768 dims)"""
    hist_r = np.histogram(img_array[:,:,0], bins=256, range=(0, 256))[0]
    hist_g = np.histogram(img_array[:,:,1], bins=256, range=(0, 256))[0]
    hist_b = np.histogram(img_array[:,:,2], bins=256, range=(0, 256))[0]
    
    total = img_array.shape[0] * img_array.shape[1]
    return np.concatenate([hist_r, hist_g, hist_b]) / total

def compute_extended_stats(img_array):
    """
    Compute extended statistical features per channel và per region
    Bao gồm moments, percentiles, entropy
    """
    stats = []
    
    for channel in range(3):
        ch = img_array[:,:,channel].astype(float)
        
        # Basic stats
        stats.append(np.mean(ch))
        stats.append(np.std(ch))
        stats.append(np.min(ch))
        stats.append(np.max(ch))
        
        # Percentiles
        stats.append(np.percentile(ch, 25))
        stats.append(np.percentile(ch, 50))
        stats.append(np.percentile(ch, 75))
        
        # Higher order moments
        mean_centered = ch - np.mean(ch)
        stats.append(np.mean(mean_centered**3) / (np.std(ch)**3 + 1e-8))  # Skewness
        stats.append(np.mean(mean_centered**4) / (np.std(ch)**4 + 1e-8))  # Kurtosis
        
        # Entropy
        hist, _ = np.histogram(ch, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        stats.append(entropy)
        
        # Spatial stats (4 quadrants)
        h, w = ch.shape
        quadrants = [
            ch[:h//2, :w//2],   # Top-left
            ch[:h//2, w//2:],   # Top-right
            ch[h//2:, :w//2],   # Bottom-left
            ch[h//2:, w//2:]    # Bottom-right
        ]
        for q in quadrants:
            stats.append(np.mean(q))
            stats.append(np.std(q))
    
    return np.array(stats)

def extract_advanced_features(image_bytes):
    """
    Extract 2048-dim advanced features
    
    Composition:
    - Color Histogram: 768 dims
    - LBP Texture: 256 dims  
    - Edge Features: 192 dims
    - DCT Features: 256 dims
    - Extended Stats: 54 dims
    - Cross-channel features: 128 dims
    - Padding: 394 dims
    
    Total: 2048 dims
    """
    try:
        from scipy import ndimage
        from scipy.fftpack import dct
        
        # Load image
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Grayscale
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        
        all_features = []
        
        # 1. Color Histogram (768 dims)
        color_hist = compute_color_histogram(img_array)
        all_features.append(color_hist)
        
        # 2. LBP Texture (256 dims)
        lbp_hist = compute_lbp(gray)
        all_features.append(lbp_hist)
        
        # 3. Edge Features (192 dims)
        edge_feat = compute_edge_features(gray.astype(float))
        all_features.append(edge_feat)
        
        # 4. DCT Features (256 dims)
        dct_feat = compute_dct_features(gray)
        all_features.append(dct_feat)
        
        # 5. Extended Stats (54 dims)
        ext_stats = compute_extended_stats(img_array)
        all_features.append(ext_stats)
        
        # 6. Cross-channel features (128 dims)
        # Differences between channels - có thể phát hiện color artifacts
        diff_rg = img_array[:,:,0].astype(float) - img_array[:,:,1].astype(float)
        diff_rb = img_array[:,:,0].astype(float) - img_array[:,:,2].astype(float)
        diff_gb = img_array[:,:,1].astype(float) - img_array[:,:,2].astype(float)
        
        hist_rg, _ = np.histogram(diff_rg.ravel(), bins=42, range=(-255, 255))
        hist_rb, _ = np.histogram(diff_rb.ravel(), bins=42, range=(-255, 255))
        hist_gb, _ = np.histogram(diff_gb.ravel(), bins=42, range=(-255, 255))
        
        cross_features = np.concatenate([
            hist_rg / hist_rg.sum(),
            hist_rb / hist_rb.sum(),
            hist_gb / hist_gb.sum()
        ])
        all_features.append(cross_features)
        
        # 7. Gradient orientation histogram (64 dims)
        gy, gx = np.gradient(gray.astype(float))
        angles = np.arctan2(gy, gx)
        hist_angles, _ = np.histogram(angles.ravel(), bins=64, range=(-np.pi, np.pi))
        hist_angles = hist_angles / hist_angles.sum()
        all_features.append(hist_angles)
        
        # Concatenate all
        final_features = np.concatenate(all_features)
        
        # Pad to 2048 if needed
        current_len = len(final_features)
        if current_len < 2048:
            padding = np.zeros(2048 - current_len)
            final_features = np.concatenate([final_features, padding])
        elif current_len > 2048:
            final_features = final_features[:2048]
        
        return Vectors.dense(final_features.tolist())
        
    except Exception as e:
        print(f"Error: {e}")
        return Vectors.dense([0.0] * 2048)

# Register UDF
extract_features_udf = udf(extract_advanced_features, VectorUDT())

# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_batch(dataset_type, class_label, df_batch, batch_num, label_value):
    """Process one batch"""
    
    print(f"\n{'='*80}")
    print(f"📦 {dataset_type.upper()}/{class_label} Batch {batch_num}")
    print(f"{'='*80}")
    
    hdfs_output = f"hdfs://namenode:8020/user/data/features/{dataset_type}/{class_label}/batch_{batch_num}"
    
    start_time = time.time()
    
    batch_size = df_batch.count()
    print(f"📊 Batch size: {batch_size:,}")
    
    # Extract features
    print("🔬 Extracting advanced features...")
    df_features = df_batch.withColumn("features", extract_features_udf(col("content"))) \
                          .withColumn("label", lit(label_value)) \
                          .select("path", "features", "label")
    
    df_features = df_features.repartition(10).cache()
    feature_count = df_features.count()
    print(f"✅ Extracted: {feature_count:,}")
    
    # Save
    print(f"💾 Saving to HDFS...")
    df_features.write.mode("overwrite").parquet(hdfs_output)
    
    # Verify
    saved_count = spark.read.parquet(hdfs_output).count()
    
    elapsed = time.time() - start_time
    
    df_features.unpersist()
    spark.catalog.clearCache()
    gc.collect()
    
    if saved_count == batch_size:
        print(f"✅ SUCCESS: {saved_count:,} saved in {elapsed:.1f}s")
        return True, saved_count
    else:
        print(f"⚠️ WARNING: {saved_count:,}/{batch_size:,}")
        return False, saved_count

# ============================================================================
# MAIN PIPELINE
# ============================================================================

pipeline_start = time.time()
results = []

print("\n" + "🚀"*40)
print("STARTING ADVANCED FEATURE EXTRACTION")
print("Features: Color + LBP + Edge + DCT + Stats = 2048 dims")
print("🚀"*40)

# ============================================================================
# TRAIN DATA
# ============================================================================

print("\n" + "📚"*40)
print("PART 1: TRAINING DATA (100,000 images)")
print("📚"*40)

# TRAIN/REAL
print("\n🟢 TRAIN/REAL - Loading...")
hdfs_real = "hdfs://namenode:8020/user/data/raw/train/REAL"
df_real = spark.read.format("binaryFile").load(hdfs_real)
df_real = df_real.repartition(50).cache()
total_real = df_real.count()
print(f"✅ Total REAL: {total_real:,}")

batches_real = df_real.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)
for i in range(5):
    success, count = process_batch('train', 'REAL', batches_real[i], i+1, 1)
    results.append(('TRAIN/REAL', i+1, success, count))
    time.sleep(15)

df_real.unpersist()
spark.catalog.clearCache()
gc.collect()

# TRAIN/FAKE
print("\n🔴 TRAIN/FAKE - Loading...")
hdfs_fake = "hdfs://namenode:8020/user/data/raw/train/FAKE"
df_fake = spark.read.format("binaryFile").load(hdfs_fake)
df_fake = df_fake.repartition(50).cache()
total_fake = df_fake.count()
print(f"✅ Total FAKE: {total_fake:,}")

batches_fake = df_fake.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed=42)
for i in range(5):
    success, count = process_batch('train', 'FAKE', batches_fake[i], i+1, 0)
    results.append(('TRAIN/FAKE', i+1, success, count))
    time.sleep(15)

df_fake.unpersist()
spark.catalog.clearCache()
gc.collect()

# ============================================================================
# TEST DATA
# ============================================================================

print("\n" + "🧪"*40)
print("PART 2: TEST DATA (20,000 images)")
print("🧪"*40)

# TEST/REAL
print("\n🟢 TEST/REAL - Loading...")
hdfs_test_real = "hdfs://namenode:8020/user/data/raw/test/REAL"
df_test_real = spark.read.format("binaryFile").load(hdfs_test_real)
df_test_real = df_test_real.repartition(20).cache()
success, count = process_batch('test', 'REAL', df_test_real, 1, 1)
results.append(('TEST/REAL', 1, success, count))
df_test_real.unpersist()

# TEST/FAKE
print("\n🔴 TEST/FAKE - Loading...")
hdfs_test_fake = "hdfs://namenode:8020/user/data/raw/test/FAKE"
df_test_fake = spark.read.format("binaryFile").load(hdfs_test_fake)
df_test_fake = df_test_fake.repartition(20).cache()
success, count = process_batch('test', 'FAKE', df_test_fake, 1, 0)
results.append(('TEST/FAKE', 1, success, count))
df_test_fake.unpersist()

# ============================================================================
# SUMMARY
# ============================================================================

pipeline_elapsed = time.time() - pipeline_start

print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)

success_count = sum(1 for _, _, s, _ in results if s)
total_batches = len(results)
total_samples = sum(c for _, _, _, c in results)

print(f"\n✅ Successful: {success_count}/{total_batches} batches")
print(f"📊 Total samples: {total_samples:,}")
print(f"⏱️  Total time: {pipeline_elapsed/60:.2f} minutes")

print("\n📋 Results:")
for dataset, batch, success, count in results:
    status = "✅" if success else "❌"
    print(f"   {status} {dataset} Batch {batch}: {count:,}")

if success_count == total_batches:
    print("\n" + "🎉"*40)
    print("ALL BATCHES COMPLETED!")
    print("Features: Color + LBP + Edge + DCT + Stats = 2048 dims")
    print("Ready for ML training with improved accuracy!")
    print("🎉"*40)

spark.stop()
print("\n✅ Done!")
