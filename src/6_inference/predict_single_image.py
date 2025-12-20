#!/usr/bin/env python3
"""Single Image Prediction Module.

Load model from HDFS and predict whether an image is REAL or FAKE.
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import io
import os


class DeepfakeDetector:
    """Deepfake detector using pre-trained model from HDFS."""
    
    def __init__(self, model_path="hdfs://namenode:8020/user/models/logistic_regression_tf", debug=False):
        """Initialize detector with model from HDFS.
        
        Args:
            model_path: Path to saved model on HDFS
            debug: If True, save intermediate images for inspection
        """
        self.model_path = model_path
        self.debug = debug
        self.spark = None
        self.lr_model = None
        self.feature_extractor = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Spark session and load models."""
        print("[INFO] Initializing Deepfake Detector...")
        
        # Initialize Spark
        self.spark = SparkSession.builder \
            .appName("Deepfake_Detector") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("ERROR")
        
        # Load Logistic Regression model from HDFS
        print(f"[INFO] Loading model from: {self.model_path}")
        try:
            self.lr_model = LogisticRegressionModel.load(self.model_path)
            print("[SUCCESS] Model loaded successfully")
            print(f"   - Features dimension: {len(self.lr_model.coefficients)}")
            print(f"   - Intercept: {self.lr_model.intercept:.6f}")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            raise
        
        # Load MobileNetV2 feature extractor
        print("[INFO] Loading MobileNetV2 feature extractor...")
        self.feature_extractor = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        print("[SUCCESS] Feature extractor loaded")
    
    def extract_features(self, image_path_or_bytes):
        """Extract 1280-dimensional features from image.
        
        IMPORTANT: Must match training preprocessing:
        - First resize: 32x32 (original training image size)
        - Second resize: 224x224 (MobileNetV2 input)
        - Interpolation: BILINEAR
        - Color mode: RGB
        - Preprocessing: MobileNetV2 preprocess_input
        
        Args:
            image_path_or_bytes: Path to image file or bytes
            
        Returns:
            np.ndarray: 1280-dimensional feature vector
        """
        # Load image
        if isinstance(image_path_or_bytes, (str, os.PathLike)):
            img = Image.open(image_path_or_bytes)
        elif isinstance(image_path_or_bytes, bytes):
            img = Image.open(io.BytesIO(image_path_or_bytes))
        else:
            img = image_path_or_bytes  # Assume PIL Image
        
        # Ensure RGB mode (convert RGBA, L, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Step 1: Resize to 32x32 (original training image size) using BILINEAR
        img = img.resize((32, 32), Image.BILINEAR)
        
        # Validate 32x32 size
        assert img.size == (32, 32), f"Image size must be 32x32, got {img.size}"
        
        # Debug: Save 32x32 image
        if self.debug:
            os.makedirs("debug_images", exist_ok=True)
            img.save("debug_images/01_resized_32x32.jpg")
            print("[DEBUG] Saved: debug_images/01_resized_32x32.jpg")
            print(f"   - Size: {img.size}, Mode: {img.mode}")
        
        # Step 2: Upscale to 224x224 (MobileNetV2 input) using BILINEAR
        img = img.resize((224, 224), Image.BILINEAR)
        
        # Validate final image properties
        assert img.size == (224, 224), f"Image size must be 224x224, got {img.size}"
        assert img.mode == 'RGB', f"Image mode must be RGB, got {img.mode}"
        
        # Debug: Save 224x224 image
        if self.debug:
            img.save("debug_images/02_resized_224x224.jpg")
            print("[DEBUG] Saved: debug_images/02_resized_224x224.jpg")
            print(f"   - Size: {img.size}, Mode: {img.mode}")
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Validate array shape
        assert img_array.shape == (224, 224, 3), f"Array shape must be (224,224,3), got {img_array.shape}"
        
        # Preprocess for MobileNetV2
        img_array = preprocess_input(img_array)
        
        # Debug: Print array stats before and after preprocessing
        if self.debug:
            print(f"[DEBUG] Array stats after preprocess_input:")
            print(f"   - Min: {img_array.min():.4f}, Max: {img_array.max():.4f}")
            print(f"   - Mean: {img_array.mean():.4f}, Std: {img_array.std():.4f}")
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Extract features
        features = self.feature_extractor.predict(img_array, verbose=0)
        
        # Flatten to 1D
        features = features.flatten()
        
        return features
    
    def predict(self, image_path_or_bytes):
        """Predict whether image is REAL or FAKE.
        
        Args:
            image_path_or_bytes: Path to image file or bytes
            
        Returns:
            dict: Prediction results with keys:
                - prediction: 'REAL' or 'FAKE'
                - confidence: float (0-1)
                - probability_real: float (0-1)
                - probability_fake: float (0-1)
                - features_shape: tuple
        """
        print("[INFO] Extracting features from image...")
        features = self.extract_features(image_path_or_bytes)
        
        print(f"   [INFO] Features extracted: {features.shape}")
        print(f"   [INFO] Feature stats: mean={features.mean():.4f}, std={features.std():.4f}")
        
        # Create Spark DataFrame
        spark_vector = Vectors.dense(features.tolist())
        
        schema = StructType([
            StructField("features", VectorUDT(), False)
        ])
        
        df = self.spark.createDataFrame([(spark_vector,)], schema)
        
        # Make prediction
        print("[INFO] Making prediction...")
        predictions = self.lr_model.transform(df)
        
        # Extract results
        result = predictions.select("prediction", "probability").collect()[0]
        
        prediction_label = int(result.prediction)
        probabilities = result.probability.toArray()
        
        prob_fake = float(probabilities[0])
        prob_real = float(probabilities[1])
        
        prediction_name = "REAL" if prediction_label == 1 else "FAKE"
        confidence = prob_real if prediction_label == 1 else prob_fake
        
        return {
            "prediction": prediction_name,
            "confidence": confidence,
            "probability_real": prob_real,
            "probability_fake": prob_fake,
            "features_shape": features.shape,
            "prediction_label": prediction_label
        }
    
    def predict_batch(self, image_paths):
        """Predict multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for i, img_path in enumerate(image_paths):
            print(f"[INFO] Processing image {i+1}/{len(image_paths)}: {img_path}")
            try:
                result = self.predict(img_path)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                print(f"   [ERROR] Error: {e}")
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        return results
    
    def close(self):
        """Close Spark session."""
        if self.spark:
            self.spark.stop()
            print("[INFO] Spark session closed")


def main():
    """Test function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_single_image.py <image_path>")
        print("Example: python predict_single_image.py test_image.jpg")
        return
    
    image_path = sys.argv[1]
    
    print("="*80)
    print("DEEPFAKE DETECTOR - SINGLE IMAGE PREDICTION")
    print("="*80)
    
    # Initialize detector
    detector = DeepfakeDetector()
    
    # Make prediction
    result = detector.predict(image_path)
    
    # Display results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    print(f"\nImage: {image_path}")
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"\nProbabilities:")
    print(f"   - REAL: {result['probability_real']*100:.2f}%")
    print(f"   - FAKE: {result['probability_fake']*100:.2f}%")
    print(f"\nFeatures: {result['features_shape']}")
    
    print("\n" + "="*80)
    if result['prediction'] == 'REAL':
        print("[RESULT] This image appears to be REAL")
    else:
        print("[RESULT] This image appears to be FAKE (Deepfake detected)")
    print("="*80)
    
    # Close
    detector.close()


if __name__ == "__main__":
    main()
